"""
Activation Segmentation → Colored 3D Meshes (V2 — Scaled Up)
=============================================================
Changes from V1:
  - DeepSDF: 256 hidden, 8 layers, skip connection at layer 4
  - Default resolution 256 marching cubes
  - More SDF samples (200K) with tighter surface band
  - More epochs (3000) for complex meshes
  - Eikonal regularization option for sharper surfaces
  - Multi-resolution sampling during training

Two modes:
  1. Analytical shapes:
     python activation_mesh_v2.py --shape snowman
     python activation_mesh_v2.py --shape all

  2. Your own OBJ file:
     python activation_mesh_v2.py --mesh_path railing.obj --n_parts 3
     python activation_mesh_v2.py --mesh_path railing.obj --n_parts 10 --epochs 5000 --resolution 256

  3. Quick mode (use v1 settings for fast iteration):
     python activation_mesh_v2.py --mesh_path model.obj --n_parts 3 --quick

Author: Aditya Jain | April 2026
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import os
import time
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from itertools import permutations
from skimage import measure
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# SDF PRIMITIVES (unchanged)
# ============================================================

def sdf_sphere(points, center=np.array([0,0,0]), radius=0.5):
    return np.linalg.norm(points - center, axis=1) - radius

def sdf_box(points, center=np.array([0,0,0]), half_extents=np.array([0.4, 0.3, 0.25])):
    d = np.abs(points - center) - half_extents
    outside = np.linalg.norm(np.maximum(d, 0), axis=1)
    inside = np.minimum(np.maximum(d[:,0], np.maximum(d[:,1], d[:,2])), 0)
    return outside + inside

def sdf_cylinder(points, center=np.array([0,0,0]), radius=0.15, half_height=0.5, axis='y'):
    p = points - center
    if axis == 'y':
        d_radial = np.sqrt(p[:,0]**2 + p[:,2]**2) - radius
        d_height = np.abs(p[:,1]) - half_height
    elif axis == 'x':
        d_radial = np.sqrt(p[:,1]**2 + p[:,2]**2) - radius
        d_height = np.abs(p[:,0]) - half_height
    elif axis == 'z':
        d_radial = np.sqrt(p[:,0]**2 + p[:,1]**2) - radius
        d_height = np.abs(p[:,2]) - half_height
    outside = np.sqrt(np.maximum(d_radial, 0)**2 + np.maximum(d_height, 0)**2)
    inside = np.minimum(np.maximum(d_radial, d_height), 0)
    return outside + inside


# ============================================================
# COMPOUND SHAPES (unchanged)
# ============================================================

def sdf_snowman(pts):
    return np.minimum(sdf_sphere(pts, [0,-0.15,0], 0.35), sdf_sphere(pts, [0,0.35,0], 0.2))
def parts_snowman():
    return [lambda p: sdf_sphere(p, [0,-0.15,0], 0.35), lambda p: sdf_sphere(p, [0,0.35,0], 0.2)]

def sdf_lollipop(pts):
    return np.minimum(sdf_sphere(pts, [0,0.3,0], 0.25), sdf_cylinder(pts, [0,-0.2,0], 0.04, 0.3, 'y'))
def parts_lollipop():
    return [lambda p: sdf_sphere(p, [0,0.3,0], 0.25), lambda p: sdf_cylinder(p, [0,-0.2,0], 0.04, 0.3, 'y')]

def sdf_barbell(pts):
    return np.minimum(np.minimum(sdf_sphere(pts, [-0.4,0,0], 0.2), sdf_sphere(pts, [0.4,0,0], 0.2)),
                      sdf_cylinder(pts, [0,0,0], 0.06, 0.4, 'x'))
def parts_barbell():
    return [lambda p: sdf_sphere(p, [-0.4,0,0], 0.2), lambda p: sdf_sphere(p, [0.4,0,0], 0.2),
            lambda p: sdf_cylinder(p, [0,0,0], 0.06, 0.4, 'x')]

def sdf_mushroom(pts):
    return np.minimum(sdf_sphere(pts, [0,0.15,0], 0.3), sdf_cylinder(pts, [0,-0.2,0], 0.08, 0.25, 'y'))
def parts_mushroom():
    return [lambda p: sdf_sphere(p, [0,0.15,0], 0.3), lambda p: sdf_cylinder(p, [0,-0.2,0], 0.08, 0.25, 'y')]

def sdf_tower(pts):
    return np.minimum(np.minimum(sdf_sphere(pts, [0,-0.35,0], 0.25), sdf_sphere(pts, [0,0.05,0], 0.2)),
                      sdf_sphere(pts, [0,0.35,0], 0.15))
def parts_tower():
    return [lambda p: sdf_sphere(p, [0,-0.35,0], 0.25), lambda p: sdf_sphere(p, [0,0.05,0], 0.2),
            lambda p: sdf_sphere(p, [0,0.35,0], 0.15)]

def sdf_table(pts):
    top = sdf_box(pts, [0,0.3,0], [0.4,0.03,0.3])
    legs = [sdf_cylinder(pts, [-0.3,0,-0.2], 0.03, 0.27, 'y'),
            sdf_cylinder(pts, [0.3,0,-0.2], 0.03, 0.27, 'y'),
            sdf_cylinder(pts, [-0.3,0,0.2], 0.03, 0.27, 'y'),
            sdf_cylinder(pts, [0.3,0,0.2], 0.03, 0.27, 'y')]
    r = top
    for l in legs: r = np.minimum(r, l)
    return r
def parts_table():
    return [lambda p: sdf_box(p, [0,0.3,0], [0.4,0.03,0.3]),
            lambda p: sdf_cylinder(p, [-0.3,0,-0.2], 0.03, 0.27, 'y'),
            lambda p: sdf_cylinder(p, [0.3,0,-0.2], 0.03, 0.27, 'y'),
            lambda p: sdf_cylinder(p, [-0.3,0,0.2], 0.03, 0.27, 'y'),
            lambda p: sdf_cylinder(p, [0.3,0,0.2], 0.03, 0.27, 'y')]

def sdf_tpipe(pts):
    return np.minimum(sdf_cylinder(pts, [0,0,0], 0.1, 0.4, 'y'),
                      sdf_cylinder(pts, [0,0.15,0], 0.1, 0.35, 'x'))
def parts_tpipe():
    return [lambda p: sdf_cylinder(p, [0,0,0], 0.1, 0.4, 'y'),
            lambda p: sdf_cylinder(p, [0,0.15,0], 0.1, 0.35, 'x')]

def sdf_l_shape(pts):
    return np.minimum(sdf_box(pts, [-0.15,0,0], [0.35,0.15,0.2]),
                      sdf_box(pts, [0.15,0.25,0], [0.15,0.25,0.2]))
def parts_l_shape():
    return [lambda p: sdf_box(p, [-0.15,0,0], [0.35,0.15,0.2]),
            lambda p: sdf_box(p, [0.15,0.25,0], [0.15,0.25,0.2])]

def sdf_chair(pts):
    seat = sdf_box(pts, [0,0,0], [0.3,0.03,0.25])
    back = sdf_box(pts, [0,0.25,-0.22], [0.3,0.22,0.02])
    legs = [sdf_cylinder(pts, [-0.25,-0.25,-0.2], 0.025, 0.22, 'y'),
            sdf_cylinder(pts, [0.25,-0.25,-0.2], 0.025, 0.22, 'y'),
            sdf_cylinder(pts, [-0.25,-0.25,0.2], 0.025, 0.22, 'y'),
            sdf_cylinder(pts, [0.25,-0.25,0.2], 0.025, 0.22, 'y')]
    r = seat
    for p in [back] + legs: r = np.minimum(r, p)
    return r
def parts_chair():
    return [lambda p: sdf_box(p, [0,0,0], [0.3,0.03,0.25]),
            lambda p: sdf_box(p, [0,0.25,-0.22], [0.3,0.22,0.02]),
            lambda p: sdf_cylinder(p, [-0.25,-0.25,-0.2], 0.025, 0.22, 'y'),
            lambda p: sdf_cylinder(p, [0.25,-0.25,-0.2], 0.025, 0.22, 'y'),
            lambda p: sdf_cylinder(p, [-0.25,-0.25,0.2], 0.025, 0.22, 'y'),
            lambda p: sdf_cylinder(p, [0.25,-0.25,0.2], 0.025, 0.22, 'y')]

SHAPES = {
    'snowman': (sdf_snowman, parts_snowman, 2),
    'lollipop': (sdf_lollipop, parts_lollipop, 2),
    'barbell': (sdf_barbell, parts_barbell, 3),
    'mushroom': (sdf_mushroom, parts_mushroom, 2),
    'tower': (sdf_tower, parts_tower, 3),
    'table': (sdf_table, parts_table, 5),
    'tpipe': (sdf_tpipe, parts_tpipe, 2),
    'l_shape': (sdf_l_shape, parts_l_shape, 2),
    'chair': (sdf_chair, parts_chair, 6),
}

def get_gt_labels(parts_fns, points):
    sdfs = np.stack([fn(points) for fn in parts_fns], axis=1)
    return np.argmin(sdfs, axis=1)


# ============================================================
# MESH SDF (for user OBJ files) — unchanged
# ============================================================

class MeshSDF:
    def __init__(self, mesh_path):
        import trimesh
        print(f"\n  Loading mesh: {mesh_path}")
        self.mesh = trimesh.load(mesh_path, force='mesh')
        print(f"  Verts: {len(self.mesh.vertices):,} | Faces: {len(self.mesh.faces):,}")
        centroid = self.mesh.centroid
        self.mesh.vertices -= centroid
        scale = np.max(np.abs(self.mesh.vertices))
        if scale > 0:
            self.mesh.vertices = self.mesh.vertices / scale * 0.45

        self._backend = None
        try:
            from pysdf import SDF
            self._sdf_fn = SDF(self.mesh.vertices, self.mesh.faces)
            self._backend = 'pysdf'
            far = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
            self._sign_flip = -1.0 if self._sdf_fn(far)[0] > 0 else 1.0
            print(f"  Backend: pysdf")
        except ImportError:
            self._backend = 'trimesh'
            if not self.mesh.is_watertight:
                self.mesh.fill_holes(); self.mesh.fix_normals()
            import trimesh.proximity
            self._proximity = trimesh.proximity.ProximityQuery(self.mesh)
            print(f"  Backend: trimesh")

    def __call__(self, points):
        points = np.asarray(points, dtype=np.float32)
        if self._backend == 'pysdf':
            n = len(points)
            if n > 500000:
                r = []
                for i in range(0, n, 500000):
                    r.append(self._sdf_fn(points[i:i+500000]))
                return (self._sign_flip * np.concatenate(r)).astype(np.float32)
            return (self._sign_flip * self._sdf_fn(points)).astype(np.float32)
        else:
            p64 = points.astype(np.float64)
            closest, dists, fids = self._proximity.on_surface(p64)
            normals = self.mesh.face_normals[fids]
            dots = np.sum((p64 - closest) * normals, axis=1)
            signs = np.where(dots < 0, -1.0, 1.0)
            return (signs * dists).astype(np.float32)


# ============================================================
# DeepSDF NETWORK V2 — SCALED UP
# ============================================================

class DeepSDFNetworkV2(nn.Module):
    """
    Bigger DeepSDF with skip connection.
    
    Architecture (default):
      Input: PE(xyz) + latent = 39 + 128 = 167 dim
      Layer 0: Linear(167, 256) + ReLU   → 256-dim activations
      Layer 1: Linear(256, 256) + ReLU   → 256-dim activations
      Layer 2: Linear(256, 256) + ReLU   → 256-dim activations
      Layer 3: Linear(256, 256) + ReLU   → 256-dim activations  [skip target]
      Layer 4: Linear(256+167, 256) + ReLU → 256-dim activations [skip input re-injected]
      Layer 5: Linear(256, 256) + ReLU   → 256-dim activations
      Layer 6: Linear(256, 256) + ReLU   → 256-dim activations
      Output:  Linear(256, 1)            → scalar SDF
      
    Total activations per point: 256 * 7 = 1792 dim (all_layers)
    Parameters: ~660K (vs 46K in v1)
    
    The skip connection at the midpoint re-injects the input (PE + latent),
    which is standard DeepSDF practice. It helps the later layers maintain
    high-frequency spatial information that would otherwise be smoothed out
    by the early layers.
    """
    def __init__(self, latent_dim=128, hidden_dim=256, num_layers=8,
                 num_frequencies=6, skip_at=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_frequencies = num_frequencies
        self.num_layers = num_layers
        self.skip_at = skip_at

        xyz_dim = 3 + 3 * 2 * num_frequencies  # 3 + 36 = 39
        self.input_dim = xyz_dim + latent_dim

        self.layers = nn.ModuleList()
        # First layer
        self.layers.append(nn.Linear(self.input_dim, hidden_dim))
        # Hidden layers
        for i in range(1, num_layers - 1):
            if i == skip_at:
                # Skip connection: concatenate input again
                self.layers.append(nn.Linear(hidden_dim + self.input_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, 1))

        self.relu = nn.ReLU()
        self._activations = {}

        # Weight initialization (important for deeper networks)
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
        # Output layer: small init for stable SDF values
        nn.init.normal_(self.layers[-1].weight, 0.0, 0.01)

    def positional_encoding(self, x):
        enc = [x]
        for f in range(self.num_frequencies):
            s = (2.0 ** f) * np.pi
            enc.append(torch.sin(s * x)); enc.append(torch.cos(s * x))
        return torch.cat(enc, dim=-1)

    def forward(self, xyz, latent, store_activations=False):
        pe = self.positional_encoding(xyz)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0).expand(xyz.shape[0], -1)
        inp = torch.cat([pe, latent], dim=-1)

        if store_activations:
            self._activations = {}

        x = inp
        for i, layer in enumerate(self.layers[:-1]):
            if i == self.skip_at:
                x = torch.cat([x, inp], dim=-1)
            x = self.relu(layer(x))
            if store_activations:
                self._activations[f'layer_{i}'] = x.detach()

        out = self.layers[-1](x)
        if store_activations:
            self._activations['pre_output'] = x.detach()
        return out

    def get_activations(self):
        return self._activations


# Keep V1 for --quick mode and analytical shapes
class DeepSDFNetworkV1(nn.Module):
    """Original small network: 128 hidden, 4 layers, no skip."""
    def __init__(self, latent_dim=64, hidden_dim=128, num_layers=4, num_frequencies=6):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        xyz_dim = 3 + 3 * 2 * num_frequencies
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(xyz_dim + latent_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, 1))
        self.relu = nn.ReLU()
        self._activations = {}

    def positional_encoding(self, x):
        enc = [x]
        for f in range(self.num_frequencies):
            s = (2.0 ** f) * np.pi
            enc.append(torch.sin(s * x)); enc.append(torch.cos(s * x))
        return torch.cat(enc, dim=-1)

    def forward(self, xyz, latent, store_activations=False):
        pe = self.positional_encoding(xyz)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0).expand(xyz.shape[0], -1)
        x = torch.cat([pe, latent], dim=-1)
        if store_activations: self._activations = {}
        for i, layer in enumerate(self.layers[:-1]):
            x = self.relu(layer(x))
            if store_activations:
                self._activations[f'layer_{i}'] = x.detach()
        out = self.layers[-1](x)
        if store_activations:
            self._activations['pre_output'] = x.detach()
        return out

    def get_activations(self):
        return self._activations


# ============================================================
# SAMPLING V2 — Multi-resolution surface sampling
# ============================================================

def sample_points_v2(n_points, sdf_func, surface_ratio=0.7):
    """
    Improved sampling with multi-band surface focus.
    - 30% uniform volume samples (learn far-field)
    - 40% tight near-surface (within 0.005 of surface)
    - 20% medium near-surface (within 0.02 of surface)
    - 10% very close (within 0.002 — for thin features)
    """
    n_uniform = int(n_points * 0.3)
    n_tight = int(n_points * 0.4)
    n_medium = int(n_points * 0.2)
    n_close = n_points - n_uniform - n_tight - n_medium

    # Uniform volume
    up = np.random.uniform(-0.55, 0.55, (n_uniform, 3)).astype(np.float32)
    ud = sdf_func(up)

    # Generate candidates and sort by distance to surface
    n_cand = n_points * 5
    cand = np.random.uniform(-0.55, 0.55, (n_cand, 3)).astype(np.float32)
    cd = sdf_func(cand)
    abs_cd = np.abs(cd)
    sorted_idx = np.argsort(abs_cd)

    # Tight band: closest to surface + small perturbation
    tight_idx = sorted_idx[:n_tight]
    tight_pts = cand[tight_idx] + np.random.normal(0, 0.005, (n_tight, 3)).astype(np.float32)
    tight_d = sdf_func(tight_pts)

    # Medium band
    med_idx = sorted_idx[:n_medium * 2]
    med_sel = np.random.choice(len(med_idx), n_medium, replace=False)
    med_pts = cand[med_idx[med_sel]] + np.random.normal(0, 0.02, (n_medium, 3)).astype(np.float32)
    med_d = sdf_func(med_pts)

    # Very close band (for thin features like baluster gaps)
    close_idx = sorted_idx[:n_close * 2]
    close_sel = np.random.choice(len(close_idx), n_close, replace=len(close_idx) < n_close)
    close_pts = cand[close_idx[close_sel]] + np.random.normal(0, 0.002, (n_close, 3)).astype(np.float32)
    close_d = sdf_func(close_pts)

    all_pts = np.concatenate([up, tight_pts, med_pts, close_pts]).astype(np.float32)
    all_d = np.concatenate([ud, tight_d, med_d, close_d]).astype(np.float32)
    return all_pts, all_d


def sample_points_v1(n_points, sdf_func, surface_ratio=0.6):
    """Original sampling (for --quick mode)."""
    n_s = int(n_points * surface_ratio)
    n_u = n_points - n_s
    up = np.random.uniform(-1, 1, (n_u, 3)).astype(np.float32)
    ud = sdf_func(up)
    cand = np.random.uniform(-1, 1, (n_s * 3, 3)).astype(np.float32)
    cd = sdf_func(cand)
    near = np.argsort(np.abs(cd))[:n_s]
    sp = cand[near] + np.random.normal(0, 0.01, (n_s, 3)).astype(np.float32)
    sd = sdf_func(sp)
    return np.concatenate([up, sp]).astype(np.float32), np.concatenate([ud, sd]).astype(np.float32)


# ============================================================
# TRAINING V2
# ============================================================

def train_deepsdf(sdf_funcs, device, latent_dim=128, hidden_dim=256, num_layers=8,
                  epochs=3000, n_points=200000, batch_size=16384,
                  use_v1=False, use_eikonal=False):
    """
    Train shared decoder on one or more SDF functions.
    
    V2 defaults: latent=128, hidden=256, 8 layers, 3000 epochs, 200K points
    V1 (--quick): latent=64, hidden=128, 4 layers, 1500 epochs, 100K points
    """
    n_shapes = len(sdf_funcs)

    if use_v1:
        model = DeepSDFNetworkV1(latent_dim=latent_dim, hidden_dim=128, num_layers=4).to(device)
        sample_fn = sample_points_v1
    else:
        model = DeepSDFNetworkV2(latent_dim=latent_dim, hidden_dim=hidden_dim,
                                  num_layers=num_layers).to(device)
        sample_fn = sample_points_v2

    latent_codes = nn.Embedding(n_shapes, latent_dim).to(device)
    nn.init.normal_(latent_codes.weight, 0.0, 0.01)

    opt = torch.optim.Adam([
        {'params': model.parameters(), 'lr': 5e-4, 'weight_decay': 1e-6},
        {'params': latent_codes.parameters(), 'lr': 5e-4},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  DeepSDF {'V1' if use_v1 else 'V2'}: {n_params:,} params | "
          f"{n_shapes} shapes | latent={latent_dim} | hidden={hidden_dim if not use_v1 else 128}")

    # Pre-sample
    all_pts, all_gt = {}, {}
    for i, fn in enumerate(sdf_funcs):
        p, d = sample_fn(n_points, fn)
        all_pts[i] = torch.tensor(p, dtype=torch.float32, device=device)
        all_gt[i] = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(-1)

    # SDF clamping value (clamp distant values to focus on near-surface)
    clamp_val = 0.1

    print(f"  Training for {epochs} epochs, {n_points:,} points/shape, batch={batch_size}...")
    best = float('inf')
    t0 = time.time()

    resample_interval = 500 if not use_v1 else 300

    for ep in range(epochs):
        # Resample periodically
        if ep > 0 and ep % resample_interval == 0:
            for i, fn in enumerate(sdf_funcs):
                p, d = sample_fn(n_points, fn)
                all_pts[i] = torch.tensor(p, dtype=torch.float32, device=device)
                all_gt[i] = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(-1)

        el, nb = 0.0, 0
        model.train()
        for si in np.random.permutation(n_shapes):
            lat = latent_codes(torch.tensor(si, device=device))
            perm = torch.randperm(len(all_pts[si]), device=device)[:batch_size]
            pts_batch = all_pts[si][perm]
            gt_batch = all_gt[si][perm]

            # Clamp SDF targets
            gt_clamped = torch.clamp(gt_batch, -clamp_val, clamp_val)

            pred = model(pts_batch, lat)
            pred_clamped = torch.clamp(pred, -clamp_val, clamp_val)

            # L1 loss on clamped values
            loss = torch.mean(torch.abs(pred_clamped - gt_clamped))

            # Eikonal regularization: |∇SDF| ≈ 1 near surface
            if use_eikonal and not use_v1:
                near_mask = (torch.abs(gt_batch.squeeze()) < 0.05)
                if near_mask.sum() > 100:
                    near_pts = pts_batch[near_mask].detach().requires_grad_(True)
                    near_pred = model(near_pts, lat.detach())
                    grad = torch.autograd.grad(
                        near_pred.sum(), near_pts,
                        create_graph=True)[0]
                    eikonal = torch.mean((grad.norm(dim=-1) - 1.0) ** 2)
                    loss = loss + 0.01 * eikonal

            # Latent regularization
            loss = loss + 1e-4 * torch.mean(lat ** 2)

            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); nb += 1

        sched.step()
        avg = el / nb; best = min(best, avg)
        if ep % 500 == 0 or ep == epochs - 1:
            elapsed = time.time() - t0
            print(f"    Ep {ep:4d}/{epochs} | {avg:.6f} | Best {best:.6f} | {elapsed:.0f}s")

    print(f"  Done in {time.time()-t0:.0f}s | Best: {best:.6f}\n")
    return model, latent_codes


# ============================================================
# ACTIVATION-BASED SEGMENTATION → MESH
# ============================================================

def get_activation_labels(model, latent, sdf_func_or_none, device,
                          n_parts, resolution=256, bounds=(-0.55, 0.55),
                          seg_mode='instance', spatial_weight=0.3):
    """
    1. Extract mesh via marching cubes from the learned SDF
    2. Get activations at each mesh vertex
    3. Cluster activations → per-vertex labels
    """
    model.eval()

    # Step 1: Build SDF grid from the LEARNED model
    print(f"  Building SDF grid ({resolution}^3)...")
    grid = np.linspace(bounds[0], bounds[1], resolution)
    xx, yy, zz = np.meshgrid(grid, grid, grid, indexing='ij')
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)

    sdf_vals = []
    with torch.no_grad():
        chunk_size = 65536
        for i in range(0, len(pts), chunk_size):
            batch = torch.tensor(pts[i:i+chunk_size], dtype=torch.float32, device=device)
            sdf_vals.append(model(batch, latent).cpu().numpy().squeeze())
    sdf_grid = np.concatenate(sdf_vals).reshape(resolution, resolution, resolution)

    # Step 2: Marching cubes
    print(f"  Extracting mesh...")
    try:
        verts, faces, normals, _ = measure.marching_cubes(sdf_grid, level=0.0)
        verts = verts / resolution * (bounds[1] - bounds[0]) + bounds[0]
    except ValueError:
        print(f"  Marching cubes failed! (no zero-crossing found)")
        return None, None, None, None

    print(f"  Mesh: {len(verts):,} verts, {len(faces):,} faces")

    # Step 3: Get activations at each vertex
    print(f"  Computing activations at vertices...")
    all_acts = {}
    with torch.no_grad():
        chunk_size = 16384
        for i in range(0, len(verts), chunk_size):
            batch = torch.tensor(verts[i:i+chunk_size].astype(np.float32),
                                 dtype=torch.float32, device=device)
            _ = model(batch, latent, store_activations=True)
            acts = model.get_activations()
            for k, v in acts.items():
                if k not in all_acts:
                    all_acts[k] = []
                all_acts[k].append(v.cpu().numpy())

    for k in all_acts:
        all_acts[k] = np.concatenate(all_acts[k], axis=0)

    # Step 4: Segmentation
    #
    # Three modes:
    #   --mode semantic
    #     Pure activation clustering. Groups by surface type.
    #     "all balusters = same color"
    #
    #   --mode instance (DEFAULT for meshes)
    #     Step A: Cluster activations into n_types semantic groups
    #     Step B: Within each group, find connected components on the mesh
    #     Each connected component = one individual part
    #     "top rail = 1 part, each baluster = 1 part, base = 1 part"
    #
    #   --mode spatial
    #     Activation + spatial coords (the old approach, kept for comparison)

    print(f"  Segmenting (mode={seg_mode})...")

    if seg_mode == 'instance':
        # --- HYBRID: semantic clustering + mesh connected components ---
        
        # Step A: Semantic clustering with few types (auto or user-specified)
        n_types = min(n_parts, 5)  # coarse type count: rail, baluster, base, etc.
        print(f"  Step A: Semantic clustering into {n_types} surface types...")
        
        best_score = float('-inf')
        best_type_labels = None
        best_layer = None
        
        candidates = list(all_acts.keys()) + ['all_layers']
        for layer_name in candidates:
            if layer_name == 'all_layers':
                feats = np.concatenate([all_acts[k] for k in sorted(all_acts.keys())], axis=1)
            else:
                feats = all_acts[layer_name]
            
            feat_std = feats.std(axis=0, keepdims=True)
            feat_std[feat_std < 1e-8] = 1.0
            feats_norm = feats / feat_std
            
            km = KMeans(n_clusters=n_types, n_init=10, random_state=42)
            labels = km.fit_predict(feats_norm)
            
            sizes = [np.sum(labels == k) for k in range(n_types)]
            min_ratio = min(sizes) / max(max(sizes), 1)
            norm_inertia = km.inertia_ / (len(feats_norm) * feats_norm.shape[1])
            score = min_ratio - 0.01 * norm_inertia
            
            if score > best_score:
                best_score = score
                best_type_labels = labels
                best_layer = layer_name
        
        print(f"  Best layer for types: {best_layer}")
        for t in range(n_types):
            count = np.sum(best_type_labels == t)
            print(f"    Type {t}: {count:,} verts ({100*count/len(verts):.1f}%)")
        
        # Step B: Build adjacency graph and find connected components per type
        print(f"  Step B: Finding connected components per type...")
        
        # Build vertex adjacency from faces
        from collections import defaultdict
        adjacency = defaultdict(set)
        for face in faces:
            for i in range(3):
                for j in range(i+1, 3):
                    adjacency[face[i]].add(face[j])
                    adjacency[face[j]].add(face[i])
        
        # For each semantic type, find connected components using BFS
        final_labels = np.full(len(verts), -1, dtype=int)
        part_id = 0
        
        for t in range(n_types):
            type_mask = (best_type_labels == t)
            type_verts = set(np.where(type_mask)[0])
            
            if not type_verts:
                continue
            
            # BFS to find connected components within this type
            visited = set()
            components = []
            
            for start in type_verts:
                if start in visited:
                    continue
                # BFS
                component = []
                queue = [start]
                visited.add(start)
                while queue:
                    v = queue.pop(0)
                    component.append(v)
                    for neighbor in adjacency[v]:
                        if neighbor not in visited and neighbor in type_verts:
                            visited.add(neighbor)
                            queue.append(neighbor)
                components.append(component)
            
            # Sort components by size (largest first)
            components.sort(key=len, reverse=True)
            
            # Filter out tiny fragments (< 1% of type's total verts)
            type_total = sum(len(c) for c in components)
            min_component = max(int(type_total * 0.01), 10)
            
            significant = [c for c in components if len(c) >= min_component]
            fragments = [c for c in components if len(c) < min_component]
            
            # Assign part IDs to significant components
            for comp in significant:
                for v in comp:
                    final_labels[v] = part_id
                part_id += 1
            
            # Assign fragments to nearest significant component
            if fragments and significant:
                # Build KD-tree of significant component centroids
                centroids = []
                comp_ids = []
                for ci, comp in enumerate(significant):
                    centroid = verts[comp].mean(axis=0)
                    centroids.append(centroid)
                    comp_ids.append(part_id - len(significant) + ci)
                centroids = np.array(centroids)
                
                for frag in fragments:
                    frag_center = verts[frag].mean(axis=0)
                    dists = np.linalg.norm(centroids - frag_center, axis=1)
                    nearest_id = comp_ids[np.argmin(dists)]
                    for v in frag:
                        final_labels[v] = nearest_id
        
        # Handle any unlabeled verts (shouldn't happen, but safety)
        unlabeled = final_labels == -1
        if unlabeled.any():
            print(f"  Warning: {unlabeled.sum()} unlabeled verts, assigning to nearest part")
            labeled_idx = np.where(~unlabeled)[0]
            unlabeled_idx = np.where(unlabeled)[0]
            tree = cKDTree(verts[labeled_idx])
            _, nearest = tree.query(verts[unlabeled_idx])
            final_labels[unlabeled_idx] = final_labels[labeled_idx[nearest]]
        
        total_parts = len(np.unique(final_labels))
        print(f"\n  Instance segmentation result: {total_parts} parts found")
        for k in range(total_parts):
            count = np.sum(final_labels == k)
            print(f"    Part {k}: {count:,} verts ({100*count/len(verts):.1f}%)")
        
        return verts, faces, normals, final_labels
    
    elif seg_mode == 'spatial':
        # --- OLD: activation + spatial weight ---
        print(f"  Clustering activations (k={n_parts})...")
        best_score = float('-inf')
        best_labels = None
        best_layer = None
        
        verts_normalized = verts.copy()
        for dim in range(3):
            v_min, v_max = verts_normalized[:, dim].min(), verts_normalized[:, dim].max()
            if v_max - v_min > 1e-6:
                verts_normalized[:, dim] = (verts_normalized[:, dim] - v_min) / (v_max - v_min)
        
        candidates = list(all_acts.keys()) + ['all_layers']
        for layer_name in candidates:
            if layer_name == 'all_layers':
                feats = np.concatenate([all_acts[k] for k in sorted(all_acts.keys())], axis=1)
            else:
                feats = all_acts[layer_name]
            feat_std = feats.std(axis=0, keepdims=True)
            feat_std[feat_std < 1e-8] = 1.0
            feats_norm = feats / feat_std
            spatial_scaled = verts_normalized * spatial_weight * feats_norm.shape[1]
            combined = np.concatenate([feats_norm, spatial_scaled], axis=1)
            
            km = KMeans(n_clusters=n_parts, n_init=10, random_state=42)
            labels = km.fit_predict(combined)
            sizes = [np.sum(labels == k) for k in range(n_parts)]
            min_ratio = min(sizes) / max(max(sizes), 1)
            norm_inertia = km.inertia_ / (len(combined) * combined.shape[1])
            score = min_ratio - 0.01 * norm_inertia
            if score > best_score:
                best_score = score
                best_labels = labels
                best_layer = layer_name
        
        print(f"  Best layer: {best_layer} (score={best_score:.4f})")
        for k in range(n_parts):
            count = np.sum(best_labels == k)
            print(f"    Part {k}: {count:,} verts ({100*count/len(verts):.1f}%)")
        return verts, faces, normals, best_labels
    
    else:
        # --- SEMANTIC: pure activation clustering ---
        print(f"  Clustering activations (k={n_parts})...")
        best_score = float('-inf')
        best_labels = None
        best_layer = None
        
        candidates = list(all_acts.keys()) + ['all_layers']
        for layer_name in candidates:
            if layer_name == 'all_layers':
                feats = np.concatenate([all_acts[k] for k in sorted(all_acts.keys())], axis=1)
            else:
                feats = all_acts[layer_name]
            feat_std = feats.std(axis=0, keepdims=True)
            feat_std[feat_std < 1e-8] = 1.0
            feats_norm = feats / feat_std
            
            km = KMeans(n_clusters=n_parts, n_init=10, random_state=42)
            labels = km.fit_predict(feats_norm)
            sizes = [np.sum(labels == k) for k in range(n_parts)]
            min_ratio = min(sizes) / max(max(sizes), 1)
            norm_inertia = km.inertia_ / (len(feats_norm) * feats_norm.shape[1])
            score = min_ratio - 0.01 * norm_inertia
            if score > best_score:
                best_score = score
                best_labels = labels
                best_layer = layer_name
        
        print(f"  Best layer: {best_layer} (score={best_score:.4f})")
        for k in range(n_parts):
            count = np.sum(best_labels == k)
            print(f"    Part {k}: {count:,} verts ({100*count/len(verts):.1f}%)")
        return verts, faces, normals, best_labels


# ============================================================
# MESH EXPORT (unchanged from v1)
# ============================================================

COLORS = [
    (231, 76, 60),    # red
    (52, 152, 219),   # blue
    (46, 204, 113),   # green
    (243, 156, 18),   # orange
    (155, 89, 182),   # purple
    (26, 188, 156),   # teal
    (230, 126, 34),   # dark orange
    (41, 128, 185),   # dark blue
    (241, 196, 15),   # yellow
    (231, 126, 106),  # salmon
    (142, 68, 173),   # deep purple
    (39, 174, 96),    # emerald
    (192, 57, 43),    # pomegranate
    (44, 62, 80),     # midnight
    (127, 140, 141),  # concrete
    (211, 84, 0),     # pumpkin
]


def export_colored_obj(verts, faces, labels, output_path, normals=None):
    """Export OBJ with vertex colors (v x y z r g b format)."""
    n_parts = len(np.unique(labels))
    print(f"  Writing colored OBJ: {output_path}")

    with open(output_path, 'w') as f:
        f.write(f"# Activation-segmented mesh | {len(verts)} verts | {n_parts} parts\n")
        for i, v in enumerate(verts):
            c = COLORS[labels[i] % len(COLORS)]
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} "
                    f"{c[0]/255:.3f} {c[1]/255:.3f} {c[2]/255:.3f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f"  Done: {len(verts):,} verts, {len(faces):,} faces")


def export_part_objs(verts, faces, labels, n_parts, output_dir, name):
    """Export individual OBJ per part."""
    for k in range(n_parts):
        mask = labels == k
        if not mask.any():
            continue
        vert_map = np.full(len(verts), -1, dtype=int)
        part_verts = verts[mask]
        vert_map[mask] = np.arange(len(part_verts))
        part_faces = []
        for face in faces:
            if mask[face[0]] and mask[face[1]] and mask[face[2]]:
                part_faces.append([vert_map[face[0]], vert_map[face[1]], vert_map[face[2]]])
        if not part_faces:
            for face in faces:
                in_part = sum(1 for fi in face if mask[fi])
                if in_part >= 2:
                    new_face = []
                    for fi in face:
                        if mask[fi]:
                            new_face.append(vert_map[fi])
                        else:
                            dists = np.linalg.norm(part_verts - verts[fi], axis=1)
                            new_face.append(np.argmin(dists))
                    part_faces.append(new_face)
        path = os.path.join(output_dir, f'{name}_part{k}.obj')
        c = COLORS[k % len(COLORS)]
        with open(path, 'w') as f:
            f.write(f"# Part {k} | {len(part_verts)} verts | {len(part_faces)} faces\n")
            for v in part_verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} "
                        f"{c[0]/255:.3f} {c[1]/255:.3f} {c[2]/255:.3f}\n")
            for face in part_faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        print(f"  Part {k}: {path} ({len(part_verts):,}v, {len(part_faces):,}f)")


def export_ply(verts, faces, labels, output_path):
    """Export PLY with vertex colors."""
    n_parts = len(np.unique(labels))
    print(f"  Writing PLY: {output_path}")
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i, v in enumerate(verts):
            c = COLORS[labels[i] % len(COLORS)]
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    print(f"  Done: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Activation Segmentation V2 — Scaled up DeepSDF for complex meshes')
    parser.add_argument('--shape', default=None,
                        choices=list(SHAPES.keys()) + ['all'],
                        help='Analytical shape name')
    parser.add_argument('--mesh_path', default=None,
                        help='Path to your OBJ/STL file')
    parser.add_argument('--n_parts', type=int, default=None,
                        help='Number of parts to segment into')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Training epochs (default: 3000 for mesh, 1500 for analytical)')
    parser.add_argument('--resolution', type=int, default=None,
                        help='Marching cubes resolution (default: 256 for mesh, 128 for analytical)')
    parser.add_argument('--output_dir', default='./activation_mesh_output')
    parser.add_argument('--quick', action='store_true',
                        help='Use V1 settings (128 hidden, 4 layers) for fast iteration')
    parser.add_argument('--eikonal', action='store_true',
                        help='Add eikonal regularization (|grad|=1) for sharper surfaces')
    parser.add_argument('--latent_dim', type=int, default=None,
                        help='Latent code dimension (default: 128 for V2, 64 for V1/quick)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden layer width (V2 only, default: 256)')
    parser.add_argument('--num_layers', type=int, default=8,
                        help='Number of layers (V2 only, default: 8)')
    parser.add_argument('--mode', default=None, choices=['instance', 'semantic', 'spatial'],
                        help='Segmentation mode: instance (semantic types + connected components), '
                             'semantic (group by surface type), spatial (old activation+position). '
                             'Default: instance for mesh, semantic for analytical')
    parser.add_argument('--spatial_weight', type=float, default=0.3,
                        help='How much spatial position influences clustering in instance mode. '
                             '0.0=pure activation, 1.0=heavy spatial. Default: 0.3')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to saved model checkpoint (skip training, only re-cluster)')
    args = parser.parse_args()

    if args.shape is None and args.mesh_path is None:
        print("ERROR: Provide --shape or --mesh_path")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Determine defaults based on mode
    is_mesh = args.mesh_path is not None
    use_v1 = args.quick or (not is_mesh)  # analytical shapes use V1 by default

    if args.epochs is None:
        args.epochs = 1500 if use_v1 else 3000
    if args.resolution is None:
        args.resolution = 128 if use_v1 else 256
    if args.latent_dim is None:
        args.latent_dim = 64 if use_v1 else 128

    if is_mesh:
        # === User OBJ mode ===
        name = os.path.splitext(os.path.basename(args.mesh_path))[0]
        if args.n_parts is None:
            print("ERROR: --n_parts required with --mesh_path")
            return

        mesh_sdf = MeshSDF(args.mesh_path)

        if args.checkpoint:
            # Load existing model — skip training
            print(f"\n  Loading checkpoint: {args.checkpoint}")
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
            ckpt_v1 = ckpt.get('use_v1', False)
            ckpt_latent = ckpt.get('latent_dim', 128)
            ckpt_hidden = ckpt.get('hidden_dim', 256)
            ckpt_layers = ckpt.get('num_layers', 8)
            if ckpt_v1:
                model = DeepSDFNetworkV1(latent_dim=ckpt_latent).to(device)
            else:
                model = DeepSDFNetworkV2(latent_dim=ckpt_latent, hidden_dim=ckpt_hidden,
                                          num_layers=ckpt_layers).to(device)
            model.load_state_dict(ckpt['model_state'])
            latent_codes = nn.Embedding(1, ckpt_latent).to(device)
            latent_codes.load_state_dict(ckpt['latent_codes'])
            latent = latent_codes(torch.tensor(0, device=device))
            print(f"  Loaded {'V1' if ckpt_v1 else 'V2'}: latent={ckpt_latent}, hidden={ckpt_hidden}, layers={ckpt_layers}")
        else:
            # Train fresh
            print(f"\n  Training DeepSDF {'V1' if use_v1 else 'V2'} on {name}...")
            print(f"  Config: epochs={args.epochs}, resolution={args.resolution}, "
                  f"latent={args.latent_dim}, hidden={args.hidden_dim}, layers={args.num_layers}")

            model, latent_codes = train_deepsdf(
                [mesh_sdf], device,
                latent_dim=args.latent_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                epochs=args.epochs,
                use_v1=use_v1,
                use_eikonal=args.eikonal)
            latent = latent_codes(torch.tensor(0, device=device))

            # Save model checkpoint
            ckpt_path = os.path.join(args.output_dir, f'{name}_model.pth')
            torch.save({
                'model_state': model.state_dict(),
                'latent_codes': latent_codes.state_dict(),
                'latent_dim': args.latent_dim,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'use_v1': use_v1,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        seg_mode = args.mode if args.mode else 'instance'
        print(f"\n  Segmenting into {args.n_parts} parts (resolution={args.resolution}, mode={seg_mode})...")
        verts, faces, normals, labels = get_activation_labels(
            model, latent, mesh_sdf, device,
            n_parts=args.n_parts, resolution=args.resolution,
            seg_mode=seg_mode, spatial_weight=args.spatial_weight)

        if verts is not None:
            actual_n_parts = len(np.unique(labels))
            export_colored_obj(verts, faces, labels,
                               os.path.join(args.output_dir, f'{name}_colored.obj'))
            export_ply(verts, faces, labels,
                       os.path.join(args.output_dir, f'{name}_colored.ply'))
            export_part_objs(verts, faces, labels, actual_n_parts, args.output_dir, name)

    else:
        # === Analytical shapes mode (uses V1 by default) ===
        shapes_to_run = list(SHAPES.keys()) if args.shape == 'all' else [args.shape]

        sdf_funcs = [SHAPES[s][0] for s in shapes_to_run]
        print(f"\n  Training shared DeepSDF on {len(sdf_funcs)} shapes...")
        model, latent_codes = train_deepsdf(
            sdf_funcs, device,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            use_v1=True)  # always V1 for analytical

        for i, sname in enumerate(shapes_to_run):
            _, parts_fn, n_parts = SHAPES[sname]
            latent = latent_codes(torch.tensor(i, device=device))

            print(f"\n{'='*50}")
            print(f"  {sname.upper()} ({n_parts} parts)")
            print(f"{'='*50}")

            verts, faces, normals, labels = get_activation_labels(
                model, latent, SHAPES[sname][0], device,
                n_parts=n_parts, resolution=args.resolution,
                seg_mode=args.mode if args.mode else 'semantic',
                spatial_weight=args.spatial_weight)

            if verts is None:
                continue

            gt = get_gt_labels(parts_fn(), verts)
            ari = adjusted_rand_score(gt, labels)
            print(f"  ARI vs ground truth: {ari:.3f}")

            export_colored_obj(verts, faces, labels,
                               os.path.join(args.output_dir, f'{sname}_colored.obj'))
            export_ply(verts, faces, labels,
                       os.path.join(args.output_dir, f'{sname}_colored.ply'))
            export_part_objs(verts, faces, labels, n_parts, args.output_dir, sname)

    print(f"\n  All outputs in: {args.output_dir}/")


if __name__ == "__main__":
    main()