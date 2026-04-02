"""
Activation Segmentation → Colored 3D Meshes
=============================================
Takes the DeepSDF hidden activation approach and produces:
  - Per-part OBJ meshes (individual parts)
  - A single colored OBJ (all parts, vertex-colored)
  - Visualization PNGs

Two modes:
  1. Analytical shapes (from probe_activations.py):
     python activation_mesh.py --shape snowman
     python activation_mesh.py --shape all

  2. Your own OBJ file:
     python activation_mesh.py --mesh_path model.obj --n_parts 3
     (Trains DeepSDF on your mesh first, then probes activations)

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
# SDF PRIMITIVES
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
# COMPOUND SHAPES (same as probe_activations.py)
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
# MESH SDF (for user OBJ files)
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
# DeepSDF NETWORK
# ============================================================

class DeepSDFNetwork(nn.Module):
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
# TRAINING
# ============================================================

def sample_points(n_points, sdf_func, surface_ratio=0.6):
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


def train_deepsdf(sdf_funcs, device, latent_dim=64, epochs=1500, n_points=100000, batch_size=8192):
    """Train shared decoder on one or more SDF functions."""
    n_shapes = len(sdf_funcs)
    model = DeepSDFNetwork(latent_dim=latent_dim).to(device)
    latent_codes = nn.Embedding(n_shapes, latent_dim).to(device)
    nn.init.normal_(latent_codes.weight, 0.0, 0.01)

    opt = torch.optim.Adam([
        {'params': model.parameters(), 'lr': 1e-3},
        {'params': latent_codes.parameters(), 'lr': 1e-3},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Pre-sample
    all_pts, all_gt = {}, {}
    for i, fn in enumerate(sdf_funcs):
        p, d = sample_points(n_points, fn)
        all_pts[i] = torch.tensor(p, dtype=torch.float32, device=device)
        all_gt[i] = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(-1)

    print(f"  Training DeepSDF: {n_shapes} shapes, {epochs} epochs...")
    best = float('inf')
    t0 = time.time()
    for ep in range(epochs):
        if ep > 0 and ep % 300 == 0:
            for i, fn in enumerate(sdf_funcs):
                p, d = sample_points(n_points, fn)
                all_pts[i] = torch.tensor(p, dtype=torch.float32, device=device)
                all_gt[i] = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(-1)
        el, nb = 0.0, 0
        model.train()
        for si in np.random.permutation(n_shapes):
            lat = latent_codes(torch.tensor(si, device=device))
            perm = torch.randperm(len(all_pts[si]), device=device)[:batch_size]
            pred = model(all_pts[si][perm], lat)
            loss = torch.mean(torch.abs(pred - all_gt[si][perm])) + 1e-4 * torch.mean(lat**2)
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); nb += 1
        sched.step()
        avg = el / nb; best = min(best, avg)
        if ep % 300 == 0 or ep == epochs - 1:
            print(f"    Ep {ep:4d}/{epochs} | {avg:.6f} | Best {best:.6f}")
    print(f"  Done in {time.time()-t0:.0f}s\n")
    return model, latent_codes


# ============================================================
# ACTIVATION-BASED SEGMENTATION → MESH
# ============================================================

def get_activation_labels(model, latent, sdf_func, device,
                          n_parts, resolution=128, bounds=(-0.8, 0.8)):
    """
    1. Extract mesh via marching cubes
    2. Get activations at each mesh vertex
    3. Cluster activations → per-vertex labels
    """
    model.eval()

    # Step 1: Build SDF grid
    print(f"  Building SDF grid ({resolution}^3)...")
    grid = np.linspace(bounds[0], bounds[1], resolution)
    xx, yy, zz = np.meshgrid(grid, grid, grid, indexing='ij')
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)

    sdf_vals = []
    with torch.no_grad():
        for i in range(0, len(pts), 65536):
            batch = torch.tensor(pts[i:i+65536], dtype=torch.float32, device=device)
            sdf_vals.append(model(batch, latent).cpu().numpy().squeeze())
    sdf_grid = np.concatenate(sdf_vals).reshape(resolution, resolution, resolution)

    # Step 2: Marching cubes
    print(f"  Extracting mesh...")
    try:
        verts, faces, normals, _ = measure.marching_cubes(sdf_grid, level=0.0)
        verts = verts / resolution * (bounds[1] - bounds[0]) + bounds[0]
    except ValueError:
        print(f"  Marching cubes failed!")
        return None, None, None, None

    print(f"  Mesh: {len(verts):,} verts, {len(faces):,} faces")

    # Step 3: Get activations at each vertex
    print(f"  Computing activations at vertices...")
    all_acts = {}
    with torch.no_grad():
        for i in range(0, len(verts), 16384):
            batch = torch.tensor(verts[i:i+16384].astype(np.float32),
                                 dtype=torch.float32, device=device)
            _ = model(batch, latent, store_activations=True)
            acts = model.get_activations()
            for k, v in acts.items():
                if k not in all_acts:
                    all_acts[k] = []
                all_acts[k].append(v.cpu().numpy())

    for k in all_acts:
        all_acts[k] = np.concatenate(all_acts[k], axis=0)

    # Step 4: Try each layer + all_layers, pick best by cluster separation
    print(f"  Clustering activations (k={n_parts})...")
    best_score = float('-inf')
    best_labels = None
    best_layer = None

    for layer_name in list(all_acts.keys()) + ['all_layers']:
        if layer_name == 'all_layers':
            feats = np.concatenate([all_acts[k] for k in sorted(all_acts.keys())], axis=1)
        else:
            feats = all_acts[layer_name]

        km = KMeans(n_clusters=n_parts, n_init=10, random_state=42)
        labels = km.fit_predict(feats)

        # Score: prefer balanced clusters (even-sized parts)
        # and tighter clusters (lower inertia, normalized by feature dim)
        sizes = [np.sum(labels == k) for k in range(n_parts)]
        min_ratio = min(sizes) / max(max(sizes), 1)
        # Normalize inertia by number of points and feature dimensions
        norm_inertia = km.inertia_ / (len(feats) * feats.shape[1])
        # Combined: balance is primary, tightness is secondary
        combined_score = min_ratio - 0.01 * norm_inertia

        print(f"    {layer_name}: balance={min_ratio:.3f} inertia={norm_inertia:.4f} score={combined_score:.4f}")

        if combined_score > best_score:
            best_score = combined_score
            best_labels = labels
            best_layer = layer_name

    print(f"  Best layer: {best_layer}")
    for k in range(n_parts):
        count = np.sum(best_labels == k)
        print(f"    Part {k}: {count:,} verts ({100*count/len(verts):.1f}%)")

    return verts, faces, normals, best_labels


# ============================================================
# MESH EXPORT
# ============================================================

# Part colors (RGB 0-255)
COLORS = [
    (231, 76, 60),    # red
    (52, 152, 219),   # blue
    (46, 204, 113),   # green
    (243, 156, 18),   # orange
    (155, 89, 182),   # purple
    (26, 188, 156),   # teal
    (230, 126, 34),   # dark orange
    (41, 128, 185),   # dark blue
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

        # Remap vertex indices
        vert_map = np.full(len(verts), -1, dtype=int)
        part_verts = verts[mask]
        vert_map[mask] = np.arange(len(part_verts))

        # Filter faces: keep only faces where ALL vertices belong to this part
        part_faces = []
        for face in faces:
            if mask[face[0]] and mask[face[1]] and mask[face[2]]:
                part_faces.append([vert_map[face[0]], vert_map[face[1]], vert_map[face[2]]])

        if not part_faces:
            # Fallback: keep faces where MAJORITY of vertices belong
            for face in faces:
                in_part = sum(1 for fi in face if mask[fi])
                if in_part >= 2:
                    new_face = []
                    for fi in face:
                        if mask[fi]:
                            new_face.append(vert_map[fi])
                        else:
                            # Find nearest part vertex
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
    """Export PLY with vertex colors (wider viewer support than colored OBJ)."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', default=None,
                        choices=list(SHAPES.keys()) + ['all'],
                        help='Analytical shape name')
    parser.add_argument('--mesh_path', default=None,
                        help='Path to your OBJ/STL file')
    parser.add_argument('--n_parts', type=int, default=None,
                        help='Number of parts to segment into (required for --mesh_path)')
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--resolution', type=int, default=128,
                        help='Marching cubes resolution')
    parser.add_argument('--output_dir', default='./activation_mesh_output')
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

    if args.mesh_path:
        # === User OBJ mode ===
        name = os.path.splitext(os.path.basename(args.mesh_path))[0]
        if args.n_parts is None:
            print("ERROR: --n_parts required with --mesh_path")
            print("  Example: --mesh_path chair.obj --n_parts 5")
            return

        mesh_sdf = MeshSDF(args.mesh_path)

        print(f"\n  Training DeepSDF on {name}...")
        model, latent_codes = train_deepsdf([mesh_sdf], device, epochs=args.epochs)
        latent = latent_codes(torch.tensor(0, device=device))

        print(f"\n  Segmenting into {args.n_parts} parts...")
        verts, faces, normals, labels = get_activation_labels(
            model, latent, mesh_sdf, device,
            n_parts=args.n_parts, resolution=args.resolution)

        if verts is not None:
            export_colored_obj(verts, faces, labels,
                               os.path.join(args.output_dir, f'{name}_colored.obj'))
            export_ply(verts, faces, labels,
                       os.path.join(args.output_dir, f'{name}_colored.ply'))
            export_part_objs(verts, faces, labels, args.n_parts, args.output_dir, name)

    else:
        # === Analytical shapes mode ===
        shapes_to_run = list(SHAPES.keys()) if args.shape == 'all' else [args.shape]

        # Train shared DeepSDF on all shapes
        sdf_funcs = [SHAPES[s][0] for s in shapes_to_run]
        print(f"\n  Training shared DeepSDF on {len(sdf_funcs)} shapes...")
        model, latent_codes = train_deepsdf(sdf_funcs, device, epochs=args.epochs)

        for i, sname in enumerate(shapes_to_run):
            _, parts_fn, n_parts = SHAPES[sname]
            latent = latent_codes(torch.tensor(i, device=device))

            print(f"\n{'='*50}")
            print(f"  {sname.upper()} ({n_parts} parts)")
            print(f"{'='*50}")

            verts, faces, normals, labels = get_activation_labels(
                model, latent, SHAPES[sname][0], device,
                n_parts=n_parts, resolution=args.resolution)

            if verts is None:
                continue

            # Evaluate against GT
            gt = get_gt_labels(parts_fn(), verts)
            ari = adjusted_rand_score(gt, labels)
            print(f"  ARI vs ground truth: {ari:.3f}")

            # Export
            export_colored_obj(verts, faces, labels,
                               os.path.join(args.output_dir, f'{sname}_colored.obj'))
            export_ply(verts, faces, labels,
                       os.path.join(args.output_dir, f'{sname}_colored.ply'))
            export_part_objs(verts, faces, labels, n_parts, args.output_dir, sname)

    print(f"\n  All outputs in: {args.output_dir}/")


if __name__ == "__main__":
    main()