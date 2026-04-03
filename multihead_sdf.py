"""
Multi-Head Part SDF — Thickened Shell Approach
================================================
Each segmented part is a thin surface sheet. We give it thickness
by defining: part_sdf = distance_to_part_surface - thickness/2

This turns each open sheet into a thin watertight solid.
Train one MLP head per part on this thickened SDF.
Union (min) of all heads = full shape.

No Houdini. No VDB. No watertight closing. Pure math + neural networks.

Usage:
  python multihead_sdf.py --name test_2 \
      --checkpoint ./activation_mesh_output/test_2_model.pth

  # Adjust shell thickness (default 0.015)
  python multihead_sdf.py --name test_2 \
      --checkpoint ./activation_mesh_output/test_2_model.pth \
      --thickness 0.02

Author: Aditya Jain | April 2026
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import os
import time
from scipy.spatial import cKDTree
from skimage import measure
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# ORIGINAL DeepSDF (for loading checkpoint)
# ============================================================

class DeepSDFNetworkV2(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256, num_layers=8,
                 num_frequencies=6, skip_at=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_frequencies = num_frequencies
        self.num_layers = num_layers
        self.skip_at = skip_at
        xyz_dim = 3 + 3 * 2 * num_frequencies
        self.input_dim = xyz_dim + latent_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, hidden_dim))
        for i in range(1, num_layers - 1):
            if i == skip_at:
                self.layers.append(nn.Linear(hidden_dim + self.input_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, 1))
        self.relu = nn.ReLU()

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
        x = inp
        for i, layer in enumerate(self.layers[:-1]):
            if i == self.skip_at: x = torch.cat([x, inp], dim=-1)
            x = self.relu(layer(x))
        return self.layers[-1](x)


class DeepSDFNetworkV1(nn.Module):
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
        for layer in self.layers[:-1]: x = self.relu(layer(x))
        return self.layers[-1](x)


# ============================================================
# PER-PART SDF HEAD
# ============================================================

class PartSDF(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4, num_frequencies=6):
        super().__init__()
        self.num_frequencies = num_frequencies
        xyz_dim = 3 + 3 * 2 * num_frequencies
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(xyz_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, 1))
        self.relu = nn.ReLU()
        for layer in self.layers[:-1]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.layers[-1].weight, 0.0, 0.01)
        nn.init.zeros_(self.layers[-1].bias)

    def positional_encoding(self, x):
        enc = [x]
        for f in range(self.num_frequencies):
            s = (2.0 ** f) * np.pi
            enc.append(torch.sin(s * x)); enc.append(torch.cos(s * x))
        return torch.cat(enc, dim=-1)

    def forward(self, xyz):
        x = self.positional_encoding(xyz)
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        return self.layers[-1](x)


# ============================================================
# LOADING
# ============================================================

def load_original_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    is_v1 = ckpt.get('use_v1', False)
    latent_dim = ckpt.get('latent_dim', 128)
    hidden_dim = ckpt.get('hidden_dim', 256)
    num_layers = ckpt.get('num_layers', 8)
    if is_v1:
        model = DeepSDFNetworkV1(latent_dim=latent_dim).to(device)
    else:
        model = DeepSDFNetworkV2(latent_dim=latent_dim, hidden_dim=hidden_dim,
                                  num_layers=num_layers).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    latent_codes = nn.Embedding(1, latent_dim).to(device)
    latent_codes.load_state_dict(ckpt['latent_codes'])
    latent = latent_codes(torch.tensor(0, device=device)).detach()
    print(f"  Loaded {'V1' if is_v1 else 'V2'}: latent={latent_dim}, hidden={hidden_dim}, layers={num_layers}")
    return model, latent


def load_segmentation_from_obj(obj_path):
    print(f"  Loading: {obj_path}")
    verts, faces, colors = [], [], []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v ') and not line.startswith('vn') and not line.startswith('vt'):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(parts) >= 7:
                    colors.append([float(parts[4]), float(parts[5]), float(parts[6])])
            elif line.startswith('f '):
                parts = line.strip().split()
                faces.append([int(p.split('/')[0]) - 1 for p in parts[1:4]])
    verts = np.array(verts, dtype=np.float32)
    faces = np.array(faces, dtype=int)
    colors = np.array(colors, dtype=np.float32)

    color_keys = np.round(colors * 1000).astype(int)
    unique = {}
    labels = np.zeros(len(verts), dtype=int)
    for i, ck in enumerate(color_keys):
        key = tuple(ck)
        if key not in unique: unique[key] = len(unique)
        labels[i] = unique[key]

    n_parts = len(unique)
    print(f"  {len(verts):,} verts, {n_parts} parts")
    for k in range(n_parts):
        count = np.sum(labels == k)
        print(f"    Part {k}: {count:,} verts ({100*count/len(verts):.1f}%)")
    return verts, faces, labels, n_parts


# ============================================================
# THICKENED SHELL SDF
# ============================================================

def compute_thickened_sdf(query_points, part_tree, thickness):
    """
    Compute the thickened shell SDF for one part.
    
    For each query point:
      dist = distance to nearest part surface vertex
      sdf = dist - thickness/2
    
    Points within thickness/2 of the surface → negative (inside shell)
    Points farther away → positive (outside shell)
    
    This turns an open surface sheet into a thin watertight solid.
    Like Houdini scatter → VDB from particles → convert, but as pure math.
    """
    dist, _ = part_tree.query(query_points)
    return (dist - thickness / 2).astype(np.float32)


# ============================================================
# TRAINING
# ============================================================

def train_part(part_verts, part_tree, device, thickness,
               hidden_dim=64, num_layers=4, epochs=2000,
               n_points=150000, batch_size=8192, part_name="part"):
    """
    Train one small MLP on the thickened shell SDF of a single part.
    """
    model = PartSDF(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    clamp_val = 0.1

    def sample_batch():
        # 40% near part surface, 30% slightly further, 30% global
        n_near = int(n_points * 0.4)
        n_mid = int(n_points * 0.3)
        n_global = n_points - n_near - n_mid

        # Near surface: jitter part vertices tightly
        idx = np.random.choice(len(part_verts), n_near)
        near = part_verts[idx] + np.random.normal(0, thickness, (n_near, 3)).astype(np.float32)

        # Mid range: slightly further from surface
        idx2 = np.random.choice(len(part_verts), n_mid)
        mid = part_verts[idx2] + np.random.normal(0, thickness * 5, (n_mid, 3)).astype(np.float32)

        # Global
        glob = np.random.uniform(-0.55, 0.55, (n_global, 3)).astype(np.float32)

        pts = np.concatenate([near, mid, glob]).astype(np.float32)
        sdf = compute_thickened_sdf(pts, part_tree, thickness)
        return pts, sdf

    pts, sdf = sample_batch()
    n_inside = np.sum(sdf < 0)
    print(f"    {part_name}: {n_params:,} params | thickness={thickness:.3f}")
    print(f"    Inside: {n_inside:,}/{len(sdf):,} ({100*n_inside/len(sdf):.1f}%)")

    pts_t = torch.tensor(pts, dtype=torch.float32, device=device)
    sdf_t = torch.tensor(sdf, dtype=torch.float32, device=device).unsqueeze(-1)
    sdf_clamped = torch.clamp(sdf_t, -clamp_val, clamp_val)

    best_loss = float('inf')
    t0 = time.time()

    for ep in range(epochs):
        if ep > 0 and ep % 500 == 0:
            pts, sdf = sample_batch()
            pts_t = torch.tensor(pts, dtype=torch.float32, device=device)
            sdf_t = torch.tensor(sdf, dtype=torch.float32, device=device).unsqueeze(-1)
            sdf_clamped = torch.clamp(sdf_t, -clamp_val, clamp_val)

        perm = torch.randperm(len(pts_t), device=device)[:batch_size]
        pred = model(pts_t[perm])
        pred_clamped = torch.clamp(pred, -clamp_val, clamp_val)
        loss = torch.mean(torch.abs(pred_clamped - sdf_clamped[perm]))

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        best_loss = min(best_loss, loss.item())

    elapsed = time.time() - t0
    print(f"    loss={best_loss:.6f}, {elapsed:.1f}s")
    return model


# ============================================================
# EXPORT
# ============================================================

COLORS = [
    (231, 76, 60), (52, 152, 219), (46, 204, 113), (243, 156, 18),
    (155, 89, 182), (26, 188, 156), (230, 126, 34), (41, 128, 185),
    (241, 196, 15), (231, 126, 106), (142, 68, 173), (39, 174, 96),
    (192, 57, 43), (44, 62, 80), (127, 140, 141), (211, 84, 0),
    (22, 160, 133), (149, 165, 166), (39, 174, 96), (52, 73, 94),
]

def export_obj(verts, faces, path, labels=None):
    with open(path, 'w') as f:
        f.write(f"# {len(verts)} verts, {len(faces)} faces\n")
        for i, v in enumerate(verts):
            if labels is not None:
                c = COLORS[labels[i] % len(COLORS)]
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} "
                        f"{c[0]/255:.3f} {c[1]/255:.3f} {c[2]/255:.3f}\n")
            else:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"  Wrote: {path} ({len(verts):,}v, {len(faces):,}f)")


def export_ply(verts, faces, labels, path):
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i, v in enumerate(verts):
            c = COLORS[labels[i] % len(COLORS)]
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Multi-head part SDF with thickened shell approach')
    parser.add_argument('--name', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--colored_obj', default=None)
    parser.add_argument('--input_dir', default='./activation_mesh_output')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--thickness', type=float, default=0.015,
                        help='Shell thickness in normalized coords. '
                             '0.01=thin, 0.02=medium, 0.03=thick')
    parser.add_argument('--min_verts', type=int, default=100)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, 'multihead_output')
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Step 1: Load original (just for reference, not used for training)
    print(f"\n{'='*60}")
    print(f"  Step 1: Load original DeepSDF (reference only)")
    print(f"{'='*60}")
    original_model, latent = load_original_model(args.checkpoint, device)

    # Step 2: Load segmentation
    print(f"\n{'='*60}")
    print(f"  Step 2: Load segmentation")
    print(f"{'='*60}")
    colored_obj = args.colored_obj
    if colored_obj is None:
        colored_obj = os.path.join(args.input_dir, f'{args.name}_colored.obj')
    verts, faces, labels, n_parts = load_segmentation_from_obj(colored_obj)

    # Step 3: Train per-part MLPs on thickened shell SDFs
    print(f"\n{'='*60}")
    print(f"  Step 3: Train {n_parts} part SDFs (thickened shell)")
    print(f"  thickness={args.thickness}, hidden={args.hidden_dim}, "
          f"layers={args.num_layers}, epochs={args.epochs}")
    print(f"{'='*60}")

    part_models = []
    part_trees = []
    total_params = 0

    for k in range(n_parts):
        mask = labels == k
        n_v = np.sum(mask)
        print(f"\n  [Part {k}/{n_parts}] {n_v:,} surface vertices")

        if n_v < args.min_verts:
            print(f"    SKIP: too small")
            part_models.append(None)
            part_trees.append(None)
            continue

        part_verts = verts[mask].astype(np.float32)
        part_tree = cKDTree(part_verts)
        part_trees.append(part_tree)

        model = train_part(
            part_verts, part_tree, device, args.thickness,
            hidden_dim=args.hidden_dim, num_layers=args.num_layers,
            epochs=args.epochs, part_name=f"part_{k}")

        part_models.append(model)
        total_params += sum(p.numel() for p in model.parameters())

    active_models = [m for m in part_models if m is not None]
    active_indices = [i for i, m in enumerate(part_models) if m is not None]
    print(f"\n  Trained: {len(active_models)} parts, {total_params:,} params total")

    # Step 4: Extract meshes
    print(f"\n{'='*60}")
    print(f"  Step 4: Extract per-part + union meshes")
    print(f"{'='*60}")

    grid = np.linspace(-0.55, 0.55, args.resolution)
    xx, yy, zz = np.meshgrid(grid, grid, grid, indexing='ij')
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)
    chunk = 65536

    print(f"  Evaluating {len(active_models)} heads on {args.resolution}^3 grid...")

    all_sdfs = np.full((len(pts), len(active_models)), 1e6, dtype=np.float32)

    for idx, (model, part_idx) in enumerate(zip(active_models, active_indices)):
        model.eval()
        vals = []
        with torch.no_grad():
            for i in range(0, len(pts), chunk):
                batch = torch.tensor(pts[i:i+chunk], dtype=torch.float32, device=device)
                vals.append(model(batch).cpu().numpy().squeeze())
        all_sdfs[:, idx] = np.concatenate(vals)
        n_inside = np.sum(all_sdfs[:, idx] < 0)
        print(f"    Part {part_idx}: {n_inside:,} inside voxels")

    # Per-part export
    for idx, part_idx in enumerate(active_indices):
        head_sdf = all_sdfs[:, idx]
        n_in = np.sum(head_sdf < 0)
        if n_in == 0 or n_in == len(head_sdf):
            print(f"  Part {part_idx}: bad SDF, skipping")
            continue
        sdf_grid = head_sdf.reshape(args.resolution, args.resolution, args.resolution)
        try:
            v, f, _, _ = measure.marching_cubes(sdf_grid, level=0.0)
            v = v / args.resolution * 1.1 + (-0.55)
            path = os.path.join(args.output_dir, f'{args.name}_part{part_idx}.obj')
            export_obj(v, f, path)
        except ValueError:
            print(f"  Part {part_idx}: marching cubes failed")

    # Union
    union_sdf = all_sdfs.min(axis=1)
    ownership = all_sdfs.argmin(axis=1)
    # Map back to original part indices
    ownership_mapped = np.array([active_indices[o] for o in ownership])

    n_inside = np.sum(union_sdf < 0)
    print(f"\n  Union: {n_inside:,} inside voxels")

    if n_inside == 0 or n_inside == len(union_sdf):
        print("  ERROR: bad union SDF")
    else:
        sdf_grid = union_sdf.reshape(args.resolution, args.resolution, args.resolution)
        own_grid = ownership_mapped.reshape(args.resolution, args.resolution, args.resolution)
        try:
            v, f, _, _ = measure.marching_cubes(sdf_grid, level=0.0)
            v = v / args.resolution * 1.1 + (-0.55)

            vg = ((v - (-0.55)) / 1.1 * (args.resolution - 1)).astype(int)
            vg = np.clip(vg, 0, args.resolution - 1)
            ulabels = own_grid[vg[:, 0], vg[:, 1], vg[:, 2]]

            export_obj(v, f, os.path.join(args.output_dir, f'{args.name}_union.obj'))
            export_obj(v, f, os.path.join(args.output_dir, f'{args.name}_union_colored.obj'), labels=ulabels)
            export_ply(v, f, ulabels, os.path.join(args.output_dir, f'{args.name}_union_colored.ply'))
            print(f"  Union: {len(v):,} verts, {len(f):,} faces")
        except ValueError:
            print("  Union marching cubes failed!")

    # Save
    torch.save({
        'n_parts': n_parts,
        'active_indices': active_indices,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'thickness': args.thickness,
        'models': [m.state_dict() for m in active_models],
    }, os.path.join(args.output_dir, f'{args.name}_multihead.pth'))

    print(f"\n  All outputs in: {args.output_dir}/")


if __name__ == "__main__":
    main()