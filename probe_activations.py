"""
Hidden Activation Probe: Do DeepSDF internal features separate parts?
=====================================================================
Pipeline:
  1. Train a shared DeepSDF decoder (MLP + latent codes) on multiple
     compound shapes simultaneously
  2. For each shape, sample surface points
  3. Forward pass through the decoder, extract HIDDEN LAYER activations
     (not the final scalar output — the intermediate 128-dim features)
  4. Cluster those features with k-means
  5. Compare to ground truth part labels

If the decoder's internal representation naturally separates parts,
then we have a path to unsupervised segmentation through any learned
SDF — no voxels, no point clouds, no forensic signals needed.

Usage:
  python probe_activations.py
  python probe_activations.py --epochs 2000 --n_shapes 10

Author: Aditya Jain | April 2026
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import os
import time
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from itertools import permutations
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
# COMPOUND SHAPES
# ============================================================

def sdf_snowman(pts):
    return np.minimum(
        sdf_sphere(pts, [0, -0.15, 0], 0.35),
        sdf_sphere(pts, [0, 0.35, 0], 0.2))
def parts_snowman():
    return [
        lambda p: sdf_sphere(p, [0, -0.15, 0], 0.35),
        lambda p: sdf_sphere(p, [0, 0.35, 0], 0.2)]

def sdf_lollipop(pts):
    return np.minimum(
        sdf_sphere(pts, [0, 0.3, 0], 0.25),
        sdf_cylinder(pts, [0, -0.2, 0], 0.04, 0.3, 'y'))
def parts_lollipop():
    return [
        lambda p: sdf_sphere(p, [0, 0.3, 0], 0.25),
        lambda p: sdf_cylinder(p, [0, -0.2, 0], 0.04, 0.3, 'y')]

def sdf_barbell(pts):
    return np.minimum(np.minimum(
        sdf_sphere(pts, [-0.4, 0, 0], 0.2),
        sdf_sphere(pts, [0.4, 0, 0], 0.2)),
        sdf_cylinder(pts, [0, 0, 0], 0.06, 0.4, 'x'))
def parts_barbell():
    return [
        lambda p: sdf_sphere(p, [-0.4, 0, 0], 0.2),
        lambda p: sdf_sphere(p, [0.4, 0, 0], 0.2),
        lambda p: sdf_cylinder(p, [0, 0, 0], 0.06, 0.4, 'x')]

def sdf_mushroom(pts):
    return np.minimum(
        sdf_sphere(pts, [0, 0.15, 0], 0.3),
        sdf_cylinder(pts, [0, -0.2, 0], 0.08, 0.25, 'y'))
def parts_mushroom():
    return [
        lambda p: sdf_sphere(p, [0, 0.15, 0], 0.3),
        lambda p: sdf_cylinder(p, [0, -0.2, 0], 0.08, 0.25, 'y')]

def sdf_tower(pts):
    return np.minimum(np.minimum(
        sdf_sphere(pts, [0, -0.35, 0], 0.25),
        sdf_sphere(pts, [0, 0.05, 0], 0.2)),
        sdf_sphere(pts, [0, 0.35, 0], 0.15))
def parts_tower():
    return [
        lambda p: sdf_sphere(p, [0, -0.35, 0], 0.25),
        lambda p: sdf_sphere(p, [0, 0.05, 0], 0.2),
        lambda p: sdf_sphere(p, [0, 0.35, 0], 0.15)]

def sdf_table(pts):
    top = sdf_box(pts, [0, 0.3, 0], [0.4, 0.03, 0.3])
    leg1 = sdf_cylinder(pts, [-0.3, 0.0, -0.2], 0.03, 0.27, 'y')
    leg2 = sdf_cylinder(pts, [0.3, 0.0, -0.2], 0.03, 0.27, 'y')
    leg3 = sdf_cylinder(pts, [-0.3, 0.0, 0.2], 0.03, 0.27, 'y')
    leg4 = sdf_cylinder(pts, [0.3, 0.0, 0.2], 0.03, 0.27, 'y')
    result = top
    for l in [leg1, leg2, leg3, leg4]:
        result = np.minimum(result, l)
    return result
def parts_table():
    return [
        lambda p: sdf_box(p, [0, 0.3, 0], [0.4, 0.03, 0.3]),
        lambda p: sdf_cylinder(p, [-0.3, 0.0, -0.2], 0.03, 0.27, 'y'),
        lambda p: sdf_cylinder(p, [0.3, 0.0, -0.2], 0.03, 0.27, 'y'),
        lambda p: sdf_cylinder(p, [-0.3, 0.0, 0.2], 0.03, 0.27, 'y'),
        lambda p: sdf_cylinder(p, [0.3, 0.0, 0.2], 0.03, 0.27, 'y')]

def sdf_tpipe(pts):
    return np.minimum(
        sdf_cylinder(pts, [0, 0, 0], 0.1, 0.4, 'y'),
        sdf_cylinder(pts, [0, 0.15, 0], 0.1, 0.35, 'x'))
def parts_tpipe():
    return [
        lambda p: sdf_cylinder(p, [0, 0, 0], 0.1, 0.4, 'y'),
        lambda p: sdf_cylinder(p, [0, 0.15, 0], 0.1, 0.35, 'x')]

def sdf_l_shape(pts):
    return np.minimum(
        sdf_box(pts, [-0.15, 0.0, 0.0], [0.35, 0.15, 0.2]),
        sdf_box(pts, [0.15, 0.25, 0.0], [0.15, 0.25, 0.2]))
def parts_l_shape():
    return [
        lambda p: sdf_box(p, [-0.15, 0.0, 0.0], [0.35, 0.15, 0.2]),
        lambda p: sdf_box(p, [0.15, 0.25, 0.0], [0.15, 0.25, 0.2])]

def sdf_chair(pts):
    seat = sdf_box(pts, [0, 0.0, 0], [0.3, 0.03, 0.25])
    back = sdf_box(pts, [0, 0.25, -0.22], [0.3, 0.22, 0.02])
    leg1 = sdf_cylinder(pts, [-0.25, -0.25, -0.2], 0.025, 0.22, 'y')
    leg2 = sdf_cylinder(pts, [0.25, -0.25, -0.2], 0.025, 0.22, 'y')
    leg3 = sdf_cylinder(pts, [-0.25, -0.25, 0.2], 0.025, 0.22, 'y')
    leg4 = sdf_cylinder(pts, [0.25, -0.25, 0.2], 0.025, 0.22, 'y')
    result = seat
    for p in [back, leg1, leg2, leg3, leg4]:
        result = np.minimum(result, p)
    return result
def parts_chair():
    return [
        lambda p: sdf_box(p, [0, 0.0, 0], [0.3, 0.03, 0.25]),
        lambda p: sdf_box(p, [0, 0.25, -0.22], [0.3, 0.22, 0.02]),
        lambda p: sdf_cylinder(p, [-0.25, -0.25, -0.2], 0.025, 0.22, 'y'),
        lambda p: sdf_cylinder(p, [0.25, -0.25, -0.2], 0.025, 0.22, 'y'),
        lambda p: sdf_cylinder(p, [-0.25, -0.25, 0.2], 0.025, 0.22, 'y'),
        lambda p: sdf_cylinder(p, [0.25, -0.25, 0.2], 0.025, 0.22, 'y')]


SHAPES = {
    'snowman':  (sdf_snowman,  parts_snowman,  2),
    'lollipop': (sdf_lollipop, parts_lollipop, 2),
    'barbell':  (sdf_barbell,  parts_barbell,  3),
    'mushroom': (sdf_mushroom, parts_mushroom, 2),
    'tower':    (sdf_tower,    parts_tower,    3),
    'table':    (sdf_table,    parts_table,    5),
    'tpipe':    (sdf_tpipe,    parts_tpipe,    2),
    'l_shape':  (sdf_l_shape,  parts_l_shape,  2),
    'chair':    (sdf_chair,    parts_chair,    6),
}

def get_gt_labels(parts_fns, points):
    sdfs = np.stack([fn(points) for fn in parts_fns], axis=1)
    return np.argmin(sdfs, axis=1)


# ============================================================
# DeepSDF NETWORK WITH ACTIVATION HOOKS
# ============================================================

class DeepSDFNetwork(nn.Module):
    """
    Shared decoder: takes (xyz + latent_code) → distance.
    Also stores intermediate activations for probing.
    """
    def __init__(self, latent_dim=64, hidden_dim=128, num_layers=4,
                 num_frequencies=6):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_frequencies = num_frequencies
        self.num_layers = num_layers

        xyz_dim = 3 + 3 * 2 * num_frequencies
        input_dim = xyz_dim + latent_dim

        # Build layers separately so we can extract activations
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, 1))

        self.relu = nn.ReLU()

        # Storage for intermediate activations
        self._activations = {}

    def positional_encoding(self, x):
        encodings = [x]
        for freq in range(self.num_frequencies):
            scale = (2.0 ** freq) * np.pi
            encodings.append(torch.sin(scale * x))
            encodings.append(torch.cos(scale * x))
        return torch.cat(encodings, dim=-1)

    def forward(self, xyz, latent, store_activations=False):
        """
        xyz: (B, 3)
        latent: (B, latent_dim) or (latent_dim,) broadcast
        """
        pe = self.positional_encoding(xyz)

        if latent.dim() == 1:
            latent = latent.unsqueeze(0).expand(xyz.shape[0], -1)

        x = torch.cat([pe, latent], dim=-1)

        if store_activations:
            self._activations = {}

        for i, layer in enumerate(self.layers[:-1]):
            x = self.relu(layer(x))
            if store_activations:
                self._activations[f'layer_{i}'] = x.detach()

        out = self.layers[-1](x)

        if store_activations:
            self._activations['pre_output'] = x.detach()

        return out

    def get_activations(self, layer_name=None):
        """Get stored activations from last forward pass."""
        if layer_name:
            return self._activations.get(layer_name, None)
        return self._activations


# ============================================================
# TRAINING
# ============================================================

def sample_points(n_points, sdf_func, bounds=(-1.0, 1.0), surface_ratio=0.6):
    n_s = int(n_points * surface_ratio)
    n_u = n_points - n_s
    up = np.random.uniform(bounds[0], bounds[1], (n_u, 3)).astype(np.float32)
    ud = sdf_func(up)
    cand = np.random.uniform(bounds[0], bounds[1], (n_s * 3, 3)).astype(np.float32)
    cd = sdf_func(cand)
    near = np.argsort(np.abs(cd))[:n_s]
    sp = cand[near] + np.random.normal(0, 0.01, (n_s, 3)).astype(np.float32)
    sd = sdf_func(sp)
    return (np.concatenate([up, sp]).astype(np.float32),
            np.concatenate([ud, sd]).astype(np.float32))


def train_deepsdf(shapes_dict, device, latent_dim=64, epochs=1500,
                  n_points=100000, batch_size=8192, lr=1e-3, lr_latent=1e-3):
    """
    Train shared decoder + per-shape latent codes.
    """
    shape_names = list(shapes_dict.keys())
    n_shapes = len(shape_names)

    model = DeepSDFNetwork(latent_dim=latent_dim).to(device)
    latent_codes = nn.Embedding(n_shapes, latent_dim).to(device)
    nn.init.normal_(latent_codes.weight, 0.0, 0.01)

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': lr},
        {'params': latent_codes.parameters(), 'lr': lr_latent},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  DeepSDF: {n_params:,} params | {n_shapes} shapes | latent_dim={latent_dim}")
    print(f"  Training for {epochs} epochs...")

    # Pre-sample all shapes
    all_points = {}
    all_dists = {}
    for i, name in enumerate(shape_names):
        sdf_func = shapes_dict[name][0]
        pts, dists = sample_points(n_points, sdf_func)
        all_points[i] = torch.tensor(pts, dtype=torch.float32, device=device)
        all_dists[i] = torch.tensor(dists, dtype=torch.float32, device=device).unsqueeze(-1)

    best_loss = float('inf')
    t0 = time.time()

    for epoch in range(epochs):
        # Resample periodically
        if epoch > 0 and epoch % 300 == 0:
            for i, name in enumerate(shape_names):
                sdf_func = shapes_dict[name][0]
                pts, dists = sample_points(n_points, sdf_func)
                all_points[i] = torch.tensor(pts, dtype=torch.float32, device=device)
                all_dists[i] = torch.tensor(dists, dtype=torch.float32, device=device).unsqueeze(-1)

        epoch_loss = 0.0
        n_batches = 0

        # Shuffle shape order each epoch
        shape_order = np.random.permutation(n_shapes)

        model.train()
        for si in shape_order:
            pts = all_points[si]
            gt = all_dists[si]
            latent = latent_codes(torch.tensor(si, device=device))

            perm = torch.randperm(len(pts), device=device)[:batch_size]
            batch_pts = pts[perm]
            batch_gt = gt[perm]

            pred = model(batch_pts, latent)
            loss = torch.mean(torch.abs(pred - batch_gt))

            # Latent regularization
            reg = 1e-4 * torch.mean(latent ** 2)
            total_loss = loss + reg

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg = epoch_loss / n_batches
        best_loss = min(best_loss, avg)

        if epoch % 300 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch:4d}/{epochs} | Loss {avg:.6f} | Best {best_loss:.6f}")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s | Best loss: {best_loss:.6f}\n")

    return model, latent_codes, shape_names


# ============================================================
# ACTIVATION PROBING
# ============================================================

def extract_surface_points(sdf_func, n_points=5000, surface_thresh=0.015):
    collected = []
    for _ in range(30):
        c = np.random.uniform(-0.85, 0.85, (n_points * 5, 3)).astype(np.float32)
        d = sdf_func(c)
        collected.append(c[np.abs(d) < surface_thresh])
        if sum(len(x) for x in collected) >= n_points:
            break
    return np.concatenate(collected)[:n_points] if collected else np.zeros((0, 3))


def probe_shape(model, latent_codes, shape_idx, sdf_func, parts_fn,
                n_true_parts, device, shape_name):
    """
    Extract hidden activations at surface points and cluster them.
    Test each hidden layer to see which one separates parts best.
    """
    # Get surface points
    surface_pts = extract_surface_points(sdf_func, n_points=5000)
    gt_labels = get_gt_labels(parts_fn(), surface_pts)
    n_pts = len(surface_pts)

    print(f"  [{shape_name}] {n_pts} surface points, {n_true_parts} true parts")

    # Forward pass with activation storage
    model.eval()
    pts_t = torch.tensor(surface_pts, dtype=torch.float32, device=device)
    latent = latent_codes(torch.tensor(shape_idx, device=device))

    with torch.no_grad():
        _ = model(pts_t, latent, store_activations=True)

    activations = model.get_activations()

    results = {}

    for layer_name, act in activations.items():
        features = act.cpu().numpy()  # (N, hidden_dim)

        # PCA
        pca = PCA(n_components=min(3, features.shape[1]))
        pca_coords = pca.fit_transform(features)
        var_explained = pca.explained_variance_ratio_[:3]

        # K-means with true k
        km = KMeans(n_clusters=n_true_parts, n_init=10, random_state=42)
        pred = km.fit_predict(features)

        ari = adjusted_rand_score(gt_labels, pred)
        acc = best_perm_accuracy(gt_labels, pred, n_true_parts)

        results[layer_name] = {
            'ari': ari, 'acc': acc,
            'pca_coords': pca_coords, 'pred': pred,
            'var_explained': var_explained,
            'features': features,
        }

        print(f"    {layer_name:>12}: ARI={ari:.3f} Acc={acc:.3f} "
              f"PCA var={var_explained.round(3)}")

    # Also test concatenation of all layers
    all_feats = np.concatenate([activations[k].cpu().numpy()
                                for k in sorted(activations.keys())], axis=1)
    pca_all = PCA(n_components=3)
    pca_c = pca_all.fit_transform(all_feats)
    km_all = KMeans(n_clusters=n_true_parts, n_init=10, random_state=42)
    pred_all = km_all.fit_predict(all_feats)
    ari_all = adjusted_rand_score(gt_labels, pred_all)
    acc_all = best_perm_accuracy(gt_labels, pred_all, n_true_parts)
    print(f"    {'ALL LAYERS':>12}: ARI={ari_all:.3f} Acc={acc_all:.3f}")

    results['all_layers'] = {
        'ari': ari_all, 'acc': acc_all,
        'pca_coords': pca_c, 'pred': pred_all,
        'features': all_feats,
    }

    # Find best layer
    best_layer = max(results, key=lambda k: results[k]['ari'])
    best_ari = results[best_layer]['ari']
    best_acc = results[best_layer]['acc']
    print(f"    BEST: {best_layer} → ARI={best_ari:.3f} Acc={best_acc:.3f}")

    return results, surface_pts, gt_labels, best_layer


def best_perm_accuracy(gt, pred, max_k):
    best = 0.0
    max_k = min(max_k, 8)
    for perm in permutations(range(max_k)):
        remapped = np.array([perm[l] if l < len(perm) else -1 for l in pred])
        best = max(best, np.mean(remapped == gt))
        if best > 0.99:
            break
    return best


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_probe(surface_pts, gt_labels, results, best_layer,
                    shape_name, n_true_parts, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    best = results[best_layer]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: XY views — GT, best layer prediction, PCA
    ax = axes[0, 0]
    for k in range(n_true_parts):
        m = gt_labels == k
        if m.any():
            ax.scatter(surface_pts[m, 0], surface_pts[m, 1],
                       c=colors[k % len(colors)], s=3, alpha=0.5)
    ax.set_title(f'Ground Truth (XY) — {n_true_parts} parts')
    ax.set_aspect('equal')

    ax = axes[0, 1]
    n_pred = len(np.unique(best['pred']))
    for k in range(n_pred):
        m = best['pred'] == k
        if m.any():
            ax.scatter(surface_pts[m, 0], surface_pts[m, 1],
                       c=colors[k % len(colors)], s=3, alpha=0.5)
    ax.set_title(f'Predicted — {best_layer}\nARI={best["ari"]:.3f} Acc={best["acc"]:.3f}')
    ax.set_aspect('equal')

    # PCA scatter
    ax = axes[0, 2]
    pca = best['pca_coords']
    for k in range(n_true_parts):
        m = gt_labels == k
        if m.any():
            ax.scatter(pca[m, 0], pca[m, 1],
                       c=colors[k % len(colors)], s=3, alpha=0.5)
    ax.set_title(f'PCA of {best_layer} features\n(colored by GT)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    # Row 2: per-layer ARI bar chart, XZ views
    ax = axes[1, 0]
    layer_names = [k for k in results if k != 'all_layers']
    aris = [results[k]['ari'] for k in layer_names]
    aris.append(results['all_layers']['ari'])
    layer_names.append('all_layers')
    bar_colors = ['#3498db'] * (len(layer_names) - 1) + ['#e74c3c']
    ax.barh(layer_names, aris, color=bar_colors)
    ax.set_xlabel('ARI')
    ax.set_title('ARI by Layer')
    ax.set_xlim(0, 1)

    # GT XZ
    ax = axes[1, 1]
    for k in range(n_true_parts):
        m = gt_labels == k
        if m.any():
            ax.scatter(surface_pts[m, 0], surface_pts[m, 2],
                       c=colors[k % len(colors)], s=3, alpha=0.5)
    ax.set_title('Ground Truth (XZ)')
    ax.set_aspect('equal')

    # Predicted XZ
    ax = axes[1, 2]
    for k in range(n_pred):
        m = best['pred'] == k
        if m.any():
            ax.scatter(surface_pts[m, 0], surface_pts[m, 2],
                       c=colors[k % len(colors)], s=3, alpha=0.5)
    ax.set_title(f'Predicted (XZ) — {best_layer}')
    ax.set_aspect('equal')

    plt.suptitle(f'{shape_name} | Best: {best_layer} | ARI={best["ari"]:.3f} | Acc={best["acc"]:.3f}',
                 fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, f'activations_{shape_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--output_dir', default='./activation_probe_output')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # ---- Step 1: Train shared DeepSDF ----
    print(f"\n{'='*60}")
    print(f"Step 1: Training shared DeepSDF decoder")
    print(f"{'='*60}")

    model, latent_codes, shape_names = train_deepsdf(
        SHAPES, device,
        latent_dim=args.latent_dim,
        epochs=args.epochs)

    # Save model
    torch.save({
        'model_state': model.state_dict(),
        'latent_codes': latent_codes.state_dict(),
        'shape_names': shape_names,
        'latent_dim': args.latent_dim,
    }, os.path.join(args.output_dir, 'deepsdf_model.pth'))

    # ---- Step 2: Probe each shape ----
    print(f"\n{'='*60}")
    print(f"Step 2: Probing hidden activations for part segmentation")
    print(f"{'='*60}")

    all_results = []

    for i, name in enumerate(shape_names):
        sdf_func, parts_fn, n_parts = SHAPES[name]

        results, surface_pts, gt_labels, best_layer = probe_shape(
            model, latent_codes, i, sdf_func, parts_fn, n_parts,
            device, name)

        best = results[best_layer]
        all_results.append({
            'shape': name,
            'n_parts': n_parts,
            'best_layer': best_layer,
            'ari': best['ari'],
            'acc': best['acc'],
        })

        visualize_probe(surface_pts, gt_labels, results, best_layer,
                        name, n_parts, args.output_dir)

    # ---- Summary ----
    print(f"\n\n{'='*60}")
    print(f"RESULTS: Hidden Activation Probe")
    print(f"{'='*60}")
    print(f"{'Shape':<12} {'Parts':>5} {'Best Layer':<14} {'ARI':>7} {'Acc':>7}")
    print(f"{'-'*55}")
    for r in all_results:
        print(f"{r['shape']:<12} {r['n_parts']:>5} {r['best_layer']:<14} "
              f"{r['ari']:>7.3f} {r['acc']:>7.3f}")

    avg_ari = np.mean([r['ari'] for r in all_results])
    success = sum(1 for r in all_results if r['ari'] > 0.3)
    print(f"\nAverage ARI: {avg_ari:.3f}")
    print(f"Shapes with ARI > 0.3: {success}/{len(all_results)}")

    if avg_ari > 0.3:
        print(f"\n✓ Hidden activations carry part-separation signal!")
        print(f"  → Worth pursuing DINO-style training on this representation")
    elif avg_ari > 0.1:
        print(f"\n~ Weak signal in activations — DINO training might amplify it")
    else:
        print(f"\n✗ Activations don't naturally separate parts")
        print(f"  → Need contrastive/DINO training to create the signal")

    print(f"\nOutputs in: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
