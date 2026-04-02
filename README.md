# Self-Supervised SDF Part Discovery via Activation-Space DINO

Unsupervised 3D shape segmentation by probing the hidden activations of a learned DeepSDF decoder. The key insight: when a neural network learns to represent 3D shapes as Signed Distance Fields, its hidden layer activations naturally encode part-level structure — without any supervision.

## Results

| Shape | Method | Parts Found | Notes |
|-------|--------|-------------|-------|
| Snowman (analytical) | Activation probe | 2 | ARI 0.634 — two spheres separated |
| Lollipop (analytical) | Activation probe | 2 | ARI 0.996 — near-perfect |
| Railing (real mesh) | Activation + connected components | 10 | Top rail, 8 balusters, base — each individually segmented |
| Head (real mesh) | Activation probe | 5 | Geometric regions: forehead, side, face, nose area, neck |

## How It Works

1. **Train DeepSDF** on a 3D shape (learns a neural SDF representation)
2. **Extract hidden activations** at each surface point (the intermediate 256-dim features from each layer)
3. **Cluster activations** to identify surface types (rail vs baluster vs base)
4. **Connected components** within each type to separate individual instances (each baluster becomes its own part)

The activation features encode what *kind* of surface a point belongs to (curvature profile, distance computation pattern). The mesh topology encodes which points are physically connected. Combining both gives instance-level segmentation.

## Files

| File | Description |
|------|-------------|
| `activation_mesh_v2.py` | **Main script** — V2 with scaled-up DeepSDF (256 hidden, 8 layers, skip connection), instance segmentation via connected components |
| `activation_mesh.py` | V1 original (128 hidden, 4 layers, k-means only) |
| `probe_activations.py` | Standalone activation probing experiment on 9 analytical shapes with ARI evaluation |
| `activation_dino_design.docx` | Technical design document for the full DINO self-distillation approach |

## Usage

### Segment your own mesh

```bash
# Full pipeline: train DeepSDF + segment (takes ~5 min on M4 iMac)
python activation_mesh_v2.py --mesh_path railing.obj --n_parts 3

# Instance mode (default): separates individual components
# Outputs: colored OBJ, colored PLY, per-part OBJs
```

### Re-cluster without retraining

```bash
# Use saved checkpoint to try different settings instantly
python activation_mesh_v2.py --mesh_path railing.obj --n_parts 3 \
    --checkpoint ./activation_mesh_output/railing_model.pth

# Semantic mode: group by surface type (all balusters = same color)
python activation_mesh_v2.py --mesh_path railing.obj --n_parts 3 \
    --checkpoint ./activation_mesh_output/railing_model.pth --mode semantic
```

### Run analytical shape experiments

```bash
# All 9 shapes with ARI evaluation
python probe_activations.py

# Single shape mesh export
python activation_mesh_v2.py --shape snowman
python activation_mesh_v2.py --shape all
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mesh_path` | — | Path to OBJ/STL file |
| `--n_parts` | — | Number of semantic types (instance mode auto-discovers final count) |
| `--checkpoint` | — | Skip training, load existing model |
| `--mode` | `instance` | `instance` (connected components), `semantic` (surface type), `spatial` (position-weighted) |
| `--resolution` | 256 | Marching cubes grid resolution |
| `--epochs` | 3000 | Training epochs |
| `--eikonal` | off | Eikonal regularization for sharper surfaces |
| `--quick` | off | Use V1 small network for fast iteration |
| `--hidden_dim` | 256 | MLP hidden layer width |
| `--num_layers` | 8 | MLP depth |
| `--spatial_weight` | 0.3 | Position influence (spatial mode only) |

## Architecture

### DeepSDF V2
- 256 hidden dim, 8 layers, skip connection at layer 4
- ~480K parameters (vs 46K in V1)
- Positional encoding (6 frequencies) + latent code (128-dim)
- Kaiming initialization, SDF clamping to [-0.1, 0.1]
- Multi-band surface sampling (tight/medium/close bands for thin features)

### Segmentation Pipeline (Instance Mode)
1. **Semantic clustering**: k-means on activation features → 3-5 surface types
2. **Connected components**: BFS on mesh adjacency within each type → individual parts
3. **Fragment cleanup**: tiny components (< 1% of type) merged to nearest significant part

## Requirements

```
torch
numpy
scikit-learn
scikit-image
scipy
trimesh
pysdf
```

## Hardware

Tested on M4 iMac (32GB, MPS). Also works on CUDA GPUs and CPU. Training a single shape takes ~5 minutes on M4, ~2 minutes on RTX 3060.

## Author

Aditya Jain (Bob) — April 2026

## Citation

Part of thesis research on neural 3D reconstruction. See `activation_dino_design.docx` for the full technical design document including the proposed DINO self-distillation extension.
