# Self-Supervised SDF Part Discovery & Reconstruction

Unsupervised 3D shape segmentation and per-part reconstruction by probing hidden activations of a learned DeepSDF decoder. The key insight: when a neural network learns to represent 3D shapes as Signed Distance Fields, its hidden layer activations naturally encode part-level structure — without any supervision.

## Pipeline

```
Input Mesh
    ↓
DeepSDF V2 (train learned SDF representation)
    ↓
Hidden Activation Probing (extract per-vertex features)
    ↓
Semantic Clustering + Connected Components (instance segmentation)
    ↓
Thickened Shell SDF per part (turn open sheets into watertight solids)
    ↓
Per-part MLP training (one small network per part)
    ↓
Boolean Union (min of all parts → full shape with sharp junctions)
```

## Results

### Segmentation
| Shape | Method | Parts Found |
|-------|--------|-------------|
| Railing (real mesh) | Activation + connected components | 12 parts: top rail, 8 balusters, base sections |
| Snowman (analytical) | Activation probe | 2 parts, ARI 0.634 |
| Lollipop (analytical) | Activation probe | 2 parts, ARI 0.996 |

### Part-wise Reconstruction
Each segmented part trained as an independent watertight SDF using the thickened shell approach. Individual balusters, rails, and base exported as clean meshes. Union of all parts reconstructs the full shape.

## How It Works

### Step 1: Learn the shape (activation_mesh_v2.py)
Train a DeepSDF V2 (256 hidden, 8 layers, skip connection) on the input mesh. The network learns a continuous SDF representation.

### Step 2: Segment via activations (activation_mesh_v2.py)
Extract hidden layer activations at each mesh vertex. Cluster by activation similarity (semantic types), then split by mesh connectivity (connected components). Each baluster becomes its own part because they're disconnected by air gaps.

### Step 3: Reconstruct per part (multihead_sdf.py)
For each part, define a thickened shell SDF: `sdf(point) = distance_to_part_surface - thickness/2`. This turns each open surface sheet into a thin watertight solid. Train a small MLP (64 hidden, 4 layers) on this SDF. Union all parts via min().

## Files

| File | Description |
|------|-------------|
| `activation_mesh_v2.py` | **Segmentation** — DeepSDF V2 + activation-based instance segmentation |
| `multihead_sdf.py` | **Per-part reconstruction** — thickened shell SDF + per-part MLP training + union |
| `activation_mesh.py` | V1 original (smaller network, k-means only) |
| `probe_activations.py` | Standalone probing experiment on 9 analytical shapes |
| `activation_dino_design.docx` | Design doc for DINO self-distillation extension |
| `activation_sdf_status.docx` | Project status and continuation document |

## Usage

### Full pipeline

```bash
# Step 1: Train DeepSDF + segment (~5 min on M4 iMac)
python activation_mesh_v2.py --mesh_path railing.obj --n_parts 3

# Step 2: Per-part reconstruction (~3 min)
python multihead_sdf.py --name railing \
    --checkpoint ./activation_mesh_output/railing_model.pth
```

### Segmentation only

```bash
# Train + segment
python activation_mesh_v2.py --mesh_path model.obj --n_parts 3

# Re-cluster with saved checkpoint (instant)
python activation_mesh_v2.py --mesh_path model.obj --n_parts 3 \
    --checkpoint ./activation_mesh_output/model_model.pth

# Semantic mode (group by surface type)
python activation_mesh_v2.py --mesh_path model.obj --n_parts 3 \
    --checkpoint ./activation_mesh_output/model_model.pth --mode semantic
```

### Per-part reconstruction options

```bash
# Adjust shell thickness (thicker = smoother, thinner = sharper)
python multihead_sdf.py --name test_2 \
    --checkpoint ./activation_mesh_output/test_2_model.pth \
    --thickness 0.02

# Bigger part networks for complex geometry
python multihead_sdf.py --name test_2 \
    --checkpoint ./activation_mesh_output/test_2_model.pth \
    --hidden_dim 128 --num_layers 6
```

### Analytical shapes

```bash
python probe_activations.py                    # All 9 shapes with ARI
python activation_mesh_v2.py --shape snowman   # Single shape mesh export
python activation_mesh_v2.py --shape all       # All shapes
```

## Key Options

### activation_mesh_v2.py
| Flag | Default | Description |
|------|---------|-------------|
| `--mesh_path` | — | Input OBJ/STL |
| `--n_parts` | — | Semantic type count (instance mode auto-discovers final parts) |
| `--checkpoint` | — | Skip training, load saved model |
| `--mode` | `instance` | `instance`, `semantic`, or `spatial` |
| `--resolution` | 256 | Marching cubes grid |
| `--epochs` | 3000 | Training epochs |
| `--hidden_dim` | 256 | DeepSDF hidden width |
| `--num_layers` | 8 | DeepSDF depth |
| `--eikonal` | off | Gradient regularization |

### multihead_sdf.py
| Flag | Default | Description |
|------|---------|-------------|
| `--name` | — | Shape name (matches activation_mesh_v2 output) |
| `--checkpoint` | — | Original DeepSDF checkpoint |
| `--thickness` | 0.015 | Shell thickness (0.01=thin, 0.02=medium) |
| `--hidden_dim` | 64 | Part MLP hidden width |
| `--num_layers` | 4 | Part MLP depth |
| `--epochs` | 2000 | Training epochs per part |
| `--resolution` | 256 | Marching cubes for export |

## Architecture

### DeepSDF V2 (full shape)
- 256 hidden, 8 layers, skip connection at layer 4
- ~480K parameters
- Positional encoding (6 frequencies) + latent code (128-dim)

### Per-part MLP (thickened shell)
- 64 hidden, 4 layers, ~11K parameters each
- No latent code (one network per part)
- Trained on: `sdf(p) = distance_to_part_vertices(p) - thickness/2`

### Thickened Shell SDF
The key innovation for per-part reconstruction. Each segmented part is an open surface sheet (no inside/outside). We give it thickness:
```
part_sdf(point) = dist(point, nearest_part_vertex) - thickness/2
```
Points within `thickness/2` of the surface are inside. Everything else is outside. This creates a thin watertight solid from an open sheet — like Houdini's scatter → VDB from particles → convert VDB, but as pure math.

## Requirements

```
torch
numpy
scikit-learn
scikit-image
scipy
trimesh
pysdf
matplotlib
```

## Hardware

Tested on M4 iMac (32GB, MPS). Full pipeline (segmentation + per-part reconstruction) takes ~8 minutes. Also works on CUDA GPUs.

## Author

Aditya Jain (Bob) — April 2026

Part of thesis research on neural 3D reconstruction. See design documents for the proposed DINO self-distillation extension and full project status.
