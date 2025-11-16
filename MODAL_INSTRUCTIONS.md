# Modal Remote Training Guide

Complete guide for training F1 Racing AI models remotely using Modal's cloud infrastructure.

## Table of Contents

1. [Overview](#overview)
2. [Setup](#setup)
3. [Track Management](#track-management)
4. [Training Workflows](#training-workflows)
5. [Evaluation](#evaluation)
6. [Advanced Features](#advanced-features)
7. [Troubleshooting](#troubleshooting)

---

## Overview

This project uses Modal to run PPO (Proximal Policy Optimization) training remotely with:
- **GPU acceleration** (optional, with automatic fallback)
- **External track management** via volumes (no image rebuilds for track changes)
- **Checkpoint saving** at configurable intervals
- **TensorBoard logging** for monitoring
- **Automatic artifact download** after training
- **Resume training** from any checkpoint
- **ONNX export** for production deployment

### Architecture

- **Local CLI**: `tools/track_cli.py` - Your main interface
- **Remote Functions**: `train_ppo_modal.py` - Modal cloud functions
- **Volumes**:
  - `f1-ppo-checkpoints` → `/root/outputs` (models, checkpoints, TensorBoard logs)
  - `f1-ppo-tracks` → `/root/tracks` (track images + metadata)
- **Environment**: Custom `RacingEnv` with external track loading

---

## Setup

### Prerequisites

1. **Install Modal in your virtual environment:**
   ```powershell
   # Activate your venv first!
   .\environment47\Scripts\Activate.ps1
   pip install modal
   ```

2. **Configure Modal authentication:**
   ```powershell
   python -m modal setup
   ```
   Follow the prompts to authenticate with your Modal account.

3. **Verify installation:**
   ```powershell
   python -m modal --version
   ```

---

## Track Management

### Initial Track Sync

Before training, you must upload your track assets to the remote volume.

**Directory structure:**
```
tracks/
├── checkpoints.json    # Track metadata (checkpoints, start position, angle)
└── circuit.png         # Track image file
```

**Sync command:**
```powershell
python tools/track_cli.py sync --dir tracks
```

This uploads all tracks defined in `checkpoints.json` to Modal's persistent volume.

### Track Commands

**List remote tracks:**
```powershell
python tools/track_cli.py list
```

**View track metadata:**
```powershell
python tools/track_cli.py meta --track circuit.png
```

**Upload a single track manually:**
```powershell
python tools/track_cli.py upload --png circuit.png --checkpoints-json tracks/checkpoints.json --track-name circuit.png
```

### Adding New Tracks

1. Create your track image (e.g., `speedway.png`)
2. Add entry to `tracks/checkpoints.json`:
   ```json
   {
     "speedway.png": {
       "checkpoints": [
         [[x1, y1], [x2, y2]],
         ...
       ],
       "start_position": [x, y],
       "start_angle": 90.0
     }
   }
   ```
3. Copy image to `tracks/` folder
4. Run sync:
   ```powershell
   python tools/track_cli.py sync --dir tracks
   ```

---

## Training Workflows

### Basic Training (CPU)

**Start a new training run:**
```powershell
python tools/track_cli.py train \
  --track-name circuit.png \
  --timesteps 1000000 \
  --save-name ppo_baseline \
  --vector-envs 8 \
  --checkpoint-freq 100000
```

**Parameters:**
- `--track-name`: Track PNG filename (must exist in remote volume)
- `--timesteps`: Total training steps
- `--save-name`: Model output name (will be `<name>.zip`)
- `--vector-envs`: Number of parallel environments (default: 4 for CPU)
- `--checkpoint-freq`: Save checkpoint every N steps

### GPU Training (Recommended)

**Add `--gpu` flag for faster training:**
```powershell
python tools/track_cli.py train \
  --track-name circuit.png \
  --timesteps 10000000 \
  --save-name ppo_10m_run \
  --gpu \
  --vector-envs 16 \
  --checkpoint-freq 500000 \
  --tensorboard \
  --download-model \
  --download-meta
```

**GPU-specific parameters:**
- `--gpu`: Use GPU (auto-selects from: L40S, A100, A10G, V100, T4)
- `--vector-envs`: Increase to 16-32 for GPU (more parallel environments)

**GPU Fallback Order:**
The system tries GPUs in this order: L40S → A100 → A10G → V100 → T4

### Resume Training (Continuation)

**Continue from a previous checkpoint:**
```powershell
python tools/track_cli.py train \
  --track-name circuit.png \
  --timesteps 5000000 \
  --save-name ppo_continued \
  --resume-from ppo_baseline.zip \
  --gpu \
  --vector-envs 16 \
  --checkpoint-freq 500000 \
  --tensorboard \
  --download-model
```

**Important notes:**
- `--resume-from`: Must be a model file already in `/root/outputs` volume
- Additional timesteps are **added** to the existing training
- TensorBoard logs continue from the same run

### Progressive Training Strategy (Recommended for Full Laps)

For complex tracks with 45+ checkpoints:

**Phase 1: Initial learning (1M steps)**
```powershell
python tools/track_cli.py train \
  --track-name circuit.png \
  --timesteps 1000000 \
  --save-name ppo_phase1 \
  --gpu \
  --vector-envs 16 \
  --checkpoint-freq 250000 \
  --tensorboard \
  --download-model
```

**Phase 2: Refinement (continue for 4M more)**
```powershell
python tools/track_cli.py train \
  --track-name circuit.png \
  --timesteps 4000000 \
  --save-name ppo_phase2 \
  --resume-from ppo_phase1.zip \
  --gpu \
  --vector-envs 16 \
  --checkpoint-freq 500000 \
  --tensorboard \
  --download-model
```

**Phase 3: Mastery (continue for 5M more = 10M total)**
```powershell
python tools/track_cli.py train \
  --track-name circuit.png \
  --timesteps 5000000 \
  --save-name ppo_phase3_final \
  --resume-from ppo_phase2.zip \
  --gpu \
  --vector-envs 16 \
  --checkpoint-freq 1000000 \
  --tensorboard \
  --download-model
```

### Training Flags Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--track-name` | Track PNG filename | `circuit.png` |
| `--timesteps` | Total training steps | Required |
| `--save-name` | Output model name | Required |
| `--gpu` | Use GPU acceleration | Off (CPU) |
| `--vector-envs` | Parallel environments | 4 (CPU), 8 (GPU) |
| `--checkpoint-freq` | Checkpoint interval | 100000 |
| `--tensorboard` | Enable TensorBoard logging | Off |
| `--resume-from` | Continue from model | None |
| `--prune-keep` | Keep latest N checkpoints | None (keep all) |
| `--prune-interval` | Also keep interval checkpoints | None |
| `--download-model` | Auto-download final model | Off |
| `--download-meta` | Auto-download metadata JSON | Off |

### Checkpoint Pruning

Save disk space by automatically pruning old checkpoints:

```powershell
python tools/track_cli.py train \
  --track-name circuit.png \
  --timesteps 10000000 \
  --save-name ppo_pruned \
  --gpu \
  --checkpoint-freq 250000 \
  --prune-keep 5 \
  --prune-interval 1000000 \
  --tensorboard
```

**Pruning logic:**
- `--prune-keep 5`: Keeps latest 5 checkpoints
- `--prune-interval 1000000`: Also keeps checkpoints at 1M, 2M, 3M... (milestones)
- Older checkpoints are deleted automatically

---

## Evaluation

### Remote Evaluation

**Evaluate a trained model remotely (headless):**
```powershell
python tools/track_cli.py evaluate \
  --model ppo_10m_run.zip \
  --episodes 10
```

**Output includes:**
- Average reward across episodes
- Average steps per episode
- Average checkpoints reached
- Per-episode breakdown

**Example output:**
```
Episode 1/10: reward=4523.21 steps=342 checkpoints=45
Episode 2/10: reward=4612.89 steps=356 checkpoints=45
...
Evaluation metrics:
{
  "avg_reward": 4567.23,
  "avg_steps": 348.5,
  "avg_checkpoints": 44.8
}
```

### Local Evaluation (Visual)

**Download and test locally with visualization:**

1. **Fetch the model:**
   ```powershell
   python tools/track_cli.py fetch \
     --path ppo_10m_run.zip \
     --out ppo_10m_run.zip
   ```

2. **Run locally:**
   ```powershell
   python play_ppo.py --model ppo_10m_run.zip
   ```

### Validation (Dry-Run)

**Test track assets locally before training:**
```powershell
python tools/track_cli.py validate --track circuit.png
```

Checks:
- Track image exists
- Checkpoints JSON is valid
- Start position and angle are defined
- Environment can load successfully

---

## Advanced Features

### TensorBoard Monitoring

**Enable TensorBoard during training:**
```powershell
python tools/track_cli.py train \
  --track-name circuit.png \
  --timesteps 5000000 \
  --save-name ppo_monitored \
  --gpu \
  --tensorboard \
  --download-model
```

**Logs are stored remotely at:**
- `/root/outputs/tb/<save_name>/`

**To view logs locally:**

1. Download TensorBoard logs from Modal volume (manual)
2. Or monitor via Modal dashboard at training URL

### ONNX Export (Production Deployment)

**Export trained policy to ONNX format:**
```powershell
python tools/track_cli.py export \
  --model ppo_10m_run.zip \
  --download
```

**Parameters:**
- `--model`: Model zip file name
- `--onnx-name`: Output ONNX filename (optional, defaults to `<model>.onnx`)
- `--opset`: ONNX opset version (default: 17)
- `--download`: Auto-download ONNX file after export

**What gets exported:**
- Policy network only (for inference)
- Input: observation vector (10 values)
- Output: action vector (2 values: steering, acceleration)

**Use case:**
Deploy to production environments without PyTorch/SB3 dependencies.

### Artifact Management

**List all remote artifacts:**
```powershell
python tools/track_cli.py list-artifacts
```

**Fetch any artifact:**
```powershell
python tools/track_cli.py fetch \
  --path checkpoints/ppo_run_500000_steps.zip \
  --out checkpoint_500k.zip
```

**Artifact structure:**
```
/root/outputs/
├── ppo_run.zip                    # Final model
├── ppo_run.meta.json              # Training metadata
├── checkpoints/
│   ├── ppo_run_250000_steps.zip
│   ├── ppo_run_500000_steps.zip
│   └── ppo_run_750000_steps.zip
└── tb/ppo_run/                    # TensorBoard logs
    └── PPO_1/
```

### Metadata Files

**Auto-generated metadata (`.meta.json`):**
```json
{
  "save_name": "ppo_10m_run",
  "final_model": "/root/outputs/ppo_10m_run.zip",
  "timesteps": 10000000,
  "vector_envs": 16,
  "checkpoint_freq": 500000,
  "tensorboard": true,
  "resume_from": null,
  "gpu_type": "L40S",
  "prune_keep": 5,
  "prune_interval": 1000000,
  "pruned_count": 8,
  "kept_count": 12,
  "kept_files": ["..."],
  "track_name": "circuit.png",
  "timestamp_utc": "2025-11-16T01:23:45Z"
}
```

**Use metadata to:**
- Track experiment parameters
- Reproduce training runs
- Compare different configurations

---

## Troubleshooting

### Common Issues

**1. "No module named modal"**
- **Cause**: Modal not installed or venv not activated
- **Solution**:
  ```powershell
  .\environment47\Scripts\Activate.ps1
  pip install modal
  ```

**2. "No checkpoints.json found in expected locations"**
- **Cause**: Track not synced to remote volume
- **Solution**:
  ```powershell
  python tools/track_cli.py sync --dir tracks
  ```

**3. "Track 'circuit.png' missing in checkpoints.json"**
- **Cause**: Track metadata not defined
- **Solution**: Add track entry to `tracks/checkpoints.json` and re-sync

**4. "Image was modified during build process"**
- **Cause**: CLI files changed, triggering rebuild
- **Solution**: Already fixed in current version (only `car.py` and `racing_env.py` are bundled)

**5. Training very slow / poor GPU utilization**
- **Expected**: PPO with MlpPolicy is CPU-bound for small networks
- **Impact**: GPU helps with larger batch sizes but won't match CNN performance
- **Mitigation**: Increase `--vector-envs` to 16-32 on GPU

**6. "charmap codec can't encode character"**
- **Cause**: Windows encoding issue with Modal CLI output
- **Solution**: Already fixed (UTF-8 enforcement in subprocess calls)

### Performance Tips

**Faster training:**
- Use GPU with `--gpu` flag
- Increase `--vector-envs` to 16-32 (GPU) or 8-12 (CPU)
- Reduce `--checkpoint-freq` if disk I/O is bottleneck

**Better convergence:**
- Train for 10M+ timesteps for full lap mastery
- Use progressive training (1M → 5M → 10M)
- Monitor TensorBoard to detect plateaus

**Disk space optimization:**
- Use `--prune-keep` and `--prune-interval`
- Download final model only with `--download-model`
- Clean old experiments from Modal volume

### Debugging Training

**Monitor progress in real-time:**
```
Nov 16  01:05:42.587
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 25.5        |  ← Episode length (steps)
|    ep_rew_mean          | 804         |  ← Average reward
| time/                   |             |
|    fps                  | 587         |  ← Training speed
|    iterations           | 9           |  ← Policy updates
|    total_timesteps      | 294912      |  ← Progress
```

**Key metrics:**
- `ep_rew_mean`: Should increase over time (negative = bad, positive = good)
- `ep_len_mean`: Longer = car survives more steps
- `fps`: Training speed (600-800 typical for GPU)

**Healthy training signs:**
- Reward steadily increasing
- Episode length growing
- Clip fraction decreasing (policy stabilizing)

**Warning signs:**
- Reward stuck or decreasing
- Very short episodes (< 10 steps) after 500K+ timesteps
- NaN values in any metric

---

## Quick Reference

### Full Training Pipeline (Start to Finish)

```powershell
# 1. Setup (one-time)
.\environment47\Scripts\Activate.ps1
pip install modal
python -m modal setup

# 2. Sync tracks
python tools/track_cli.py sync --dir tracks

# 3. Validate
python tools/track_cli.py validate --track circuit.png

# 4. Train (10M steps for full mastery)
python tools/track_cli.py train \
  --track-name circuit.png \
  --timesteps 10000000 \
  --save-name ppo_final \
  --gpu \
  --vector-envs 16 \
  --checkpoint-freq 500000 \
  --tensorboard \
  --prune-keep 5 \
  --prune-interval 1000000 \
  --download-model \
  --download-meta

# 5. Evaluate
python tools/track_cli.py evaluate --model ppo_final.zip --episodes 10

# 6. Test locally
python play_ppo.py --model ppo_final.zip

# 7. Export for production
python tools/track_cli.py export --model ppo_final.zip --download
```

### Cost Optimization

**Modal pricing considerations:**
- GPU time is billed per second
- Volumes storage is free (within limits)
- Minimize idle time by batching experiments

**Tips:**
- Use CPU for quick tests (< 1M steps)
- Reserve GPU for long runs (5M+ steps)
- Prune checkpoints to save storage
- Download artifacts and delete old experiments

---

## Summary

You now have a complete remote training pipeline:

✅ **Externalized track management** (no rebuilds for new tracks)  
✅ **GPU acceleration** with automatic fallback  
✅ **Resume training** from any checkpoint  
✅ **TensorBoard logging** for monitoring  
✅ **Automatic pruning** to save space  
✅ **Metadata tracking** for reproducibility  
✅ **ONNX export** for production deployment  
✅ **Local validation** before remote runs  
✅ **Unified CLI** for all operations  

Happy training! 🏎️💨
