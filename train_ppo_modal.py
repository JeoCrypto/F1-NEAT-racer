"""Remote PPO training & evaluation on Modal (GPU, resume, TensorBoard).

Setup (once):
    pip install modal
    python -m modal setup

Training (CPU) example:
    modal run train_ppo_modal.py::remote_train --timesteps 2000000 --save-name ppo_modal_run --vector-envs 8 --checkpoint-freq 200000

Training (GPU) example:
    modal run train_ppo_modal.py::remote_train_gpu --timesteps 5000000 --save-name ppo_modal_gpu --vector-envs 16 --checkpoint-freq 250000 --tensorboard --prune-keep 6 --prune-interval 500000

Resume training:
    modal run train_ppo_modal.py::remote_train_gpu --resume-from ppo_modal_gpu.zip --timesteps 1000000 --save-name ppo_modal_gpu_cont --tensorboard

List artifacts:
    modal run train_ppo_modal.py::remote_list_artifacts

Evaluate a saved model (headless):
    modal run train_ppo_modal.py::remote_evaluate --model-name ppo_modal_gpu.zip --episodes 5

Download model artifact bytes (prints size):
    modal run train_ppo_modal.py::remote_fetch_artifact --path ppo_modal_gpu.zip > ppo_modal_gpu.zip

Artifacts:
    - Stored in Modal Volume `f1-ppo-checkpoints` under /root/outputs.
    - Checkpoints: /root/outputs/checkpoints/*
    - Final model: /root/outputs/<save>.zip

Notes:
    - Uses uv for faster, cached dependency installation during image build.
    - Rendering disabled (SDL_VIDEODRIVER=dummy).
    - GPU optional; small MLP often CPU-bound; larger batch/env counts benefit.
    - Fallback GPU preference order logged (GPU_FALLBACK_ORDER) when using remote_train_gpu.
    - Checkpoint pruning optional: retain latest N plus interval milestones.
    - TensorBoard logs stored under /root/outputs/tb/<save>/; download final model if needed.
    - External tracks stored in volume /root/tracks with checkpoints.json.
    - Manage tracks via remote_upload_track / remote_list_tracks / remote_get_track_meta.
    - ONNX export available via remote_export_onnx (policy network only).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import modal


# --------------------------- Modal Configuration --------------------------- #

APP_NAME = "f1-ppo-trainer"
CHECKPOINT_VOLUME_NAME = "f1-ppo-checkpoints"
TRACKS_VOLUME_NAME = "f1-ppo-tracks"

REPO_ROOT = "/root/project"  # POSIX path string to avoid Windows Path backslashes

# Ensure remote project directory is importable
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

PYTHON_VERSION = "3.11"

BASE_PACKAGES = [
    "stable-baselines3==2.7.0",
    "torch>=2.2.0",
    "gymnasium==1.1.1",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "matplotlib>=3.7.0",
    "pillow>=10.0.0",
    "pygame>=2.6.0",
    "tensorboard>=2.14.0",
]

GPU_FALLBACK_ORDER = ["L40S", "A100", "A10G", "V100", "T4"]

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("curl", "git")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        # Install all base packages with uv into system site-packages
        f"~/.local/bin/uv pip install --system {' '.join(BASE_PACKAGES)}"
    )
    # Only add required source dependencies to avoid rebuild errors if tools/ changes
    .add_local_file(Path("racing_env.py"), remote_path=f"{REPO_ROOT}/racing_env.py")
    .add_local_file(Path("car.py"), remote_path=f"{REPO_ROOT}/car.py")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)
tracks_volume = modal.Volume.from_name(TRACKS_VOLUME_NAME, create_if_missing=True)

# GPU image (same packages). Could customize for CUDA wheels if needed.
gpu_image = image


# Mounts deprecated in current modal version; source baked into image.


# --------------------------- Remote Training Logic ------------------------- #

def _train_impl(
    timesteps: int,
    save_name: str,
    vector_envs: int,
    checkpoint_freq: int,
    tensorboard: bool,
    resume_from: str | None,
    prune_keep: int | None,
    prune_interval: int | None,
    track_name: str,
) -> str:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from racing_env import RacingEnv

    def make_env():
        env = RacingEnv(render_mode=None, max_steps=1000, track_name=track_name)
        return Monitor(env)

    if vector_envs > 1:
        env = DummyVecEnv([make_env for _ in range(vector_envs)])
    else:
        env = make_env()

    tb_log_dir = f"/root/outputs/tb/{save_name}" if tensorboard else None

    print("=" * 70)
    print(
        f"Modal PPO Training | timesteps={timesteps:,} envs={vector_envs} resume={'yes' if resume_from else 'no'}"
    )
    print("=" * 70)
    print(f"Save base name: {save_name}")
    print(f"Checkpoint frequency: {checkpoint_freq:,}")
    if tensorboard:
        print(f"TensorBoard log dir: {tb_log_dir}")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    print(f"Track: {track_name}")
    print()

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path="/root/outputs/checkpoints",
        name_prefix=save_name,
    )

    if resume_from:
        resume_path = Path(f"/root/outputs/{resume_from}")
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume model not found: {resume_path}")
        print("Loading existing model for resume...")
        model = PPO.load(str(resume_path), env=env, tensorboard_log=tb_log_dir)
    else:
        print("Creating new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=tb_log_dir,
        )

    try:
        model.learn(total_timesteps=timesteps, callback=checkpoint_callback, progress_bar=False)
    except KeyboardInterrupt:
        print("Training interrupted by user.")

    final_path = f"/root/outputs/{save_name}.zip"
    model.save(final_path)
    print(f"Saved final model to {final_path}")

    # Checkpoint pruning logic
    pruned = []
    kept = []
    cp_dir = Path("/root/outputs/checkpoints")
    if prune_keep is not None and cp_dir.exists():
        candidates = []
        for fp in cp_dir.glob(f"{save_name}_*_steps.zip"):
            name = fp.name
            try:
                # expected pattern: <save_name>_<steps>_steps.zip
                middle = name[len(save_name)+1:]
                step_str = middle.split("_steps")[0]
                steps_val = int(step_str)
                candidates.append((steps_val, fp))
            except Exception:
                print(f"Skipping unparsable checkpoint filename: {name}")
        candidates.sort(key=lambda t: t[0])
        if candidates:
            latest = candidates[-prune_keep:] if prune_keep > 0 else []
            latest_set = {fp for _, fp in latest}
            interval_set = set()
            if prune_interval is not None and prune_interval > 0:
                interval_set = {fp for steps_val, fp in candidates if steps_val % prune_interval == 0}
            keep_set = latest_set | interval_set
            for steps_val, fp in candidates:
                if fp in keep_set:
                    kept.append(fp)
                else:
                    try:
                        fp.unlink()
                        pruned.append(fp)
                    except Exception as e:
                        print(f"Failed to delete {fp}: {e}")
            print(f"Pruning complete: kept={len(kept)} pruned={len(pruned)} (latest={len(latest_set)} interval_kept={len(interval_set)})")
        else:
            print("No checkpoints found to prune.")
    else:
        if prune_keep is not None:
            print("Checkpoint directory missing; skipping pruning.")

    # Write metadata JSON
    import json, time, datetime
    meta = {
        "save_name": save_name,
        "final_model": final_path,
        "timesteps": timesteps,
        "vector_envs": vector_envs,
        "checkpoint_freq": checkpoint_freq,
        "tensorboard": tensorboard,
        "resume_from": resume_from,
        "gpu_type": os.environ.get("MODAL_GPU_TYPE"),
        "prune_keep": prune_keep,
        "prune_interval": prune_interval,
        "pruned_count": len(pruned),
        "kept_count": len(kept),
        "kept_files": [p.name for p in kept],
        "track_name": track_name,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    meta_path = f"/root/outputs/{save_name}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata file {meta_path}")

    for p in Path("/root/outputs").rglob("*"):
        print("ARTIFACT:", p)

    env.close()
    return final_path


@app.function(
    image=image,
    timeout=60 * 60 * 8,
    volumes={"/root/outputs": volume, "/root/tracks": tracks_volume},
)
def remote_train(
    timesteps: int,
    save_name: str,
    vector_envs: int = 4,
    checkpoint_freq: int = 100_000,
    tensorboard: bool = False,
    resume_from: str | None = None,
    prune_keep: int | None = None,
    prune_interval: int | None = None,
    track_name: str = "circuit.png",
) -> str:
    return _train_impl(timesteps, save_name, vector_envs, checkpoint_freq, tensorboard, resume_from, prune_keep, prune_interval, track_name)


@app.function(
    image=gpu_image,
    gpu="any",
    timeout=60 * 60 * 24,  # 24 hours for long training runs
    volumes={"/root/outputs": volume, "/root/tracks": tracks_volume},
)
def remote_train_gpu(
    timesteps: int,
    save_name: str,
    vector_envs: int = 8,
    checkpoint_freq: int = 100_000,
    tensorboard: bool = False,
    resume_from: str | None = None,
    prune_keep: int | None = None,
    prune_interval: int | None = None,
    track_name: str = "circuit.png",
) -> str:
    import torch
    allocated = os.environ.get("MODAL_GPU_TYPE")
    print(f"GPU allocated: {allocated} | availability: cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
    print(f"GPU fallback preference order: {GPU_FALLBACK_ORDER}")
    return _train_impl(timesteps, save_name, vector_envs, checkpoint_freq, tensorboard, resume_from, prune_keep, prune_interval, track_name)


@app.function(
    image=image,
    timeout=60 * 30,  # 30 minutes
    volumes={"/root/outputs": volume, "/root/tracks": tracks_volume},
)
def remote_evaluate(
    model_name: str,
    episodes: int = 5,
    max_steps: int = 1000,
) -> dict:
    """Evaluate a saved PPO model for a number of episodes.

    Returns metrics dict: avg_reward, avg_steps, avg_checkpoints, per_episode list.
    """
    import json
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from stable_baselines3 import PPO
    from racing_env import RacingEnv

    model_path = f"/root/outputs/{model_name}"
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model for evaluation: {model_path}")
    model = PPO.load(model_path)

    episode_rewards = []
    episode_steps = []
    episode_checkpoints = []

    for ep in range(episodes):
        env = RacingEnv(render_mode=None, max_steps=max_steps)
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        last_cp = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            last_cp = info.get("checkpoint", last_cp)
        env.close()
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        episode_checkpoints.append(last_cp)
        print(f"Episode {ep+1}/{episodes}: reward={total_reward:.2f} steps={steps} checkpoints={last_cp}")

    metrics = {
        "model": model_name,
        "episodes": episodes,
        "avg_reward": sum(episode_rewards) / episodes,
        "avg_steps": sum(episode_steps) / episodes,
        "avg_checkpoints": sum(episode_checkpoints) / episodes,
        "per_episode": [
            {"reward": r, "steps": s, "checkpoints": c}
            for r, s, c in zip(episode_rewards, episode_steps, episode_checkpoints)
        ],
    }
    print("Evaluation metrics:\n" + json.dumps(metrics, indent=2))
    return metrics


@app.function(
    image=image,
    timeout=60 * 5,
    volumes={"/root/outputs": volume},
)
def remote_list_artifacts(prefix: str | None = None) -> list[str]:
    """List artifact files stored in the volume."""
    base = Path("/root/outputs")
    files = []
    for p in base.rglob("*"):
        if p.is_file():
            rel = p.relative_to(base).as_posix()
            if prefix is None or rel.startswith(prefix):
                files.append(rel)
    print(f"Found {len(files)} artifact files")
    return sorted(files)


@app.function(
    image=image,
    timeout=60 * 10,
    volumes={"/root/outputs": volume},
)
def remote_fetch_artifact(path: str) -> bytes:
    """Fetch artifact bytes for a given relative path under /root/outputs."""
    base = Path("/root/outputs")
    full = base / path
    if not full.exists() or not full.is_file():
        raise FileNotFoundError(f"Artifact not found: {path}")
    data = full.read_bytes()
    print(f"Fetched artifact {path} size={len(data)} bytes")
    return data


# --------------------------- Track Management ----------------------------- #

@app.function(
    image=image,
    timeout=60 * 5,
    volumes={"/root/tracks": tracks_volume},
)
def remote_upload_track(
    track_name: str,
    image_base64: str,
    checkpoints_json: str,
    start_position: str = "375,410",
    start_angle: float = 240.8,
) -> str:
    """Upload a track PNG and checkpoint metadata.

    Parameters:
        track_name: Filename to store (e.g., circuit.png)
        image_base64: Base64-encoded PNG bytes.
        checkpoints_json: JSON array of checkpoint pairs [[[x1,y1],[x2,y2]], ...]
        start_position: Comma separated start position.
        start_angle: Starting heading angle.
    Returns path stored.
    """
    import base64, json
    track_dir = Path("/root/tracks")
    track_dir.mkdir(parents=True, exist_ok=True)
    img_path = track_dir / track_name
    with open(img_path, "wb") as f:
        f.write(base64.b64decode(image_base64))
    checkpoints_path = track_dir / "checkpoints.json"
    if checkpoints_path.exists():
        with open(checkpoints_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}
    cps = json.loads(checkpoints_json)
    sx, sy = [int(p.strip()) for p in start_position.split(",")]
    meta[track_name] = {
        "checkpoints": cps,
        "start_position": [sx, sy],
        "start_angle": start_angle,
    }
    with open(checkpoints_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return img_path.as_posix()


@app.function(
    image=image,
    timeout=60 * 2,
    volumes={"/root/tracks": tracks_volume},
)
def remote_list_tracks() -> list[str]:
    """List available track names from checkpoints.json."""
    import json
    path = Path("/root/tracks/checkpoints.json")
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return sorted(meta.keys())


@app.function(
    image=image,
    timeout=60 * 2,
    volumes={"/root/tracks": tracks_volume},
)
def remote_get_track_meta(track_name: str) -> dict:
    """Return metadata dict for a given track name."""
    import json
    path = Path("/root/tracks/checkpoints.json")
    if not path.exists():
        raise FileNotFoundError("checkpoints.json missing")
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if track_name not in meta:
        raise KeyError(f"Track {track_name} not found")
    return meta[track_name]


@app.function(
    image=image,
    timeout=60 * 10,
    volumes={"/root/outputs": volume},
)
def remote_export_onnx(
    model_name: str,
    onnx_name: str | None = None,
    opset: int = 17,
) -> str:
    """Export a trained PPO policy to ONNX.

    Parameters:
        model_name: Existing model zip filename under /root/outputs.
        onnx_name: Output ONNX base name (defaults to model_name without .zip + .onnx).
        opset: ONNX opset version.
    Returns path to ONNX file.
    """
    from stable_baselines3 import PPO
    import torch, json
    from pathlib import Path
    base = Path("/root/outputs")
    model_path = base / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = PPO.load(str(model_path))
    obs_shape = model.observation_space.shape
    if obs_shape is None:
        raise ValueError("Could not determine observation space shape for ONNX export")
    obs_dim = obs_shape[0]

    class PolicyWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy
        def forward(self, obs):  # obs: (batch, obs_dim)
            obs = obs.float()
            features = self.policy.extract_features(obs)
            latent_pi, _ = self.policy.mlp_extractor(features)
            dist = self.policy._get_action_dist_from_latent(latent_pi)
            if hasattr(dist, "mean"):
                return dist.mean
            # discrete case: return logits
            return dist.distribution.logits

    wrapper = PolicyWrapper(model.policy).eval()
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    out_name = onnx_name or model_name.replace(".zip", "") + ".onnx"
    onnx_path = base / out_name
    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=opset,
    )
    print(f"Exported ONNX policy to {onnx_path}")
    return onnx_path.as_posix()


# Local entrypoint removed; use direct function invocation syntax shown above.
