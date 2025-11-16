#!/usr/bin/env python3
"""Track & Remote Training Management CLI.

Unifies local track asset handling with remote Modal training functions to create
an agent-like interface for managing PPO experiments.

Commands:
    upload        Upload a single track PNG + checkpoint metadata
    list          List remote tracks
    meta          Show metadata for a remote track
    sync          Sync all local tracks defined in local checkpoints.json
    validate      Local dry-run validation of track assets + metadata
    train         Launch remote training (CPU or GPU) with optional auto-download
    evaluate      Evaluate a remote model
    fetch         Fetch an artifact from remote outputs volume
    export        Export model policy to ONNX (remote) with optional download

Examples:
    python tools/track_cli.py upload --png circuit.png --checkpoints-json tracks/checkpoints.json
    python tools/track_cli.py list
    python tools/track_cli.py meta --track circuit.png
    python tools/track_cli.py sync --dir tracks
    python tools/track_cli.py train --track circuit.png --timesteps 1000000 --save-name ppo_ext --gpu --vector-envs 16 --checkpoint-freq 250000 --tensorboard --download-model --download-meta
    python tools/track_cli.py evaluate --model ppo_ext.zip --episodes 5
    python tools/track_cli.py fetch --path ppo_ext.meta.json --out local_meta.json
    python tools/track_cli.py export --model ppo_ext.zip --download

For multi-agent style communication: this CLI speaks the same "contract" as the
remote Modal functions, translating local assets into remote calls.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any
import subprocess
import shlex
import os

# Ensure project root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# We previously attempted direct Python invocation of Modal functions (.call), but
# current Modal version does not expose that attribute. We now shell out to the
# Modal CLI (`modal run file.py::function --args ...`) for all remote operations.


def modal_run(function: str, **kwargs) -> subprocess.CompletedProcess:
    """Invoke a remote Modal function via CLI and return CompletedProcess.

    Boolean kwargs become flags; other values produce --key value pairs.
    Underscores in keys are converted to hyphens to match argparse dests.
    """
    cmd = [sys.executable, "-m", "modal", "run", f"train_ppo_modal.py::{function}"]
    for k, v in kwargs.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        elif v is not None:
            cmd.append(flag)
            cmd.append(str(v))
    # Use text mode; binary artifacts handled separately.
    env = os.environ.copy()
    # Force UTF-8 to avoid Windows charmap encode issues with Unicode (e.g. ✓)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # Disable rich formatting if supported to reduce special glyph usage
    env.setdefault("MODAL_RICH", "0")
    result = subprocess.run(cmd, capture_output=True, env=env)
    # Decode with replacement to avoid crashes
    try:
        result.stdout = result.stdout.decode("utf-8", errors="replace")
    except AttributeError:
        pass
    try:
        result.stderr = result.stderr.decode("utf-8", errors="replace")
    except AttributeError:
        pass
    return result


def encode_png_base64(png_path: Path) -> str:
    data = png_path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def load_checkpoints_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cmd_upload(args: argparse.Namespace) -> None:
    png_path = Path(args.png)
    if not png_path.exists():
        raise FileNotFoundError(f"PNG file not found: {png_path}")

    if args.checkpoints_json.endswith(".json") and Path(args.checkpoints_json).exists():
        meta = load_checkpoints_json(Path(args.checkpoints_json))
        if args.track_name not in meta:
            raise KeyError(f"Track '{args.track_name}' missing in provided checkpoints JSON")
        track_entry = meta[args.track_name]
        cps = track_entry["checkpoints"]
        start_position = ",".join(str(x) for x in track_entry.get("start_position", [375, 410]))
        start_angle = track_entry.get("start_angle", 240.8)
    else:
        # Raw JSON array provided
        cps = json.loads(args.checkpoints_json)
        start_position = args.start_position
        start_angle = args.start_angle

    image_b64 = encode_png_base64(png_path)
    result = modal_run(
        "remote_upload_track",
        track_name=args.track_name,
        image_base64=image_b64,
        checkpoints_json=json.dumps(cps),
        start_position=start_position,
        start_angle=start_angle,
    )
    if result.returncode != 0:
        print("UPLOAD FAILED:\n" + result.stderr.strip())
    else:
        print(result.stdout.strip())


def cmd_list(args: argparse.Namespace) -> None:
    result = modal_run("remote_list_tracks")
    if result.returncode != 0:
        print("LIST FAILED:\n" + result.stderr.strip())
        return
    # Attempt to parse JSON or Python list from stdout
    out = result.stdout.strip()
    try:
        data = json.loads(out)
    except Exception:
        import ast
        try:
            data = ast.literal_eval(out)
        except Exception:
            data = out.splitlines()
    if not data:
        print("No tracks found remotely.")
        return
    for t in data:
        print(t)


def cmd_meta(args: argparse.Namespace) -> None:
    result = modal_run("remote_get_track_meta", track_name=args.track)
    if result.returncode != 0:
        print("META FAILED:\n" + result.stderr.strip())
        return
    print(result.stdout.strip())


def cmd_sync(args: argparse.Namespace) -> None:
    """Sync tracks by creating a temporary Modal function with local tracks embedded."""
    dir_path = Path(args.dir)
    checkpoints_path = dir_path / "checkpoints.json"
    if not checkpoints_path.exists():
        raise FileNotFoundError(f"Local checkpoints.json not found at {checkpoints_path}")
    
    # Create inline helper with tracks baked into image
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tf:
        abs_tracks = str(dir_path.resolve())
        helper_code = f'''import modal
from pathlib import Path

app = modal.App("track-sync")
tracks_vol = modal.Volume.from_name("f1-ppo-tracks", create_if_missing=True)

# Embed local tracks directory into image
image = modal.Image.debian_slim(python_version="3.11").add_local_dir(
    Path(r"{abs_tracks}"),
    remote_path="/embedded_tracks"
)

@app.function(image=image, volumes={{"/root/tracks": tracks_vol}}, timeout=600)
def do_sync():
    import shutil
    from pathlib import Path
    src = Path("/embedded_tracks")
    dst = Path("/root/tracks")
    dst.mkdir(parents=True, exist_ok=True)
    uploaded = []
    for item in src.iterdir():
        if item.is_file():
            shutil.copy(item, dst / item.name)
            uploaded.append(item.name)
    tracks_vol.commit()
    return uploaded
'''
        tf.write(helper_code)
        temp_script = tf.name
    
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        result = subprocess.run(
            [sys.executable, "-m", "modal", "run", temp_script + "::do_sync"],
            capture_output=True,
            env=env,
        )
        if result.returncode != 0:
            stderr_text = result.stderr.decode('utf-8', errors='replace')
            print(f"SYNC FAILED:\\n{stderr_text}")
        else:
            stdout_text = result.stdout.decode('utf-8', errors='replace')
            print(stdout_text)
            print("Sync complete!")
    finally:
        Path(temp_script).unlink(missing_ok=True)


def cmd_train(args: argparse.Namespace) -> None:
    fn_name = "remote_train_gpu" if args.gpu else "remote_train"
    result = modal_run(
        fn_name,
        timesteps=args.timesteps,
        save_name=args.save_name,
        vector_envs=args.vector_envs,
        checkpoint_freq=args.checkpoint_freq,
        tensorboard=args.tensorboard,
        resume_from=args.resume_from,
        prune_keep=args.prune_keep,
        prune_interval=args.prune_interval,
        track_name=args.track_name,
    )
    if result.returncode != 0:
        print("TRAIN FAILED:\n" + result.stderr.strip())
        return
    print(result.stdout.strip())
    base_name = args.save_name
    if args.download_model:
        # Binary fetch via modal run and redirect not robust; instruct user if failure
        fetch = subprocess.run(["modal", "run", "train_ppo_modal.py::remote_fetch_artifact", "--path", f"{base_name}.zip"], capture_output=True)
        if fetch.returncode == 0:
            Path(f"{base_name}.zip").write_bytes(fetch.stdout)
            print(f"Downloaded model artifact {base_name}.zip")
        else:
            print("Model download failed; you can retry manually with:\n  modal run train_ppo_modal.py::remote_fetch_artifact --path {base_name}.zip > {base_name}.zip")
    if args.download_meta:
        meta_name = f"{base_name}.meta.json"
        meta_fetch = subprocess.run(["modal", "run", "train_ppo_modal.py::remote_fetch_artifact", "--path", meta_name], capture_output=True)
        if meta_fetch.returncode == 0:
            Path(meta_name).write_bytes(meta_fetch.stdout)
            print(f"Downloaded metadata {meta_name}")
        else:
            print(f"Metadata download failed; retry manually with modal run train_ppo_modal.py::remote_fetch_artifact --path {meta_name} > {meta_name}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    result = modal_run(
        "remote_evaluate",
        model_name=args.model,
        episodes=args.episodes,
        max_steps=args.max_steps,
    )
    if result.returncode != 0:
        print("EVALUATE FAILED:\n" + result.stderr.strip())
    else:
        print(result.stdout.strip())


def cmd_fetch(args: argparse.Namespace) -> None:
    # Use raw binary stdout
    cmd = ["modal", "run", "train_ppo_modal.py::remote_fetch_artifact", "--path", args.path]
    fetch = subprocess.run(cmd, capture_output=True)
    if fetch.returncode != 0:
        print("FETCH FAILED:\n" + fetch.stderr.decode(errors='ignore'))
        return
    out_path = Path(args.out)
    out_path.write_bytes(fetch.stdout)
    print(f"Fetched {args.path} -> {out_path} ({len(fetch.stdout)} bytes)")


def cmd_validate(args: argparse.Namespace) -> None:
    """Local dry-run validation of track PNG and checkpoints JSON entry."""
    png = Path(args.png)
    if not png.exists():
        print(f"FAIL: PNG '{png}' not found")
        return
    print(f"OK: Found PNG {png}")
    cpath = Path(args.checkpoints_json)
    if not cpath.exists():
        print(f"FAIL: checkpoints.json '{cpath}' not found")
        return
    meta = load_checkpoints_json(cpath)
    if args.track_name not in meta:
        print(f"FAIL: Track '{args.track_name}' missing in checkpoints.json")
        return
    entry = meta[args.track_name]
    cps = entry.get("checkpoints")
    if not isinstance(cps, list) or not cps:
        print("FAIL: checkpoints list invalid or empty")
        return
    # Basic structural checks
    bad = []
    for i, cp in enumerate(cps):
        if not (isinstance(cp, list) and len(cp) == 2 and all(isinstance(pair, list) and len(pair) == 2 for pair in cp)):
            bad.append(i)
    if bad:
        print(f"FAIL: Malformed checkpoints at indices {bad}")
        return
    print(f"OK: {len(cps)} checkpoints structurally valid")
    sp = entry.get("start_position", [375, 410])
    if not (isinstance(sp, list) and len(sp) == 2):
        print("FAIL: start_position invalid")
        return
    print(f"OK: start_position={sp} start_angle={entry.get('start_angle', 240.8)}")
    print("VALIDATION PASSED")


def cmd_export(args: argparse.Namespace) -> None:
    result = modal_run(
        "remote_export_onnx",
        model_name=args.model,
        onnx_name=args.onnx_name,
        opset=args.opset,
    )
    if result.returncode != 0:
        print("EXPORT FAILED:\n" + result.stderr.strip())
        return
    print(result.stdout.strip())
    if args.download:
        # Determine onnx filename from stdout (last token containing .onnx)
        name = None
        for token in result.stdout.split():
            if token.endswith('.onnx'):
                name = Path(token).name
        if not name:
            print("Could not infer ONNX filename from output; fetch manually.")
            return
        fetch = subprocess.run(["modal", "run", "train_ppo_modal.py::remote_fetch_artifact", "--path", name], capture_output=True)
        if fetch.returncode == 0:
            Path(name).write_bytes(fetch.stdout)
            print(f"Downloaded ONNX artifact {name} ({len(fetch.stdout)} bytes)")
        else:
            print(f"ONNX download failed; retry manually with modal run train_ppo_modal.py::remote_fetch_artifact --path {name} > {name}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Track & Remote Training Management CLI")
    sub = p.add_subparsers(dest="command", required=True)

    # upload
    up = sub.add_parser("upload", help="Upload a single track")
    up.add_argument("--png", required=True, help="PNG file path")
    up.add_argument("--track-name", required=True, help="Track name (filename to store)")
    up.add_argument("--checkpoints-json", required=True, help="Path to checkpoints.json OR raw JSON array")
    up.add_argument("--start-position", default="375,410", help="Start position 'x,y'")
    up.add_argument("--start-angle", type=float, default=240.8, help="Start angle heading")
    up.set_defaults(func=cmd_upload)

    # list
    lp = sub.add_parser("list", help="List remote tracks")
    lp.set_defaults(func=cmd_list)

    # meta
    mp = sub.add_parser("meta", help="Show metadata for track")
    mp.add_argument("--track", required=True, help="Track name")
    mp.set_defaults(func=cmd_meta)

    # sync
    sp = sub.add_parser("sync", help="Sync all local tracks")
    sp.add_argument("--dir", default="tracks", help="Directory containing tracks + checkpoints.json")
    sp.add_argument("--force", action="store_true", help="Overwrite existing remote tracks")
    sp.set_defaults(func=cmd_sync)

    # validate
    vp = sub.add_parser("validate", help="Local dry-run validation of track")
    vp.add_argument("--png", required=True, help="Local PNG path")
    vp.add_argument("--track-name", required=True, help="Track name key in checkpoints.json")
    vp.add_argument("--checkpoints-json", required=True, help="Path to checkpoints.json")
    vp.set_defaults(func=cmd_validate)

    # train
    tp = sub.add_parser("train", help="Launch remote training")
    tp.add_argument("--track-name", default="circuit.png", help="Track name to use")
    tp.add_argument("--timesteps", type=int, required=True, help="Total timesteps")
    tp.add_argument("--save-name", required=True, help="Model save base name")
    tp.add_argument("--vector-envs", type=int, default=8, help="Number of vector envs")
    tp.add_argument("--checkpoint-freq", type=int, default=100000, help="Checkpoint frequency")
    tp.add_argument("--tensorboard", action="store_true", help="Enable tensorboard logs")
    tp.add_argument("--resume-from", help="Resume from existing model.zip")
    tp.add_argument("--gpu", action="store_true", help="Use GPU training")
    tp.add_argument("--prune-keep", type=int, help="Keep latest N checkpoints")
    tp.add_argument("--prune-interval", type=int, help="Keep checkpoints at interval multiples")
    tp.add_argument("--download-model", action="store_true", help="Auto-download final model after training")
    tp.add_argument("--download-meta", action="store_true", help="Auto-download metadata after training")
    tp.set_defaults(func=cmd_train)

    # evaluate
    ep = sub.add_parser("evaluate", help="Evaluate a saved model")
    ep.add_argument("--model", required=True, help="Model filename in volume /root/outputs")
    ep.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    ep.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    ep.set_defaults(func=cmd_evaluate)

    # fetch
    fp = sub.add_parser("fetch", help="Fetch an artifact")
    fp.add_argument("--path", required=True, help="Artifact relative path under outputs volume")
    fp.add_argument("--out", required=True, help="Local output path")
    fp.set_defaults(func=cmd_fetch)

    # export
    exp = sub.add_parser("export", help="Export model policy to ONNX")
    exp.add_argument("--model", required=True, help="Model zip filename")
    exp.add_argument("--onnx-name", help="Override output ONNX filename")
    exp.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    exp.add_argument("--download", action="store_true", help="Download ONNX artifact after export")
    exp.set_defaults(func=cmd_export)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
