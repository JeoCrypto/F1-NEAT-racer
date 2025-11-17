"""Quick script to list volume contents."""
import modal

app = modal.App("list-vol")
vol = modal.Volume.from_name("f1-ppo-checkpoints")

@app.function(volumes={"/root/outputs": vol}, timeout=300)
def list_all():
    from pathlib import Path
    files = []
    for p in Path("/root/outputs").rglob("*"):
        if p.is_file():
            size = p.stat().st_size
            rel_path = str(p.relative_to("/root/outputs"))
            files.append((rel_path, size))
    files.sort()
    
    print("\n" + "=" * 80)
    print("Files in f1-ppo-checkpoints volume:")
    print("=" * 80)
    for path, size in files:
        print(f"{path:60s} {size:>15,} bytes")
    print("=" * 80)
    
    return files

if __name__ == "__main__":
    with app.run():
        files = list_all.remote()
        print(f"\nTotal files: {len(files)}")
        print("\nCheckpoint files found:")
        for path, size in files:
            if 'checkpoint' in path or path.endswith('.zip'):
                print(f"  {path}")
