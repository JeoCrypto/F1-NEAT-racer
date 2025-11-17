import importlib
import sys

mods = ["torch", "gymnasium", "stable_baselines3", "cv2"]
ok = True
for m in mods:
    try:
        importlib.import_module(m)
        print(f"ok: {m}")
    except Exception as e:
        ok = False
        print(f"fail: {m}: {e}")

sys.exit(0 if ok else 1)
