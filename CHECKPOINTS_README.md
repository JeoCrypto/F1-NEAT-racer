# Checkpoint Editor - Quick Start

## Launch the Editor

```bash
python3 checkpoint_editor.py
```

**Note**: The editor may show a font warning - this is OK! The editor works fine without fonts; you just won't see text labels.

## How It Works

The editor opens a window showing:
- **Left side**: Your circuit image
- **Right side**: Help panel (if fonts are available)

## Controls

### Placing Checkpoints

1. **Click once** on the left edge of the track (you'll see a yellow circle)
2. **Click again** on the right edge of the track
3. A line appears:
   - **GREEN** = Both points are on track (good!)
   - **RED** = One or both points are off-track (needs fixing)

### Other Controls

- **Right Click**: Delete the nearest checkpoint
- **S Key**: Save checkpoints to `checkpoints_new.txt`
- **L Key**: Load existing checkpoints from `main.py`
- **C Key**: Clear all checkpoints
- **H Key**: Toggle help (if fonts work)
- **Q or ESC**: Quit

## Visual Guide

### Good Checkpoint (GREEN)
```
    Dark Pixel ←──GREEN LINE──→ Dark Pixel
    (on track)                   (on track)
```

### Bad Checkpoint (RED)
```
    White Pixel ←──RED LINE──→ Dark Pixel
    (off-track)                 (on track)
```

## Workflow

1. **Start at the finish line**
   - This will be checkpoint 0
   - Click left edge, then right edge of the track

2. **Follow the racing line**
   - Move forward along the track
   - Place checkpoints every 50-80 pixels
   - In corners: place more checkpoints (every 40px)
   - On straights: fewer checkpoints (every 80-100px)

3. **Watch the colors**
   - GREEN = good, keep going
   - RED = delete it (right-click) and try again

4. **Complete the lap**
   - Place checkpoints all the way around
   - Last checkpoint should be near the start

5. **Save your work**
   - Press **S** to save
   - File saved as `checkpoints_new.txt`

## After Creating Checkpoints

1. Open `checkpoints_new.txt`
2. Copy the checkpoint list
3. Paste into `main.py` around line 76
4. Update `STARTING_POSITION` to be 40-50 pixels before checkpoint 0

## Tips

- **Both endpoints must be on dark pixels** (not white)
- Aim for 50-60 checkpoints total for this circuit size
- More checkpoints in tricky sections = better AI learning
- Save frequently (press S)

## Troubleshooting

**Q: Editor shows font warning**
A: This is normal. The editor works fine without fonts - you just won't see checkpoint numbers or help text.

**Q: How do I know which endpoint is which?**
A: The first click shows a yellow circle. The second click completes the checkpoint.

**Q: Checkpoint keeps showing RED**
A: One or both points are on white (off-track) pixels. Delete it (right-click) and place both points on dark track areas.

**Q: Can't see the help panel**
A: If fonts aren't working, the help panel won't display. Use this README instead for controls.

## Quick Reference Card

| Mouse | Action |
|-------|--------|
| Left Click (1st) | Place first point (yellow circle appears) |
| Left Click (2nd) | Place second point (checkpoint created) |
| Right Click | Delete nearest checkpoint |

| Keyboard | Action |
|----------|--------|
| S | Save to file |
| L | Load from main.py |
| C | Clear all |
| H | Toggle help |
| Q/ESC | Quit |

| Color | Meaning |
|-------|---------|
| 🟢 GREEN | Valid checkpoint - both points on track |
| 🔴 RED | Invalid checkpoint - fix or delete |
| 🟡 YELLOW | Currently placing first point |

## Example Session

```
1. Launch: python3 checkpoint_editor.py
2. Click left side of start/finish line (dark pixel)
3. Click right side of start/finish line (dark pixel)
   → GREEN line appears ✓
4. Move forward ~60 pixels
5. Click left edge of track
6. Click right edge of track
   → GREEN line appears ✓
7. Continue around entire track...
8. Press 'S' to save
9. Copy from checkpoints_new.txt to main.py
10. Done!
```

## Need More Help?

See [CHECKPOINT_GUIDE.md](CHECKPOINT_GUIDE.md) for comprehensive documentation.

Happy checkpoint placing! 🏁
