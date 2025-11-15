# Checkpoint Placement Guide

## Overview

This guide will help you create properly-placed checkpoints for your racing track. Good checkpoint placement is crucial for the AI to learn the track properly.

## Tools Available

### 1. **checkpoint_editor.py** (Interactive Editor)

The main tool for placing and editing checkpoints.

**Features:**
- Click-and-place checkpoint creation
- Visual validation (green = valid, red = invalid)
- Load existing checkpoints from main.py
- Save to formatted output file
- Delete and edit checkpoints

**Usage:**
```bash
python3 checkpoint_editor.py
```

**Controls:**
- **Left Click (1st)**: Place first endpoint of checkpoint
- **Left Click (2nd)**: Place second endpoint (completes checkpoint)
- **Right Click**: Delete nearest checkpoint
- **S**: Save checkpoints to file
- **L**: Load existing checkpoints from main.py
- **C**: Clear all checkpoints
- **H**: Toggle help overlay
- **Q/ESC**: Quit

### 2. **auto_checkpoint_generator.py** (Automatic Generation)

Programmatically generates checkpoints (experimental).

**Usage:**
```bash
python3 auto_checkpoint_generator.py --spacing 80
```

Note: Auto-generated checkpoints will likely need manual adjustment.

## Best Practices for Checkpoint Placement

### Rule 1: Both Endpoints Must Be On Track

✓ **CORRECT**: Both endpoints on dark track surface
```
    Track Edge  ←→  Track Edge
    Dark pixel  ←→  Dark pixel
```

✗ **WRONG**: Endpoints extending into white areas
```
    Off-track   ←→  Off-track
    White pixel ←→  White pixel
```

### Rule 2: Checkpoints Should Span Track Width

Checkpoints should go from one side of the track to the other, perpendicular to the racing line.

```
         |
    -----+-----  ← Checkpoint spans full width
         |
```

### Rule 3: Spacing

- **Straights**: 80-120 pixels apart
- **Corners**: 40-60 pixels apart (more checkpoints in turns)
- **Complex sections**: 30-50 pixels apart

### Rule 4: Coverage

- Cover the entire track from start to finish
- No large gaps between checkpoints
- Extra checkpoints in tricky sections

### Rule 5: Checkpoint 0 Special Case

Checkpoint 0 is the start/finish line:
- Should be placed where cars cross to complete a lap
- Starting position should be 30-50 pixels BEFORE this checkpoint
- Car must be able to reach it easily from start

## Step-by-Step Workflow

### Method 1: Manual Placement (Recommended)

1. **Launch the editor:**
   ```bash
   python3 checkpoint_editor.py
   ```

2. **Start at the finish line:**
   - This will be checkpoint 0
   - Click on left edge of track (dark pixel)
   - Click on right edge of track (dark pixel)
   - You should see a GREEN line if both points are valid

3. **Follow the track:**
   - Move along the track in racing direction
   - Place checkpoints at regular intervals
   - Watch the color: GREEN = good, RED = needs adjustment

4. **Extra checkpoints in corners:**
   - Place 2-3 checkpoints per corner
   - Ensures AI learns the racing line through turns

5. **Complete the lap:**
   - End near checkpoint 0
   - The last checkpoint should lead back to start

6. **Save your work:**
   - Press 'S' to save
   - File saved as `checkpoints_new.txt`

7. **Copy to main.py:**
   - Open `checkpoints_new.txt`
   - Copy the checkpoint list
   - Paste into `main.py` replacing old checkpoints

### Method 2: Load and Edit Existing

1. **Launch editor and load current checkpoints:**
   ```bash
   python3 checkpoint_editor.py
   ```
   Press 'L' to load from main.py

2. **Fix invalid checkpoints:**
   - Red checkpoints need adjustment
   - Right-click to delete bad ones
   - Left-click to place new ones

3. **Save when done:**
   - Press 'S'

## Validation Checklist

Before using your checkpoints, verify:

- [ ] All checkpoints are GREEN (valid)
- [ ] Checkpoints cover entire track
- [ ] No large gaps in coverage
- [ ] Checkpoint 0 is at start/finish line
- [ ] Checkpoints numbered in racing order
- [ ] Corners have extra checkpoints
- [ ] Both endpoints on dark pixels

## Common Issues

### Issue: Checkpoint shows as RED

**Cause**: One or both endpoints are on white (off-track) pixels

**Fix**:
1. Right-click to delete it
2. Place new checkpoint with both points on dark track surface

### Issue: Too many/too few checkpoints

**Recommended counts:**
- Small track (like karting circuit): 40-60 checkpoints
- Medium track: 60-100 checkpoints
- Large track: 100+ checkpoints

### Issue: Car not detecting checkpoints

**Possible causes:**
1. Starting position inside checkpoint 0 collision zone
   - Fix: Move start position 40+ pixels before CP0
2. Checkpoints too far apart
   - Fix: Add more checkpoints
3. Checkpoints in wrong order
   - Fix: Verify numbering follows racing line

## Technical Details

### Checkpoint Format

```python
checkpoints: list[tuple[tuple[int, int], tuple[int, int]]] = [
    ((x1, y1), (x2, y2)),  # 0 - start/finish
    ((x1, y1), (x2, y2)),  # 1
    # ...
]
```

Each checkpoint is a tuple of two points:
- First point: (x1, y1)
- Second point: (x2, y2)

### Track Detection

The editor considers pixels "on track" if:
- RGB value < 100 (dark pixels)
- This matches the main.py detection logic

### Collision Detection

In main.py, cars detect checkpoints within 20 pixels of the checkpoint midpoint:
```python
threshold: float = 20.0
```

## Example: Creating Checkpoints for Braga Circuit

1. **Start at the main straight:**
   - Checkpoint 0 at start/finish line

2. **Follow into turn 1 (first left):**
   - Place 3 checkpoints through the turn
   - Space them 40-50px apart

3. **Through the S-curves:**
   - Place checkpoints every 40px
   - Ensure they follow the racing line

4. **Back straight:**
   - Wider spacing (80px)
   - Fewer checkpoints needed

5. **Final corners back to start:**
   - Resume closer spacing (50px)

6. **Total checkpoints:**
   - Aim for 50-60 for this circuit size

## Files Generated

- `checkpoints_new.txt` - Your new checkpoints (from editor)
- `checkpoints_auto.txt` - Auto-generated checkpoints
- `main.py` - Copy checkpoints here for use in training

## Updating main.py

After creating checkpoints:

1. Open `checkpoints_new.txt`
2. Copy the entire checkpoints list
3. Open `main.py`
4. Find the checkpoints section (around line 76)
5. Replace old checkpoints with new ones
6. Verify starting position is appropriate (30-50px before CP0)

## Quick Reference

| Key | Action |
|-----|--------|
| Left Click | Place checkpoint point |
| Right Click | Delete nearest checkpoint |
| S | Save checkpoints |
| L | Load from main.py |
| C | Clear all |
| H | Toggle help |
| Q | Quit |

| Color | Meaning |
|-------|---------|
| Green | Valid checkpoint (both points on track) |
| Red | Invalid checkpoint (point(s) off track) |
| Yellow | Currently placing |
| Cyan | Next target checkpoint |

## Tips

- Work methodically from start to finish
- Check your work as you go (watch for RED checkpoints)
- Save frequently (press 'S')
- Test with training after placing ~20 checkpoints to verify they work
- You can always add more checkpoints later

## Need Help?

If checkpoints aren't working:
1. Check that all are GREEN in the editor
2. Verify checkpoint 0 is at start/finish
3. Confirm starting position is 40+ pixels from CP0
4. Ensure no gaps in coverage

Good luck! 🏁
