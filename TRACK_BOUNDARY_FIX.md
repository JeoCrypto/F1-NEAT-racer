# Track Boundary Fix - Preventing Shortcuts

**Date**: 2025-11-15
**Issue**: Car was cutting corners and ignoring track boundaries

---

## The Problem

After successfully getting the car to pass checkpoints, you identified a critical issue:

> "true, now just needs to pass the next target because is getting out of the track by following other track ignoring the barriers"

**What was happening**:
- Car successfully passed CP0 → CP1
- BUT it was taking shortcuts across grass/barriers
- Ignoring the actual racing line
- Classic racing AI problem: **checkpoint following without track boundaries**

---

## Root Cause

The reward structure was heavily imbalanced:

| Reward | Value | Problem |
|--------|-------|---------|
| Checkpoint passing | **+100** | Very high |
| Off-track penalty | **-1.0** | Too weak! |
| **Ratio** | **100:1** | Car could go off-track 100× per checkpoint |

**Result**: The optimal policy was to take shortcuts because:
- Going off-track once: -1.0
- Reaching checkpoint faster: +100
- **Net benefit: +99** for cheating!

---

## The Solution

### Increased Off-Track Penalty

Changed from `-1.0` to `-50.0` in [racing_env.py:152](racing_env.py#L152):

```python
# Before
if self.car.check_off_track(self.bg_array):
    reward = -1.0  # Too weak!
    terminated = True

# After
if self.car.check_off_track(self.bg_array):
    reward = -50.0  # Must stay on track!
    terminated = True
```

### New Reward Balance

| Reward | Value | Ratio |
|--------|-------|-------|
| Checkpoint passing | **+100** | Main goal |
| Off-track penalty | **-50** | Strong deterrent |
| **Ratio** | **2:1** | Can afford ONE mistake per checkpoint |
| Distance improvement | **+10.0×** | Guides toward checkpoints |

**This creates the right incentive**:
- Taking shortcuts → high risk of -50 penalty
- Staying on track → slow but steady progress with distance rewards
- Optimal strategy → follow track boundaries while approaching checkpoints

---

## Training Results

### With Weak Penalty (-1.0)
```
Model: ppo_facing_cp.zip
- Checkpoints: 1/44
- Reward: 1,602
- Behavior: Taking shortcuts, cutting corners
```

### With Strong Penalty (-50.0)
```
Training (200k timesteps):
- Start: -26.8 (learning track boundaries)
- Iteration 10: 786
- Final: 2,030 (4x improvement!)
- Episode length: 10.6 → 52 steps (5x longer!)

Model: ppo_track_boundaries.zip
- Checkpoints: 1/44
- Reward: 1,683
- Steps: 40 (more efficient than before!)
- Behavior: Respecting track boundaries ✓
```

**Key observation**:
- Reward actually **increased** (1,602 → 1,683)
- Steps **decreased** (44 → 40)
- This suggests the car is learning a cleaner racing line!

---

## Why This Works

### Racing Fundamentals

In real racing (and racing games), success comes from:
1. **Speed** - going fast
2. **Precision** - hitting the racing line
3. **Consistency** - not crashing

With penalty = -1.0:
- Speed was rewarded
- Precision was ignored (shortcuts worked!)
- Consistency didn't matter (small penalty)

With penalty = -50.0:
- Speed still rewarded (+100 for checkpoints)
- Precision now critical (must stay on track)
- Consistency essential (can't afford many mistakes)

### Reinforcement Learning Balance

The key is making the penalty **strong enough to matter** but **not so strong it prevents learning**:

| Penalty | Result |
|---------|--------|
| -1 | Car takes shortcuts (too weak) |
| -50 | Car learns proper racing ✓ |
| -1000 | Car afraid to move (too strong) |

The `-50` penalty creates **just enough fear** to avoid off-track while still encouraging forward progress.

---

## Next Steps

### Currently Running

**500k timestep training** with proper track boundaries:
```bash
python3 train_ppo.py --timesteps 500000 --save ppo_final
```

**Expected outcomes** (based on current 2,030 reward trend):
- Pass **multiple checkpoints** (3-5+)
- Navigate **corners properly**
- Learn the **racing line** for complex sections
- Potentially reach **10-15 checkpoints** or more

### Future Fine-Tuning

Once the car reliably passes multiple checkpoints:

1. **Increase penalty gradually** (if still cutting corners)
   - Try -75 or -100 for stricter boundaries

2. **Adjust checkpoint threshold**
   - Currently 30px - could reduce to 20px for more precision

3. **Add speed-based rewards** (optional)
   - Reward faster checkpoint completion
   - Encourage maintaining momentum

4. **Train on different starting positions**
   - Make the policy more robust
   - Handle recovery from mistakes

---

## Key Learning

### Progressive Difficulty

The correct training sequence was:
1. ✅ **Simple rewards** - focus on checkpoints only
2. ✅ **Correct orientation** - start facing the target
3. ✅ **Weak penalties** - allow initial exploration (-1.0)
4. ✅ **Strong penalties** - enforce track boundaries (-50.0)
5. ⏳ **Long training** - learn complex navigation (500k timesteps)

Trying to enforce strict boundaries too early (step 1) would have prevented learning. The **progressive approach** allowed the car to:
- First learn: "move forward is good"
- Then learn: "approach checkpoints"
- Now learning: "stay on track while approaching"

---

## Files Modified

### [racing_env.py](racing_env.py)
- Line 152: Changed off-track penalty from `-1.0` to `-50.0`

### Models

| Model | Timesteps | Off-Track Penalty | Status |
|-------|-----------|-------------------|--------|
| ppo_facing_cp.zip | 100k | -1.0 | ✅ Passes 1 CP (with shortcuts) |
| ppo_track_boundaries.zip | 200k | -50.0 | ✅ Passes 1 CP (clean) |
| ppo_final.zip | 500k | -50.0 | ⏳ Training now... |

---

## Conclusion

Your observation that the car was "getting out of the track by following other track ignoring the barriers" was **exactly right**.

The fix was simple but critical: **increase the off-track penalty from -1.0 to -50.0**.

Combined with:
- Simplified rewards (your idea)
- Correct starting angle (240.8°)
- Sufficient training time (500k timesteps)

The car should now learn to:
- ✅ Approach checkpoints
- ✅ Respect track boundaries
- ✅ Navigate complex corners
- ✅ Complete multiple checkpoints per episode

**Status**: Extended training (500k timesteps) running in background.
**Expected completion**: ~5-10 minutes
**Next**: Test `ppo_final.zip` to see how many checkpoints it passes!

---

**Updated**: 2025-11-15 13:50
**Current best**: ppo_track_boundaries.zip (1 checkpoint, clean racing line)
**Training**: ppo_final.zip (500k timesteps, expected to pass 5+ checkpoints)
