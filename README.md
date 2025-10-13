# 🏎️ NEAT F1 Car AI — Teaching an AI to Drive with Python

This project uses **NEAT (NeuroEvolution of Augmenting Topologies)** and **Pygame** to teach an AI how to drive a simple car around an **F1-style racetrack**.  
The AI learns steering and acceleration through evolution — no human driving data or supervision required.


---

## 🧠 How It Works

The simulation evolves neural networks using the [NEAT-Python](https://neat-python.readthedocs.io/en/latest/) library.  
Each generation, cars are tested on the track, and their fitness is determined by how far they go and how many checkpoints they reach without leaving the road.

### Inputs
The car’s **neural network inputs** are:
- Vision ray distances (4–8 directions)
- Car speed
- Car angle  

### Outputs
The network outputs:
- **Steering** value (left/right)
- **Acceleration** value (forward/backward)

---

## 🏁 Fitness Function

The AI is rewarded for staying on track and passing checkpoints, and penalized for crashing or wasting time.

| Condition | Reward / Penalty |
|------------|------------------|
| Reaches next checkpoint | +100 |
| Goes off track | −1000 |
| Moves backward (previous checkpoint) | −50 |
| Each frame alive | −1 |

---

## ⚙️ Configuration

Example NEAT configuration (`config-feedforward.txt`):

```ini
[NEAT]
fitness_criterion     = max
fitness_threshold     = 50000
pop_size              = 150
reset_on_extinction   = False
