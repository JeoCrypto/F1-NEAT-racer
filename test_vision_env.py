"""
Test Vision Environment Locally

Quick validation script for the vision-based environment.
"""

from racing_env_vision import RacingEnvVision
import numpy as np

def test_vision_env():
    """Test vision environment creation and basic functionality."""
    print("🧪 Testing Vision-Based Environment...")
    
    # Create environment
    env = RacingEnvVision(
        track_name="circuit.png",
        camera_height=84,
        camera_width=84,
        use_grayscale=False,
        include_telemetry=True,
        telemetry_history=1,
    )
    
    print(f"✓ Environment created")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Reset
    obs, info = env.reset()
    print(f"\n✓ Reset successful")
    print(f"  Vision shape: {obs['vision'].shape}")
    print(f"  Telemetry shape: {obs['telemetry'].shape if 'telemetry' in obs else 'N/A'}")
    
    # Take random steps
    print(f"\n🏃 Taking 10 random steps...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:.2f}, checkpoint={info['checkpoint']}")
        
        if terminated or truncated:
            print(f"  Episode ended (terminated={terminated}, truncated={truncated})")
            break
    
    # Close
    env.close()
    print(f"\n✅ Test completed!")

if __name__ == "__main__":
    test_vision_env()
