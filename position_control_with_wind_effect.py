import numpy as np
import os
import math

from aerial_gym.utils.logging import CustomLogger
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.helpers import get_args
from isaacgym import gymapi  # Import the gymapi module
import torch

logger = CustomLogger(__name__)

# === CONFIG ===
wind_start_step = 1000  # When to start applying wind
wind_velocity = 2.0  # m/s in x-direction
hover_height = 1.0  # meters

## initialize env
args = get_args()
logger.warning("This example demonstrates a hover scenario with wind applied in x-direction mid-flight.")

env_manager = SimBuilder().build_env(
    sim_name="base_sim",
    env_name="empty_env",
    robot_name="base_quadrotor",
    controller_name="lmf2_position_control",
    args=None,
    device="cuda:0",
    num_envs=1,
    headless=args.headless,
    use_warp=args.use_warp,
)

env_manager.reset()
dt = env_manager.sim_config().sim.dt

## wind setup!
IGE_env = env_manager.IGE_env
env_handle = IGE_env.env_handles[0]
actor_handle = env_manager.robot_manager.robot_handles[0]
body_index = IGE_env.gym.find_actor_rigid_body_handle(env_handle, actor_handle, "base_link")

## setup for logging
all_positions = []
all_orientations = []
all_euler_angles = []

## start of sim loop
actions = torch.zeros((1, 4), device="cuda:0")
actions[0, 0] = 0.0   # x
actions[0, 1] = 0.0   # y
actions[0, 2] = hover_height  # z
actions[0, 3] = 0.0   # yaw

wind_applied = False

for i in range(2000):
    if i == wind_start_step and not wind_applied:
        logger.warning(f"Applying wind in x-direction at step {i}")
        
        try:
           
            # Calculate drag force
            drag_coefficient = 0.1  # Adjust based on your robot
            frontal_area = 0.01     # m^2, approximate frontal area
            air_density = 1.225     # kg/m^3
            drag_force = 0.5 * air_density * wind_velocity**2 * drag_coefficient * frontal_area
            
            # Create force tensor
            force = torch.tensor([drag_force, 0.0, 0.0], device="cuda:0").view(1, 3)
            force_pos = torch.tensor([0.0, 0.0, 0.0], device="cuda:0").view(1, 3)
            
            # Apply force at the body's center of mass
            IGE_env.gym.apply_rigid_body_force_tensors(
                env_handle, 
                force, 
                force_pos, 
                gymapi.ENV_SPACE
            )
            wind_applied = True
        except Exception as e:
            logger.error(f"Could not apply wind force: {str(e)}")
            # Fallback method - modify actions directly
            try:
                logger.warning("Trying action modification fallback")
                actions[0, 0] = 0.5  # Add constant offset to x-position command
                wind_applied = True
            except Exception as e2:
                logger.error(f"Fallback also failed: {str(e2)}")

    env_manager.step(actions=actions)

    obs = env_manager.get_obs()
    all_positions.append(obs["robot_position"].cpu().numpy())
    all_orientations.append(obs["robot_orientation"].cpu().numpy())
    all_euler_angles.append(obs["robot_euler_angles"].cpu().numpy())

## saving ...
os.makedirs("collected_data", exist_ok=True)
np.savez("collected_data/positions_with_wind.npz", positions=np.stack(all_positions))
np.savez("collected_data/orientations_with_wind.npz", orientations=np.stack(all_orientations))
np.savez("collected_data/euler_angles_with_wind.npz", euler_angles=np.stack(all_euler_angles))

logger.info("Saved hover trajectory data with wind disturbance to collected_data/")


