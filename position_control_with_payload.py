import numpy as np
import os
import math

from isaacgym import gymapi
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.helpers import get_args
import torch

logger = CustomLogger(__name__)

# === CONFIG ===
payload_step = 1000
new_mass = 0.5  # New mass in kg (updated mid-flight)
hover_height = 1.0  # Hover altitude in meters

# === INIT ENVIRONMENT ===
args = get_args()
logger.warning("This example demonstrates runtime payload attachment using CPU pipeline.")

env_manager = SimBuilder().build_env(
    sim_name="base_sim",
    env_name="empty_env",
    robot_name="base_quadrotor",
    controller_name="lmf2_position_control",
    args=None,
    device="cpu",               # Force CPU pipeline
    num_envs=1,
    headless=args.headless,
    use_warp=False,             # Avoid GPU usage
)

env_manager.reset()
dt = env_manager.sim_config().sim.dt
IGE_env = env_manager.IGE_env

env_handle = IGE_env.env_handles[0]
actor_handle = env_manager.robot_manager.robot_handles[0]

## Setup FOr logging
all_positions = []
all_orientations = []
all_euler_angles = []

## Setup for Action
actions = torch.zeros((1, 4))
actions[0, 0] = 0.0   # x
actions[0, 1] = 0.0   # y
actions[0, 2] = hover_height
actions[0, 3] = 0.0   # yaw

mass_changed = False

## SImulation loop for the file
for i in range(2000):
    # Apply payload update
    if i == payload_step and not mass_changed:
        logger.warning(f"Applying new mass at step {i}")

        # body_props = IGE_env.gym.get_actor_rigid_body_properties(IGE_env.sim, env_handle, actor_handle)
        body_props = IGE_env.gym.get_actor_rigid_body_properties(env_handle, actor_handle)

        original_mass = body_props[0].mass

        payload_mass = 0.005

        new_mass = original_mass + payload_mass

        body_props[0].mass = new_mass  # Only base_link (body index 0)

        scale = new_mass / original_mass
        inertia_tensor = body_props[0].inertia
        inertia_tensor.x *= scale
        inertia_tensor.y *= scale
        inertia_tensor.z *= scale
        body_props[0].inertia = inertia_tensor

        # Apply updated properties (must use CPU pipeline!!)
        IGE_env.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=False)

        mass_changed = True

    env_manager.step(actions=actions)

    obs = env_manager.get_obs()
    # Extract safely
    pos = obs["robot_position"].clone().cpu().detach().numpy()
    ori = obs["robot_orientation"].clone().cpu().detach().numpy()
    eul = obs["robot_euler_angles"].clone().cpu().detach().numpy()

    
    all_positions.append(pos.copy())
    all_orientations.append(ori.copy())
    all_euler_angles.append(eul.copy())

## saving ...
os.makedirs("collected_data", exist_ok=True)
np.savez("collected_data/positions_with_mass_change.npz", positions=np.stack(all_positions))
np.savez("collected_data/orientations_with_mass_change.npz", orientations=np.stack(all_orientations))
np.savez("collected_data/euler_angles_with_mass_change.npz", euler_angles=np.stack(all_euler_angles))

logger.info("Saved hover trajectory data with runtime mass change to collected_data/")





