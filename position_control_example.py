from aerial_gym.utils.logging import CustomLogger
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import math
import numpy as np
from aerial_gym.utils.helpers import get_args
import os

logger = CustomLogger(__name__)

if __name__ == "__main__":
    args = get_args()
    logger.warning("This example demonstrates the use of geometric controllers for a quadrotor.")

    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="empty_env",
        ## Large robot
        # robot_name="base_octarotor",
        # controller_name="lee_position_control_octarotor",

        ## Small robot
        # robot_name="lmf2",
        # controller_name="lee_position_control",

        ## Medium-size robot
        robot_name="base_quadrotor",
        controller_name="lmf2_position_control",
        args=None,
        device="cuda:0",
        num_envs=1,
        headless=args.headless,
        use_warp=args.use_warp,
    )
    print(env_manager.sim_config.__dict__)
    print(type(env_manager.sim_config))
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    env_manager.reset()

    all_positions = []
    all_orientations = []
    all_euler_angles = []

    # Parameters for figure-8 trajectory
    A, B = 1.0, 1.0  # amplitude in x and y
    omega = 0.2      # angular frequency
    z_fixed = 1.0    # constant altitude

    for i in range(10000):
        # t = i * env_manager.dt
        dt = env_manager.sim_config().sim.dt
        t = i * dt

        # Define figure-8 trajectory
        x = A * math.sin(omega * t)
        y = B * math.sin(2 * omega * t)
        z = z_fixed

        actions[:, 0] = x
        actions[:, 1] = y
        actions[:, 2] = z
        actions[:, 3] = 0.0  # yaw angle

        env_manager.step(actions=actions)

        obs = env_manager.get_obs()

        # Collect the data @ each time step
        all_positions.append(obs['robot_position'].cpu().numpy())
        all_orientations.append(obs['robot_orientation'].cpu().numpy())
        all_euler_angles.append(obs['robot_euler_angles'].cpu().numpy())

    # Stack all timesteps
    all_positions = np.stack(all_positions, axis=0)        # [timesteps, num_envs, 3]
    all_orientations = np.stack(all_orientations, axis=0)  # [timesteps, num_envs, 4]
    all_euler_angles = np.stack(all_euler_angles, axis=0)  # [timesteps, num_envs, 3]

    # Save each to a separate .npz file
    os.makedirs("collected_data", exist_ok=True)
    np.savez("collected_data/positions.npz", positions=all_positions)
    np.savez("collected_data/orientations.npz", orientations=all_orientations)
    np.savez("collected_data/euler_angles.npz", euler_angles=all_euler_angles)

    logger.info("Saved position, orientation, and euler angle data to collected_data/")
