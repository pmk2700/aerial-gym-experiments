# from aerial_gym.utils.logging import CustomLogger

# logger = CustomLogger(__name__)
# from aerial_gym.sim.sim_builder import SimBuilder
# import torch
# from aerial_gym.utils.helpers import get_args

# if __name__ == "__main__":
#     args = get_args()
#     logger.warning("This example demonstrates the use of geometric controllers for a quadrotor.")
#     env_manager = SimBuilder().build_env(
#         sim_name="base_sim",
#         env_name="empty_env",
#         robot_name="base_quadrotor",
#         controller_name="lee_position_control",
#         args=None,
#         device="cuda:0",
#         num_envs=args.num_envs,
#         headless=args.headless,
#         use_warp=args.use_warp,
#     )
#     actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
#     env_manager.reset()
#     for i in range(10000):
#         if i % 1000 == 0:
#             logger.info(f"Step {i}, changing target setpoint.")
#             actions[:, 0:3] = 2.0 * (torch.rand_like(actions[:, 0:3]) * 2 - 1)
#             actions[:, 3] = torch.pi * (torch.rand_like(actions[:, 3]) * 2 - 1)
#             env_manager.reset()
#         env_manager.step(actions=actions)

# from aerial_gym.utils.logging import CustomLogger

# logger = CustomLogger(__name__)
# from aerial_gym.sim.sim_builder import SimBuilder
# import torch
# from aerial_gym.utils.helpers import get_args

# if __name__ == "__main__":
#     args = get_args()
#     logger.warning("This example demonstrates the use of geometric controllers for a quadrotor.")
#     env_manager = SimBuilder().build_env(
#         sim_name="base_sim",
#         env_name="empty_env",
#         robot_name="base_quadrotor",
#         controller_name="lee_position_control",
#         args=None,
#         device="cuda:0",
#         num_envs=args.num_envs,
#         headless=args.headless,
#         use_warp=args.use_warp,
#     )
#     actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
#     env_manager.reset()
#     for i in range(10000):
#         if i % 1000 == 0:
#             logger.info(f"Step {i}, changing target setpoint.")
#             actions[:, 0:3] = 2.0 * (torch.rand_like(actions[:, 0:3]) * 2 - 1)
#             actions[:, 3] = torch.pi * (torch.rand_like(actions[:, 3]) * 2 - 1)
#             env_manager.reset()
#         env_manager.step(actions=actions)
#         #print(type(env_manager.get_obs())) # it is a dictionary
#         obs = env_manager.get_obs()
#         #print(obs.keys()) # got all possible keys
#         #print(f"THe robot positions are: {obs['robot_position']}\n") 64x3 matrix
#         #print(f"THe robot orientations are: {obs['robot_orientation']}\n") 64x4 matrix
#         #print(f"THe robot euler angles are: {obs['robot_euler_angles']}\n") 64x3 matrix


# from aerial_gym.utils.logging import CustomLogger
# from aerial_gym.sim.sim_builder import SimBuilder
# import torch
# import math
# import numpy as np
# from aerial_gym.utils.helpers import get_args
# import os

# logger = CustomLogger(__name__)

# if __name__ == "__main__":
#     args = get_args()
#     logger.warning("This example demonstrates the use of geometric controllers for a quadrotor.")

#     env_manager = SimBuilder().build_env(
#         sim_name="base_sim",
#         env_name="empty_env",
#         ## Large robot
#         # robot_name="base_octarotor",
#         # controller_name="lee_position_control_octarotor",
#         ## SMall robot
#         robot_name="lmf2",
#         controller_name="lee_position_control",
#         ## Mefium-size robot
#         # robot_name="base_quadrotor",
#         # controller_name="lmf2_position_control",
#         args=None,
#         device="cuda:0",
#         # num_envs=args.num_envs,
#         num_envs=1,
#         headless=args.headless,
#         use_warp=args.use_warp,
#     )

#     actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
#     env_manager.reset()

#     all_positions = []
#     all_orientations = []
#     all_euler_angles = []

#     for i in range(10000):
#         if i % 1000 == 0:
#             logger.info(f"Step {i}, changing target setpoint.")
#             actions[:, 0:3] = 2.0 * (torch.rand_like(actions[:, 0:3]) * 2 - 1)
#             actions[:, 3] = torch.pi * (torch.rand_like(actions[:, 3]) * 2 - 1)
#             env_manager.reset()

#         env_manager.step(actions=actions)

#         obs = env_manager.get_obs()

#         # Collectinf the data @ each time step
#         all_positions.append(obs['robot_position'].cpu().numpy())
#         all_orientations.append(obs['robot_orientation'].cpu().numpy())
#         all_euler_angles.append(obs['robot_euler_angles'].cpu().numpy())

#     # Stack all timesteps
#     all_positions = np.stack(all_positions, axis=0)        # for my info: [timesteps, num_envs, 3]
#     all_orientations = np.stack(all_orientations, axis=0)  # for my info: [timesteps, num_envs, 4]
#     all_euler_angles = np.stack(all_euler_angles, axis=0)  # for my info: [timesteps, num_envs, 3]
#     print(obs)
#     # Save each to a separate .npz file
#     os.makedirs("collected_data", exist_ok=True)
#     np.savez("collected_data/positions.npz", positions=all_positions)
#     np.savez("collected_data/orientations.npz", orientations=all_orientations)
#     np.savez("collected_data/euler_angles.npz", euler_angles=all_euler_angles)

#     logger.info("Saved position, orientation, and euler angle data to collected_data/")



#### 8 shape ####

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

# import numpy as np
# import os

# import math

# from aerial_gym.utils.logging import CustomLogger
# from aerial_gym.sim.sim_builder import SimBuilder
# from aerial_gym.utils.helpers import get_args
# import torch

# logger = CustomLogger(__name__)

# # === CONFIG ===
# payload_step = 1000
# payload_mass = 5.0  # kg
# hover_height = 1.0  # meters

# # === INIT ENVIRONMENT ===
# args = get_args()
# logger.warning("This example demonstrates a hover scenario with payload force applied mid-flight.")

# env_manager = SimBuilder().build_env(
#     sim_name="base_sim",
#     env_name="empty_env",
#     robot_name="base_quadrotor",
#     controller_name="lmf2_position_control",
#     args=None,
#     device="cuda:0",
#     num_envs=1,
#     headless=args.headless,
#     use_warp=args.use_warp,
# )

# env_manager.reset()
# dt = env_manager.sim_config().sim.dt

# # === FORCE SETUP ===
# IGE_env = env_manager.IGE_env
# num_bodies = IGE_env.num_rigid_bodies_per_env
# global_force_tensor = IGE_env.global_tensor_dict["global_force_tensor"].view(1, num_bodies, 3)
# payload_force = torch.tensor([0.0, 0.0, -payload_mass * 9.81], device="cuda:0")

# # Get base link index
# env_handle = IGE_env.env_handles[0]
# actor_handle = env_manager.robot_manager.robot_handles[0]
# body_index = IGE_env.gym.find_actor_rigid_body_handle(env_handle, actor_handle, "base_link")

# # === LOGGING SETUP ===
# all_positions = []
# all_orientations = []
# all_euler_angles = []

# # === SIMULATION LOOP ===
# actions = torch.zeros((1, 4), device="cuda:0")
# actions[0, 0] = 0.0   # x
# actions[0, 1] = 0.0   # y
# actions[0, 2] = hover_height  # z
# actions[0, 3] = 0.0   # yaw

# for i in range(2000):
#     if i == payload_step:
#         logger.warning(f"Payload attached at step {i}")

#     if i >= payload_step:
#         IGE_env.global_tensor_dict["global_force_tensor"][body_index] = payload_force
#     else:
#         IGE_env.global_tensor_dict["global_force_tensor"][body_index] = torch.tensor([0.0, 0.0, 0.0], device="cuda:0")

#     env_manager.step(actions=actions)

#     obs = env_manager.get_obs()
#     all_positions.append(obs["robot_position"].cpu().numpy())
#     all_orientations.append(obs["robot_orientation"].cpu().numpy())
#     all_euler_angles.append(obs["robot_euler_angles"].cpu().numpy())

# # === SAVE ===
# os.makedirs("collected_data", exist_ok=True)
# np.savez("collected_data/positions_with_payload.npz", positions=np.stack(all_positions))
# np.savez("collected_data/orientations_with_payload.npz", orientations=np.stack(all_orientations))
# np.savez("collected_data/euler_angles_with_payload.npz", euler_angles=np.stack(all_euler_angles))

# logger.info("Saved hover trajectory data with payload to collected_data/")


