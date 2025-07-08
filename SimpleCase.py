import gym
from stable_baselines3 import PPO
from satellite_control_env import SatelliteControlEnv
import matplotlib.pyplot as plt
import numpy as np

# TRAINING
env = SatelliteControlEnv()  # Create the environment
model = PPO("MlpPolicy", env, tensorboard_log="./ppo_tensorboard/", clip_range=0.2, clip_range_vf=0.2, n_steps=4096, vf_coef=0.25, gamma=0.9, learning_rate = 3e-3, verbose=1, gae_lambda=0.9)  # Instantiate the PPO agent
model.learn(total_timesteps=1000000)  # Training the model
model.save("rl_model")  # Save the trained model

print('TRAINING COMPLETE')
print("="*6)

# EVALUATION & DATA COLLECTION ACROSS MULTIPLE EPISODES
n_eval_episodes = 50  # Number of episodes to evaluate
max_steps_per_episode = 10000  # Maximum timesteps per episode

# Lists to store episode-level metrics
episode_rewards = []
episode_altitudes = []
episode_velocities = []
episode_thrusts = []
episode_final_altitudes = []
episode_fuel_consumed = []
episode_altitude_errors = []
episode_drag_forces = []  # New list to store drag forces per episode

# Variables to track the best episode
best_reward = float('-inf')
best_reward_idx = -1
best_altitude_error = float('inf')
best_altitude_idx = -1

for episode in range(n_eval_episodes):  # Loop through each episode
    # Reset for new episode
    obs = env.reset()
    done = False

    episode_reward = 0  # Initialize reward for the episode
    altitudes = []  # List to store altitude values for each timestep in the episode
    velocities = []  # List to store velocity values for each timestep in the episode
    thrusts = []  # List to store thrust values for each timestep in the episode
    drag_forces = []  # New list to store drag forces for each timestep

    initial_mass = obs[2]  # Get initial mass
    step_count = 0

    # Loop through each timestep in the episode
    while not done and step_count < max_steps_per_episode:
        action, _states = model.predict(obs)

        obs, reward, done, info = env.step(action)

        # print(f"Episode {episode + 1}: Action={action}, Altitude={obs[0]}, Velocity={obs[1]}, Mass={obs[2]}, Reward={reward}, Drag Force={info['drag_force']}")

        # Store data
        altitudes.append(obs[0])  # Altitude
        velocities.append(obs[1])  # Velocity
        thrusts.append(action[0])  # Thrust
        drag_forces.append(info['drag_force'])  # Drag force

        episode_reward += reward
        step_count += 1

    # Calculate altitude error (average distance from target) for the given episode
    altitude_error = np.mean([abs(alt - 400000) for alt in altitudes])
    episode_altitude_errors.append(altitude_error)

    # Calculate average drag force for the episode
    average_drag_force = np.mean(drag_forces) if drag_forces else 0
    episode_drag_forces.append(average_drag_force)

    # Store episode results
    episode_rewards.append(episode_reward)
    episode_altitudes.append(altitudes)
    episode_velocities.append(velocities)
    episode_thrusts.append(thrusts)
    episode_final_altitudes.append(altitudes[-1])  # Final altitude
    episode_fuel_consumed.append(initial_mass - obs[2])  # Fuel consumed

    # Track best episodes
    if episode_reward > best_reward:
        best_reward = episode_reward
        best_reward_idx = episode

    if altitude_error < best_altitude_error:
        best_altitude_error = altitude_error
        best_altitude_idx = episode

    # print(f"\nEpisode {episode + 1} Summary")
    # print(f"Reward: {episode_reward:.2f}, Steps: {step_count}, Final Altitude: {altitudes[-1]:.2f}m")
    # print(f"Altitude Error: {altitude_error:.2f}m, Fuel Consumed: {initial_mass - obs[2]:.2f}kg")
    # print(f"Average Drag Force: {average_drag_force:.2e} N")
    # print(f"Altitudes: {[f'{alt:.2f}' for alt in altitudes]}")
    # print(f"Velocities: {[f'{vel:.2f}' for vel in velocities]}")
    # print(f"Thrusts: {[f'{thr:.2f}' for thr in thrusts]}")
    # print(f"Drag Forces: {[f'{df:.2e}' for df in drag_forces]}")

    print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Altitude Error={altitude_error:.2f}, Velocity Error={abs(velocities[-1] - 7665.98):.2f}")

# Print information about the best episodes
print("\nBest Episodes Summary:")
print(f"Best by reward: Episode {best_reward_idx + 1}, Reward={episode_rewards[best_reward_idx]:.2f}, Altitude Error={episode_altitude_errors[best_reward_idx]:.2f}")
print(f"Best by altitude: Episode {best_altitude_idx + 1}, Reward={episode_rewards[best_altitude_idx]:.2f}, Altitude Error={episode_altitude_errors[best_altitude_idx]:.2f}")

# VISUALIZATION ACROSS EPISODES

# 1. Plot reward progression across episodes
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_eval_episodes + 1), episode_rewards)
plt.axvline(x=best_reward_idx + 1, color='g', linestyle='--', label='Best Reward Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward Progression Across Episodes')
plt.legend()
plt.grid(True)
plt.savefig("reward_progression.png")
plt.show()

# 2. Plot final altitude for each episode
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_eval_episodes + 1), episode_final_altitudes)
plt.axhline(y=400000, color='r', linestyle='--', label='Target Altitude')
plt.axvline(x=best_altitude_idx + 1, color='b', linestyle='--', label='Best Altitude Episode')
plt.xlabel('Episode')
plt.ylabel('Final Altitude (m)')
plt.title('Final Altitude Achieved in Each Episode')
plt.legend()
plt.grid(True)
plt.savefig("final_altitudes.png")
plt.show()

# 3. Plot fuel consumption across episodes
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_eval_episodes + 1), episode_fuel_consumed)
plt.axvline(x=best_reward_idx + 1, color='g', linestyle='--', label='Best Reward Episode')
plt.xlabel('Episode')
plt.ylabel('Fuel Consumed (kg)')
plt.title('Fuel Efficiency Across Episodes')
plt.legend()
plt.grid(True)
plt.savefig("fuel_consumed.png")
plt.show()

# 4. Plot average drag force across episodes
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_eval_episodes + 1), episode_drag_forces)
plt.axvline(x=best_reward_idx + 1, color='g', linestyle='--', label='Best Reward Episode')
plt.xlabel('Episode')
plt.ylabel('Average Drag Force (N)')
plt.title('Average Drag Force per Episode')
plt.legend()
plt.grid(True)
plt.savefig("average_drag_force.png")
plt.show()

# 5. Plot trajectory of the best reward episode
plt.figure(figsize=(15, 12))

# Plot altitude trajectory
plt.subplot(4, 1, 1)
plt.plot(episode_altitudes[best_reward_idx], linewidth=0.5)
plt.axhline(y=400000, color='r', linestyle='--', label='Target Altitude')
plt.title(f"Altitude Trajectory - Best Reward Episode {best_reward_idx + 1}")
plt.ylabel('Altitude (m)')
plt.legend()
plt.grid(True)

# Plot velocity
plt.subplot(4, 1, 2)
plt.plot(episode_velocities[best_reward_idx], linewidth=0.5)
plt.title(f"Velocity - Best Reward Episode {best_reward_idx + 1}")
plt.ylabel('Velocity (m/s)')
plt.grid(True)

# Plot thrust actions
plt.subplot(4, 1, 3)
plt.plot(episode_thrusts[best_reward_idx], linewidth=0.5)
plt.title(f"Control Actions (Thrust) - Best Reward Episode {best_reward_idx + 1}")
plt.xlabel('Timestep')
plt.ylabel('Thrust (m/s²)')
plt.grid(True)

# Plot drag force
plt.subplot(4, 1, 4)
plt.plot(episode_drag_forces[best_reward_idx], linewidth=0.5)  # Note: episode_drag_forces contains averages, use drag_forces from the episode
plt.title(f"Drag Force - Best Reward Episode {best_reward_idx + 1}")
plt.xlabel('Timestep')
plt.ylabel('Drag Force (N)')
plt.grid(True)

plt.tight_layout()
plt.savefig("best_reward_episode_trajectory.png")
plt.show()

# 6. Plot trajectory of the best altitude episode
plt.figure(figsize=(15, 12))

# Plot altitude trajectory
plt.subplot(4, 1, 1)
plt.plot(episode_altitudes[best_altitude_idx], linewidth=0.5)
plt.axhline(y=400000, color='r', linestyle='--', label='Target Altitude')
plt.title(f"Altitude Trajectory - Best Altitude Episode {best_altitude_idx + 1}")
plt.ylabel('Altitude (m)')
plt.legend()
plt.grid(True)

# Plot velocity
plt.subplot(4, 1, 2)
plt.plot(episode_velocities[best_altitude_idx], linewidth=0.5)
plt.title(f"Velocity - Best Altitude Episode {best_altitude_idx + 1}")
plt.ylabel('Velocity (m/s)')
plt.grid(True)

# Plot thrust actions
plt.subplot(4, 1, 3)
plt.plot(episode_thrusts[best_altitude_idx], linewidth=0.5)
plt.title(f"Control Actions (Thrust) - Best Altitude Episode {best_altitude_idx + 1}")
plt.xlabel('Timestep')
plt.ylabel('Thrust (m/s²)')
plt.grid(True)

# Plot drag force
plt.subplot(4, 1, 4)
plt.plot(drag_forces[best_altitude_idx], linewidth=0.5)  # Note: Use drag_forces from the episode
plt.title(f"Drag Force - Best Altitude Episode {best_altitude_idx + 1}")
plt.xlabel('Timestep')
plt.ylabel('Drag Force (N)')
plt.grid(True)

plt.tight_layout()
plt.savefig("best_altitude_episode_trajectory.png")
plt.show()