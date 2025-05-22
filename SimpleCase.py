import gym
from stable_baselines3 import PPO
from satellite_control_env import SatelliteControlEnv
import matplotlib.pyplot as plt
import numpy as np

# TRAINING
env = SatelliteControlEnv()  # Create the environment
model = PPO("MlpPolicy", env, learning_rate=0.0001, verbose=1)  # Instantiate the PPO agent
model.learn(total_timesteps=5000000)  # Training the model
model.save("rl_model")  # Save the trained model

# EVALUATION & DATA COLLECTION ACROSS MULTIPLE EPISODES
n_eval_episodes = 50  # Number of episodes to evaluate
max_steps_per_episode = 3000  # Maximum timesteps per episode

# Lists to store episode-level metrics
episode_rewards = []
episode_altitudes = []
episode_velocities = []
episode_thrusts = []
episode_final_altitudes = []
episode_fuel_consumed = []
episode_altitude_errors = []  # New list to track altitude errors

# Variables to track the best episode
best_reward = float('-inf')
best_altitude_error = float('inf')
best_reward_idx = -1
best_altitude_idx = -1

for episode in range(n_eval_episodes): # Loop through each episode
    # Reset for new episode
    obs = env.reset()
    done = False

    episode_reward = 0 # Initialize reward for the episode
    altitudes = [] # list to store altitude values for each timestep in the episode
    velocities = [] # list to store velocity values for each timestep in the episode
    thrusts = [] # list to store thrust values for each timestep in the episode

    initial_mass = obs[2]  # Get initial mass
    step_count = 0

    # Run the actual episode by calculating the action to take at each timestep and then actually taking it
    while not done and step_count < max_steps_per_episode: # while the episode is not done and the step count is less than the max steps
        action, _states = model.predict(obs)  # giving the current observed state of the satellite to the model, which then feeds that to the RL agent and gets an action

        obs, reward, done, info = env.step(action)  # actually implementing the action / updating environment by one timestep based on agent's action

        print(f"Episode {episode + 1}: Action={action}, Altitude={obs[0]}, Velocity={obs[1]}, Mass={obs[2]}, Reward={reward}")

        # Store data
        altitudes.append(obs[0])  # Altitude
        velocities.append(obs[1])  # Velocity
        thrusts.append(action[0])  # Thrust

        episode_reward += reward # Accumulate reward
        step_count += 1

    # Calculate altitude error (average distance from target)
    altitude_error = np.mean([abs(alt - 400000) for alt in altitudes]) # altitudes is a list of altitude values recorded at each timestep during the episode; for each altitude alt in the list, the absolute difference from
                                                                        # the target altitude is computed; then np.mean() calculates the mean of these absolute differences for that episode
    episode_altitude_errors.append(altitude_error)

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

    print(f"\nEpisode {episode + 1} Summary")
    print(f"Reward: {episode_reward:.2f}, Steps: {step_count}, Final Altitude: {altitudes[-1]:.2f}m")
    print(f"Altitude Error: {altitude_error:.2f}m, Fuel Consumed: {initial_mass - obs[2]:.2f}kg")
    print(f"Altitudes: {[f'{alt:.2f}' for alt in altitudes]}")
    print(f"Velocities: {[f'{vel:.2f}' for vel in velocities]}")
    print(f"Thrusts: {[f'{thr:.2f}' for thr in thrusts]}")

    print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Steps={step_count}, Final Altitude={altitudes[-1]:.2f}, Altitude Error={altitude_error:.2f}")

# Calculate the best overall episode (weighted combination)
# Normalize metrics to 0-1 range
norm_rewards = (np.array(episode_rewards) - min(episode_rewards)) / (max(episode_rewards) - min(episode_rewards)) if max(episode_rewards) > min(episode_rewards) else np.zeros(n_eval_episodes)
norm_errors = 1 - (np.array(episode_altitude_errors) - min(episode_altitude_errors)) / (max(episode_altitude_errors) - min(episode_altitude_errors)) if max(episode_altitude_errors) > min(episode_altitude_errors) else np.zeros(n_eval_episodes)

# Calculate combined score (70% reward, 30% altitude maintenance)
combined_scores = 0.7 * norm_rewards + 0.3 * norm_errors
best_overall_idx = np.argmax(combined_scores)

# Print information about the best episodes
print("\nBest Episodes Summary:")
print(
    f"Best by reward: Episode {best_reward_idx + 1}, Reward={episode_rewards[best_reward_idx]:.2f}, Altitude Error={episode_altitude_errors[best_reward_idx]:.2f}")
print(
    f"Best by altitude: Episode {best_altitude_idx + 1}, Reward={episode_rewards[best_altitude_idx]:.2f}, Altitude Error={episode_altitude_errors[best_altitude_idx]:.2f}")
print(
    f"Best overall: Episode {best_overall_idx + 1}, Reward={episode_rewards[best_overall_idx]:.2f}, Altitude Error={episode_altitude_errors[best_overall_idx]:.2f}")

# VISUALIZATION ACROSS EPISODES

# 1. Plot reward progression across episodes
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_eval_episodes + 1), episode_rewards)
plt.axvline(x=best_overall_idx + 1, color='g', linestyle='--', label='Best Overall Episode')
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
plt.axvline(x=best_overall_idx + 1, color='g', linestyle='--', label='Best Overall Episode')
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
plt.axvline(x=best_overall_idx + 1, color='g', linestyle='--', label='Best Overall Episode')
plt.xlabel('Episode')
plt.ylabel('Fuel Consumed (kg)')
plt.title('Fuel Efficiency Across Episodes')
plt.legend()
plt.grid(True)
plt.savefig("fuel_consumed.png")
plt.show()

# 4. NEW: Plot trajectory of the best overall episode
plt.figure(figsize=(15, 12))

# Plot altitude trajectory
plt.subplot(3, 1, 1)
plt.plot(episode_altitudes[best_overall_idx], linewidth=0.5)
plt.axhline(y=400000, color='r', linestyle='--', label='Target Altitude')
plt.title(f"Altitude Trajectory - Best Episode {best_overall_idx + 1}")
plt.ylabel('Altitude (m)')
plt.legend()
plt.grid(True)

# Plot velocity
plt.subplot(3, 1, 2)
plt.plot(episode_velocities[best_overall_idx], linewidth=0.5)
plt.title(f"Velocity - Best Episode {best_overall_idx + 1}")
plt.ylabel('Velocity (m/s)')
plt.grid(True)

# Plot thrust actions
plt.subplot(3, 1, 3)
plt.plot(episode_thrusts[best_overall_idx], linewidth=0.5)
plt.title(f"Control Actions (Thrust) - Best Episode {best_overall_idx + 1}")
plt.xlabel('Timestep')
plt.ylabel('Thrust (m/s²)')
plt.grid(True)

plt.tight_layout()
plt.savefig("best_episode_trajectory.png")
plt.show()