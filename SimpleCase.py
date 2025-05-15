import gym
from stable_baselines3 import PPO
from satellite_control_env import SatelliteControlEnv
import matplotlib.pyplot as plt


# Create the environment
env = SatelliteControlEnv()
# Instantiate the PPO agent
model = PPO("MlpPolicy", env, verbose=1)
# Train the model
model.learn(total_timesteps=100000)
# Save the trained model
model.save("satellite_rl_model")
# Optionally, test the model after training
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    episode_rewards = []  # List to store reward per episode
    for episode in range(100000):
        total_reward = 0
        state = env.reset()
        done = False
        while not done:
            action = model.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward  # Accumulate reward for the episode
            state = next_state
        episode_rewards.append(total_reward)  # Add the total reward of this episode
    # print(f"Action: {action}, New state: {obs}, Reward: {rewards}")

altitudes = []
velocities = []
fuel_usage = []
# Simulate for visualization
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    altitudes.append(obs[0])
    velocities.append(obs[1])
    fuel_usage.append(action[0])

# Plot the results
plt.figure(figsize=(10,6))
plt.subplot(3, 1, 1)
plt.plot(altitudes)
plt.title("Altitude vs. Time")
plt.subplot(3, 1, 2)
plt.plot(velocities)
plt.title("Velocity vs. Time")
plt.subplot(3, 1, 3)
plt.plot(fuel_usage)
plt.title("Fuel Usage (Thrust) vs. Time")
plt.tight_layout()
plt.show()

plt.plot(episode_rewards)  # Plot the total reward for each episode
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward per Episode')
plt.grid(True)
plt.show()






# import numpy as np
# from scipy.integrate import odeint
# import gymnasium as gym
# from gymnasium import spaces
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.logger import configure
# import matplotlib.pyplot as plt
#
# model = PPO("MlpPolicy", "CartPole-v1").learn(total_timesteps=1_000) # When you train an agent using SB3, you pass a total_timesteps parameter to the learn() method which defines the training budget for the agent (how many interaction with the environment are allowed)
#
#
# # Custom Gym Environment for Satellite in LEO
# class SatelliteEnv(gym.Env):
#     def __init__(self):
#         super(SatelliteEnv, self).__init__()
#         # State: [angle (theta), angular velocity (omega), signal strength]
#         self.observation_space = spaces.Box(
#             low=np.array([-np.pi, -10.0, 0.0]),
#             high=np.array([np.pi, 10.0, 1.0]),
#             dtype=np.float32
#         )
#         # Action: torque (tau)
#         self.action_space = spaces.Box(
#             low=np.array([-1.0]),
#             high=np.array([1.0]),
#             dtype=np.float32
#         )
#         # Satellite parameters
#         self.I = 1.0  # Moment of inertia (kg·m²)
#         self.max_steps = 1000  # Maximum steps per episode
#         self.step_count = 0
#         self.state = None
#         self.dt = 0.1  # Time step (seconds)
#
#     def reset(self):
#         # Initialize state: [angle, angular velocity, signal strength]
#         self.state = np.array([
#             np.random.uniform(-np.pi, np.pi),  # Random initial angle
#             np.random.uniform(-1.0, 1.0),      # Random initial angular velocity
#             0.0                                # Initial signal strength
#         ])
#         self.step_count = 0
#         return self.state
#
#     def step(self, action):
#         theta, omega, _ = self.state
#         tau = np.clip(action[0], -1.0, 1.0)  # Torque
#
#         # Update dynamics: angular acceleration = torque / moment of inertia
#         alpha = tau / self.I
#         omega_new = omega + alpha * self.dt
#         theta_new = theta + omega_new * self.dt
#
#         # Keep angle in [-pi, pi]
#         theta_new = ((theta_new + np.pi) % (2 * np.pi)) - np.pi
#
#         # Calculate signal strength (Gaussian centered at theta = 0)
#         signal = np.exp(-theta_new**2 / 0.5)
#
#         # Update state
#         self.state = np.array([theta_new, omega_new, signal])
#
#         # Calculate reward
#         reward = signal - 0.1 * tau**2  # Maximize signal, penalize torque
#
#         self.step_count += 1
#         done = self.step_count >= self.max_steps
#
#         info = {}
#         return self.state, reward, done, info
#
#     def render(self, mode='human'):
#         pass  # Optional: Add visualization later
#
# # Main script
# if __name__ == "__main__":
#     # Create environment
#     env = SatelliteEnv()
#
#     # Check environment for compatibility
#     check_env(env)
#
#     # Set up logging for TensorBoard
#     logger = configure("./ppo_satellite_logs", ["tensorboard"])
#
#     # Initialize PPO model
#     model = PPO(
#         policy="MlpPolicy",  # Multi-layer perceptron policy
#         env=env,
#         learning_rate=3e-4,
#         n_steps=2048,
#         batch_size=64,
#         n_epochs=10,
#         gamma=0.99,
#         verbose=1
#     )
#     model.set_logger(logger)
#
#     # Train the model
#     model.learn(total_timesteps=100000, log_interval=10)
#
#     # Save the model
#     model.save("ppo_satellite")
#
#     # Test the model with visualization
#     env = SatelliteEnv()
#     obs = env.reset()
#     states = [obs]
#     rewards = []
#     total_reward = 0
#
#     for _ in range(1000):
#         action, _ = model.predict(obs)
#         obs, reward, done, _ = env.step(action)
#         states.append(obs)
#         rewards.append(reward)
#         total_reward += reward
#         if done:
#             break
#
#     # Print results
#     print(f"Total reward: {total_reward}")
#     print(f"Final state: angle={obs[0]:.2f} rad, velocity={obs[1]:.2f} rad/s, signal={obs[2]:.2f}")
#
#     # Plot results
#     states = np.array(states)
#     t = np.arange(len(states)) * env.dt
#
#     plt.figure(figsize=(12, 8))
#     plt.subplot(3, 1, 1)
#     plt.plot(t, states[:, 0])
#     plt.ylabel("Angle (rad)")
#     plt.subplot(3, 1, 2)
#     plt.plot(t, states[:, 1])
#     plt.ylabel("Angular Velocity (rad/s)")
#     plt.subplot(3, 1, 3)
#     plt.plot(t, states[:, 2])
#     plt.ylabel("Signal Strength")
#     plt.xlabel("Time (s)")
#     plt.tight_layout()
#     plt.savefig("satellite_performance.png")
#     plt.show()