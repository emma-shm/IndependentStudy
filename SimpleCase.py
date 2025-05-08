import numpy as np
class OrbitMaintenanceEnv:
    def __init__(self):
        self.h_min = 180      # min altitude in km (crash)
        self.h_max = 220      # max altitude for reward
        self.h_target = 200   # desired orbit altitude
        self.v_max = 1.0      # max allowed velocity (km/s)
        self.dt = 1.0         # time step in seconds
        self.max_steps = 500
        self.g = 0.001        # effective gravity (km/s²)
        self.fuel_capacity = 100.0
        self.fuel_burn_rate = 1.0
        self.max_thrust = 0.01
        self.reset()
    def reset(self):
        self.h = np.random.uniform(195, 205)  # initial altitude
        self.v = 0.0
        self.fuel = self.fuel_capacity
        self.steps = 0
        return self._get_state()
    def _drag(self, h):
        return 0.0005 * np.exp(-(h - 180) / 10.0)  # simplified drag
    def _get_state(self):
        return np.array([self.h, self.v, self.fuel], dtype=np.float32)

    def step(self, action):
        # action ∈ [0, 1]: proportion of max thrust
        thrust = np.clip(action, 0, 1) * self.max_thrust
        if self.fuel <= 0:
            thrust = 0
        # Update physics
        drag = self._drag(self.h)
        self.v += (thrust - self.g - drag) * self.dt
        self.h += self.v * self.dt
        self.fuel -= thrust * self.fuel_burn_rate * self.dt
        self.fuel = max(self.fuel, 0)
        self.steps += 1
        # Compute reward
        done = False
        reward = 0
        deviation = abs(self.h - self.h_target)  # Initialize deviation
        if self.h < self.h_min or self.steps >= self.max_steps:
            reward = -100
            done = True
        else:
            if deviation < 10:
                reward += 1
            elif deviation > 10:
                reward -= 0.1
        return self._get_state(), deviation, reward, done, {}

env = OrbitMaintenanceEnv()

for episode in range(10000):
    state = env.reset()
    total_reward = 0
    for t in range(500):
        action = np.random.rand()  # random thrust
        state, deviation, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    print(f"Episode {episode}, deviation: {deviation}, total reward: {total_reward:.2f}")

# # creating a plot of episodes vs rewards
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 5))
# plt.plot()