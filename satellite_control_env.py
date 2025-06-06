import gym
from gym import spaces
import numpy as np


class SatelliteControlEnv(gym.Env):
    def __init__(self, target_altitude=400000, max_thrust=50, max_mass=1000):
        super(SatelliteControlEnv, self).__init__()
        self.target_altitude = target_altitude
        self.max_thrust = max_thrust
        self.max_mass = max_mass
        self.R_E = 6378137
        self.G = 6.67430e-11
        self.M_E = 5.972e24
        self.rho_0 = 1.225
        self.C_D = 2.2
        self.A = 10
        self.H_lower = 7500
        self.H_upper = 42000
        self.altitude_transition = 100000
        self.rho_100 = self.rho_0 * np.exp(-self.altitude_transition / self.H_lower)

        self.state = np.array([self.target_altitude + 10000, 0.0, 0.0, 7660.33, self.max_mass])

        # MODIFIED: 1D action space for radial thrust
        # Positive = thrust away from Earth, Negative = thrust toward Earth
        self.action_space = spaces.Box(low=-self.max_thrust, high=self.max_thrust, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([1000000, 10000, self.max_mass]),
            dtype=np.float32)

        self.episode_count = 0
        self.step_count = 0

    def step(self, action):
        # Current state
        x, y, vx, vy, m = self.state
        r = np.sqrt(x ** 2 + y ** 2)
        altitude = r - self.R_E
        v_magnitude = np.sqrt(vx ** 2 + vy ** 2)

        # MODIFIED: Convert 1D radial thrust to 2D components
        radial_thrust = action[0]  # Single value: positive = away from Earth

        # Unit vector pointing away from Earth (radial direction)
        r_hat_x = x / r
        r_hat_y = y / r

        # Apply thrust in radial direction
        thrust_x = radial_thrust * r_hat_x
        thrust_y = radial_thrust * r_hat_y
        thrust_magnitude = abs(radial_thrust)

        # Calculate acceleration from gravity
        a_gravity_x = -self.G * self.M_E * x / r ** 3
        a_gravity_y = -self.G * self.M_E * y / r ** 3

        # Calculate atmospheric drag
        if altitude < self.altitude_transition:
            rho = self.rho_0 * np.exp(-altitude / self.H_lower)
        else:
            rho = self.rho_100 * np.exp(-(altitude - self.altitude_transition) / self.H_upper)

        drag_magnitude = 0.5 * rho * v_magnitude ** 2 * self.C_D * self.A / m if v_magnitude > 0 else 0
        drag_x = -drag_magnitude * vx / v_magnitude if v_magnitude > 0 else 0
        drag_y = -drag_magnitude * vy / v_magnitude if v_magnitude > 0 else 0

        # Calculate total acceleration
        ax = a_gravity_x + drag_x + thrust_x
        ay = a_gravity_y + drag_y + thrust_y

        # Update velocity and position using Euler integration
        dt = 0.01
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        x_new = x + vx_new * dt
        y_new = y + vy_new * dt

        # Update mass based on thrust
        m_new = max(10, m - 0.005 * thrust_magnitude)

        # New state
        self.state = np.array([x_new, y_new, vx_new, vy_new, m_new])

        # Calculate new altitude and velocity
        r_new = np.sqrt(x_new ** 2 + y_new ** 2)
        altitude_new = r_new - self.R_E
        v_magnitude_new = np.sqrt(vx_new ** 2 + vy_new ** 2)

        # Reward
        altitude_error = abs(altitude_new - self.target_altitude)
        fuel_penalty = 0.1 * thrust_magnitude
        reward = -1 * altitude_error - 0.1 * fuel_penalty

        # Check termination
        done = altitude_new <= 0 or m_new <= 10

        observation = np.array([altitude_new, v_magnitude_new, m_new])
        self.step_count += 1

        return observation, reward, done, {'radial_thrust': radial_thrust}

    def reset(self):
        desired_radius = self.R_E + self.target_altitude + 500
        x = desired_radius
        y = 0.0
        v_magnitude = 7665.7
        vx = 0.0
        vy = v_magnitude
        m = self.max_mass
        self.state = np.array([x, y, vx, vy, m])
        observation = np.array([self.target_altitude + 500, v_magnitude, m])
        self.episode_count += 1
        self.step_count = 0
        print(f"\nEpisode {self.episode_count} Reset")
        print(f"Initial State: x={x:.2f}m, y={y:.2f}m, vx={vx:.2f}m/s, vy={vy:.2f}m/s, m={m:.2f}kg")
        print(f"Desired Radius: {desired_radius:.2f}m, Initial Altitude: {self.target_altitude + 500:.2f}m")
        return observation

    def render(self):
        pass