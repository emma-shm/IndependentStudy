import gym
from gym import spaces
import numpy as np
class SatelliteControlEnv(gym.Env):         #creating class that has the basic structure and functionality of the parent class, gym.Env, which is just how environment classes are defined for SB3
    '''
    Custom Environment for controlling a satellite's altitude and velocity that follows gym interface
    https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
    '''
    def __init__(self, target_altitude=400000, max_thrust=10, max_mass=1000):
        super(SatelliteControlEnv, self).__init__()
        self.target_altitude = target_altitude  # Desired altitude in meters (400000 m = 400 km)
        self.max_thrust = max_thrust  # Maximum acceleration due to thrust (m/s^2)
        self.max_mass = max_mass  # Maximum mass of the satellite (kg)
        self.R_E = 6378137  # Radius of Earth (m)
        self.G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
        self.M_E = 5.972e24  # Mass of Earth (kg)
        self.rho_0 = 1.225  # Sea level air density (kg/m^3)
        self.C_D = 2.2  # Drag coefficient (assumed)
        self.A = 10  # Cross-sectional area (m^2)
        self.H = 58700  # Scale height (m) for atmospheric model
        self.R_air = 287.05  # Specific gas constant for dry air (J/(kg·K))
        self.T0 = 288.15  # Sea level temperature (K)
        self.L = 0.0065  # Lapse rate (K/m)
        self.g = 9.81  # Gravitational acceleration (m/s^2)
        self.state = np.array([self.target_altitude + 10000, 7660.33, self.max_mass])  # Initial state placeholder
        self.action_space = spaces.Box(low=-self.max_thrust, high=self.max_thrust, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([1000000, 10000, self.max_mass]),
            dtype=np.float32)
        # Initialize episode and step counters
        self.episode_count = 0
        self.step_count = 0

    def step(self, action):
        '''
        Advances the environment by one timestep based on agent's action. Environment includes:
        - State space
        - Action space
        - Transition dynamics (how state changes from current state to new state as result of actions)
        - Reward function
        - Initial states
        When step() is called for a given action (thrust), the method takes the action and state variables and uses the physics equations to calculate the new state of the satellite in response.
        The method then uses new state and constraint parameters to calculate reward, and a boolean indicates if the episode is done.
        '''

        # defining altitude, velocity, and mass locally at given step using the instance variable self.state, which at any point in time should give the current state of the satellite
        # Current state
        x, y, vx, vy, m = self.state
        r = np.sqrt(x ** 2 + y ** 2)
        altitude = r - self.R_E
        v_magnitude = np.sqrt(vx ** 2 + vy ** 2)

        # # Print initial state and action
        # print(f"\nTimestep Start - Episode: {self.episode_count}, Step: {self.step_count}")
        # print(f"Initial State: x={x:.2f}m, y={y:.2f}m, vx={vx:.2f}m/s, vy={vy:.2f}m/s, m={m:.2f}kg")
        # print(f"Action: thrust_x={action[0]:.2f}m/s^2, thrust_y={action[1]:.2f}m/s^2")
        # print(f"Current Radius: r={r:.2f}m, Altitude={altitude:.2f}m, Velocity Magnitude={v_magnitude:.2f}m/s")

        # Apply thrust
        thrust_x = action[0]
        thrust_y = action[1]
        thrust_magnitude = np.sqrt(thrust_x ** 2 + thrust_y ** 2)
        # print(f"Thrust Magnitude: {thrust_magnitude:.2f}m/s^2")

        # Calculate acceleration from gravity
        a_gravity_x = -self.G * self.M_E * x / r ** 3
        a_gravity_y = -self.G * self.M_E * y / r ** 3
        # print(f"Gravity Acceleration: ax={a_gravity_x:.2f}m/s^2, ay={a_gravity_y:.2f}m/s^2")

        # Calculate atmospheric drag
        rho = self.rho_0 * np.exp(-altitude / self.H)
        drag_magnitude = 0.5 * rho * v_magnitude ** 2 * self.C_D * self.A / m if v_magnitude > 0 else 0
        drag_x = -drag_magnitude * vx / v_magnitude if v_magnitude > 0 else 0
        drag_y = -drag_magnitude * vy / v_magnitude if v_magnitude > 0 else 0
        # print(f"Atmospheric Density: rho={rho:.6f}kg/m^3, Drag Magnitude={drag_magnitude:.2f}N")
        # print(f"Drag Acceleration: ax={drag_x:.2f}m/s^2, ay={drag_y:.2f}m/s^2")

        # Calculate total acceleration
        ax = a_gravity_x + drag_x + thrust_x
        ay = a_gravity_y + drag_y + thrust_y
        # print(f"Total Acceleration: ax={ax:.2f}m/s^2, ay={ay:.2f}m/s^2")

        # Update velocity and position using Euler integration
        dt = 0.01
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        x_new = x + vx_new * dt
        y_new = y + vy_new * dt
        # print(f"Euler Integration: dt={dt}s, vx_new={vx_new:.2f}m/s, vy_new={vy_new:.2f}m/s")
        # print(f"Position Update: x_new={x_new:.2f}m, y_new={y_new:.2f}m")

        # Update mass based on thrust
        m_new = max(10, m - 0.01 * thrust_magnitude)
        # print(f"Mass Update: m_new={m_new:.2f}kg, Fuel Consumed={m - m_new:.2f}kg")

        # New state
        self.state = np.array([x_new, y_new, vx_new, vy_new, m_new])

        # Calculate new altitude and velocity
        r_new = np.sqrt(x_new ** 2 + y_new ** 2)
        altitude_new = r_new - self.R_E
        v_magnitude_new = np.sqrt(vx_new ** 2 + vy_new ** 2)
        # print(f"New State: x={x_new:.2f}m, y={y_new:.2f}m, vx={vx_new:.2f}m/s, vy={vy_new:.2f}m/s, m={m_new:.2f}kg")
        # print(f"New Radius: r_new={r_new:.2f}m, Altitude_new={altitude_new:.2f}m, Velocity Magnitude_new={v_magnitude_new:.2f}m/s")

        # Reward
        altitude_error = abs(altitude_new - self.target_altitude)
        # v_ideal = np.sqrt(self.G * self.M_E / (self.target_altitude + self.R_E))
        # velocity_error = abs(v_magnitude_new - v_ideal)
        fuel_penalty = 0.1 * thrust_magnitude
        reward = -1 * altitude_error - 0.1 * fuel_penalty
        # print(f"Reward Components: altitude_error={altitude_error:.2f}m, v_ideal={v_ideal:.2f}m/s, velocity_error={velocity_error:.2f}m/s")
        # print(f"Fuel Penalty: {fuel_penalty:.2f}, Total Reward: {reward:.2f}")

        # Check termination
        done = altitude_new <= 0 or m_new <= 10
        # print(f"Done: {done}, Altitude Check: {altitude_new <= 0}, Mass Check: {m_new <= 10}")

        observation = np.array([altitude_new, v_magnitude_new, m_new])

        # Increment step counter
        self.step_count += 1

        return observation, reward, done, {}

    def reset(self):
        '''
        Resets the environment to an initial state and returns the initial observation, so this is the first method called when starting a new episode
        and defines the initial conditions of the environment.
        :return:
        '''
        # Desired radius: Radius of earth + (target_altitude + 10000)
        desired_radius = self.R_E + self.target_altitude + 10000  # 410,000 m above surface
        x = desired_radius
        y = 0.0
        v_magnitude = 7660.33  # Initial velocity (m/s) for circular orbit
        vx = 0.0
        vy = v_magnitude
        m = self.max_mass
        self.state = np.array([x, y, vx, vy, m])
        observation = np.array([self.target_altitude + 10000, v_magnitude, m])
        self.episode_count += 1  # Increment episode counter
        self.step_count = 0  # Reset step counter
        print(f"\nEpisode {self.episode_count} Reset")
        print(f"Initial State: x={x:.2f}m, y={y:.2f}m, vx={vx:.2f}m/s, vy={vy:.2f}m/s, m={m:.2f}kg")
        print(f"Desired Radius: {desired_radius:.2f}m, Initial Altitude: {self.target_altitude + 10000:.2f}m")

        return observation
    
    def render(self):
        pass  # Optionally, render the environment (not implemented here)

