import gym
from gym import spaces
import numpy as np
class SatelliteControlEnv(gym.Env):         #creating class that has the basic structure and functionality of the parent class, gym.Env, which is just how environment classes are defined for SB3
    '''
    Custom Environment for controlling a satellite's altitude and velocity that follows gym interface
    https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
    '''
    def __init__(self, target_altitude=400000, max_thrust=5, max_mass=1000):
        super(SatelliteControlEnv, self).__init__()   # initializing the parent class (setting up attributes and structures)                        
        # Parameters for the environment
        self.target_altitude = target_altitude  # Desired altitude in meters (400000 m = 400 km)
        self.max_thrust = max_thrust  # Maximum acceleration due to thrust (m/s^2)
        self.max_mass = max_mass  # Maximum mass of the satellite (kg)
        # Constants
        self.R_E = 6378137  # Radius of Earth (m)
        self.G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
        self.M_E = 5.972e24  # Mass of Earth (kg)
        self.rho_0 = 1.225  # Sea level air density (kg/m^3)
        self.C_D = 2.2  # Drag coefficient (assumed)
        self.A = 10  # Cross-sectional area (m^2)
        self.H = 8500  # Scale height (m) for atmospheric model
        self.R_air = 287.05  # Specific gas constant for dry air (J/(kg·K))
        self.T0 = 288.15  # Sea level temperature (K)
        self.L = 0.0065  # Lapse rate (K/m)
        self.g = 9.81  # Gravitational acceleration (m/s^2)
        # State variables: altitude (m), velocity (m/s), mass (kg)
        self.state = np.array([self.target_altitude + 10000, 7800, self.max_mass])  # Starting 400km + 10km, 7800 m/s orbital speed
        # Action space: thrust (m/s^2)
        self.action_space = spaces.Box(low=0, high=self.max_thrust, shape=(1,), dtype=np.float32)
        # Observation space: altitude (m), velocity (m/s), mass (kg)
        # Observation space: altitude (m), velocity magnitude (m/s), mass (kg)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([1000000, 10000, self.max_mass]),
            dtype=np.float32)

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
        x, y, vx, vy, m = self.state # state is x and y position in meters measured relative to a fixed, geocentric coordinate frame; x and y velocity in meters/sec; mass in kg
        r = np.sqrt(x ** 2 + y ** 2) # radius is the distance from the center of the Earth to the satellite

        # Apply thrust in the velocity direction
        thrust_magnitude = action[0]
        v_magnitude = np.sqrt(vx ** 2 + vy ** 2)
        if v_magnitude == 0:  # Avoid division by zero
            thrust_x, thrust_y = 0, 0
        else:
            thrust_x = thrust_magnitude * vx / v_magnitude
            thrust_y = thrust_magnitude * vy / v_magnitude

        # Calculate acceleration from gravity
        a_gravity_x = -self.G * self.M_E * x / r ** 3
        a_gravity_y = -self.G * self.M_E * y / r ** 3

        # Calculate atmospheric drag
        altitude = r - self.R_E
        rho = self.rho_0 * np.exp(-altitude / self.H)
        drag_magnitude = 0.5 * rho * v_magnitude ** 2 * self.C_D * self.A / m if v_magnitude > 0 else 0
        drag_x = -drag_magnitude * vx / v_magnitude if v_magnitude > 0 else 0
        drag_y = -drag_magnitude * vy / v_magnitude if v_magnitude > 0 else 0

        # Calculate total acceleration
        ax = a_gravity_x + drag_x + thrust_x
        ay = a_gravity_y + drag_y + thrust_y

        # Update velocity and position using Euler integration
        dt = 0.1  # Time step
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        x_new = x + vx_new * dt
        y_new = y + vy_new * dt

        # Update mass based on thrust
        m_new = max(10, m - 0.01 * thrust_magnitude)

        # New state
        self.state = np.array([x_new, y_new, vx_new, vy_new, m_new])

        # print(self.state)

        # Calculate altitude and velocity for observation
        r_new = np.sqrt(x_new ** 2 + y_new ** 2)
        print(f"r_new: {r_new}")
        altitude_new = r_new - self.R_E
        v_magnitude_new = np.sqrt(vx_new ** 2 + vy_new ** 2)

        # Observation: [altitude, velocity_magnitude, mass]
        observation = np.array([altitude_new, v_magnitude_new, m_new])

        # Reward
        altitude_error = abs(altitude_new - self.target_altitude)
        fuel_penalty = 0.1 * thrust_magnitude
        reward = -1.0 * altitude_error - 0.1 * fuel_penalty

        # Check termination
        done = altitude_new <= 0 or m_new <= 10

        return observation, reward, done, {}

    def reset(self):
        # Desired radius: R_E + (target_altitude + 10000)
        desired_radius = self.R_E + self.target_altitude + 10000  # 410,000 m above surface
        # Place satellite on x-axis (y = 0)
        x = desired_radius
        y = 0.0
        # Circular orbit velocity: v = sqrt(G * M_E / r)
        v_magnitude = 7670.02 # Initial velocity (m/s) for circular orbit
        # Velocity is perpendicular to radius, so along y-axis (vx = 0, vy = v)
        vx = 0.0
        vy = v_magnitude
        # Initial mass
        m = self.max_mass
        self.state = np.array([x, y, vx, vy, m])
        # Observation: [altitude, velocity_magnitude, mass]
        altitude = desired_radius - self.R_E
        observation = np.array([altitude, v_magnitude, m])
        return observation
    
    def render(self):
        pass  # Optionally, render the environment (not implemented here)

