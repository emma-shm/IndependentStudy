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
        self.target_altitude = target_altitude  # Desired altitude in meters (400 km)
        self.max_thrust = max_thrust  # Maximum acceleration due to thrust (m/s^2)
        self.max_mass = max_mass  # Maximum mass of the satellite (kg)
        # Constants
        self.R_E = 6371000  # Radius of Earth (m)
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
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([1000000, 10000, self.max_mass]), dtype=np.float32)
    
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
        h = self.state[0] 
        v = self.state[1]
        m = self.state[2]

        thrust = action[0] # defining thrust to be the first part of the action argument given

        # Taking the current state variables and calculating how they will change in response to action (thrust) based on physics/orbital mechanics
        T = self.T0 - self.L * h  # Temperature at altitude (K)
        rho = self.rho_0 * np.exp(-h / self.H)  # calculating atmospheric density (kg/m^3) of satellite at current altitude
        # Drag force and acceleration
        F_drag = 0.5 * rho * v**2 * self.C_D * self.A # calculating drag force                                                                                                                                                                                                                                                                                                                           
        a_drag = F_drag / m
        # Update velocity (v) based on drag and thrust
        v_new = v - a_drag * 1 + thrust
        # Calculate new orbit radius from velocity
        r_new = self.G * self.M_E / v_new**2
        h_new = r_new - self.R_E # calculate new altitude from radius
        # Update mass based on thrust (fuel consumption)
        m_new = max(10, m - 0.01 * thrust)  # Mass can't go below 10kg

        # Reward: penalize deviation from target altitude and fuel usage
        altitude_error = abs(h_new - self.target_altitude)
        fuel_penalty = 0.1 * thrust
        reward = -altitude_error - fuel_penalty # Negative reward for deviation from target altitude and fuel usage

        # Check if the satellite is in a valid state
        done = False
        if h_new <= 0 or m_new <= 10:  # Satellite crashes or runs out of fuel
            done = True
        self.state = np.array([h_new, v_new, m_new]) # Update state with new values
        return self.state, reward, done, {}
    
    def reset(self):
        '''
        Resets the environment to its initial state. This method is called at the beginning of each episode.
        '''
        # Reset state to initial values
        self.state = np.array([self.target_altitude + 10000, 7800, self.max_mass])  # Same initial conditions
        return self.state
    
    def render(self):
        pass  # Optionally, render the environment (not implemented here)

