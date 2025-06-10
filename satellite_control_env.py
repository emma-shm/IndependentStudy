import gym
from gym import spaces
import numpy as np


class SatelliteControlEnv(gym.Env):
    def __init__(self, target_altitude=400000, max_thrust=100, max_mass=1000):
        super(SatelliteControlEnv, self).__init__()

        # Mission parameters
        self.target_altitude = target_altitude
        self.max_thrust = max_thrust
        self.max_mass = max_mass

        # Physical constants
        self.R_E = 6378137  # Earth radius (m)
        self.G = 6.67430e-11  # Gravitational constant
        self.M_E = 5.972e24  # Earth mass (kg)

        # Satellite parameters
        self.C_D = 2.2  # Drag coefficient
        self.A = 4.0  # Cross-sectional area (m²)

        # Atmospheric model parameters
        self.rho_0 = 1.225  # Sea level density (kg/m³)

        # Reference densities at key altitudes (CORRECTED for realistic values)
        self.alt_refs = [0, 100e3, 150e3, 200e3, 300e3, 400e3, 500e3, 600e3]
        self.rho_refs = [1.225, 5.5e-7, 2e-9, 2.5e-10, 1.9e-11, 2.8e-12, 3.8e-13, 1.0e-13]
        self.scale_heights = [7500, 22000, 29000, 37000, 45000, 53000, 62000, 71000]

        # Solar activity parameters
        self.F107 = 150  # Solar flux index (70=min, 150=moderate, 250=max)

        # Episode tracking
        self.episode_count = 0
        self.step_count = 0

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-self.max_thrust,
            high=self.max_thrust,
            shape=(1,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([1000000, 10000, self.max_mass]),
            dtype=np.float32
        )

        # Initialize state
        self.state = None
        self.reset()

    def calculate_base_density(self, altitude):
        """Function that calculates atmospheric density at a given altitude using
        piecewise exponential atmosphere model"""

        # Find the appropriate altitude bracket
        for i in range(len(self.alt_refs) - 1): # looping through the altitude references
            if altitude <= self.alt_refs[i + 1]: # checking to see if the altitude is less than or equal to the next altitude reference
                # if current altitude is less than the next altitude reference, then we know what range the altitude is in, so we can calculate the density
                h_ref = self.alt_refs[i]
                rho_ref = self.rho_refs[i]
                H = self.scale_heights[i]
                rho = rho_ref * np.exp(-(altitude - h_ref) / H)
                return rho

        # Above highest reference
        h_ref = self.alt_refs[-1]
        rho_ref = self.rho_refs[-1]
        H = self.scale_heights[-1]
        rho = rho_ref * np.exp(-(altitude - h_ref) / H)
        return rho

    def calculate_density_with_solar_activity(self, altitude, F107=None):
        """Calculate atmospheric density with solar activity effects"""
        if F107 is None:
            F107 = self.F107 # using the default solar activity index if not provided

        # Base density calculation
        rho_base = self.calculate_base_density(altitude)

        # Enhanced solar activity multiplier for more realistic effects
        if altitude > 300000:  # Above 300km, very strong solar effect
            # Exponential increase with F107
            solar_multiplier = 0.5 * np.exp((F107 - 70) / 100)  # Range: 0.5x to 4.5x
        elif altitude > 200000:  # 200-300km, strong effect
            solar_multiplier = 0.7 + (F107 - 70) / 150  # Range: 0.7x to 2.0x
        elif altitude > 100000:  # 100-200km, moderate effect
            solar_multiplier = 0.85 + (F107 - 70) / 300  # Range: 0.85x to 1.4x
        else:
            solar_multiplier = 1.0  # Minimal effect below 100km

        return rho_base * solar_multiplier

    def calculate_decay_rate(self, state):
        """Calculate instantaneous orbital decay rate in meters per day"""
        x, y, vx, vy, m = state
        r = np.sqrt(x ** 2 + y ** 2)
        altitude = r - self.R_E
        v = np.sqrt(vx ** 2 + vy ** 2)

        # Atmospheric density
        rho = self.calculate_density_with_solar_activity(altitude)

        # Ballistic coefficient
        B = m / (self.C_D * self.A)

        # Simplified decay rate for near-circular orbits
        # da/dt = -2 * pi * a * rho * v * A / m
        decay_rate = -2 * np.pi * r * rho * v * self.A * self.C_D / m * 86400  # m/day

        return decay_rate

    def step(self, action):
        # Current state
        x, y, vx, vy, m = self.state
        r = np.sqrt(x ** 2 + y ** 2)
        altitude = r - self.R_E
        v_magnitude = np.sqrt(vx ** 2 + vy ** 2)

        # Convert 1D radial thrust to 2D components
        radial_thrust = action[0]  # Positive = away from Earth

        # Unit vector pointing away from Earth (radial direction)
        r_hat_x = x / r # dividing the x coordinate of the satellite relative to the center of the Earth by the distance from the center of the Earth
                        # to the satellite to get the unit vector in the x direction
        r_hat_y = y / r # dividing the y coordinate of the satellite relative to the center of the Earth by the distance from the center of the Earth
                        # to the satellite to get the unit vector in the y direction


        # Apply thrust in radial direction
        thrust_x = radial_thrust * r_hat_x / m  # multiplying the radial thrust by the unit vector in the x direction and dividing by the mass to get the acceleration in the x direction
        thrust_y = radial_thrust * r_hat_y / m
        thrust_magnitude = abs(radial_thrust)

        # Calculate acceleration from gravity
        a_gravity_x = -self.G * self.M_E * x / r ** 3
        a_gravity_y = -self.G * self.M_E * y / r ** 3

        # Calculate atmospheric density with solar activity
        rho = self.calculate_density_with_solar_activity(altitude, self.F107)

        # Calculate ballistic coefficient
        B = m / (self.C_D * self.A)

        # Calculate drag force first, then drag acceleration
        F_drag = 0.5 * rho * v_magnitude ** 2 * self.C_D * self.A if v_magnitude > 0 else 0
        a_drag_magnitude = F_drag / m if m > 0 else 0  # Convert to acceleration

        # Apply drag in opposite direction of velocity
        drag_x = -a_drag_magnitude * vx / v_magnitude if v_magnitude > 0 else 0
        drag_y = -a_drag_magnitude * vy / v_magnitude if v_magnitude > 0 else 0

        # Calculate total acceleration
        ax = a_gravity_x + drag_x + thrust_x
        ay = a_gravity_y + drag_y + thrust_y

        # Update velocity and position using Euler integration
        dt = 1.0  # 1 second timestep (was 0.01, too small for orbital dynamics)
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        x_new = x + vx_new * dt
        y_new = y + vy_new * dt

        # Update mass based on thrust (fuel consumption)
        specific_impulse = 300  # seconds (typical for chemical propulsion)
        g0 = 9.81  # standard gravity
        mass_flow_rate = thrust_magnitude / (specific_impulse * g0)
        m_new = max(10, m - mass_flow_rate * dt)

        # New state
        self.state = np.array([x_new, y_new, vx_new, vy_new, m_new])

        # Calculate new altitude and velocity
        r_new = np.sqrt(x_new ** 2 + y_new ** 2)
        altitude_new = r_new - self.R_E
        v_magnitude_new = np.sqrt(vx_new ** 2 + vy_new ** 2)

        # Reward function - encourage staying close to target altitude
        altitude_error = abs(altitude_new - self.target_altitude)
        fuel_penalty = 0.1 * thrust_magnitude
        reward = -1 * altitude_error - 0.1 * fuel_penalty

        # Check termination conditions
        done = altitude_new <= 0 or m_new <= 10 or altitude_new > 500000

        # Create observation
        observation = np.array([altitude_new, v_magnitude_new, m_new])

        # Log decay rate periodically
        if self.step_count % 1000 == 0 and self.step_count > 0:
            decay_rate = self.calculate_decay_rate(self.state)
            print(f"Step {self.step_count}: Alt={altitude_new / 1000:.1f}km, "
                  f"V={v_magnitude_new:.1f}m/s, M={m_new:.1f}kg, "
                  f"B={B:.1f}kg/m², Decay={decay_rate:.2f}m/day")

        self.step_count += 1

        info = {
            'radial_thrust': radial_thrust,
            'ballistic_coefficient': B,
            'atmospheric_density': rho,
            'decay_rate': self.calculate_decay_rate(self.state) if not done else 0,
            'drag_force': F_drag  # Add drag force to info
        }

        return observation, reward, done, info

    def reset(self):
        # Initial conditions
        initial_altitude = self.target_altitude + 10000  # Start 10km above target
        radius = self.R_E + initial_altitude

        # Circular orbit velocity at this altitude
        v_circular = np.sqrt(self.G * self.M_E / radius)

        # Start at x-axis, velocity in y-direction (circular orbit)
        x = radius
        y = 0.0
        vx = 0.0
        vy = v_circular
        m = self.max_mass

        self.state = np.array([x, y, vx, vy, m])

        # Create observation
        observation = np.array([initial_altitude, v_circular, m])

        # Update episode tracking
        self.episode_count += 1
        self.step_count = 0

        # Calculate initial orbital parameters
        B = m / (self.C_D * self.A)
        rho = self.calculate_density_with_solar_activity(initial_altitude)
        decay_rate = self.calculate_decay_rate(self.state)

        print(f"\n{'=' * 60}")
        print(f"Episode {self.episode_count} Reset")
        print(f"Initial altitude: {initial_altitude / 1000:.1f} km")
        print(f"Target altitude: {self.target_altitude / 1000:.1f} km")
        print(f"Circular velocity: {v_circular:.1f} m/s")
        print(f"Initial mass: {m:.1f} kg")
        print(f"Ballistic coefficient: {B:.1f} kg/m²")
        print(f"Atmospheric density: {rho:.2e} kg/m³")
        print(f"Initial decay rate: {decay_rate:.2f} m/day")
        print(f"Solar activity (F10.7): {self.F107}")
        print(f"{'=' * 60}")

        return observation

    def render(self):
        """Optional: Add visualization here"""
        pass

    def set_solar_activity(self, F107):
        """Allow changing solar activity during training"""
        self.F107 = np.clip(F107, 70, 250)  # Keep within realistic bounds