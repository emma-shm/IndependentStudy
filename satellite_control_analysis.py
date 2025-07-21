import sympy as sp
import numpy as np
from controllability_observability_check import linearize_system, check_controllability_observability, design_state_feedback_controller, print_controller_summary
from scipy.signal import place_poles
from satellite_control_env import SatelliteControlEnv

# Initialize environment
env = SatelliteControlEnv(target_altitude=400000, max_thrust=1000, max_mass=100)

# Define symbolic variables
x, y, vx, vy, m = sp.symbols('x y vx vy m')  # State variables
u = sp.Symbol('u')  # Control input (radial_thrust)
state_vars = [x, y, vx, vy, m]
input_vars = [u]

# Physical constants
R_E = env.R_E
G = env.G
M_E = env.M_E
C_D = env.C_D
A = env.A
specific_impulse = 300
g0 = 9.81

# Define radius and altitude
r = sp.sqrt(x**2 + y**2)
altitude = r - R_E

# Atmospheric density at equilibrium (400 km)
rho = env.calculate_density_with_solar_activity(400000)
v_magnitude = sp.sqrt(vx**2 + vy**2)

# Gravitational acceleration
a_gravity_x = -G * M_E * x / r**3
a_gravity_y = -G * M_E * y / r**3

# Drag force and acceleration
F_drag = 0.5 * rho * v_magnitude**2 * C_D * A
a_drag_magnitude = F_drag / m
drag_x = -a_drag_magnitude * vx / v_magnitude
drag_y = -a_drag_magnitude * vy / v_magnitude

# Thrust acceleration
r_hat_x = x / r
r_hat_y = y / r
thrust_x = u * r_hat_x / m
thrust_y = u * r_hat_y / m

# Mass flow rate (smoothed to avoid non-differentiability)
epsilon = 1e-6
mass_flow_rate = sp.sqrt(u**2 + epsilon) / (specific_impulse * g0)

# System dynamics
f = [
    vx,
    vy,
    a_gravity_x + drag_x + thrust_x,
    a_gravity_y + drag_y + thrust_y,
    -mass_flow_rate
]

# Equilibrium point for circular orbit at 400 km
r_eq = R_E + 400000
v_circular = np.sqrt(G * M_E / r_eq)
x_e = r_eq
y_e = 0
vx_e = 0
vy_e = v_circular
m_e = 100
# Calculate equilibrium thrust to counteract drag
F_drag_eq = 0.5 * rho * v_circular**2 * C_D * A
u_e = F_drag_eq  # Thrust in x-direction to balance drag

# Verify equilibrium
eq_point = {x: x_e, y: y_e, vx: vx_e, vy: vy_e, m: m_e, u: u_e}
f_subs = sp.Matrix(f).subs(eq_point).evalf()
print("Equilibrium check (f(x_e, u_e) should be near zero except for dy/dt):")
print(f_subs)

# Linearize the system
A, B = linearize_system(f, state_vars, input_vars, [x_e, y_e, vx_e, vy_e, m_e], [u_e])
A_num = np.array(A.evalf(), dtype=float)
B_num = np.array(B.evalf(), dtype=float)

print("\nLinearized A matrix:")
print(A_num)
print("\nLinearized B matrix:")
print(B_num)

# Check controllability and observability
C = sp.Matrix([
    [x/r, y/r, 0, 0, 0],
    [0, 0, vx/v_magnitude, vy/v_magnitude, 0],
    [0, 0, 0, 0, 1]
]).subs({x: x_e, y: y_e, vx: vx_e, vy: vy_e, m: m_e, r: r_eq, v_magnitude: v_circular})
C_num = np.array(C.evalf(), dtype=float)

controllable, observable = check_controllability_observability(A, B, C)
print(f"\nSystem is controllable: {controllable}")
print(f"System is observable: {observable}")

# Design state feedback controller
desired_poles = [-0.1, -0.2, -0.3, -0.4, -0.5]  # Faster poles for quicker response
controller = design_state_feedback_controller(
    A, B, desired_poles, state_vars, input_vars,
    [x_e, y_e, vx_e, vy_e, m_e], [u_e], reference_tracking=True
)

if controller is None:
    print("Controller design failed. Exiting.")
    exit()

# Print controller summary
print_controller_summary(controller)

# Test controller
K = controller['K_matrix']
kf = controller['feedforward_gain']
x_e = np.array([x_e, y_e, vx_e, vy_e, m_e], dtype=float)
u_e = np.array([u_e], dtype=float)

def apply_controller(state, reference_altitude=400000):
    state_np = np.array(state, dtype=float)
    r = reference_altitude - 400000
    delta_x = state_np - x_e
    u = u_e - K @ delta_x
    if kf is not None:
        u += kf * r
    return u[0]

# Simulate with plotting
import matplotlib.pyplot as plt
altitudes = []
times = []
obs = env.reset()
done = False
steps = 0
max_steps = 1000

while not done and steps < max_steps:
    action = apply_controller(env.state)
    action = np.clip(action, -env.max_thrust, env.max_thrust)
    obs, reward, done, info = env.step([action])
    altitudes.append(obs[0]/1000)
    times.append(steps)
    steps += 1
    if steps % 100 == 0:
        print(f"Step {steps}: Altitude={obs[0]/1000:.1f}km, Velocity={obs[1]:.1f}m/s, Mass={obs[2]:.1f}kg")

print(f"Simulation ended after {steps} steps. Final altitude: {obs[0]/1000:.1f}km")

# Plot results
plt.plot(times, altitudes, label='Altitude')
plt.axhline(y=400, color='r', linestyle='--', label='Target Altitude (400 km)')
plt.xlabel('Time (steps)')
plt.ylabel('Altitude (km)')
plt.title('Satellite Altitude Over Time')
plt.grid(True)
plt.legend()
plt.show()