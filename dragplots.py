import numpy as np
import matplotlib.pyplot as plt

# Parameters from your satellite environment
rho_0 = 1.225  # Sea level air density (kg/m^3)
C_D = 2.2      # Drag coefficient
A = 10         # Cross-sectional area (m^2)
mass = 1000    # Satellite mass (kg)
velocity = 7800  # Orbital velocity (m/s)

# Current atmospheric model parameters
H_lower = 7500   # Scale height for <100 km
H_upper = 42000  # Scale height for >100 km
altitude_transition = 100000  # 100 km
rho_100 = rho_0 * np.exp(-altitude_transition / H_lower)

# Altitude range (0 to 800 km)
altitudes = np.linspace(0, 800000, 1000)

# Calculate atmospheric density
densities = []
for alt in altitudes:
    if alt < altitude_transition:
        rho = rho_0 * np.exp(-alt / H_lower)
    else:
        rho = rho_100 * np.exp(-(alt - altitude_transition) / H_upper)
    densities.append(rho)

# Calculate drag acceleration
drag_accelerations = []
for rho in densities:
    drag_acc = 0.5 * rho * velocity**2 * C_D * A / mass
    drag_accelerations.append(drag_acc)

# Plot
plt.figure(figsize=(10, 6))
plt.semilogy(altitudes/1000, drag_accelerations)
plt.xlabel('Altitude (km)')
plt.ylabel('Drag Acceleration (m/s²)')
plt.title('Atmospheric Drag vs Altitude')
plt.grid(True, alpha=0.3)
plt.axvline(x=400, color='red', linestyle='--', label='Target Altitude (400 km)')
plt.legend()
plt.show()

# Print drag at target altitude
target_idx = np.argmin(np.abs(altitudes - 400000))
print(f"Drag acceleration at 400 km: {drag_accelerations[target_idx]:.2e} m/s²")