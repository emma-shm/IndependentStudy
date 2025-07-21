import sympy as sp
import numpy as np
from scipy.signal import place_poles


def find_equilibrium(f, x, u, u_vals=None):
    """
    Finds equilibrium points of a nonlinear dynamical system by solving f(x, u) = 0.

    Parameters:
    f: list of sympy expressions, the nonlinear system dynamics (dx/dt = f(x, u))
    x: list of sympy symbols, state variables
    u: list of sympy symbols, input variables (can be empty)
    u_vals: list of floats or None, fixed input values for equilibrium (None if solving for u)

    Returns:
    eq_points: list of dicts, equilibrium points as {x_i: val, u_j: val} (if u_vals is None)
                or {x_i: val} (if u_vals is provided)
    """
    f = sp.Matrix(f)

    # If u_vals is provided, substitute into f
    if u_vals is not None:
        subs_dict = {ui: ui_val for ui, ui_val in zip(u, u_vals)} # creating a dictionary for substitution of inputs
        f = f.subs(subs_dict) # substituting the inputs into the function
        vars_to_solve = x # Only solve for state variables x
    else: # If u_vals is None, we solve for both state and input variables
        vars_to_solve = x + u # Solve for both state and input variables

    # Solve f(x, u) = 0
    eq_solutions = sp.solve(f, vars_to_solve, dict=True)

    return eq_solutions

def linearize_system(f, x, u, x_e, u_e):
    """
    Linearize a nonlinear system around an equilibrium point.

    Parameters:
    f (function): Nonlinear system function (as list of sympy expressions)
    x (list): List of state variables (as sympy symbols)
    u (list): List of input variables (as sympy symbols)
    x_e (list): Equilibrium state values (as sympy expressions)
    u_e (list): Equilibrium input values (as sympy expressions)

    Returns:
    A: State matrix (sympy.Matrix) that is the Jacobian of the system with respect to the state variables evaluated at the equilibrium
    B: Input matrix (sympy.Matrix) that is the Jacobian of the system with respect to the input variables evaluated at the equilibrium
    """

    # Convert f to a sympy Matrix for vectorized operations
    f = sp.Matrix(f)

    # Compute Jacobian A = df/dx
    A = f.jacobian(x)

    # Substitute equilibrium points into A
    subs_dict = {xi: xei for xi, xei in zip(x, x_e)}
    if u:  # Only include inputs if u is non-empty
        subs_dict.update({ui: uei for ui, uei in zip(u, u_e)})
    A_eq = A.subs(subs_dict)

    # Compute Jacobian B = df/du if inputs exist
    B = None
    if u:
        B = f.jacobian(u)
        B_eq = B.subs(subs_dict)
    else:
        B_eq = None

    return A_eq, B_eq




def check_controllability_observability(A, B, C):
    """
    Check the controllability and observability of a linear dynamical system defined by matrices A, B, and C.

    Parameters:
    A (sympy.Matrix): State matrix.
    B (sympy.Matrix): Input matrix.
    C (sympy.Matrix): Output matrix.

    Returns:
    tuple: (controllable, observable)
    """
    n = A.shape[0]

    # Controllability matrix
    controllability_matrix = B
    for i in range(1, n): # looping through the number of states
        controllability_matrix = controllability_matrix.row_join(A**i * B) # adding the new column to the controllability matrix

    # Observability matrix
    observability_matrix = C
    for i in range(1, n): # looping through the number of states
        observability_matrix = observability_matrix.col_join(C * A**i) # adding the new row to the observability matrix

    # Check rank
    controllable = controllability_matrix.rank() == n
    observable = observability_matrix.rank() == n

    return controllable, observable


def design_state_feedback_controller(A, B, desired_poles, x, u, x_e, u_e, reference_tracking=True):
    """
    Design a state feedback controller for pole placement and optional reference tracking.

    Parameters:
    A (sympy.Matrix): State matrix from linearization
    B (sympy.Matrix): Input matrix from linearization
    desired_poles (list): Desired closed-loop pole locations
    x (list): List of state variables (sympy symbols)
    u (list): List of input variables (sympy symbols)
    x_e (list): Equilibrium state values
    u_e (list): Equilibrium input values
    reference_tracking (bool): Whether to include feedforward gain for tracking

    Returns:
    dict: Controller parameters and control law
    """

    # Convert sympy matrices to numpy for numerical computation
    A_num = np.array(A.evalf(), dtype=float)
    B_num = np.array(B.evalf(), dtype=float)

    # Check if system is SISO or MIMO
    m = B_num.shape[1]  # number of inputs
    n = A_num.shape[0]  # number of states

    if len(desired_poles) != n: # checking if the number of desired poles matches the system order
        raise ValueError(f"Number of desired poles ({len(desired_poles)}) must equal system order ({n})")

    # Design feedback gain matrix K using pole placement
    if m == 1:  # SISO system
        result = place_poles(A_num, B_num, desired_poles)
        K = result.gain_matrix
    else:  # MIMO system
        result = place_poles(A_num, B_num, desired_poles)
        K = result.gain_matrix

    # Design feedforward gain for reference tracking (optional)
    kf = None
    if reference_tracking:
        # For step reference tracking: kf = -1/[C(A-BK)^(-1)B]
        # Assuming we want to track the first state (modify as needed)
        C_track = np.zeros((1, n))
        C_track[0, 0] = 1  # Track first state - modify this as needed

        A_cl = A_num - B_num @ K  # Closed-loop A matrix
        try:
            kf = -1 / (C_track @ np.linalg.inv(A_cl) @ B_num)[0, 0]
        except np.linalg.LinAlgError:
            print("Warning: Could not compute feedforward gain (singular matrix)")
            kf = 1.0

    # Create symbolic control law
    # u = u_e - K*(x - x_e) + kf*r
    delta_x = [xi - xei for xi, xei in zip(x, x_e)]

    # Convert K back to sympy for symbolic representation
    K_sym = sp.Matrix(K)

    # Create control law expression
    if len(u) == 1:  # SISO
        if reference_tracking:
            r = sp.Symbol('r')  # reference signal
            control_law = u_e[0] - (K_sym @ sp.Matrix(delta_x))[0] + kf * r
        else:
            control_law = u_e[0] - (K_sym @ sp.Matrix(delta_x))[0]
    else:  # MIMO
        if reference_tracking:
            r = [sp.Symbol(f'r_{i}') for i in range(m)]
            control_law = [u_e[i] - (K_sym @ sp.Matrix(delta_x))[i] + (kf * r[i] if kf is not None else 0)
                           for i in range(m)]
        else:
            control_law = [u_e[i] - (K_sym @ sp.Matrix(delta_x))[i] for i in range(m)]

    # Verify closed-loop poles
    A_cl = A_num - B_num @ K
    actual_poles = np.linalg.eigvals(A_cl)

    controller_info = {
        'K_matrix': K,
        'K_symbolic': K_sym,
        'feedforward_gain': kf,
        'control_law': control_law,
        'desired_poles': desired_poles,
        'actual_poles': actual_poles,
        'closed_loop_A': A_cl,
        'pole_placement_successful': np.allclose(np.sort(actual_poles), np.sort(desired_poles), rtol=1e-6)
    }

    return controller_info


def print_controller_summary(controller_info):
    """
    Print a nice summary of the designed controller.
    """
    print("=" * 50)
    print("STATE FEEDBACK CONTROLLER DESIGN SUMMARY")
    print("=" * 50)

    print(f"Feedback Gain Matrix K:")
    print(controller_info['K_matrix'])
    print()

    if controller_info['feedforward_gain'] is not None:
        print(f"Feedforward Gain kf: {controller_info['feedforward_gain']:.4f}")
        print()

    print("Control Law:")
    if isinstance(controller_info['control_law'], list):
        for i, law in enumerate(controller_info['control_law']):
            print(f"u_{i} = {law}")
    else:
        print(f"u = {controller_info['control_law']}")
    print()

    print("Pole Placement Results:")
    print(f"Desired poles: {controller_info['desired_poles']}")
    print(f"Actual poles:  {[f'{pole:.4f}' for pole in controller_info['actual_poles']]}")
    print(f"Placement successful: {controller_info['pole_placement_successful']}")
    print("=" * 50)


# Example usage function
def complete_state_feedback_design(f, x, u, u_vals, desired_poles, reference_tracking=True):
    """
    Complete workflow: find equilibrium -> linearize -> check controllability -> design controller
    """
    print("Step 1: Finding equilibrium...")
    eq_points = find_equilibrium(f, x, u, u_vals)
    print(f"Equilibrium points found: {eq_points}")

    if not eq_points:
        print("No equilibrium points found!")
        return None

    # Use first equilibrium point
    eq_point = eq_points[0]
    x_e = [eq_point[xi] for xi in x]
    u_e = u_vals if u_vals is not None else [eq_point[ui] for ui in u]

    print(f"\nStep 2: Linearizing around equilibrium...")
    print(f"x_e = {x_e}, u_e = {u_e}")
    A, B = linearize_system(f, x, u, x_e, u_e)
    print(f"A matrix:\n{A}")
    print(f"B matrix:\n{B}")

    print(f"\nStep 3: Checking controllability...")
    # Create dummy C matrix for controllability check
    C = sp.eye(len(x))
    controllable, _ = check_controllability_observability(A, B, C)
    print(f"System is controllable: {controllable}")

    if not controllable:
        print("System is not controllable! Cannot design arbitrary pole placement controller.")
        return None

    print(f"\nStep 4: Designing state feedback controller...")
    controller = design_state_feedback_controller(A, B, desired_poles, x, u, x_e, u_e, reference_tracking)

    print_controller_summary(controller)

    return {
        'equilibrium': {'x_e': x_e, 'u_e': u_e},
        'linearization': {'A': A, 'B': B},
        'controllable': controllable,
        'controller': controller
    }