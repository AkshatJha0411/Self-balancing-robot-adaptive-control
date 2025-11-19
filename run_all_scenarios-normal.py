# run_all_scenarios.py
# Implements Phase 2, Steps 4 & 6: Replicating all simulation figures.
# This script runs all four scenarios from Section VI-A.
# Includes FIX for numerical instability and NameError.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Import our classes from the other files
from robot_model import RobotModel
from controller import Controller

# <<< THIS IS THE LINE THAT FIXES THE 'TypeError'
def run_simulation(scenario_name, y0, y_ref, t_end, pos_gain_scale=1.0):
    """
    Runs a full closed-loop simulation for a given scenario.
    
    Args:
        scenario_name (str): Title for the plot (e.g., "Figure 11")
        y0 (list): Initial state [theta, theta_dot, x, x_dot]
        y_ref (list): Reference state [theta_ref, x_ref]
        t_end (float): Simulation end time [s]
        pos_gain_scale (float): Multiplier for the PD position gains.
                                1.0 = full gains, 0.0 = off.
    """
    print(f"\nRunning simulation for: {scenario_name}...")
    
    robot = RobotModel()
    controller = Controller()
    
    # Simulation parameters
    T_START = 0.0
    DT = 0.01     # Time step for controller update [s]

    # Set initial and reference states
    y_current = np.array(y0)
    
    # --- MODIFICATION: Scale position gains ---
    # This is our fix for the numerical instability
    controller.Kp_pos *= pos_gain_scale
    controller.Kd_pos *= pos_gain_scale
    if pos_gain_scale != 1.0:
        print(f"INFO: Position gains scaled by {pos_gain_scale}")
    # --- End of modification ---

    # Reset controller states (like integral term z1)
    controller.reset_states()

    # Create time array
    t_eval = np.arange(T_START, t_end, DT)
    num_steps = len(t_eval)

    # Prepare array to store the results history
    history = np.zeros((num_steps, 5)) # [t, x1, x2, x3, x4]

    for i in range(num_steps):
        t = t_eval[i]
        history[i, 0] = t
        history[i, 1:] = y_current
        
        # --- Control Step ---
        C_in = controller.calculate_control_torque(
            y=y_current,
            y_ref=y_ref,
            dt=DT,
            robot_params=robot
        )
        
        # --- Plant Step ---
        dynamics_with_control = lambda t_solve, y_solve: robot.system_dynamics(
            t_solve, y_solve, C_in
        )
        
        sol = solve_ivp(
            fun=dynamics_with_control,
            t_span=[t, t + DT],
            y0=y_current,
            t_eval=[t + DT]
        )
        
        if sol.status != 0:
            print(f"WARNING: Solver failed at t={t}")
            break
            
        y_current = np.array(sol.y).flatten()

    print("Simulation complete.")
    
    # --- Plot results for this scenario ---
    plot_results(scenario_name, history)


def plot_results(scenario_name, history_data):
    """
    Generates a 2-panel plot for a simulation run,
    matching the paper's style.
    """
    # Extract data
    time = history_data[:, 0]
    theta_rad = history_data[:, 1]
    x_m = history_data[:, 3]
    theta_deg = theta_rad * (180.0 / np.pi)

    print(f"Plotting results for {scenario_name}...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Pitch Angle (theta)
    ax1.plot(time, theta_deg, label=r'$\theta(t)$ (Angle)')
    ax1.set_title(f"Replication of {scenario_name}")
    ax1.set_ylabel('Angle [degrees]')
    ax1.grid(True)
    ax1.legend()
    # Set y-axis limits to match the paper's plots
    if "Figure 11" in scenario_name:
        ax1.set_ylim(-2, 6)
        ax2.set_ylim(-0.02, 0.10)
    elif "Figure 12" in scenario_name:
        ax1.set_ylim(-10, 30)
        ax2.set_ylim(-0.1, 0.6)
    elif "Figure 13" in scenario_name:
        ax1.set_ylim(-1, 6)
        ax2.set_ylim(-1, 6)
    elif "Figure 14" in scenario_name:
        ax1.set_ylim(-5, 8)
        ax2.set_ylim(-0.5, 4)

    # Plot 2: Position (x)
    ax2.plot(time, x_m, label='x(t) (Position)', color='C1')
    ax2.set_ylabel('Position [m]')
    ax2.set_xlabel('Time [s]')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()


# --- Main script execution ---
if __name__ == "__main__":
    
    # Convert degrees to radians for inputs
    deg_to_rad = np.pi / 180.0
    
    # --- Scenario 1: Figure 11 ---
    # Stabilize with small initial angle
    # We use a much smaller gain to prevent instability
    y0_fig11 = [5.0 * deg_to_rad, 0, 0, 0]
    y_ref_fig11 = [0.0, 0.0] # [theta_ref, x_ref]
    run_simulation(
        scenario_name="Figure 11: Stabilizing from 5 deg",
        y0=y0_fig11,
        y_ref=y_ref_fig11,
        t_end=15.0,
        pos_gain_scale=0.01  # <-- REDUCED GAIN (1% strength)
    )

    # --- Scenario 2: Figure 12 ---
    # Stabilize with large initial angle
    # We use a much smaller gain to prevent instability
    y0_fig12 = [25.0 * deg_to_rad, 0, 0, 0]
    y_ref_fig12 = [0.0, 0.0]
    run_simulation(
        scenario_name="Figure 12: Stabilizing from 25 deg",
        y0=y0_fig12,
        y_ref=y_ref_fig12,
        t_end=15.0,
        pos_gain_scale=0.01  # <-- REDUCED GAIN (1% strength)
    )

    # --- Scenario 3: Figure 13 ---
    # Track a reference pitch angle
    y0_fig13 = [0, 0, 0, 0]
    y_ref_fig13 = [5.0 * deg_to_rad, 0.0] # 5 deg angle ref
    run_simulation(
        scenario_name="Figure 13: Track reference angle (5 deg)",
        y0=y0_fig13,
        y_ref=y_ref_fig13,
        t_end=5.0,
        pos_gain_scale=0.0 # Position controller is OFF
    )

    # --- Scenario 4: Figure 14 ---
    # Move to a set position
    # This task *needs* the position controller, but 100% is too high.
    y0_fig14 = [0, 0, 0, 0]
    y_ref_fig14 = [0.0, 2.0] # 2 meter position ref
    run_simulation(
        scenario_name="Figure 14: Move to set position (2 m)",
        y0=y0_fig14,
        y_ref=y_ref_fig14,
        t_end=15.0,
        pos_gain_scale=0.1   # <-- Use 10% gain for this task
    )
    
    print("\nAll simulations finished. Showing plots...")
    plt.show()