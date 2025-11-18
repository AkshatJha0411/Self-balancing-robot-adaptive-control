# run_all_scenarios.py
# MODIFIED to remove the 'pos_gain_scale' hack.
# The adaptive controller should now handle this automatically.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Import our classes from the other files
from robot_model import RobotModel
from controller import Controller # This is now the ADAPTIVE controller

# Note: The 'pos_gain_scale' argument is REMOVED
def run_simulation(scenario_name, y0, y_ref, t_end, disable_pos_control=False):
    """
    Runs a full closed-loop simulation with the new ADAPTIVE controller.
    
    Args:
        scenario_name (str): Title for the plot (e.g., "Figure 11")
        y0 (list): Initial state [theta, theta_dot, x, x_dot]
        y_ref (list): Reference state [theta_ref, x_ref]
        t_end (float): Simulation end time [s]
        disable_pos_control (bool): Flag to manually turn off position control
    """
    print(f"\nRunning ADAPTIVE simulation for: {scenario_name}...")
    
    robot = RobotModel()
    controller = Controller() # This is the new adaptive one
    
    # Simulation parameters
    T_START = 0.0
    DT = 0.01     # Time step for controller update [s]

    # Set initial and reference states
    y_current = np.array(y0)
    
    # Reset controller states (like integral term z1)
    controller.reset_states()
    
    # If scenario 3, manually disable position control part
    if disable_pos_control:
        controller.K_P_POS_RANGE = [0, 0]
        controller.K_D_POS_RANGE = [0, 0]

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
        # C_in is now calculated by the adaptive controller
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
            t_eval=[t + DT],
            method='RK45' # Use a robust solver
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

    # Plot 2: Position (x)
    ax2.plot(time, x_m, label='x(t) (Position)', color='C1')
    ax2.set_ylabel('Position [m]')
    ax2.set_xlabel('Time [s]')
    ax2.grid(True)
    ax2.legend()

    # Set y-axis limits to match the paper's plots
    if "Figure 11" in scenario_name:
        ax1.set_ylim(-2, 6)
        ax2.set_ylim(-0.1, 0.2)
    elif "Figure 12" in scenario_name:
        ax1.set_ylim(-10, 30)
        ax2.set_ylim(-0.5, 0.5)
    elif "Figure 13" in scenario_name:
        ax1.set_ylim(-1, 6)
        ax2.set_ylim(-1, 10)
    elif "Figure 14" in scenario_name:
        ax1.set_ylim(-40, 40)
        ax2.set_ylim(-0.5, 5) # Use the y-limit you set
    
    plt.tight_layout()


# --- Main script execution ---
if __name__ == "__main__":
    
    # Convert degrees to radians for inputs
    deg_to_rad = np.pi / 180.0
    
    # --- Scenario 1: Figure 11 ---
    y0_fig11 = [5.0 * deg_to_rad, 0, 0, 0]
    y_ref_fig11 = [0.0, 0.0] # [theta_ref, x_ref]
    run_simulation(
        scenario_name="Figure 11 (Adaptive): Stabilizing from 5 deg",
        y0=y0_fig11,
        y_ref=y_ref_fig11,
        t_end=15.0
    )

    # --- Scenario 2: Figure 12 ---
    y0_fig12 = [25.0 * deg_to_rad, 0, 0, 0]
    y_ref_fig12 = [0.0, 0.0]
    run_simulation(
        scenario_name="Figure 12 (Adaptive): Stabilizing from 25 deg",
        y0=y0_fig12,
        y_ref=y_ref_fig12,
        t_end=15.0
    )

    # --- Scenario 3: Figure 13 ---
    # To replicate the paper, we will manually disable position control
    y0_fig13 = [0, 0, 0, 0]
    y_ref_fig13 = [5.0 * deg_to_rad, 0.0] # 5 deg angle ref
    run_simulation(
        scenario_name="Figure 13 (Adaptive) (No Position Control thus no change): Track reference angle (5 deg)",
        y0=y0_fig13,
        y_ref=y_ref_fig13,
        t_end=5.0,
        disable_pos_control=True # Manually turn off PD part
    )

    # --- Scenario 4: Figure 14 ---
    y0_fig14 = [0, 0, 0, 0]
    y_ref_fig14 = [0.0, 2.0] # 2 meter position ref
    run_simulation(
        scenario_name="Figure 14 (Adaptive): Move to set position (2 m)",
        y0=y0_fig14,
        y_ref=y_ref_fig14,
        t_end=15.0
    )
    
    print("\nAll simulations finished. Showing plots...")
    plt.show()

# # run_all_scenarios.py
# # Implements Phase 2, Steps 4 & 6: Replicating all simulation figures.
# # This script runs all four scenarios from Section VI-A.
# # FINAL TUNING 2 for Figure 14.

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp

# # Import our classes from the other files
# from robot_model import RobotModel
# from controller import Controller

# def run_simulation(scenario_name, y0, y_ref, t_end, pos_gain_scale=1.0):
#     """
#     Runs a full closed-loop simulation for a given scenario.
    
#     Args:
#         scenario_name (str): Title for the plot (e.g., "Figure 11")
#         y0 (list): Initial state [theta, theta_dot, x, x_dot]
#         y_ref (list): Reference state [theta_ref, x_ref]
#         t_end (float): Simulation end time [s]
#         pos_gain_scale (float): Multiplier for the PD position gains.
#                                 1.0 = full gains, 0.0 = off.
#     """
#     print(f"\nRunning simulation for: {scenario_name}...")
    
#     robot = RobotModel()
#     controller = Controller()
    
#     # Simulation parameters
#     T_START = 0.0
#     DT = 0.01     # Time step for controller update [s]

#     # Set initial and reference states
#     y_current = np.array(y0)
    
#     # --- MODIFICATION: Scale position gains ---
#     controller.Kp_pos *= pos_gain_scale
#     controller.Kd_pos *= pos_gain_scale
#     if pos_gain_scale != 1.0:
#         print(f"INFO: Position gains scaled by {pos_gain_scale}")
#     # --- End of modification ---

#     # Reset controller states (like integral term z1)
#     controller.reset_states()

#     # Create time array
#     t_eval = np.arange(T_START, t_end, DT)
#     num_steps = len(t_eval)

#     # Prepare array to store the results history
#     history = np.zeros((num_steps, 5)) # [t, x1, x2, x3, x4]

#     for i in range(num_steps):
#         t = t_eval[i]
#         history[i, 0] = t
#         history[i, 1:] = y_current
        
#         # --- Control Step ---
#         # NOTE: Assumes controller.py has the sign inversion fix
#         # C_x = - (self.Kp_pos * e_pos + self.Kd_pos * e_vel)
#         C_in = controller.calculate_control_torque(
#             y=y_current,
#             y_ref=y_ref,
#             dt=DT,
#             robot_params=robot
#         )
        
#         # --- Plant Step ---
#         dynamics_with_control = lambda t_solve, y_solve: robot.system_dynamics(
#             t_solve, y_solve, C_in
#         )
        
#         sol = solve_ivp(
#             fun=dynamics_with_control,
#             t_span=[t, t + DT],
#             y0=y_current,
#             t_eval=[t + DT]
#         )
        
#         if sol.status != 0:
#             print(f"WARNING: Solver failed at t={t}")
#             break
            
#         y_current = np.array(sol.y).flatten()

#     print("Simulation complete.")
    
#     # --- Plot results for this scenario ---
#     plot_results(scenario_name, history)


# def plot_results(scenario_name, history_data):
#     """
#     Generates a 2-panel plot for a simulation run,
#     matching the paper's style.
#     """
#     # Extract data
#     time = history_data[:, 0]
#     theta_rad = history_data[:, 1]
#     x_m = history_data[:, 3]
#     theta_deg = theta_rad * (180.0 / np.pi)

#     print(f"Plotting results for {scenario_name}...")
    
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

#     # Plot 1: Pitch Angle (theta)
#     ax1.plot(time, theta_deg, label=r'$\theta(t)$ (Angle)')
#     ax1.set_title(f"Replication of {scenario_name}")
#     ax1.set_ylabel('Angle [degrees]')
#     ax1.grid(True)
#     ax1.legend()

#     # Plot 2: Position (x)
#     ax2.plot(time, x_m, label='x(t) (Position)', color='C1')
#     ax2.set_ylabel('Position [m]')
#     ax2.set_xlabel('Time [s]')
#     ax2.grid(True)
#     ax2.legend()

#     # Set y-axis limits to match the paper's plots
#     if "Figure 11" in scenario_name:
#         ax1.set_ylim(-2, 6)
#         ax2.set_ylim(-0.1, 0.2)
#     elif "Figure 12" in scenario_name:
#         ax1.set_ylim(-10, 30)
#         ax2.set_ylim(-0.1, 3)
#     elif "Figure 13" in scenario_name:
#         ax1.set_ylim(-1, 6)
#         ax2.set_ylim(-1, 10)
#     elif "Figure 14" in scenario_name:
#         ax1.set_ylim(-5, 6)
#         ax2.set_ylim(-0.5, 5) # Use the y-limit you set
    
#     plt.tight_layout()


# # --- Main script execution ---
# if __name__ == "__main__":
    
#     # Convert degrees to radians for inputs
#     deg_to_rad = np.pi / 180.0
    
#     # --- Scenario 1: Figure 11 ---
#     y0_fig11 = [5.0 * deg_to_rad, 0, 0, 0]
#     y_ref_fig11 = [0.0, 0.0] # [theta_ref, x_ref]
#     run_simulation(
#         scenario_name="Figure 11: Stabilizing from 5 deg",
#         y0=y0_fig11,
#         y_ref=y_ref_fig11,
#         t_end=15.0,
#         pos_gain_scale=0.01  # 1% strength
#     )

#     # --- Scenario 2: Figure 12 ---
#     y0_fig12 = [25.0 * deg_to_rad, 0, 0, 0]
#     y_ref_fig12 = [0.0, 0.0]
#     run_simulation(
#         scenario_name="Figure 12: Stabilizing from 25 deg",
#         y0=y0_fig12,
#         y_ref=y_ref_fig12,
#         t_end=15.0,
#         pos_gain_scale=0.01  # 1% strength
#     )

#     # --- Scenario 3: Figure 13 ---
#     y0_fig13 = [0, 0, 0, 0]
#     y_ref_fig13 = [5.0 * deg_to_rad, 0.0] # 5 deg angle ref
#     run_simulation(
#         scenario_name="Figure 13: Track reference angle (5 deg)",
#         y0=y0_fig13,
#         y_ref=y_ref_fig13,
#         t_end=5.0,
#         pos_gain_scale=0.0 # Position controller is OFF
#     )

#     # --- Scenario 4: Figure 14 ---
#     # <<< THIS IS THE NEW FIX >>>
#     # Let's try 3% gain
#     y0_fig14 = [0, 0, 0, 0]
#     y_ref_fig14 = [0.0, 2.0] # 2 meter position ref
#     run_simulation(
#         scenario_name="Figure 14: Move to set position (2 m)",
#         y0=y0_fig14,
#         y_ref=y_ref_fig14,
#         t_end=15.0,
#         pos_gain_scale=0.03   # <-- NEW GAIN (3% strength)
#     )
    
#     print("\nAll simulations finished. Showing plots...")
#     plt.show()