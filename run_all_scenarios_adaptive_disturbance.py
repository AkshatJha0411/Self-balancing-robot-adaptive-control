# run_all_scenarios.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Import our classes from the other files
from robot_model import RobotModel
from controller import Controller 

def calculate_metrics(t_array, y_array, c_array, target_val, type_label):
    """
    Calculates Settling Time, Rise Time, Overshoot, Steady State Error, 
    and Integral Error Metrics (IAE, ISE, ITAE, TCE).
    """
    # Shift time to start at 0 relative to this segment
    t = t_array - t_array[0]
    dt = t[1] - t[0] if len(t) > 1 else 0.01 
    e_t = y_array - target_val # Error
    
    # --- 1. Steady State Error ---
    final_val = np.mean(y_array[int(len(y_array)*0.95):])
    ss_error = abs(final_val - target_val)
    
    # --- 2. Peak & Overshoot ---
    abs_error = np.abs(e_t)
    peak_error = np.max(abs_error)
    step_size = abs(y_array[0] - target_val)
    if step_size < 1e-3: step_size = 1.0 # Default normalization
    
    overshoot_pct = 0.0
    # Calculate % Overshoot only if target is non-zero
    if abs(target_val) > 1e-3:
        if (target_val > 0 and np.max(y_array) > target_val):
            overshoot_pct = ((np.max(y_array) - target_val) / abs(target_val)) * 100.0
        elif (target_val < 0 and np.min(y_array) < target_val):
            # For negative targets, measure undershoot past target
            overshoot_pct = ((target_val - np.min(y_array)) / abs(target_val)) * 100.0
    
    # --- 3. Settling Time (2% criterion) ---
    tolerance = 0.10 * step_size 
    outside_bounds = abs_error > tolerance
    settling_time = t[-1] 
    if np.any(outside_bounds):
        last_idx = np.where(outside_bounds)[0][-1]
        settling_time = t[last_idx]
    
    # --- 4. Rise Time (10% to 90%) ---
    rise_time = -1.0
    if step_size > 1e-2 and abs(y_array[0] - target_val) > 1e-2: # Significant movement required
        # Determine movement direction for threshold calculation
        start_val = y_array[0]
        if target_val > start_val: # Positive step
            t10 = target_val - 0.9 * (target_val - start_val)
            t90 = target_val - 0.1 * (target_val - start_val)
            try:
                idx_10 = np.where(y_array >= t10)[0][0]
                idx_90 = np.where(y_array >= t90)[0][0]
                rise_time = t[idx_90] - t[idx_10]
            except:
                rise_time = -1.0
        elif target_val < start_val: # Negative step/Stabilization
            # Use 'fall time' 90% to 10%
            t10 = target_val + 0.9 * (start_val - target_val)
            t90 = target_val + 0.1 * (start_val - target_val)
            try:
                idx_90 = np.where(y_array <= t90)[0][0] # Time reaches 90% of drop
                rise_time = t[idx_90] 
            except:
                rise_time = -1.0


    # --- 5. Integral Error Metrics (using trapezoidal rule for numerical integration) ---
    if len(e_t) > 1:
        IAE = np.trapz(abs_error, dx=dt)
        ISE = np.trapz(e_t**2, dx=dt)
        ITAE = np.trapz(t * abs_error, dx=dt)
    else:
        IAE, ISE, ITAE = 0.0, 0.0, 0.0
        
    # --- 6. Total Control Effort (TCE) ---
    if len(c_array) > 1:
        TCE = np.trapz(c_array**2, dx=dt)
    else:
        TCE = 0.0

    return {
        "Peak_Dev": peak_error,
        "Overshoot_Pct": overshoot_pct,
        "Settling_Time": settling_time,
        "SS_Error": ss_error,
        "Rise_Time": rise_time,
        "IAE": IAE,
        "ISE": ISE,
        "ITAE": ITAE,
        "TCE": TCE,
    }

def print_metrics_block(metrics, title):
    print(f"\n  --- {title} ---")
    print(f"  Settling Time (2%):   {metrics['Settling_Time']:.3f} s")
    if metrics['Rise_Time'] > 0:
        print(f"  Rise/Fall Time:       {metrics['Rise_Time']:.3f} s")
    print(f"  Steady State Error:   {metrics['SS_Error']:.4f}")
    if metrics['Overshoot_Pct'] > 0.1:
        print(f"  Overshoot:            {metrics['Overshoot_Pct']:.2f} %")
    else:
        print(f"  Peak Deviation:       {metrics['Peak_Dev']:.4f}")
    print(f"  IAE (Abs Error):      {metrics['IAE']:.4f}")
    print(f"  ITAE (Time-Wgt Err):  {metrics['ITAE']:.4f}")
    print(f"  TCE (Control Energy): {metrics['TCE']:.4f}")

def run_simulation_with_analysis(scenario_name, y0, y_ref, t_disturb=14.0, t_end=25.0, disturbance_type=None, disturbance_val=0.0, pos_gain_scale=1.0):
    """
    Runs simulation and calculates metrics for specified dimensions.
    """
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*80}")
    
    # Determine which dimensions to analyze based on user request
    if "1:" in scenario_name or "2:" in scenario_name:
        analysis_dims = [(3, y_ref[1], "Position [m]")] # Only Position (col_idx, target, name)
    elif "4:" in scenario_name:
        analysis_dims = [
            (3, y_ref[1], "Position [m]"), # Position
            (1, y_ref[0], "Angle [rad]")    # Angle
        ]
    else:
        # Fallback to the primary dimension if not explicitly 1, 2, or 4
        analysis_dims = [(3, y_ref[1], "Position [m]")]


    # --- 1. Run Simulations (Fixed & Adaptive) ---
    # Adaptive
    t_full_adapt, h_adapt, c_adapt = _run_single_pass(
        RobotModel(), Controller(), y0, y_ref, t_disturb, t_end, 
        disturbance_type, disturbance_val, 'adaptive', pos_gain_scale
    )
    # Fixed
    t_full_fixed, h_fixed, c_fixed = _run_single_pass(
        RobotModel(), Controller(), y0, y_ref, t_disturb, t_end, 
        disturbance_type, disturbance_val, 'fixed', pos_gain_scale
    )

    # Identify split index (t >= t_disturb)
    split_idx = np.searchsorted(t_full_adapt, t_disturb)
    
    # --- 2. Analyze Data for Each Requested Dimension ---
    for col_idx, target, dim_name in analysis_dims:
        print(f"\n{'*'*30} Analyzing Dimension: {dim_name} (Target: {target:.2f}) {'*'*30}")
        
        # --- ADAPTIVE ANALYSIS ---
        # Pre-Disturbance
        metrics_adapt_pre = calculate_metrics(
            t_full_adapt[:split_idx], h_adapt[:split_idx, col_idx], c_adapt[:split_idx], target, "Adaptive Pre"
        )
        # Post-Disturbance (Recovery)
        metrics_adapt_post = calculate_metrics(
            t_full_adapt[split_idx:], h_adapt[split_idx:, col_idx], c_adapt[split_idx:], target, "Adaptive Post"
        )

        # --- FIXED ANALYSIS ---
        # Pre-Disturbance
        metrics_fixed_pre = calculate_metrics(
            t_full_fixed[:split_idx], h_fixed[:split_idx, col_idx], c_fixed[:split_idx], target, "Fixed Pre"
        )
        # Post-Disturbance
        metrics_fixed_post = calculate_metrics(
            t_full_fixed[split_idx:], h_fixed[split_idx:, col_idx], c_fixed[split_idx:], target, "Fixed Post"
        )

        # --- 3. Print Results in Requested Order ---
        print_metrics_block(metrics_adapt_pre, "1. Adaptive Gain - No Disturbance (Initial)")
        print_metrics_block(metrics_adapt_post, "2. Adaptive Gain - With Disturbance (Recovery)")
        print_metrics_block(metrics_fixed_pre, "3. Fixed Gain - No Disturbance (Initial)")
        print_metrics_block(metrics_fixed_post, "4. Fixed Gain - With Disturbance (Recovery)")


    # --- 4. Plot Comparison ---
    plot_disturbance_comparison(scenario_name, t_full_fixed, h_fixed, h_adapt, t_disturb)


def _run_single_pass(robot, controller, y0, y_ref, t_disturb, t_end, dist_type, dist_val, mode, pos_gain_scale):
    """ Helper to run one full simulation pass with disturbance. Returns state and control history. """
    
    # Setup Controller Mode
    if mode == 'fixed':
        controller.K_P_POS_RANGE = [60.0 * pos_gain_scale, 60.0 * pos_gain_scale]
        controller.K_D_POS_RANGE = [7.5 * pos_gain_scale, 7.5 * pos_gain_scale]
        controller.K1_RANGE = [110.5, 110.5]
        controller.C1_RANGE = [3.0, 3.0]
    else:
        # Restore Adaptive Ranges
        controller.K_P_POS_RANGE = [0, 80.0]
        controller.K_D_POS_RANGE = [0, 15.0]
        controller.K1_RANGE = [80.0, 150.0]
        controller.C1_RANGE = [0, 5.0]

    controller.reset_states()
    y_current = np.array(y0)
    
    DT = 0.01 # Defined internally
    t_eval = np.arange(0, t_end, DT)
    history = np.zeros((len(t_eval), 5))
    c_history = np.zeros(len(t_eval)) # Added Control History
    
    disturbance_applied = False

    for i, t in enumerate(t_eval):
        # Inject Disturbance
        if t >= t_disturb and not disturbance_applied:
            if dist_type == 'angle':
                y_current[0] += dist_val
            elif dist_type == 'position':
                y_current[2] += dist_val
            elif dist_type == 'mixed':
                y_current[0] += dist_val[0]
                y_current[2] += dist_val[1]
            disturbance_applied = True

        history[i, 0] = t
        history[i, 1:] = y_current
        
        C_in = controller.calculate_control_torque(y_current, y_ref, DT, robot) # Use internal DT
        c_history[i] = C_in # Record control input
        
        sol = solve_ivp(lambda t, y: robot.system_dynamics(t, y, C_in), 
                        [t, t+DT], y_current, t_eval=[t+DT], method='RK45')
        y_current = np.array(sol.y).flatten()
        
    return t_eval, history, c_history


def plot_disturbance_comparison(scenario_name, time, h_fixed, h_adapt, t_dist):
    """ Plots comparison of Fixed vs Adaptive recovery """
    theta_fixed = h_fixed[:,1] * (180/np.pi)
    theta_adapt = h_adapt[:,1] * (180/np.pi)
    x_fixed = h_fixed[:,3]
    x_adapt = h_adapt[:,3]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(time, theta_fixed, 'b--', label='Fixed Gain', alpha=0.7)
    ax1.plot(time, theta_adapt, 'r-', label='Adaptive Gain', linewidth=1.5)
    ax1.axvline(x=t_dist, color='k', linestyle=':', alpha=0.5, label='Disturbance')
    ax1.set_title(f"Disturbance Rejection: {scenario_name}")
    ax1.set_ylabel('Angle [degrees]')
    ax1.grid(True)
    ax1.legend(loc='upper right')
    ax1.set_ylim(-30, 30) 

    ax2.plot(time, x_fixed, 'b--', label='Fixed Gain', alpha=0.7)
    ax2.plot(time, x_adapt, 'r-', label='Adaptive Gain', linewidth=1.5)
    ax2.axvline(x=t_dist, color='k', linestyle=':', alpha=0.5)
    ax2.set_ylabel('Position [m]')
    ax2.set_xlabel('Time [s]')
    ax2.grid(True)
    
    plt.tight_layout()


# --- Main Execution ---
if __name__ == "__main__":
    deg_to_rad = np.pi / 180.0
    
    # --- Scenario 1: Analyze Position only (Target 0.0m) ---
    run_simulation_with_analysis(
        scenario_name="Scenario 1: 15deg Angle Push", 
        y0=[5.0 * deg_to_rad, 0, 0, 0], y_ref=[0.0, 0.0], 
        t_disturb=14.0, t_end=25.0, 
        disturbance_type='angle', disturbance_val=15.0*deg_to_rad,
        pos_gain_scale=0.01 
    )

    # --- Scenario 2: Analyze Position only (Target 0.0m) ---
    run_simulation_with_analysis(
        scenario_name="Scenario 2: -1.0m Position Shove", 
        y0=[25.0 * deg_to_rad, 0, 0, 0], y_ref=[0.0, 0.0], 
        t_disturb=14.0, t_end=25.0, 
        disturbance_type='position', disturbance_val=-1.0,
        pos_gain_scale=0.01 
    )

    # --- Scenario 4: Analyze Position and Angle (Target 2.0m) ---
    run_simulation_with_analysis(
        scenario_name="Scenario 4: Mixed (+10deg, +0.5m)", 
        y0=[0, 0, 0, 0], y_ref=[0.0, 2.0], 
        t_disturb=14.0, t_end=25.0, 
        disturbance_type='mixed', 
        disturbance_val=(10.0 * deg_to_rad, 0.5),
        pos_gain_scale=0.03 
    )
    
    plt.show()