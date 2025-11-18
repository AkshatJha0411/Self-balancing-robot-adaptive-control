# controller.py
# MODIFIED for Adaptive Control using Fuzzy Logic
# Added divide-by-zero check for e1_dot

import numpy as np
from fuzzy_tuner import create_pd_tuner, create_pi_tuner

class Controller:
    """
    Implements the ADAPTIVE PID Backstepping controller.
    Gains are no longer fixed but are tuned by fuzzy logic engines.
    """
    def __init__(self):
        # --- Create Fuzzy Tuner Instances ---
        self.pd_tuner = create_pd_tuner()
        self.pi_tuner = create_pi_tuner()

        # --- Define Gain RANGES (min/max values) ---
        # These replace the fixed gains from Table IV.
        # We now define the *search space* for the fuzzy tuners.
        self.K_P_POS_RANGE = [0, 80.0]  # Max Kp_pos
        self.K_D_POS_RANGE = [0, 15.0]  # Max Kd_pos
        
        self.K1_RANGE = [80.0, 150.0]  # Min/Max k1
        self.C1_RANGE = [0, 5.0]       # Min/Max c1
        
        # We still need a fixed k2
        self.k2 = 21.4 # from Table IV

        # --- Controller State Variables ---
        self.z1 = 0.0  # z1 = \int e1 d\tau
        self.e1_prev = 0.0 # For calculating e1_dot

    def reset_states(self):
        """ Resets integral terms and error history. """
        self.z1 = 0.0
        self.e1_prev = 0.0

    def _scale_output(self, fuzzy_output, gain_range):
        """ Helper to scale a fuzzy output [0, 1] to a gain range [min, max] """
        return gain_range[0] + (fuzzy_output * (gain_range[1] - gain_range[0]))

    def calculate_control_torque(self, y, y_ref, dt, robot_params):
        """
        Calculates the total control torque C_in = C_theta + C_x
        using gains from the fuzzy tuners.
        """
        
        # Unpack current state
        x1, x2, x3, x4 = y
        
        # Unpack reference state
        x1_ref, x3_ref = y_ref
        
        # Unpack necessary robot parameters
        L = robot_params.L
        Mb = robot_params.Mb
        
        # --- 1. Calculate Errors ---
        e1 = x1_ref - x1
        e_pos = x3_ref - x3
        e_vel = 0.0 - x4 # Reference velocity is 0
        
        # Approximate e1_dot (FIX: prevent divide by zero on first step)
        if dt > 0:
            e1_dot = (e1 - self.e1_prev) / dt
        else:
            e1_dot = 0.0
        self.e1_prev = e1
        
        # --- 2. Run Fuzzy Tuners ---
        
        # Run Adaptive PI Tuner (for Backstepping gains k1, c1)
        self.pi_tuner.input['e1'] = e1
        self.pi_tuner.input['e1_dot'] = np.clip(e1_dot, -2.0, 2.0) # Clip input to antecedent range
        self.pi_tuner.compute()
        
        k1_fuzzy = self.pi_tuner.output['k1_out']
        c1_fuzzy = self.pi_tuner.output['c1_out']
        
        # Scale fuzzy outputs [0, 1] to our desired gain ranges
        k1_new = self._scale_output(k1_fuzzy, self.K1_RANGE)
        c1_new = self._scale_output(c1_fuzzy, self.C1_RANGE)

        # Run Adaptive PD Tuner (for Position gains Kp, Kd)
        self.pd_tuner.input['e_pos'] = np.clip(e_pos, -2.5, 2.5) # Clip input to antecedent range
        self.pd_tuner.input['e_vel'] = np.clip(e_vel, -1.0, 1.0) # Clip input to antecedent range
        self.pd_tuner.compute()
        
        kp_fuzzy = self.pd_tuner.output['kp_out']
        kd_fuzzy = self.pd_tuner.output['kd_out']
        
        # Scale fuzzy outputs
        Kp_new = self._scale_output(kp_fuzzy, self.K_P_POS_RANGE)
        Kd_new = self._scale_output(kd_fuzzy, self.K_D_POS_RANGE)

        # --- 3. Backstepping (Balance) Controller (Section V-B) ---
        # This now uses the *adaptive* gains k1_new and c1_new
        
        self.z1 += e1 * dt
        x1_ref_dot = 0.0
        x1_ref_ddot = 0.0
        
        alpha = k1_new * e1 + c1_new * self.z1 + x1_ref_dot
        e2 = alpha - x2
        
        # Re-calculate f1, f2, g1 from the model
        sin_x1 = np.sin(x1)
        cos_x1 = np.cos(x1)
        Mw, R, g = robot_params.Mw, robot_params.R, robot_params.g
        DENOM_1 = (0.75 * (Mw * R + Mb * L * cos_x1) * cos_x1 / ((2 * Mw + Mb) * L)) - 1
        
        # Add check for denominator singularity
        if abs(DENOM_1) < 1e-6: DENOM_1 = 1e-6

        f1 = (-0.75 * g * sin_x1 / L) / DENOM_1
        f2 = (0.75 * Mb * L * sin_x1 * cos_x1 * (x2**2) / ((2 * Mw + Mb) * L)) / DENOM_1
        g1_num = (0.75 * (1 + sin_x1**2) / (Mb * L**2)) + (0.75 * cos_x1 / ((2 * Mw + Mb) * R * L))
        g1 = g1_num / DENOM_1
        
        if abs(g1) < 1e-6: g1 = 1e-6 # Avoid divide by zero

        # Calculate C_theta using Equation (31) and new gains
        C_theta_num = (1 + c1_new - k1_new**2) * e1 + (k1_new + self.k2) * e2 - k1_new * c1_new * self.z1 + x1_ref_ddot - f1 - f2
        C_theta = C_theta_num / g1
        
        
        # --- 4. PD (Position) Controller (Section V-C) ---
        # This now uses the *adaptive* gains Kp_new and Kd_new
        
        # Apply the sign inversion fix we found earlier
        C_x = - (Kp_new * e_pos + Kd_new * e_vel)
        
        
        # --- 5. Total Control Torque ---
        C_in = C_theta + C_x
        
        return C_in