# fuzzy_tuner.py
# This new file defines the fuzzy logic engines for adaptive control.
# CORRECTED to fix the 'e_vel' value error.

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def create_pd_tuner():
    """
    Creates the Fuzzy Logic engine for the Adaptive PD Position Controller.
    
    Inputs:
        - e_pos (Position Error)
        - e_vel (Position Velocity Error)
    Outputs:
        - Kp (new Proportional gain)
        - Kd (new Derivative gain)
    """
    # Define fuzzy antecedents (inputs)
    e_pos = ctrl.Antecedent(np.arange(-2.5, 2.6, 0.1), 'e_pos')
    e_vel = ctrl.Antecedent(np.arange(-1, 1.1, 0.1), 'e_vel')

    # Define fuzzy consequents (outputs)
    kp_out = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'kp_out')
    kd_out = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'kd_out')

    # Define membership functions for inputs
    e_pos.automf(names=['NB', 'NS', 'ZE', 'PS', 'PB'])
    e_vel.automf(names=['NB', 'NS', 'ZE', 'PS', 'PB'])

    # Define membership functions for outputs
    kp_out['S'] = fuzz.trimf(kp_out.universe, [0, 0, 0.5])
    kp_out['M'] = fuzz.trimf(kp_out.universe, [0, 0.5, 1.0])
    kp_out['B'] = fuzz.trimf(kp_out.universe, [0.5, 1.0, 1.0])
    
    kd_out['S'] = fuzz.trimf(kd_out.universe, [0, 0, 0.5])
    kd_out['M'] = fuzz.trimf(kd_out.universe, [0, 0.5, 1.0])
    kd_out['B'] = fuzz.trimf(kd_out.universe, [0.5, 1.0, 1.0])

    # --- FIX: Define the Fuzzy Rules (A 2D rule base) ---
    
    # Rule 1: Error is Big (NB or PB). Use Big P (Kp) and Small D (Kd).
    rule1 = ctrl.Rule(e_pos['NB'] | e_pos['PB'], (kp_out['B'], kd_out['S']))
    
    # Rule 2: Error is Small (NS or PS). Use Medium gains.
    rule2 = ctrl.Rule(e_pos['NS'] | e_pos['PS'], (kp_out['M'], kd_out['M']))
    
    # Rule 3: Error is Zero (ZE) AND Velocity is Zero (ZE).
    # This is steady-state. Use Small P (Kp) and Big D (Kd) to hold position.
    rule3 = ctrl.Rule(e_pos['ZE'] & e_vel['ZE'], (kp_out['S'], kd_out['B']))

    # Rule 4: Error is Zero (ZE) but Velocity is NOT Zero (oscillating).
    # Use Small P (Kp) and Medium D (Kd) to damp oscillation.
    rule4 = ctrl.Rule(e_pos['ZE'] & (e_vel['NB'] | e_vel['NS'] | e_vel['PS'] | e_vel['PB']),
                      (kp_out['S'], kd_out['M']))

    # Rule 5: Error is Small (NS or PS) but Velocity is Big (NB or PB).
    # Approaching target too fast. Use Small P (Kp) and Big D (Kd).
    rule5 = ctrl.Rule((e_pos['NS'] | e_pos['PS']) & (e_vel['NB'] | e_vel['PB']),
                      (kp_out['S'], kd_out['B']))

    # Create the control system
    # Pass ALL rules to the system. This ensures e_vel is included.
    pd_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    pd_tuner = ctrl.ControlSystemSimulation(pd_ctrl)
    
    return pd_tuner

def create_pi_tuner():
    """
    Creates the Fuzzy Logic engine for the Adaptive "PI" Backstepping Controller.
    
    Inputs:
        - e1 (Angle Error)
        - e1_dot (Angle Error Rate)
    Outputs:
        - k1 (new 'P' gain)
        - c1 (new 'I' gain)
    """
    # Define fuzzy antecedents (inputs)
    e1 = ctrl.Antecedent(np.arange(-0.5, 0.51, 0.01), 'e1') # Approx -30 to +30 deg in rad
    e1_dot = ctrl.Antecedent(np.arange(-2.0, 2.1, 0.1), 'e1_dot') # <-- This is the input

    # Define fuzzy consequents (outputs)
    k1_out = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'k1_out')
    c1_out = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'c1_out')

    # Define membership functions
    e1.automf(names=['NB', 'NS', 'ZE', 'PS', 'PB'])
    e1_dot.automf(names=['NB', 'NS', 'ZE', 'PS', 'PB']) # <-- Use 5 levels for e1_dot
    
    k1_out['S'] = fuzz.trimf(k1_out.universe, [0, 0, 0.5])
    k1_out['M'] = fuzz.trimf(k1_out.universe, [0, 0.5, 1.0])
    k1_out['B'] = fuzz.trimf(k1_out.universe, [0.5, 1.0, 1.0])

    c1_out['S'] = fuzz.trimf(c1_out.universe, [0, 0, 0.5])
    c1_out['M'] = fuzz.trimf(c1_out.universe, [0, 0.5, 1.0])
    c1_out['B'] = fuzz.trimf(c1_out.universe, [0.5, 1.0, 1.0])

    # --- FIX: Define the Fuzzy Rules (A 2D rule base) ---
    
    # Rule 1: Error is Big (NB or PB). Use Big P (k1) and Small I (c1).
    rule1 = ctrl.Rule(e1['NB'] | e1['PB'], (k1_out['B'], c1_out['S']))
    
    # Rule 2: Error is Small (NS or PS). Use Medium gains.
    rule2 = ctrl.Rule(e1['NS'] | e1['PS'], (k1_out['M'], c1_out['M']))
    
    # Rule 3: Error is Zero (ZE) AND Error Rate is Zero (ZE).
    # This is steady-state. Use Big I (c1) to crush error, Medium P (k1).
    rule3 = ctrl.Rule(e1['ZE'] & e1_dot['ZE'], (k1_out['M'], c1_out['B']))

    # Rule 4: Error is Zero (ZE) but Error Rate is NOT Zero (oscillating).
    # Use Small gains for P (k1) and I (c1) to let it stabilize.
    rule4 = ctrl.Rule(e1['ZE'] & (e1_dot['NB'] | e1_dot['NS'] | e1_dot['PS'] | e1_dot['PB']),
                      (k1_out['S'], c1_out['S']))

    # Create the control system
    # Pass ALL rules to the system. This ensures e1_dot is included.
    pi_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4]) 
    pi_tuner = ctrl.ControlSystemSimulation(pi_ctrl)
    
    return pi_tuner