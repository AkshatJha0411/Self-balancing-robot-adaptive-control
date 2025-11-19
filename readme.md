
# PD + Backstepping Controller for Two-Wheeled Self-Balancing Robot

This project recreates the control architecture from the 2010 paper *A PID Backstepping Controller for Two-Wheeled Self-Balancing Robot* and extends it with a real-time adaptive fuzzy gain scheduler. The aim is to evaluate how fixed-gain backstepping + PD behaves under strong nonlinearities (PI excluded from our simulations due to 2D constraints), and how an adaptive fuzzy layer improves stability, disturbance rejection, and tracking without any manual gain tuning.

## What this project includes
- Full nonlinear model of a two-wheeled self-balancing robot  
- Backstepping torque control for angle stabilization  
- PD controller for position  
- Adaptive fuzzy gain scheduler updating $K_{P}$ (Position Proportional Gain), $K_{D}$ (Position Derivative Gain), $K_{1}$ (Angle Proportional Gain), $C_{1}$ (Angle Derivative Gain) continuously based on error + error rate.  
- Automated performance metrics: settling time, rise time, overshoot, steady state error, IAE, ISE, ITAE, control effort  
- Disturbance injection (angle, position, mixed)  
- Side by side comparison of fixed vs adaptive controller  
- Plots showing recovery speed, stability, and tracking behavior  
- All plots, analysis and methodology are given in more detail in the report. 

## Overall system: Cascaded Control

The robot uses a cascaded control structure because a single torque input cannot independently control both stability and position without accounting for their natural coupling.

- The outer controller (position loop) computes the desired tilt angle $\theta_{ref}$ needed to move toward the target position.

- The inner controller (stability loop) treats this $\theta_{ref}$ as its setpoint and computes the final input torque $C_{in}$ to keep the robot balanced while following the desired motion.


This architecture is essential because the robot inherently uses tilt to create forward or backward motion, meaning movement and balance are physically intertwined.

## Scenarios simulated
Main file for running is *run_all_scenarios_adaptive_disturbance.py*
1. Stabilization from a small tilt with a sudden 15° push  
2. Recovery from a 25° initial tilt combined with a -1 m push  
3. Track Reference Angle of 5 degree (only in *run_all_scenarios-adaptive.py* and *run_all_scenarios-normal.py*)
4. Tracking a 2 m target under a mixed angle + position disturbance  

Each scenario is run twice (fixed and adaptive), and the metrics highlight where adaptive control makes a measurable difference.

### Result of scenario 4

![Adaptive Disturbance Response](reports/Figure_4-adaptive-disturbance.png)

## Installation
You can optionally create a fresh virtual environment:
```
python3 -m venv venv
source venv/bin/activate

```
Install dependencies:
```
pip install numpy scipy matplotlib
```

## Running the project
Execute the main script:
```
python3 run_all_scenarios.py

```
You will get printed metrics for each scenario along with comparison plots for fixed vs adaptive gains. (Scenario 1,2 and 4)

## Methodology:
1. Define the full nonlinear robot model with four states: angle, angular velocity, position, and velocity.
2. Implement the system dynamics function that returns state derivatives based on the current state and applied control torque.
3. Implement the controller containing the Backstepping law for angle stabilization, the PD law for position tracking, and the Adaptive Fuzzy Gain Scheduler that updates KP, KD, k1, and c1 using error and error rate.
4. Allow switching between Fixed Gains (constant values) and Adaptive Gains (bounded ranges updated every timestep).
5. Configure each scenario by selecting the initial state, reference values, disturbance type, disturbance magnitude, disturbance application time, and total simulation duration.
6. Run the simulation loop for both Fixed and Adaptive modes by iterating through time, applying the disturbance at the specified moment, computing the control torque, integrating system dynamics with solve_ivp, and recording state and control histories.
7. Split the recorded data into Pre-Disturbance and Post-Disturbance segments to separately analyze baseline behavior and recovery performance.
8. Compute performance metrics including Settling Time, Rise Time, Overshoot, Steady-State Error, IAE, ISE, ITAE, and Total Control Effort using the stored arrays.
9. Print the metrics in a consistent structured order to enable direct comparison between Fixed and Adaptive controllers.
10. Plot both Angle and Position trajectories for Fixed vs Adaptive control, marking the disturbance time for clear visual interpretation.

## Repository
Full source code, report, and simulation figures:  
https://github.com/AkshatJha0411/self-balancing-robot-adaptive-control/



