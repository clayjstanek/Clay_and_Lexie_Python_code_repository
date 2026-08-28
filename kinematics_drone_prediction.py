# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 19:15:43 2026

@author: annil
"""

import numpy as np
import matplotlib.pyplot as plt

"""
Kinematic Predictor for noisy drone data
"""


# =======================================
# Drone flight simulation 
# =======================================

def flight_simulation():
    # Time step size and number of steps
    dt = 0.1
    T = 300  

    # True position and velocity
    x_true = np.zeros(T)
    y_true = np.zeros(T)
    vx_true = np.zeros(T)
    vy_true = np.zeros(T)

    # Random accelerations (simulate random drone maneuvers)
    ax = np.random.uniform(-1, 1, T)
    ay = np.random.uniform(-1, 1, T)

    # True position and velocity for each time t
    for t in range(1, T):
        vx_true[t] = vx_true[t-1] + ax[t] * dt
        vy_true[t] = vy_true[t-1] + ay[t] * dt
    
        x_true[t] = x_true[t-1] + vx_true[t] * dt
        y_true[t] = y_true[t-1] + vy_true[t] * dt

    # Add Gaussian noise to the flight data
    noise_pos = 0.8   # meters
    noise_vel = 0.3   # m/s

    x_meas = x_true + np.random.normal(0, noise_pos, T)
    y_meas = y_true + np.random.normal(0, noise_pos, T)
    vx_meas = vx_true + np.random.normal(0, noise_vel, T)
    vy_meas = vy_true + np.random.normal(0, noise_vel, T)

    # Predict with kinematics model
    x_pred = np.zeros(T)
    y_pred = np.zeros(T)
    vx_pred = np.zeros(T)
    vy_pred = np.zeros(T)

    # Initialize prediction with first noisy measurement
    x_pred[0], y_pred[0] = x_meas[0], y_meas[0]
    vx_pred[0], vy_pred[0] = vx_meas[0], vy_meas[0]

    # Continue predicting at each time
    for t in range(1, T):
        # Acceleration is calculated from noisy velocity data
        ax_est = (vx_meas[t] - vx_meas[t-1]) / dt
        ay_est = (vy_meas[t] - vy_meas[t-1]) / dt
    
        # Predict velocity based on estimated acceleration
        vx_pred[t] = vx_pred[t-1] + ax_est * dt
        vy_pred[t] = vy_pred[t-1] + ay_est * dt
        
        # Predict position using estimated acceleration
        x_pred[t] = x_pred[t-1] + vx_pred[t] * dt 
        y_pred[t] = y_pred[t-1] + vy_pred[t] * dt 


    # Plot results 
    
    plt.figure(figsize=(10,6))
    plt.plot(x_true, y_true, label="True Path", linewidth=3)
    plt.scatter(x_meas, y_meas, s=10, alpha=0.4, label="Noisy Measurements")
    plt.plot(x_pred, y_pred, label="Kinematic Prediction")
    plt.legend()
    plt.title("Drone Trajectory: True vs Noisy vs Predicted")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.grid()
    plt.show()
    
    # Calculate state loss for this model using mean-squared error (MSE)
    state_true = np.vstack([x_true, y_true, vx_true, vy_true]).T
    state_pred = np.vstack([x_pred, y_pred, vx_pred, vy_pred]).T

    mse_state = np.mean((state_true - state_pred)**2)

    print("MSE (full state):", mse_state)


def main():
    flight_simulation()
    

if __name__ == '__main__':
    main()