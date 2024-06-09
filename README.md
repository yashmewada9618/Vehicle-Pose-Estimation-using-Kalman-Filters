# Vehicle-Pose-Estimation-using-Kalman-Filters
Implemented different filters like Kalman Filters, Complementary Filters and Extended Kalman Filters, including dead reckoning to estimate the pose of the vehicle by fusing IMU and GPS data.

# Ouput of EKF vs KF vs Dead Reckoning

![EKF](/Output/EKF.png)
![Dead Reckoning](/Output/Estimated%20Trajectory.png)
![KF](/Output/kf.png)
References: https://github.com/motokimura/kalman_filter_with_kitti/blob/master/demo.ipynb

This implementation demonstrates the use of an EKF to fuse IMU and GPS data for vehicle trajectory estimation. The provided code can be adapted and extended for various applications involving sensor fusion and state estimation.

During certain time intervals, the GPS signal may be lost or degraded due to signal blockage or multipath effects. In such cases, the EKF can provide a reliable estimate of the vehicle's trajectory by fusing IMU data with the last known GPS position. The motion model used in this implementation is a simple constant velocity model, which assumes that the vehicle moves with a constant velocity between GPS updates.

The EKF algorithm consists of two main steps: prediction and update. In the prediction step, the state estimate is propagated forward in time using the motion model and the process noise covariance matrix. In the update step, the predicted state estimate is corrected using the measurement (GPS) data and the measurement noise covariance matrix.


The main idea behind sensor fusion is to estimate the pose of vehicle using multiple sensors. As long as the GPS measurements are available, the EKF uses the GPS measurements to correct the state estimate. When the GPS measurements are not available, the EKF relies on the IMU measurements to propagate the state estimate forward in time. This allows the EKF to provide a reliable estimate of the vehicle's trajectory even when the GPS signal is lost or degraded.

# Motion Model

The motion model used in this implementation is a simple constant velocity model, which assumes that the vehicle moves with a constant velocity between GPS updates. The state vector consists of the vehicle's position (x, y) and velocity (vx, vy). The state transition matrix A and the process noise covariance matrix Q are defined as follows:

```math
X = [x, y, \theta, v]
```

Control input vector u is defined as:

```math
u = [v, \omega]
```

The state transition model can be expressed as:
    
```math
\begin{equation}
x_{t+1} = Fx_t + Bu
\end{equation}
```

where F is the state transition matrix, B is the control input matrix, and u is the control input vector.
```math
\begin{equation}
F = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}
\end{equation}
```

```math
\begin{equation}
B = \begin{bmatrix} dt & 0 \\ 0 & dt \\ 0 & 0 \\ 0 & 0 \end{bmatrix}
\end{equation}
```
where dt is the time step between GPS updates.

Using these:

```math
\begin{equation}
\begin{aligned}
    x_{t+1} &= x_t + v \cdot \Delta t \cdot \cos(\theta) \\
    y_{t+1} &= y_t + v \cdot \Delta t \cdot \sin(\theta) \\
    \theta_{t+1} &= \theta_t + \omega \cdot \Delta t \\
    v_{t+1} &= v_t
\end{aligned}
\end{equation}
```

The Jacobian matrix of the motion model is calculated as follows:

```math
\mathbf{F} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = 
\begin{bmatrix}
1 & 0 & -v \cdot \Delta t \cdot \sin(\theta) & \Delta t \cdot \cos(\theta) \\
0 & 1 & v \cdot \Delta t \cdot \cos(\theta) & \Delta t \cdot \sin(\theta) \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
```
Derivatives:

```math
\begin{aligned}
    \frac{\partial x_{t+1}}{\partial \theta} &= -v \cdot \Delta t \cdot \sin(\theta) \\
    \frac{\partial x_{t+1}}{\partial v} &= \Delta t \cdot \cos(\theta) \\
    \frac{\partial y_{t+1}}{\partial \theta} &= v \cdot \Delta t \cdot \cos(\theta) \\
    \frac{\partial y_{t+1}}{\partial v} &= \Delta t \cdot \sin(\theta)
\end{aligned}
```
