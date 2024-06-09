#!/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt, detrend, lfilter
from scipy import integrate
import tqdm
from test import KalmanFilter as kf
import math
import tqdm
import seaborn as sns
from pyproj import Proj, transform


class deadReckoning:
    def __init__(self, gpsCsv, imuCsv):
        self.gpsCsv = gpsCsv
        self.imuCsv = imuCsv
        self.dt = 1.0 / 200
        self.imu_data = pd.read_csv(self.imuCsv)
        self.gps_data = pd.read_csv(self.gpsCsv)

    def getfwdVel2(self, plot=False):
        linX = (
            self.imu_data["linear_acceleration.x"].to_numpy()
            - self.imu_data["linear_acceleration.x"][0]
        )
        linY = (
            self.imu_data["linear_acceleration.y"].to_numpy()
            - self.imu_data["linear_acceleration.y"][0]
        )
        linZ = (
            self.imu_data["linear_acceleration.z"].to_numpy()
            - self.imu_data["linear_acceleration.z"][0]
        )

        nyquist = 0.5 * self.dt
        cutoff = 0.001  # 1 Hz
        normal_cutoff = cutoff / nyquist
        print("Normal Cutoff: ", normal_cutoff)
        b, a = butter(2, normal_cutoff, btype="high", analog=False)
        filtered_data = lfilter(b, a, linX)

        static = []
        for i in tqdm.tqdm(range(1, len(linX))):
            delX = linX[i] - linX[i - 1]
            delY = linY[i] - linY[i - 1]
            delZ = linZ[i] - linZ[i - 1]
            if abs(delX) < 0.3 and abs(delY) < 0.3 and abs(delZ) < 0.3:
                static.append(filtered_data[i])

        static = np.array(static)
        avg = np.mean(static)
        filtered_data = filtered_data - avg

        fwdVel = integrate.cumtrapz(filtered_data, dx=self.dt, initial=0)
        fwdVel[fwdVel < 0.0] = 0.0

        if plot:
            plt.plot(linX, label="Unfiltered")
            plt.plot(filtered_data, label="Filtered")
            plt.legend()
            plt.grid()
            plt.savefig("Filtered Forward Velocity.png")
            plt.show()
        return fwdVel

    def getfwdVel(self, plot=False):
        linX = detrend(
            self.imu_data["linear_acceleration.x"].to_numpy()
            - self.imu_data["linear_acceleration.x"][0]
        )
        linY = detrend(
            self.imu_data["linear_acceleration.y"].to_numpy()
            - self.imu_data["linear_acceleration.y"][0]
        )
        linZ = detrend(
            self.imu_data["linear_acceleration.z"].to_numpy()
            - self.imu_data["linear_acceleration.z"][0]
        )

        staticfwdVel = []
        for i in tqdm.tqdm(range(1, len(linX))):
            delX = linX[i] - linX[i - 1]
            delY = linY[i] - linY[i - 1]
            delZ = linZ[i] - linZ[i - 1]
            delAcc = np.sqrt(delX**2 + delY**2 + delZ**2)
            if abs(delAcc) < 0.3:
                staticfwdVel.append(self.imu_data["linear_acceleration.x"][i])

        staticfwdVel = np.array(staticfwdVel)
        avgAngVel = np.mean(staticfwdVel)

        angVel = []
        for i in tqdm.tqdm(range(len(linX))):
            angVel.append(self.imu_data["linear_acceleration.x"][i] - avgAngVel)

        fwdVelAdjusted = integrate.cumtrapz(angVel, dx=self.dt, initial=0)
        fwdVelAdjusted[fwdVelAdjusted < 0] = 0

        linXBias = np.mean(self.imu_data["linear_acceleration.x"])
        linX = self.imu_data["linear_acceleration.x"] - linXBias
        fwdvelUnadjusted = integrate.cumtrapz(linX, dx=self.dt, initial=0)
        fwdvelUnadjusted[fwdvelUnadjusted < 0] = 0.0

        if plot:
            plt.plot(fwdVelAdjusted, label="Resulant Filtered Velocity")
            plt.plot(fwdvelUnadjusted, label="Resulant UnFiltered Velocity")
            plt.title("Filtered Forward Velocity")
            plt.xlabel("Time")
            plt.ylabel("Velocity m/s")
            plt.legend()
            plt.grid()
            plt.savefig("Filtered Forward Velocity.png")
            plt.show()

        return fwdVelAdjusted

    def to_eulers(self, w, x, y, z):
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.degrees(yaw)

    def removeBiasYaw(self, plot=False):
        linX = detrend(
            self.imu_data["linear_acceleration.x"].to_numpy()
            - self.imu_data["linear_acceleration.x"][0]
        )
        linY = detrend(
            self.imu_data["linear_acceleration.y"].to_numpy()
            - self.imu_data["linear_acceleration.y"][0]
        )
        linZ = detrend(
            self.imu_data["linear_acceleration.z"].to_numpy()
            - self.imu_data["linear_acceleration.z"][0]
        )

        staticYaw = []
        for i in tqdm.tqdm(range(1, len(linX))):
            delX = linX[i] - linX[i - 1]
            delY = linY[i] - linY[i - 1]
            delZ = linZ[i] - linZ[i - 1]
            delAcc = np.sqrt(delX**2 + delY**2 + delZ**2)
            w, x, y, z = (
                self.imu_data["orientation.w"][i],
                self.imu_data["orientation.x"][i],
                self.imu_data["orientation.y"][i],
                self.imu_data["orientation.z"][i],
            )
            if (
                abs(delX) < 0.3
                and abs(delY) < 0.3
                and abs(delZ) < 0.3
                and abs(delAcc) < 0.3
            ):
                staticYaw.append(self.to_eulers(w, x, y, z))

        # Bias removal from static yaw estimates
        staticYaw = np.array(staticYaw)
        avgYaw = np.mean(staticYaw)

        yaw = []
        unfilteredYaw = []
        print("Avg Yaw: ", avgYaw)
        for i in tqdm.tqdm(range(len(linX))):
            w, x, y, z = (
                self.imu_data["orientation.w"][i],
                self.imu_data["orientation.x"][i],
                self.imu_data["orientation.y"][i],
                self.imu_data["orientation.z"][i],
            )
            yaw.append(self.to_eulers(w, x, y, z) - avgYaw)
            unfilteredYaw.append(self.to_eulers(w, x, y, z))

        if plot:
            plt.plot(yaw, label="Yaw")
            plt.plot(unfilteredYaw, label="Unfiltered Yaw")
            plt.xlabel("Time")
            plt.ylabel("Yaw Angle degrees")
            plt.legend()
            plt.grid()
            plt.savefig("YawFiltered.png")
            plt.show()

        return yaw

    def converttoUTM(self, gps):
        c = 0
        utmX = []
        utmY = []
        lat = gps["latitude"]
        long = gps["longitude"]
        myProj = Proj(proj="utm", zone="19", ellps="WGS84")
        for i in range(len(lat)):
            if np.isnan(lat[i]) != True:
                utmx, utmy = myProj(long[i], lat[i])
                if c == 0:
                    iniX = utmx
                    iniY = utmy
                    c += 1
                utmX.append(utmx - iniX)
                utmY.append(utmy - iniY)
            else:
                utmX.append(long[i])
                utmY.append(lat[i])

        return utmX, utmY

    def estTraj(self, plot=False):
        fv = np.unwrap(self.getfwdVel())
        yaw = self.removeBiasYaw()[: len(fv)]

        print("Yaw: ", len(yaw))
        print("FV: ", len(fv))
        Vn = fv * np.cos(yaw) + np.sin(yaw) * fv
        Ve = fv * np.sin(yaw) - np.cos(yaw) * fv

        Xe = np.zeros_like(Ve)
        Xe[0] = Ve[0]
        Xe[1:] = integrate.cumtrapz(Ve, dx=self.dt)
        Xe = Xe/3
        Xn = np.zeros_like(Vn)
        Xn[0] = Vn[0]
        Xn[1:] = integrate.cumtrapz(Vn, dx=self.dt)
        Xn = Xn/3

        utmx, utmy = self.converttoUTM(self.gps_data)
        if plot:
            plt.plot(Xe, Xn, label="Estimated IMU Trajectory")
            plt.plot(utmx, utmy, label="GPS Trajectory")
            plt.xlabel("East [m]")
            plt.ylabel("North [m]")
            plt.legend()
            plt.grid()
            plt.title("Estimated Trajectory using Dead Reckoning")
            plt.savefig("Estimated Trajectory.png")
            plt.show()

        X = Xe
        Y = Xn
        return X, Y

    def getGPSVel(self, plot=False):
        utmx, utmy = self.converttoUTM(self.gps_data)
        print("UTMX: ", utmx)
        time = (
            self.gps_data["header.stamp.secs"] - self.gps_data["header.stamp.secs"][0]
        )
        velX = []
        velY = []
        finalVel = []
        for i in range(1, len(utmx)):
            # print("utmX: ", utmx[i])
            if not np.isnan(utmx[i]) and not np.isnan(utmy[i]):
                dt = time[i] - time[i - 1]
                velX.append((utmx[i] - utmx[i - 1]) / dt)
                velY.append((utmy[i] - utmy[i - 1]) / dt)

        for i in range(len(velX)):
            finalVel.append(np.sqrt(velX[i] ** 2 + velY[i] ** 2))

        gpsVel = np.array(finalVel)
        print("GPS Vel: ", gpsVel)
        fv = np.unwrap(self.getfwdVel())

        if plot:
            plt.plot(gpsVel, label="GPS Velocity")
            # plt.plot(fv, label="IMU Velocity")
            plt.xlabel("Time")
            plt.ylabel("Velocity m/s")
            plt.legend()
            plt.grid()
            plt.title("GPS Velocity")
            plt.savefig("GPS Velocity.png")
            plt.show()
        return gpsVel

    def getangVel(self, plot):
        linX = detrend(
            self.imu_data["linear_acceleration.x"].to_numpy()
            - self.imu_data["linear_acceleration.x"][0]
        )
        linY = detrend(
            self.imu_data["linear_acceleration.y"].to_numpy()
            - self.imu_data["linear_acceleration.y"][0]
        )
        linZ = detrend(
            self.imu_data["linear_acceleration.z"].to_numpy()
            - self.imu_data["linear_acceleration.z"][0]
        )

        staticAngVel = []
        for i in tqdm.tqdm(range(1, len(linX))):
            delX = linX[i] - linX[i - 1]
            delY = linY[i] - linY[i - 1]
            delZ = linZ[i] - linZ[i - 1]
            delAcc = np.sqrt(delX**2 + delY**2 + delZ**2)
            if (
                abs(delX) < 0.3
                and abs(delY) < 0.3
                and abs(delZ) < 0.3
                and abs(delAcc) < 0.3
            ):
                staticAngVel.append(self.imu_data["angular_velocity.z"][i])

        staticAngVel = np.array(staticAngVel)
        avgAngVel = np.mean(staticAngVel)

        angVel = []
        for i in tqdm.tqdm(range(len(linX))):
            angVel.append(self.imu_data["angular_velocity.z"][i] - avgAngVel)

        if plot:
            plt.plot(angVel, label="Angular Velocity")
            plt.xlabel("Time")
            plt.ylabel("Angular Velocity")
            plt.legend()
            plt.grid()
            plt.title("Angular Velocity")
            plt.savefig("Angular Velocity.png")
            plt.show()

        return angVel


class KalmanFilter:
    def __init__(self, X, P):
        self.x = X
        self.P = P

    def predict(self, U, Q, dt):
        x, y, theta = self.x
        v, w = U
        r = v / w
        F = np.array(
            [
                [1.0, 0.0, -r * np.cos(theta) + r * np.cos(theta + w * dt)],
                [0.0, 1.0, -r * np.sin(theta) + r * np.sin(theta + w * dt)],
                [0.0, 0.0, 1.0],
            ]
        )

        B = np.array([dt * np.cos(theta), dt * np.sin(theta)])

        dtheta = w * dt
        dx = -r * np.sin(theta) + r * np.sin(theta + dtheta)
        dy = +r * np.cos(theta) - r * np.cos(theta + dtheta)

        self.x = np.dot(F, self.x) + np.dot(B, U.T)
        self.P = F @ self.P @ F.T + Q
        return self.x, self.P

    def update(self, Z, R):
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        y = Z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = self.P - K @ H @ self.P
        return self.x, self.P


from numpy import dot
from numpy.linalg import inv


class Kalman_filter:
    def __init__(self, X, P):
        self.X = X
        self.P = P
        self.K = 0

    def predict(self, A, Q, B, U):
        x, y, theta = self.X
        v, w = U
        r = v / w
        A = np.array(
            [
                [1.0, 0.0, -r * np.cos(theta) + r * np.cos(theta + w * dt)],
                [0.0, 1.0, -r * np.sin(theta) + r * np.sin(theta + w * dt)],
                [0.0, 0.0, 1.0],
            ]
        )

        self.X = dot(A, self.X) + dot(B, U)
        self.P = dot(A, dot(self.P, A.T)) + Q
        return (self.X, self.P)

    def update(self, Y, H, R):
        IM = dot(H, self.X)
        IS = R + dot(H, dot(self.P, H.T))
        self.K = dot(self.P, dot(H.T, inv(IS)))
        self.X = self.X + dot(self.K, (Y - IM))
        self.P = self.P - dot(self.K, dot(IS, self.K.T))
        return (self.X, self.P, self.K)


def trail2(utmx, utmy, fwdVel, yaw, yawRates, dt, noise):
    X = np.array([0.0, 0.0, 0.0])
    print("X: ", X)
    P = np.array(
        [
            [noise, 0.0, 0.0],
            [0.0, noise, 0.0],
            [0.0, 0.0, noise],
        ]
    )
    term1 = 0.1
    term2 = 0.1
    Q = np.array([[term1, 0, term2], [0, term1, 0], [term2, 0, term1]])
    R = np.array(
        [
            [noise, 0.0, 0.0],
            [0.0, noise, 0.0],
            [0.0, 0.0, noise],
        ]
    )

    mykf = KalmanFilter(X, P)
    mu_x = [
        X[0],
    ]
    mu_y = [
        X[1],
    ]
    mu_theta = [
        X[2],
    ]

    var_x = [
        P[0, 0],
    ]
    var_y = [
        P[1, 1],
    ]
    var_theta = [
        P[2, 2],
    ]

    t_last = 0
    t = dk.gps_data["header.stamp.secs"] - dk.gps_data["header.stamp.secs"][0]
    print("Length of t: ", len(t))
    k = 0
    for i in tqdm.tqdm(range(1, len(fwdVel))):
        v = fwdVel[i]
        w = yawRates[i]
        # del_t = t[i] - t_last
        U = np.array([v, w])

        R_ = R * (dt**2.0)

        # Propagate KF
        mykf.predict(U, Q, dt)
        if k < len(utmx) and not np.isnan(utmx[k]) and not np.isnan(utmY[k]):
            Z = np.array([utmx[k], utmy[k], yaw[i]])
            mykf.update(Z, R_)
            k += 1
        mu_x.append(mykf.x[0])
        mu_y.append(mykf.x[1])
        mu_theta.append(mykf.x[2])

        var_x.append(mykf.P[0, 0])
        var_y.append(mykf.P[1, 1])
        var_theta.append(mykf.P[2, 2])

    plt.plot(mu_x, mu_y, label="Estimated Trajectory")
    plt.plot(utmx, utmY, label="GPS Trajectory")
    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.legend()
    plt.grid()
    plt.title("Estimated Trajectory using Kalman Filter")
    plt.savefig("Estimated Trajectory using Kalman Filter.png")
    plt.show()


def trail3(utmx, utmy, fwdVel, yaw, yawRates, dt, noise):
    import seaborn

    vehcilegps = pd.read_csv("/home/mewada/Documents/AFR/CSV_data/vehicle-gps-fix.csv")
    gndutmx, gnduty = dk.converttoUTM(vehcilegps)
    X,Y = dk.estTraj()
    X_kf = []
    Y_kf = []
    Theta_kf = []
    dt = 1.0 / 200
    # Initialization of state matrices
    X = np.array([0, 0, 0])
    P = np.eye(3) * 1e-4
    my_kf = Kalman_filter(X, P)
    A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    Q = np.eye(X.shape[0]) * 1e-4
    R = np.eye(X.shape[0]) * 1e-2
    c = 0
    k = 0
    velocity = fwdVel
    yaw_rate = yawRates
    dist_x, dist_y = utmx, utmy
    pxkf = []
    pykf = []
    pthethakf = []
    # Applying the Kalman Filter
    for i in tqdm.tqdm(range(len(velocity))):
        v = velocity[i]
        theta = yaw_rate[i]
        if theta == 0:
            B = np.array([dt * np.cos(X[2]), dt * np.sin(X[2]), 0])
            U = np.array([v])
            (X, P) = my_kf.predict(A, Q, B, U)
        else:
            B = np.array(
                [
                    [(np.sin(X[2] + theta * dt) - np.sin(X[2])) / theta, 0],
                    [(np.cos(X[2]) - np.cos(X[2] + theta * dt)) / theta, 0],
                    [0, dt],
                ]
            )
            U = np.array([v, theta])
            (X, P) = my_kf.predict(A, Q, B, U)
        if k < 30758:
            if c != 20:
                if np.isnan(dist_x[k]) == False and np.isnan(dist_y[k]) == False:
                    Y = np.array([dist_x[k], dist_y[k], yaw[i]])
                    H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                    (X, P, K) = my_kf.update(Y, H, R)
            else:
                if np.isnan(dist_x[k]) == False and np.isnan(dist_y[k]) == False:
                    Y = np.array([dist_x[k], dist_y[k], yaw[i]])
                    H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                    (X, P, K) = my_kf.update(Y, H, R)
                k += 1
                c = 0
        c += 1
        X_kf.append(X[0])
        Y_kf.append(X[1])
        Theta_kf.append(X[2])
        
        pxkf.append(P[0, 0])
        pykf.append(P[1, 1])
        pthethakf.append(P[2, 2])
        
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plt.plot(X_kf, Y_kf, label="Estimated Trajectory")
    plt.plot(utmx, utmY, label="GPS Trajectory")
    plt.plot(gndutmx, gnduty, label="Ground Truth Trajectory")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    plt.title("Estimated path")
    plt.legend()
    plt.grid()
    plt.savefig("Estimated Trajectory using Kalman Filter.png")
    plt.show()

    # ax.set_xlabel("X Position")
    # ax.set_ylabel("Y Position")
    # plt.title("Estimated path")
    # plt.legend()
    # plt.grid()
    # # plt.savefig("Ground Truth Trajectory.png")
    # plt.show()
    
    plt.plot(pxkf, label="X Variance")
    plt.plot(pykf, label="Y Variance")
    plt.plot(pthethakf, label="Theta Variance")
    plt.legend()
    plt.grid()
    plt.savefig("Variance.png")
    plt.show()


if __name__ == "__main__":
    plot = True
    gpsCsv = "/home/mewada/Documents/AFR/CSV_data/gps-fix.csv"
    imuCsv = "/home/mewada/Documents/AFR/CSV_data/imu-imu_uncompensated.csv"
    dk = deadReckoning(gpsCsv, imuCsv)

    fwdVel = dk.getfwdVel2(plot)
    # fwdVel = dk.getfwdVel(plot)
    yaw = dk.removeBiasYaw(plot)
    utmx, utmY = dk.converttoUTM(dk.gps_data)
    yawRates = dk.getangVel(plot)
    dk.estTraj(plot)
    # dk.getGPSVel(plot)

    dt = 1.0 / 200
    noise = 5.0 * dt
    trail3(utmx, utmY, fwdVel, yaw, yawRates, dt, noise)

    # x = np.array([0.0,0.0,0.0,0.0])
    # p = np.eye(4)
    # F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
    # term1 = (dt**4)/4*noise
    # term2 = (dt**3)/2*noise
    # term3 = (dt**2)*noise
    # Q = np.array([[term1,0,term2,0],[0,term1,0,term2],[term2,0,term3,0],[0,term2,0,term3]])
    # Z = np.array([0.0,0.0])
    # H = np.array([[1,0,0,0],[0,1,0,0]])
    # R = np.eye(2)*noise
    # B = np.zeros(4)
    # U = np.zeros(4)

    # estimatedStates = []
    # estimatedTheta = []

    # mykf = KalmanFilter(x,p)

    # for i in tqdm.tqdm(range(len(fwdVel))):
    #     x,P = mykf.predict(F,Q,np.zeros(4),np.zeros(4))
    #     x,P = mykf.update(np.array([fwdVel[i],yaw[i]]),H,R)
    #     estimatedStates.append(x)
    #     estimatedTheta.append(x[2])

    # estimatedStates = np.array(estimatedStates)
    # plt.plot(estimatedStates[:,0],label="Estimated X")
    # plt.plot(estimatedStates[:,1],label="Estimated Y")
    # # plt.plot(estimatedTheta,label="Estimated Theta")
    # plt.legend()
    # plt.grid()
    # plt.show()
