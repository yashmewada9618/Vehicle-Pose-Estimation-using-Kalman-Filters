import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import math
import matplotlib.pyplot as plt
import numpy as np

from utils.plot import plot_covariance_ellipse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt, detrend, lfilter
from scipy import integrate
import tqdm
import math
import tqdm
import seaborn as sns
from pyproj import Proj, transform

# Covariance for EKF simulation
Q = (
    np.diag(
        [
            0.1,  # variance of location on x-axis
            0.1,  # variance of location on y-axis
            np.deg2rad(1.0),  # variance of yaw angle
            1.0,  # variance of velocity
        ]
    )
    ** 2
)  # predict state covariance
R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance

#  Simulation parameter
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
GPS_NOISE = np.diag([0.5, 0.5]) ** 2

DT = 0.005  # time tick [s]
SIM_TIME = 5000.0  # simulation time [s]

show_animation = False


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
        Xe = Xe / 3
        Xn = np.zeros_like(Vn)
        Xn[0] = Vn[0]
        Xn[1:] = integrate.cumtrapz(Vn, dx=self.dt)
        Xn = Xn / 3

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


def calc_input():
    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    u = np.array([[v], [yawrate]])
    return u


def observation(xTrue, xd, u):
    xTrue = motion_model(xTrue, u)

    # add noise to gps x-y
    # z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)
    # print
    # add noise to input
    ud = u + INPUT_NOISE @ np.random.randn(2, 1)

    xd = motion_model(xd, ud)

    return xTrue, xd, ud


def motion_model(x, u):
    F = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 0]])

    B = np.array(
        [
            [DT * math.cos(x[2, 0]), 0],
            [DT * math.sin(x[2, 0]), 0],
            [0.0, DT],
            [1.0, 0.0],
        ]
    )

    x = F @ x + B @ u

    return x


def observation_model(x):
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    z = H @ x

    return z


def jacob_f(x, u):
    """
    Jacobian of Motion Model

    motion model
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    so
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
    """
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array(
        [
            [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
            [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    return jF


def jacob_h():
    # Jacobian of Observation Model
    jH = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    return jH


def ekf_estimation(xEst, PEst, z, u):
    #  Predict
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst, u)
    PPred = jF @ PEst @ jF.T + Q

    #  Update
    jH = jacob_h()
    zPred = observation_model(xPred)
    y = z - zPred
    S = jH @ PPred @ jH.T + R
    K = PPred @ jH.T @ np.linalg.inv(S)
    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
    return xEst, PEst


def main():
    gpsCsv = "/home/mewada/Documents/AFR/CSV_data/gps-fix.csv"
    imuCsv = "/home/mewada/Documents/AFR/CSV_data/imu-imu_uncompensated.csv"
    gdnTruth = "/home/mewada/Documents/AFR/CSV_data/vehicle-gps-fix.csv"
    gndTruth = pd.read_csv(gdnTruth)
    dk = deadReckoning(gpsCsv, imuCsv)
    plot = False
    fwdVel = dk.getfwdVel2(plot)
    # fwdVel = dk.getfwdVel(plot)
    yaw = dk.removeBiasYaw(plot)
    utmx, utmY = dk.converttoUTM(dk.gps_data)
    gndX, gndY = dk.converttoUTM(gndTruth)
    yawRates = dk.getangVel(plot)
    print(__file__ + " start!!")

    time = 0.0

    # State Vector [x y yaw v]'
    xEst = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    PEst = np.eye(4)

    xDR = np.zeros((4, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))

    k = 0
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    z = np.array([[0], [0], [0], [0]])
    z = H @ z
    for i in tqdm.tqdm(range(len(fwdVel))):
        v = fwdVel[i]
        yawrate = yawRates[i]
        u = np.array([[v], [yawrate]])
        xTrue, xDR, ud = observation(xTrue, xDR, u)
        if i < len(utmx):
            if np.isnan(utmx[i]) != True and np.isnan(utmY[i]) != True:
                z = np.array([[utmx[i]], [utmY[k]], [yaw[i]], [v]])
                z = H @ z
                xEst, PEst = ekf_estimation(xEst, PEst, z, ud)
            k += 1
            # store data history
            hxEst = np.hstack((hxEst, xEst))
            hxDR = np.hstack((hxDR, xDR))
            hxTrue = np.hstack((hxTrue, xTrue))
            hz = np.hstack((hz, z))
        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )
            plt.plot(hz[0, :], hz[1, :], ".g")
            plt.plot(hxTrue[0, :].flatten(), hxTrue[1, :].flatten(), "-b")
            plt.plot(hxDR[0, :].flatten(), hxDR[1, :].flatten(), "-k")
            plt.plot(hxEst[0, :].flatten(), hxEst[1, :].flatten(), "-r")
            plot_covariance_ellipse(xEst[0, 0], xEst[1, 0], PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)

    plt.plot(hz[0, :], hz[1, :], ".g")
    plt.plot(hxTrue[0, :].flatten(), hxTrue[1, :].flatten(), "-b", label="GPS")
    plt.plot(hxDR[0, :].flatten(), hxDR[1, :].flatten(), "-k", label="Dead Reckoning")
    plt.plot(hxEst[0, :].flatten(), hxEst[1, :].flatten(), "-r", label="EKF")
    plt.plot(gndX, gndY, "-y", label="Ground Truth")
    plot_covariance_ellipse(xEst[0, 0], xEst[1, 0], PEst)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.savefig("EKF.png")
    plt.show()


if __name__ == "__main__":
    main()
