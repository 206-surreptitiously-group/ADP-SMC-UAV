import numpy as np
import matplotlib.pyplot as plt
from environment.envs.UAV.uav import UAV, uav_param
from common.common_func import *
import cv2 as cv

DT = 0.01
uav_param = uav_param()
uav_param.m = 0.8
uav_param.g = 9.8
uav_param.J = np.array([4.212e-3, 4.212e-3, 8.255e-3])
uav_param.d = 0.12
uav_param.CT = 2.168e-6
uav_param.CM = 2.136e-8
uav_param.J0 = 1.01e-5
uav_param.kr = 1e-3
uav_param.kt = 1e-3
uav_param.pos0 = np.array([0, 0, 0])
uav_param.vel0 = np.array([0, 0, 0])
uav_param.angle0 = np.array([0, 0, 0])
uav_param.pqr0 = np.array([0, 0, 0])
uav_param.dt = DT
uav_param.time_max = 30
uav_param.pos_zone = np.atleast_2d([[-3, 3], [-3, 3], [0, 3]])
uav_param.att_zone = np.atleast_2d([[deg2rad(-45), deg2rad(45)], [deg2rad(-45), deg2rad(45)], [deg2rad(-120), deg2rad(120)]])
uav = UAV(uav_param)

if __name__ == '__main__':
    T = 2
    t = np.linspace(0, 10 * T, 1000)
    x = 2 * np.sin(2 * np.pi / T * t)
    y = 2 * np.cos(2 * np.pi / T * t)
    z = 1.5 + 0.5 * np.sin(4 * 2 * np.pi / T * t)
    trajectory = np.vstack((x, y, z)).T
    points = np.atleast_2d([[0, 0, 0], [0, 0, 1], [1, 1, 1], [1, 2, 3], [-2, -3, 1]]).astype(float)
    uav.draw_init_image()
    uav.draw_3d_trajectory_projection(trajectory)
    uav.draw_3d_points_projection(points)
    cv.imshow('yyf', uav.image)
    cv.waitKey(0)
