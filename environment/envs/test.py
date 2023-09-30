import datetime
import os

import matplotlib.pyplot as plt
import numpy as np

from UAV.FNTSMC import fntsmc_param
from UAV.ref_cmd import *
from UAV.uav import uav_param
from UAV.uav_att_ctrl import uav_att_ctrl
from UAV.uav_pos_ctrl import uav_pos_ctrl

'''Parameter list of the quadrotor'''
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
'''Parameter list of the quadrotor'''

'''Parameter list of the attitude controller'''
att_ctrl_param = fntsmc_param()
att_ctrl_param.k1 = np.array([25, 25, 40])
att_ctrl_param.k2 = np.array([0.1, 0.1, 0.2])
att_ctrl_param.alpha = np.array([2.5, 2.5, 2.5])
att_ctrl_param.beta = np.array([0.99, 0.99, 0.99])
att_ctrl_param.gamma = np.array([1.5, 1.5, 1.2])
att_ctrl_param.lmd = np.array([2.0, 2.0, 2.0])
att_ctrl_param.dim = 3
att_ctrl_param.dt = DT
att_ctrl_param.ctrl0 = np.array([0., 0., 0.])
'''Parameter list of the attitude controller'''

'''Parameter list of the position controller'''
pos_ctrl_param = fntsmc_param()
pos_ctrl_param.k1 = np.array([1.2, 0.8, 0.5])
pos_ctrl_param.k2 = np.array([0.2, 0.6, 0.5])
pos_ctrl_param.alpha = np.array([1.2, 1.5, 1.2])
pos_ctrl_param.beta = np.array([0.3, 0.3, 0.5])
pos_ctrl_param.gamma = np.array([0.2, 0.2, 0.2])
pos_ctrl_param.lmd = np.array([2.0, 2.0, 2.0])
pos_ctrl_param.dim = 3
pos_ctrl_param.dt = DT
pos_ctrl_param.ctrl0 = np.array([0., 0., 0.])
'''Parameter list of the position controller'''

TEST_ATT_CTRL = False
TEST_POS_CTRL = ~TEST_ATT_CTRL

if __name__ == '__main__':
    if TEST_ATT_CTRL:
        '''1. Define a controller'''
        att_ctrl = uav_att_ctrl(uav_param, att_ctrl_param)

        '''2. Define parameters for signal generator'''
        ref_amplitude = np.array([np.pi / 3, np.pi / 3, np.pi / 2])
        ref_period = np.array([4, 4, 4])
        ref_bias_a = np.array([0, 0, 0])
        ref_bias_phase = np.array([0., np.pi / 2, np.pi / 2])

        '''3. Control'''
        while att_ctrl.time < att_ctrl.time_max:
            if att_ctrl.n % 1000 == 0:
                print('time: ', att_ctrl.n * att_ctrl.dt)
            '''3.1. generate reference signal'''
            rhod, dot_rhod, dot2_rhod, dot3_rhod = ref_inner(att_ctrl.time, ref_amplitude, ref_period, ref_bias_a, ref_bias_phase)

            '''3.2. control'''
            torque = att_ctrl.att_control(ref=rhod, dot_ref=dot_rhod, dot2_ref=dot2_rhod)
            att_ctrl.update(action=torque)
        print('Finish...')
        new_path = '../../datasave/' + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '/'
        SAVE = False
        if SAVE:
            os.mkdir(new_path)
            att_ctrl.collector.package2file(path=new_path)
        att_ctrl.collector.plot_att()
        att_ctrl.collector.plot_torque()
        plt.show()
    elif TEST_POS_CTRL:
        '''1. Define a controller'''
        pos_ctrl = uav_pos_ctrl(uav_param, att_ctrl_param, pos_ctrl_param)

        '''2. Define parameters for signal generator'''
        ref_amplitude = np.array([2, 2, 1, np.pi / 2])  # x y z psi
        ref_period = np.array([5, 5, 4, 5])
        ref_bias_a = np.array([2, 2, 1, 0])
        ref_bias_phase = np.array([np.pi / 2, 0, 0, 0])

        phi_d = phi_d_old = 0.
        theta_d = theta_d_old = 0.
        dot_phi_d = (phi_d - phi_d_old) / pos_ctrl.dt
        dot_theta_d = (theta_d - theta_d_old) / pos_ctrl.dt
        throttle = pos_ctrl.m * pos_ctrl.g

        '''3. Control'''
        while pos_ctrl.time < pos_ctrl.time_max:
            if pos_ctrl.n % 1000 == 0:
                print('time: ', pos_ctrl.n * pos_ctrl.dt)

            '''3.1 generate '''
            ref, dot_ref, dot2_ref, _ = ref_uav(pos_ctrl.time, ref_amplitude, ref_period, ref_bias_a, ref_bias_phase)  # xd yd zd psid
            uncertainty = generate_uncertainty(time=pos_ctrl.time, is_ideal=True)
            obs = np.zeros(3)

            '''3.2 outer-loop control'''
            phi_d_old = phi_d
            theta_d_old = theta_d
            phi_d, theta_d, throttle = pos_ctrl.pos_control(ref[0:3], dot_ref[0:3], dot2_ref[0:3], obs)
            dot_phi_d = (phi_d - phi_d_old) / pos_ctrl.dt
            dot_theta_d = (theta_d - theta_d_old) / pos_ctrl.dt

            '''3.3 inner-loop control'''
            rho_d = np.array([phi_d, theta_d, ref[3]])
            dot_rho_d = np.array([dot_phi_d, dot_theta_d, dot_ref[3]])
            torque = pos_ctrl.att_control(rho_d, dot_rho_d, np.zeros(3), att_only=False)

            '''3.4 update state'''
            action_4_uav = np.array([throttle, torque[0], torque[1], torque[2]])
            pos_ctrl.update(action=action_4_uav, dis=uncertainty)
        print('Finish...')
        new_path = '../../datasave/' + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '/'
        SAVE = False
        if SAVE:
            os.mkdir(new_path)
            pos_ctrl.collector.package2file(path=new_path)
        pos_ctrl.collector.plot_att()
        pos_ctrl.collector.plot_torque()
        pos_ctrl.collector.plot_pos()
        pos_ctrl.collector.plot_vel()
        pos_ctrl.collector.plot_throttle()
        pos_ctrl.collector.plot_outer_obs()
        plt.show()
    else:
        pass
