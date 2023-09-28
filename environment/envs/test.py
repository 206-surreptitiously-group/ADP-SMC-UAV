from UAV.uav_att_ctrl import uav_att_ctrl
from UAV.uav import uav_param
from UAV.FNTSMC import fntsmc_param
from UAV.ref_cmd import *
import numpy as np
import datetime, os
import matplotlib.pyplot as plt

DT = 0.01

'''Parameter list of the quadrotor'''
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

if __name__ == '__main__':
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
    # os.mkdir(new_path)
    # att_ctrl.collector.package2file(path=new_path)
    att_ctrl.collector.plot_att()
    att_ctrl.collector.plot_torque()
    plt.show()
