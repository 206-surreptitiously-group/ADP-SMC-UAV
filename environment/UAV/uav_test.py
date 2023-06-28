import numpy as np
import matplotlib.pyplot as plt
from uav import UAV


if __name__ == '__main__':
    uav = UAV()

    save_time = []
    save_state = []

    while uav.time < uav.time_max:
        save_time.append(uav.time)
        save_state.append(uav.uav_state_call_back())
        uav.control_ideal = np.array([uav.m * uav.g, 0., 0., 0.])
        dis = np.zeros(6)
        uav.rk44(action=uav.control_ideal, dis=dis, n=10)

    save_time = np.array(save_time)
    save_state = np.array(save_state)

    plt.figure(0)
    plt.plot(save_time, save_state[:, 0])
    plt.xlabel('time(s)')
    plt.ylabel('x(m)')
    plt.title('x')

    plt.figure(1)
    plt.plot(save_time, save_state[:, 1])
    plt.xlabel('time(s)')
    plt.ylabel('y(m)')
    plt.title('y')

    plt.figure(2)
    plt.plot(save_time, save_state[:, 2])
    plt.xlabel('time(s)')
    plt.ylabel('z(m)')
    plt.title('z')

    plt.figure(3)
    plt.plot(save_time, save_state[:, 6] * 180 / np.pi)
    plt.xlabel('time(s)')
    plt.ylabel('phi')
    plt.title('phi')

    plt.figure(4)
    plt.plot(save_time, save_state[:, 7] * 180 / np.pi)
    plt.xlabel('time(s)')
    plt.ylabel('theta')
    plt.title('theta')

    plt.figure(5)
    plt.plot(save_time, save_state[:, 8] * 180 / np.pi)
    plt.xlabel('time(s)')
    plt.ylabel('psi')
    plt.title('psi')

    plt.show()