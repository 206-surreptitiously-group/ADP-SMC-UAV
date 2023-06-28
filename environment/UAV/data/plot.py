import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    path = './save1/'
    controlData = pd.read_csv(path + 'control.csv', header=0).to_numpy()
    observeData = pd.read_csv(path + 'observe.csv', header=0).to_numpy()
    ref_cmdData = pd.read_csv(path + 'ref_cmd.csv', header=0).to_numpy()
    uav_stateData = pd.read_csv(path + 'uav_state.csv', header=0).to_numpy()

    time = controlData[:, 0]

    in_control = controlData[:, 1: 5]

    in_delta = observeData[:, 1: 5]
    in_delta_obs = observeData[:, 5: 9]

    ref_pos = ref_cmdData[:, 1: 4]
    ref_angle = ref_cmdData[:, 4: 7]

    uav_pos = uav_stateData[:, 1: 4]
    uav_angle = uav_stateData[:, 7: 10]

    print('   time:    ', time.shape)
    print('in_control: ', in_control.shape)
    print('   pos:     ', uav_pos.shape)
    print('  angle:    ', uav_angle.shape)
    print('  delta:    ', in_delta.shape)

    plt.figure(0)
    plt.plot(time, ref_pos[:, 2], 'red')
    plt.plot(time, uav_pos[:, 2], 'blue')
    plt.xlabel('time(s)')
    plt.title('Z')

    plt.figure(1)
    plt.plot(time, ref_angle[:, 0], 'red')
    plt.plot(time, uav_angle[:, 0], 'blue')
    plt.ylim((-60, 60))
    plt.yticks(np.arange(-60, 60, 15))
    plt.xlabel('time(s)')
    plt.title('roll  $\phi$')

    plt.figure(2)
    plt.plot(time, ref_angle[:, 1], 'red')
    plt.plot(time, uav_angle[:, 1], 'blue')
    plt.ylim((-60, 60))
    plt.yticks(np.arange(-60, 60, 15))
    plt.xlabel('time(s)')
    plt.title('pitch  $\Theta$')

    plt.figure(3)
    plt.plot(time, ref_angle[:, 2], 'red')
    plt.plot(time, uav_angle[:, 2], 'blue')
    plt.ylim((-90, 90))
    plt.yticks(np.arange(-90, 90, 15))
    plt.xlabel('time(s)')
    plt.title('yaw  $\psi$')

    plt.figure(4)
    plt.plot(time, in_control[:, 0], 'red')   # 油门
    plt.xlabel('time(s)')
    plt.title('throttle')

    plt.figure(5)
    plt.subplot(2, 2, 1)
    plt.plot(time, in_delta[:, 0], 'red')
    plt.plot(time, in_delta_obs[:, 0], 'blue')
    plt.xlabel('time(s)')
    plt.ylim((-4, 4))
    plt.title('observe error Fdz')

    plt.subplot(2, 2, 2)
    plt.plot(time, in_delta[:, 1], 'red')
    plt.plot(time, in_delta_obs[:, 1], 'blue')
    plt.xlabel('time(s)')
    plt.ylim((-2, 2))
    plt.title('observe error dp')

    plt.subplot(2, 2, 3)
    plt.plot(time, in_delta[:, 2], 'red')
    plt.plot(time, in_delta_obs[:, 2], 'blue')
    plt.xlabel('time(s)')
    plt.ylim((-2, 2))
    plt.title('observe error dq')

    plt.subplot(2, 2, 4)
    plt.plot(time, in_delta[:, 3], 'red')
    plt.plot(time, in_delta_obs[:, 3], 'blue')
    plt.xlabel('time(s)')
    plt.ylim((-3, 3))
    plt.title('observe error dr')

    plt.show()