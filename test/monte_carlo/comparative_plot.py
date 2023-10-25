import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
	res_smc = pd.read_csv('./monte_carlo_test_smc.csv', header=0).to_numpy()
	res_RL_400 = pd.read_csv('./monte_carlo_test_RL_400.csv', header=0).to_numpy()
	res_RL_800 = pd.read_csv('./monte_carlo_test_RL_800.csv', header=0).to_numpy()

	# print(res_smc.shape)
	# print(res_RL_400.shape)
	# print(res_RL_800.shape)

	'''出界个数统计'''
	print('res_smc:', np.sum(res_smc[:, 1]))
	print('res_RL_400:', np.sum(res_RL_400[:, 1]))
	print('res_RL_800:', np.sum(res_RL_800[:, 1]))
	'''出界个数统计'''

	plt.figure()
	plt.plot(np.arange(res_smc.shape[0]), res_smc[:, 0], '.')
	plt.plot(np.arange(res_RL_400.shape[0]), res_RL_400[:, 0], '.')
	plt.plot(np.arange(res_RL_800.shape[0]), res_RL_800[:, 0], '.')

	plt.show()