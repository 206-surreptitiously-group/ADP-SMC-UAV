import pandas as pd
import numpy as np


if __name__ == '__main__':
	param = []
	A = np.linspace(0, 1.5, 16)
	T = np.linspace(5, 10, 51)
	for _A in A:
		for _T in T:
			_param = [_A, _A, _A, 0] + [_T, _T, _T, _T] + [np.pi / 2, 0., 0., 0.] + [0., 0., 1.5, 0.]
			param.append(_param)

	(pd.DataFrame(param,
				  columns=['A_x', 'A_y', 'A_z', 'A_psi'] +
						  ['T_x', 'T_y', 'T_z', 'T_psi'] +
						  ['phi0_x', 'phi0_y', 'phi0_z', 'phi0_psi'] +
						  ['bias_x', 'bias_y', 'bias_z', 'bias_psi']).to_csv('./systemic_mesh.csv', sep=',', index=False))
