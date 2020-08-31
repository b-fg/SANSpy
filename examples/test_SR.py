"""
Test script using type 2 model with 4 inputs and 2 outputs: M4i2o_2
"""
import numpy as np
from postproc import io, calc
from sanspy.model import Model
import os


# Constants
sh = (1216, 540) # Shape of 2D fields
L = 90 # Scaling parameter
window = [(0,12), (-2,2)] # Window for wake mask
root_data = '/dataset/evaluate/'
params_m = {'data_path': root_data,
            'weights_load': 'weights/weights_SR_1_060.h5',
            'M_type': 'M4i2o_2',
	        'n_in': 4,
	        'n_out': 2,
	        'skip': ['uu', 'uv', 'vv'],
            'loss': 'sse',
            'max_filters': 64,
            'activation': 'relu',
            'kernel_size': 5,
            'norm_in': True,
            'pp': 3,
            'sigma':0.5,
            'l2': 0,}


# Util functions
def time_series(path):
	def absolute_file_paths(directory):
		files = []
		for dir_path, _, file_names in os.walk(directory):
			for f in file_names:
				files.append(os.path.abspath(os.path.join(dir_path, f)))
		return files
	files = absolute_file_paths(path)
	return sorted([f.split('_')[-1].split('.dat')[0] for f in files], key=lambda x: float(x))


# Main
def main():
	model = Model(**params_m)
	t_sorted = time_series(root_data + 'x/U/')
	C_SRx_list, C_SRy_list = [], []
	for t in t_sorted:
		print('\n',t)
		P = io.read_data(root_data + 'x/P/P_' + str(t) + '.dat', sh, ncomponents=1, dtype=np.single, periodic=(0,0,0))
		U = io.read_data(root_data + 'x/U/U_' + str(t) + '.dat', sh, ncomponents=1, dtype=np.single, periodic=(0,0,0))
		V = io.read_data(root_data + 'x/V/V_' + str(t) + '.dat', sh, ncomponents=1, dtype=np.single, periodic=(0,0,0))
		SRx = io.read_data(root_data + 'y/SRx/SRx_' + str(t) + '.dat', sh, ncomponents=1, dtype=np.single, periodic=(0,0,0))
		SRy = io.read_data(root_data + 'y/SRy/SRy_' + str(t) + '.dat', sh, ncomponents=1, dtype=np.single, periodic=(0,0,0))

		# ML Predictions from X
		wake = calc.wake_width_length(calc.vortZ(U, V), eps=0.001)
		if params_m['norm_in']:
			P = (P - np.mean(P)) / np.std(P)
			U = (U - np.mean(U)) / np.std(U)
			V = (V - np.mean(V)) / np.std(V)
		x_dict = {'x1': P[np.newaxis, ..., np.newaxis],
		          'x2': U[np.newaxis, ..., np.newaxis],
		          'x3': V[np.newaxis, ..., np.newaxis],
		          'x4': wake[np.newaxis, ..., np.newaxis]}

		Yp = model.predict_from_x(x_dict)
		SRx_p, SRy_p = Yp['y1'], Yp['y2']

		SRx = calc.crop(SRx, window, scaling=L, shift=(-1.5, -2 / L))
		SRy = calc.crop(SRy, window, scaling=L, shift=(-1.5, -2 / L))
		SRx_p = calc.crop(SRx_p, window, scaling=L, shift=(-1.5, -2 / L))
		SRy_p = calc.crop(SRy_p, window, scaling=L, shift=(-1.5, -2 / L))

		C_SRx = calc.corr(SRx, SRx_p)
		C_SRy = calc.corr(SRy, SRy_p)
		C_SRx_list.append(C_SRx)
		C_SRy_list.append(C_SRy)
		print("\nC(SRx, SRx_p) = {:.2f}".format(C_SRx))
		print("C(SRy, SRy_p) = {:.2f}".format(C_SRy))

	print("\nMean C(SRx, SRx_p) = {:.2f}".format(np.mean(C_SRx_list)))
	print("Mean C(SRy, SRy_p) = {:.2f}".format(np.mean(C_SRy_list)))

if __name__ == '__main__':
	main()