from postproc import io, calc
import numpy as np

sh = (1216, 540, 1) # Input and output fields size
scaling = 90 # Grid scaling length
shift = (-1.5, -2 / 90) # Shift coordinates to center body at (0,0)
radius = 0.5 # Radius of the cylinder

def G(df, n_in, n_out, stacked_in=False, batch_size=4, shuffle=False, seed=None,
      norm_in=True, **kwargs):
	"""
	Generator for traininig, validation, test
	Args:
		df: Pandas Dataframe containing input (x/...) and output (y/...) file names
		n_in: Number of inputs
		n_out: Number of outputs
		stacked_in: Boolean to stack inputs or not
		batch_size: Number of examples per optimization step
		shuffle: Boolean for dataframe shuffle (eg. for training)
		seed: Shuffle seed integer
		norm_in: Boolean to normalize input fields with their own standard score
	Returns: Keras Generator

	"""
	n_batches = np.ceil(df.shape[0] / batch_size)

	if shuffle:
		df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

	it = 0  # Iterations counter
	while True:
		idx_start = batch_size * it
		idx_end = batch_size * (it + 1)
		if idx_end > len(df):
			idx_end = len(df)
			batch_size = idx_end - idx_start

		d = {}
		d['x'] = np.empty((batch_size,) + (sh[0], sh[1], n_in), dtype=np.single)
		d['y'] = np.empty((batch_size,) + (sh[0], sh[1], n_out), dtype=np.single)

		for i, sample_files in enumerate(df.iloc[idx_start:idx_end].values):
			t = '.'.join(sample_files[0].split('_')[-1].split('.')[:2])
			all_t = ['.'.join(zz.split('_')[-1].split('.')[:2]) for zz in sample_files]
			assert all([tt == t for tt in all_t]), 'Not all files belong to the same snapshot.'

			for j, fname in enumerate(sample_files):
				f_str = '/'.join(fname.split('/')[-3:-1]) # eg 'x/U' from '.../x/U_1193.0.dat'
				p = io.read_data(fname, shape=sh[:2], ncomponents=1)
				if f_str.startswith('x'):
					d['x'][i][..., j] = p
				else:
					d['y'][i][..., j-(n_in)] = p

			# Normalize input data
			if norm_in:
				for j in range(n_in):
					p = d['x'][i][..., j]
					d['x'][i][..., j] = (p - np.mean(p)) / np.std(p)
		it += 1

		x_dic, y_dic = {}, {}
		if stacked_in:
			x_dic['x1'] = d['x']
		else:
			for j in range(n_in):
				x_dic['x' + str(j+1)] = d['x'][..., j, np.newaxis]
		for j in range(n_out):
			y_dic['y' + str(j+1)] = d['y'][..., j, np.newaxis]

		yield x_dic, y_dic

		if it == n_batches:
			df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
			it = 0

def Gww(df, n_in, n_out, stacked_in=False, batch_size=4, shuffle=False, seed=None,
        norm_in=True, ww=None, **kwargs):
	"""
	Generator for traininig, validation, test including a wake mask (ww)
	Args:
		df: Pandas Dataframe containing input (x/...) and output (y/...) file names
		n_in: Number of inputs
		n_out: Number of outputs
		stacked_in: Boolean to stack inputs or not
		batch_size: Number of examples per optimization step
		shuffle: Boolean for dataframe shuffle (eg. for training)
		seed: Shuffle seed integer
		norm_in: Boolean to normalize input fields with their own standard score
		ww: Type of wake detection layer. 'length' for wake width and 'eps' for threshhold value.
	Returns: Keras Generator

	"""
	n_batches = np.ceil(df.shape[0] / batch_size)

	if shuffle:
		df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

	it = 0  # Iterations counter
	while True:
		idx_start = batch_size * it
		idx_end = batch_size * (it + 1)
		if idx_end > len(df):
			idx_end = len(df)
			batch_size = idx_end - idx_start

		d = {}
		d['x'] = np.empty((batch_size,) + (sh[0], sh[1], n_in), dtype=np.single)
		d['y'] = np.empty((batch_size,) + (sh[0], sh[1], n_out), dtype=np.single)

		for i, sample_files in enumerate(df.iloc[idx_start:idx_end].values):
			t = '.'.join(sample_files[0].split('_')[-1].split('.')[:2])
			all_t = ['.'.join(zz.split('_')[-1].split('.')[:2]) for zz in sample_files]
			assert all([tt == t for tt in all_t]), 'Not all files belong to the same snapshot.'

			for j, fname in enumerate(sample_files):
				f_str = '/'.join(fname.split('/')[-3:-1]) # eg 'x/U' from '.../x/U_1193.0.dat'
				p = io.read_data(fname, shape=sh[:2], ncomponents=1)
				if f_str.startswith('x'):
					d['x'][i][..., j] = p
				else:
					d['y'][i][..., j-(n_in-1)] = p

			# Wake width detection
			vort = calc.vortZ(d['x'][i][..., 1], d['x'][i][..., 2])
			if ww == 'eps':
				d['x'][i][..., 3] = calc.wake_width_eps(vort, eps=0.001)
			elif ww == 'length':
				d['x'][i][..., 3] = calc.wake_width_length(vort, eps=0.001)
			else:
				raise(ValueError('No ww mode detected.'))

			# Normalize input data
			if norm_in:
				for j in range(n_in-1):
					p = d['x'][i][..., j]
					d['x'][i][..., j] = (p - np.mean(p)) / np.std(p)
		it += 1

		x_dic, y_dic = {}, {}
		if stacked_in:
			x_dic['x1'] = d['x']
		else:
			for j in range(n_in):
				x_dic['x' + str(j+1)] = d['x'][..., j, np.newaxis]
		for j in range(n_out):
			y_dic['y' + str(j+1)] = d['y'][..., j, np.newaxis]

		yield x_dic, y_dic

		if it == n_batches:
			df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
			it = 0