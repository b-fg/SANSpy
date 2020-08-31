import os
import pandas as pd
import sanspy.losses as losses
import sanspy.k_models as k_models
import sanspy.k_generators as k_generator


# Constants
alias = {'mse': losses.mse,
		 'mae': losses.mae,
		 'sse': losses.sse,
		 'cc':  losses.cc,
		 'ccp': losses.ccp,
		 'sae': losses.sae,
		 'M3i2o_1': (k_models.M3i2o_1, k_generator.G, False),  # [model, generator, stacked_in]
		 'M2i2o_1': (k_models.M2i2o_1, k_generator.G, False),
		 'M6i2o_1': (k_models.M6i2o_1, k_generator.G3, False),
		 'M3i2o_2': (k_models.M3i2o_2, k_generator.G, False),
		 'M3i3o_2': (k_models.M3i3o_2, k_generator.G, False),
		 'M4i2o_2': (k_models.M4i2o_2, k_generator.Gww, False),
		 'M4i3o_2': (k_models.M4i3o_2, k_generator.Gww, False),
		 'M3i2o_2_stack': (k_models.M3i2o_2_stack, k_generator.G, True),
		 'M6i2o_2': (k_models.M6i2o_2, k_generator.G3, False),
         }


# Functions
def get_df(data_path, dataset, truncate=None, skip=None):
	"""
	Generate a Pandas Dataframe for the generators
	Args:
		data_path: Root path of data. Should be organised as: data_path/dataset/x/x1, x/x2, ..., y/y1, y/y1, ...
		dataset: `fit`, `validate`, `evaluate`, `predict`
		truncate: Samples to keep. If `None`, all data is included.
		skip: List of strings with inputs/outputs to not take into account. Eg: ['x/x1','x/x3'].
			If `None`, all inputs are considered.
	Returns: Pandas Dataframe with columns such as: 'x/x1', 'x/x2', 'y/y1', 'y/y2', ...
	"""
	dic = {}
	if skip is None: skip = []

	if dataset == 'fit':
		path = data_path + 'fit/'
	elif dataset == 'validate':
		path = data_path + 'validate/'
	elif dataset == 'evaluate':
		path = data_path + 'evaluate/'
	elif dataset == 'predict':
		path = data_path + 'predict/'
	else:
		raise ValueError('Incorrect dataset.')

	x_tags = os.listdir(path + 'x/')
	y_tags = os.listdir(path + 'y/')

	for sf in skip: # Remove unwanted fields from dataframe
		x_tags = [i for i in x_tags if not i in sf]
		y_tags = [i for i in y_tags if not i in sf]

	for tag in x_tags:
		files = absolute_file_paths(path + 'x/' + tag)
		time_sorted_files = sorted(files, key=lambda f: float(f.split('/')[-1].split('_')[-1].split('.dat')[0]))
		dic['x/' + tag] = time_sorted_files

	for tag in y_tags:
		files = absolute_file_paths(path + 'y/' + tag)
		time_sorted_files = sorted(files, key=lambda f: float(f.split('/')[-1].split('_')[-1].split('.dat')[0]))
		dic['y/' + tag] = time_sorted_files

	df = pd.DataFrame(dic).truncate(after=truncate) # Create dataframe
	df = df.reindex(sorted(df.columns), axis=1) # Sort columns alphabetically
	return df


def absolute_file_paths(directory):
	"""
	Get absolute path
	Args:
		directory: A (string) directory to obtain an absolute path from.
	Returns: Absolute path (string)
	"""
	files = []
	for dir_path, _, file_names in os.walk(directory):
		for f in file_names:
			files.append(os.path.abspath(os.path.join(dir_path, f)))
	return files


def mem_usage(batch_size, model):
	"""
	Get an approximate memory requirement for your Keras model.
	Args:
		batch_size: Number of samples per batch.
		model: Keras model
	Returns: Memory requirement
	"""
	import numpy as np
	from keras import backend as K

	shapes_mem_count = 0
	for l in model.layers:
		single_layer_mem = 1
		for s in l.output_shape:
			if s is None:
				continue
			single_layer_mem *= s
		shapes_mem_count += single_layer_mem

	trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
	non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

	number_size = 4.0
	if K.floatx() == 'float16':
		number_size = 2.0
	if K.floatx() == 'float64':
		number_size = 8.0

	total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
	gbytes = np.round(total_memory / (1024.0 ** 3), 3)
	return gbytes