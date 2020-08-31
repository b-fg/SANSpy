from keras.layers import concatenate
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import json
from sanspy.utils import get_df, alias, mem_usage
from sanspy.callbacks import SaveHistoryEpochEnd

class Model(object):
	"""
	Model class as a wrapper for Keras. Contains all the information required for building the model, training and testing.
	"""
	def __init__(self, data_path=None, weights_load=None, weights_save='weights.h5', history_path='history.json',
	             score_path='score.json',
				 M_type='M1', lr=0.0001, patience=5, l2=None, skip=None, norm_in=False,
				 optimizer=optimizers.adam, loss='mse', merge=concatenate,
				 max_filters=64, activation='relu', kernel_size=3, n_in=4, n_out=3, pp=3, ww=None, sigma=0.5,
				 n_epochs=100, batch_size=6, queue_size=2, truncate=None, verbose=True):
		"""
		Model initializer
		Args:
			data_path: Root path of the dataset.
			weights_load: Weights filename required to restart a model from some training state.
			weights_save: Weigths filename to save a training state.
			history_path: Training history output file.
			score_path: Test score output file.
			M_type: Model type. See sanspy.utils.alias. Eg: 'M3i2o_1'.
			lr: Learning rate.
			patience: Number of epochs of patience before early stop is triggered during training.
			l2: Regularization value.
			skip: Input fields to skip in the data_path. Eg: ['x/P', 'y/ww'].
			norm_in: Boolean for normalizing input fields.
			optimizer: Keras optimizer.
			loss: Keras loss whch is selected from sanspy.losses. Eg: 'mse'.
			merge: Keras merging operator.
			max_filters: Number of maximum filters on the thickest layer.
			activation: Keras activation function.
			kernel_size: Convolutional kernel size.
			n_in: Number of inputs.
			n_out: Number of outputs.
			pp: Last layer processing operation:
				0: No processing (linear output)
				1: Linear + Wake mask
				2: Linear + Gaussian filter
				3: Linear + Wake mask + Gaussian filter
			ww: Type of wake detection layer. 'length' for wake width and 'eps' for threshhold value.
			sigma: Gaussian filter layer sigma.
			n_epochs: Max number of epochs for training.
			batch_size: Size of the training batch.
			queue_size: Generetor queue, ie how many samples are loaded in the queue as the batches are being dynamically generated.
			truncate: Number of samples to use. If `None`, all samples in the dataset folders are used.
			verbose: Boolean to print Keras model information and memory requirements.
		"""
		self.data_path = data_path
		self.weights_load = weights_load
		self.weights_save = weights_save
		self.history_path = history_path
		self.score_path = score_path
		self.lr = lr
		self.lr_decay = lr*0.1
		self.patience = patience
		self.optimizer = optimizer
		self.loss = alias.get(loss, None)
		self.merge = merge
		self.max_filters = max_filters
		self.activation = activation
		self.kernel_size = kernel_size
		self.n_in = n_in
		self.n_out = n_out
		self.pp = pp
		self.ww = ww
		self.sigma = sigma
		self.n_epochs = n_epochs
		self.batch_size = batch_size
		self.queue_size = queue_size
		self.truncate = truncate
		self.model = None
		self.history = None
		self.eval_score = None
		self.M_type = M_type
		self.l2 = l2
		self.skip = skip
		self.norm_in = norm_in

		model_type = alias.get(M_type, None)[0]
		self.stacked_in = alias.get(M_type, None)[2]
		if self.stacked_in: self.merge = None
		model_params = {'merge': self.merge,
		                'max_filters': self.max_filters,
		                'activation': self.activation,
		                'kernel_size': self.kernel_size,
						'optimizer': self.optimizer,
						'loss': self.loss,
						'lr': self.lr,
						'lr_decay': self.lr_decay,
						'l2': self.l2,
		                'pp': self.pp,
		                'ww': self.ww,
		                'sigma': self.sigma,}

		self.model = model_type(**model_params)
		if verbose:
			print(self.model.summary())
			print('Mem usage = {:.2} GB'.format(mem_usage(self.batch_size, self.model)))
			print(M_type)
		return


	def fit(self):
		"""
		Fit model.
		Returns: History object containing the training metrics at each epoch.
		"""
		df_fit = get_df(self.data_path, 'fit', truncate=self.truncate, skip=self.skip)
		df_val = get_df(self.data_path, 'validate', truncate=self.truncate, skip=self.skip)
		print(list(df_fit))

		# Generators
		gen = alias.get(self.M_type, None)[1]
		gen_fit = gen(df_fit, n_in=self.n_in, n_out=self.n_out, stacked_in=self.stacked_in,
		              batch_size=self.batch_size, shuffle=True, norm_in=self.norm_in,
		              ww=self.ww)
		gen_val = gen(df_val, n_in=self.n_in, n_out=self.n_out, stacked_in=self.stacked_in,
		              batch_size=self.batch_size, shuffle=False, norm_in=self.norm_in,
		              ww=self.ww)

		# Callbacks
		es = EarlyStopping(monitor='val_loss', patience=self.patience, verbose=1)
		mc = ModelCheckpoint(self.weights_save+'_{epoch:03d}.h5', save_weights_only=True, verbose=0) # Save weights after every epoch
		she = SaveHistoryEpochEnd(history_path=self.history_path) # Save history after every epoch

		steps_per_epoch_train = np.ceil(df_fit.shape[0] / self.batch_size)
		steps_per_epoch_val = np.ceil(df_val.shape[0] / self.batch_size)
		if self.weights_load is not None: self.model.load_weights(self.weights_load)

		self.history = self.model.fit_generator(generator=gen_fit,
												steps_per_epoch=steps_per_epoch_train,
												epochs=self.n_epochs,
												validation_data=gen_val,
												validation_steps=steps_per_epoch_val,
												max_queue_size=self.queue_size,
												callbacks=[es, mc, she],
												verbose=1)

		return self.history


	def evaluate(self, dataset='evaluate', shuffle=False, seed=None):
		"""
		Test model.
		Args:
			dataset: Dataset within the data_path: 'evaluate' or 'predict'.
			shuffle: Boolean to shuffle the samples order.
			seed: Seed for shuffle.
		Returns: A dictionary containing the test score.
		"""
		df_eval = get_df(self.data_path, dataset, truncate=self.truncate, skip=self.skip)
		print(list(df_eval))

		gen = alias.get(self.M_type, None)[1]
		gen_eval = gen(df_eval, n_in=self.n_in, n_out=self.n_out, stacked_in=self.stacked_in,
		               batch_size=self.batch_size, shuffle=shuffle, seed=seed, norm_in=self.norm_in,
		               ww=self.ww)

		if self.weights_load is not None:
			self.model.load_weights(self.weights_load)
		else:
			raise ValueError('Incorrect path of weights to load.')

		steps_eval = np.ceil(df_eval.shape[0] / self.batch_size)

		score = self.model.evaluate_generator(generator=gen_eval,
											  steps=steps_eval,
											  max_queue_size=self.queue_size,
											  verbose=1)
		self.eval_score = {}
		print('')
		for m, s in zip(self.model.metrics_names, score):
			self.eval_score[m] = s
			if 'loss' == m:
				print('{}: {:.4e}'.format(m,s))
			elif 'cc' in m:
				print('{}: {:.4f}'.format(m,s))

		if self.score_path is not None:
			with open(self.score_path, 'w') as f:
				json.dump(self.eval_score, f)

		return self.eval_score


	def predict(self, dataset='predict',  shuffle=False, seed=None):
		"""
		Similar to `evaluate` but for the 'predict' dataset which returns the predictions (output fields).
		Args:
			dataset: Dataset within the data_path: 'evaluate' or 'predict'.
			shuffle: Boolean to shuffle the samples order.
			seed: Seed for shuffle.
		Returns: (X, Yt, Yp), where X is the input fields, Yt the target output, and Yp the predicted output.
			These are dictionaries containing keys of input or output fields: 'x1', 'y1', ...
			For each key, there is an array with each dimension being: [snapshot_index, snapshot (2D)].
		"""
		df = get_df(self.data_path, dataset=dataset, truncate=self.truncate)

		gen = alias.get(self.M_type, None)[1]
		gen_predict = gen(df, n_in=self.n_in, n_out=self.n_out, stacked_in=self.stacked_in,
		                  batch_size=self.batch_size, shuffle=shuffle, seed=seed, norm_in=self.norm_in,
		                  ww=self.ww)

		if self.weights_load is not None:
			self.model.load_weights(self.weights_load)
		else:
			raise ValueError('Incorrect path of weights to load.')

		steps_predict = np.ceil(df.shape[0] / self.batch_size)

		X, Yp, Yt = {}, {}, {} # Key== x1, x2, y1, y2... val= all predictions (batched flattened and appended in list)
		for out in self.model.inputs:
			X[out.name.split(':')[0]] = []
		for out in self.model.outputs:
			Yp[out.name.split('/')[0]] = []
		for out in self.model.outputs:
			Yt[out.name.split('/')[0]] = []

		for i, x_y in enumerate(gen_predict):
			if i == steps_predict:
				break
			print('{:.2f}%'.format(i / steps_predict * 100))

			x, y_t = x_y[0], x_y[1]  # Dictionary of input batch and outputs batch from the generator
			p = self.model.predict(x, batch_size=self.batch_size)
			y_p = {out.name.split('/')[0]: p[i] for i, out in enumerate(self.model.outputs)}  # Dictionary of predicted outputs
			for m in range(self.batch_size):
				for k, v in x.items():
					X[k].append(x[k][m, ..., 0])
				for k, v in y_t.items():
					Yt[k].append(y_t[k][m, ..., 0])
				for k, v in y_p.items():
					Yp[k].append(y_p[k][m, ..., 0])
		return X, Yt, Yp


	def predict_from_x(self, x_dict):
		"""
		Similar to `predict` but using an input dictionary instead of a directory path.
		Args:
			x_dict: Input dictionary as created by a generator. Eg:
				x_dict = {'x1': P[np.newaxis, ..., np.newaxis],
				          'x2': U[np.newaxis, ..., np.newaxis],
				          'x3': V[np.newaxis, ..., np.newaxis]}
		Returns: Predicted output (Yp). See `predict` for output structure.
		"""
		if self.weights_load is not None:
			self.model.load_weights(self.weights_load)
		else:
			raise ValueError('Incorrect path of weights to load.')

		Yp = {}
		p = self.model.predict(x_dict, batch_size=1)
		y_p = {out.name.split('/')[0]: p[i] for i, out in enumerate(self.model.outputs)}
		for k, v in y_p.items():
			Yp[k] = y_p[k][0, ..., 0]

		return Yp