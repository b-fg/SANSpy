from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Activation, BatchNormalization, InputSpec
from keras.layers import concatenate, Lambda
from keras.models import Model as K_model
from keras import optimizers, regularizers
import sanspy.losses as losses
import sanspy.metrics as metrics
from keras import backend as K
from scipy.ndimage import gaussian_filter
import numpy as np
import tensorflow as tf

sh = (1216, 540, 1) # Input and output fields shape
D = 90 # Grid scaling factor

def M3i2o_1(merge=concatenate, max_filters=64, activation='relu', kernel_size=3,
			optimizer=optimizers.adam, loss=losses.sse, lr=0.0001, lr_decay=None, l2=None,
            pp=0, sigma=1, **kwargs):
	"""
	CNN with 3 inputs and 2 outputs. Model type 1 (autoencoder before merge only).
	Args:
		merge: Branch merge type (Keras function).
		max_filters: Number of maximum filters on the thickest layer.
		activation: Keras activation function.
		kernel_size: Convolutional kernel size.
		optimizer: Keras optimizer.
		loss: Type of loss. See sanspy.losses.
		lr: Learning rate.
		lr_decay: Learning rate decay.
		l2: Regularization factor.
		pp: Output layer processing (0, 1, 2, 3). See sanspy.model.Model for more details.
		sigma: When pp=2 or pp=3, select the sigma value of the Gaussian function to filter high freqs in output.
		**kwargs:
	Returns: Keras model
	"""
	if l2 is None or l2 == 0:
		reg = None
	else:
		reg = regularizers.l2(l2)

	# Column 1
	i1 = Input(shape=sh, name='x1')
	x1 = Conv2D(filters=int(int(max_filters/4)), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)
	x1 = MaxPooling2D(pool_size=2)(x1)

	x1 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)
	x1 = MaxPooling2D(pool_size=2)(x1)

	x1 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = UpSampling2D(size=2)(x1)
	x1 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = UpSampling2D(size=2)(x1)
	x1 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	# Column 2
	i2 = Input(shape=sh, name='x2')
	x2 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)
	x2 = MaxPooling2D(pool_size=2)(x2)

	x2 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)
	x2 = MaxPooling2D(pool_size=2)(x2)

	x2 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = UpSampling2D(size=2)(x2)
	x2 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = UpSampling2D(size=2)(x2)
	x2 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	# Column 3
	i3 = Input(shape=sh, name='x3')
	x3 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)
	x3 = MaxPooling2D(pool_size=2)(x3)

	x3 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)
	x3 = MaxPooling2D(pool_size=2)(x3)

	x3 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	x3 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	x3 = UpSampling2D(size=2)(x3)
	x3 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	x3 = UpSampling2D(size=2)(x3)
	x3 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	# Merge
	x = merge([x1, x2, x3])
	x = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, activation=activation, padding='same', kernel_initializer='he_normal')(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	# Outputs
	if pp == 1:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y1 = Lambda(clipFunc, name='y1')([y1, i2, i3])
		y2 = Lambda(clipFunc, name='y2')([y2, i2, i3])
	elif pp == 2:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y1 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y1')(y1)
		y2 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y2')(y2)
	elif pp == 3:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y1 = Lambda(clipFunc)([y1, i2, i3])
		y2 = Lambda(clipFunc)([y2, i2, i3])
		y1 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y1')(y1)
		y2 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y2')(y2)
	else:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear', name='y1')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear', name='y2')(x)

	# Compile model
	ins = [i1, i2, i3]
	outs = [y1, y2]
	model = K_model(inputs=ins, outputs=outs)
	opt = optimizer(lr=lr, decay=lr_decay)
	model.compile(optimizer=opt, loss=[loss] * len(outs), metrics=[metrics.cc])
	return model


def M2i2o_1(merge=concatenate, max_filters=64, activation='relu', kernel_size=3,
			optimizer=optimizers.adam, loss=losses.sse, lr=0.0001, lr_decay=None, l2=None,
            pp=0, sigma=1, **kwargs):
	"""
	CNN with 2 inputs and 2 outputs. Model type 1 (autoencoder before merge only).

	Args:
		merge: Branch merge type (Keras function).
		max_filters: Number of maximum filters on the thickest layer.
		activation: Keras activation function.
		kernel_size: Convolutional kernel size.
		optimizer: Keras optimizer.
		loss: Type of loss. See sanspy.losses.
		lr: Learning rate.
		lr_decay: Learning rate decay.
		l2: Regularization factor.
		pp: Output layer processing (0, 2). See sanspy.model.Model for more details.
		sigma: When pp=2, select the sigma value of the Gaussian function to filter high freqs in output.
		**kwargs:
	Returns: Keras model
	"""
	if l2 is None or l2 == 0:
		reg = None
	else:
		reg = regularizers.l2(l2)

	# Column 1
	i1 = Input(shape=sh, name='x1')
	x1 = Conv2D(filters=int(int(max_filters/4)), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)
	x1 = MaxPooling2D(pool_size=2)(x1)

	x1 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)
	x1 = MaxPooling2D(pool_size=2)(x1)

	x1 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = UpSampling2D(size=2)(x1)
	x1 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = UpSampling2D(size=2)(x1)
	x1 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	# Column 2
	i2 = Input(shape=sh, name='x2')
	x2 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)
	x2 = MaxPooling2D(pool_size=2)(x2)

	x2 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)
	x2 = MaxPooling2D(pool_size=2)(x2)

	x2 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = UpSampling2D(size=2)(x2)
	x2 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = UpSampling2D(size=2)(x2)
	x2 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	# Merge
	x = merge([x1, x2])
	x = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, activation=activation, padding='same', kernel_initializer='he_normal')(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	# Outputs
	if pp == 2:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y1 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y1')(y1)
		y2 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y2')(y2)
	else:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear', name='y1')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear', name='y2')(x)

	# Compile model
	ins = [i1, i2]
	outs = [y1, y2]
	model = K_model(inputs=ins, outputs=outs)
	opt = optimizer(lr=lr, decay=lr_decay)
	model.compile(optimizer=opt, loss=[loss] * len(outs), metrics=[metrics.cc])
	return model


def M6i2o_1(merge=concatenate, max_filters=64, activation='relu', kernel_size=3,
			optimizer=optimizers.adam, loss=losses.sse, lr=0.0001, lr_decay=None, l2=None,
            pp=0, sigma=1, **kwargs):
	"""
	CNN with 6 inputs and 2 outputs. Model type 1 (autoencoder before merge only).
	Args:
		merge: Branch merge type (Keras function).
		max_filters: Number of maximum filters on the thickest layer.
		activation: Keras activation function.
		kernel_size: Convolutional kernel size.
		optimizer: Keras optimizer.
		loss: Type of loss. See sanspy.losses.
		lr: Learning rate.
		lr_decay: Learning rate decay.
		l2: Regularization factor.
		pp: Output layer processing (0, 2). See sanspy.model.Model for more details.
		sigma: When pp=2, select the sigma value of the Gaussian function to filter high freqs in output.
		**kwargs:
	Returns: Keras model
	"""
	if l2 is None or l2 == 0:
		reg = None
	else:
		reg = regularizers.l2(l2)

	# Column 1
	i1 = Input(shape=sh, name='x1')
	x1 = Conv2D(filters=int(int(max_filters/4)), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)
	x1 = MaxPooling2D(pool_size=2)(x1)

	x1 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)
	x1 = MaxPooling2D(pool_size=2)(x1)

	x1 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = UpSampling2D(size=2)(x1)
	x1 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = UpSampling2D(size=2)(x1)
	x1 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	# Column 2
	i2 = Input(shape=sh, name='x2')
	x2 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)
	x2 = MaxPooling2D(pool_size=2)(x2)

	x2 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)
	x2 = MaxPooling2D(pool_size=2)(x2)

	x2 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = UpSampling2D(size=2)(x2)
	x2 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = UpSampling2D(size=2)(x2)
	x2 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	# Column 3
	i3 = Input(shape=sh, name='x3')
	x3 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)
	x3 = MaxPooling2D(pool_size=2)(x3)

	x3 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)
	x3 = MaxPooling2D(pool_size=2)(x3)

	x3 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	x3 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	x3 = UpSampling2D(size=2)(x3)
	x3 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	x3 = UpSampling2D(size=2)(x3)
	x3 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	# Column 4
	i4 = Input(shape=sh, name='x4')
	x4 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i4)
	x4 = BatchNormalization()(x4)
	x4 = Activation(activation)(x4)
	x4 = MaxPooling2D(pool_size=2)(x4)

	x4 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x4)
	x4 = BatchNormalization()(x4)
	x4 = Activation(activation)(x4)
	x4 = MaxPooling2D(pool_size=2)(x4)

	x4 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x4)
	x4 = BatchNormalization()(x4)
	x4 = Activation(activation)(x4)

	x4 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x4)
	x4 = BatchNormalization()(x4)
	x4 = Activation(activation)(x4)

	x4 = UpSampling2D(size=2)(x4)
	x4 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x4)
	x4 = BatchNormalization()(x4)
	x4 = Activation(activation)(x4)

	x4 = UpSampling2D(size=2)(x4)
	x4 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x4)
	x4 = BatchNormalization()(x4)
	x4 = Activation(activation)(x4)

	# Column 5
	i5 = Input(shape=sh, name='x5')
	x5 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i5)
	x5 = BatchNormalization()(x5)
	x5 = Activation(activation)(x5)
	x5 = MaxPooling2D(pool_size=2)(x5)

	x5 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x5)
	x5 = BatchNormalization()(x5)
	x5 = Activation(activation)(x5)
	x5 = MaxPooling2D(pool_size=2)(x5)

	x5 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x5)
	x5 = BatchNormalization()(x5)
	x5 = Activation(activation)(x5)

	x5 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x5)
	x5 = BatchNormalization()(x5)
	x5 = Activation(activation)(x5)

	x5 = UpSampling2D(size=2)(x5)
	x5 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x5)
	x5 = BatchNormalization()(x5)
	x5 = Activation(activation)(x5)

	x5 = UpSampling2D(size=2)(x5)
	x5 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x5)
	x5 = BatchNormalization()(x5)
	x5 = Activation(activation)(x5)

	# Column 6
	i6 = Input(shape=sh, name='x6')
	x6 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i6)
	x6 = BatchNormalization()(x6)
	x6 = Activation(activation)(x6)
	x6 = MaxPooling2D(pool_size=2)(x6)

	x6 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x6)
	x6 = BatchNormalization()(x6)
	x6 = Activation(activation)(x6)
	x6 = MaxPooling2D(pool_size=2)(x6)

	x6 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x6)
	x6 = BatchNormalization()(x6)
	x6 = Activation(activation)(x6)

	x6 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x6)
	x6 = BatchNormalization()(x6)
	x6 = Activation(activation)(x6)

	x6 = UpSampling2D(size=2)(x6)
	x6 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x6)
	x6 = BatchNormalization()(x6)
	x6 = Activation(activation)(x6)

	x6 = UpSampling2D(size=2)(x6)
	x6 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x6)
	x6 = BatchNormalization()(x6)
	x6 = Activation(activation)(x6)

	# Merge
	x = merge([x1, x2, x3, x4, x5, x6])
	x = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, activation=activation, padding='same', kernel_initializer='he_normal')(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	# Outputs
	if pp == 2:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y1 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y1')(y1)
		y2 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y2')(y2)
	else:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear', name='y1')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear', name='y2')(x)

	# Compile model
	ins = [i1, i2, i3, i4, i5, i6]
	outs = [y1, y2]
	model = K_model(inputs=ins, outputs=outs)
	opt = optimizer(lr=lr, decay=lr_decay)
	model.compile(optimizer=opt, loss=[loss] * len(outs), metrics=[metrics.cc])
	return model


def M3i3o_2(merge=concatenate, max_filters=64, activation='relu', kernel_size=3,
			optimizer=optimizers.adam, loss=losses.sse, lr=0.0001, lr_decay=None, l2=None,
            pp=3, sigma=0.5, **kwargs):
	"""
	CNN with 3 inputs and 3 outputs. Model type 2 (autoencoder before and after merge).
	Args:
		merge: Branch merge type (Keras function).
		max_filters: Number of maximum filters on the thickest layer.
		activation: Keras activation function.
		kernel_size: Convolutional kernel size.
		optimizer: Keras optimizer.
		loss: Type of loss. See sanspy.losses.
		lr: Learning rate.
		lr_decay: Learning rate decay.
		l2: Regularization factor.
		pp: Output layer processing (0, 1, 2, 3). See sanspy.model.Model for more details.
		sigma: When pp=2 or pp=3, select the sigma value of the Gaussian function to filter high freqs in output.
		**kwargs:
	Returns: Keras model
	"""
	if l2 is None or l2 == 0:
		reg = None
	else:
		reg = regularizers.l2(l2)

	# Column 1
	i1 = Input(shape=sh, name='x1')
	x1 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)
	x1 = MaxPooling2D(pool_size=2)(x1)

	x1 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)
	x1 = MaxPooling2D(pool_size=2)(x1)

	x1 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = UpSampling2D(size=2)(x1)
	x1 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = UpSampling2D(size=2)(x1)
	x1 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	# Column 2
	i2 = Input(shape=sh, name='x2')
	x2 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)
	x2 = MaxPooling2D(pool_size=2)(x2)

	x2 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)
	x2 = MaxPooling2D(pool_size=2)(x2)

	x2 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = UpSampling2D(size=2)(x2)
	x2 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = UpSampling2D(size=2)(x2)
	x2 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	# Column 3
	i3 = Input(shape=sh, name='x3')
	x3 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)
	x3 = MaxPooling2D(pool_size=2)(x3)

	x3 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)
	x3 = MaxPooling2D(pool_size=2)(x3)

	x3 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	x3 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	x3 = UpSampling2D(size=2)(x3)
	x3 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	x3 = UpSampling2D(size=2)(x3)
	x3 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	# Merge
	x = merge([x1, x2, x3])
	x = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)
	x = MaxPooling2D(pool_size=2)(x)

	x = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)
	x = MaxPooling2D(pool_size=2)(x)

	x = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	x = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	x = UpSampling2D(size=2)(x)
	x = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	x = UpSampling2D(size=2)(x)
	x = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	# Outputs
	if pp == 1:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y3 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y1 = Lambda(clipFunc_and_positive, name='y1')([y1, i2, i3])
		y2 = Lambda(clipFunc, name='y2')([y2, i2, i3])
		y3 = Lambda(clipFunc_and_positive, name='y3')([y3, i2, i3])
	elif pp == 2:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y3 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y1 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y1')(y1)
		y2 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y2')(y2)
		y3 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y3')(y3)
	elif pp == 3:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y3 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y1 = Lambda(clipFunc_and_positive)([y1, i2, i3])
		y2 = Lambda(clipFunc)([y2, i2, i3])
		y3 = Lambda(clipFunc_and_positive)([y3, i2, i3])
		y1 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y1')(y1)
		y2 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y2')(y2)
		y3 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y3')(y3)
	else:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear', name='y1')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear', name='y2')(x)
		y3 = Conv2D(filters=1, kernel_size=1, activation='linear', name='y3')(x)

	# Compile model
	ins = [i1, i2, i3]
	outs = [y1, y2, y3]
	model = K_model(inputs=ins, outputs=outs)
	opt = optimizer(lr=lr, decay=lr_decay)
	model.compile(optimizer=opt, loss=[loss] * len(outs), metrics=[metrics.cc])
	return model


def M4i3o_2(merge=concatenate, max_filters=64, activation='relu', kernel_size=3,
			optimizer=optimizers.adam, loss=losses.sse, lr=0.0001, lr_decay=None, l2=None,
            pp=3, sigma=0.5, **kwargs):
	"""
	CNN with 4 inputs and 3 outputs. The 4th input is used as the wake mask computed in k_generators.Gww.
	Model type 2 (autoencoder before and after merge).
	Args:
		merge: Branch merge type (Keras function).
		max_filters: Number of maximum filters on the thickest layer.
		activation: Keras activation function.
		kernel_size: Convolutional kernel size.
		optimizer: Keras optimizer.
		loss: Type of loss. See sanspy.losses.
		lr: Learning rate.
		lr_decay: Learning rate decay.
		l2: Regularization factor.
		pp: Output layer processing (0, 1, 2, 3). See sanspy.model.Model for more details.
		sigma: When pp=2 or pp=3, select the sigma value of the Gaussian function to filter high freqs in output.
		**kwargs:
	Returns: Keras model
	"""
	if l2 is None or l2 == 0:
		reg = None
	else:
		reg = regularizers.l2(l2)

	# Column 1
	i1 = Input(shape=sh, name='x1')
	x1 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)
	x1 = MaxPooling2D(pool_size=2)(x1)

	x1 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)
	x1 = MaxPooling2D(pool_size=2)(x1)

	x1 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = UpSampling2D(size=2)(x1)
	x1 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = UpSampling2D(size=2)(x1)
	x1 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	# Column 2
	i2 = Input(shape=sh, name='x2')
	x2 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)
	x2 = MaxPooling2D(pool_size=2)(x2)

	x2 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)
	x2 = MaxPooling2D(pool_size=2)(x2)

	x2 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = UpSampling2D(size=2)(x2)
	x2 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = UpSampling2D(size=2)(x2)
	x2 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	# Column 3
	i3 = Input(shape=sh, name='x3')
	x3 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)
	x3 = MaxPooling2D(pool_size=2)(x3)

	x3 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)
	x3 = MaxPooling2D(pool_size=2)(x3)

	x3 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	x3 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	x3 = UpSampling2D(size=2)(x3)
	x3 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	x3 = UpSampling2D(size=2)(x3)
	x3 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	# Merge
	x = merge([x1, x2, x3])
	x = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)
	x = MaxPooling2D(pool_size=2)(x)

	x = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)
	x = MaxPooling2D(pool_size=2)(x)

	x = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	x = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	x = UpSampling2D(size=2)(x)
	x = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	x = UpSampling2D(size=2)(x)
	x = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	# Outputs
	i4 = Input(shape=sh, name='x4')
	if pp == 1:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y3 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y1 = Lambda(clip_wake_and_positive, name='y1')([y1, i4])
		y2 = Lambda(clip_wake, name='y2')([y2, i4])
		y3 = Lambda(clip_wake_and_positive, name='y3')([y3, i4])
	elif pp == 2:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y3 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y1 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y1')(y1)
		y2 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y2')(y2)
		y3 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y3')(y3)
	elif pp == 3:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y3 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y1 = Lambda(clip_wake_and_positive)([y1, i4])
		y2 = Lambda(clip_wake)([y2, i4])
		y3 = Lambda(clip_wake_and_positive)([y3, i4])
		y1 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y1')(y1)
		y2 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y2')(y2)
		y3 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y3')(y3)
	else:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear', name='y1')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear', name='y2')(x)
		y3 = Conv2D(filters=1, kernel_size=1, activation='linear', name='y3')(x)

	# Compile model
	ins = [i1, i2, i3, i4]
	outs = [y1, y2, y3]
	model = K_model(inputs=ins, outputs=outs)
	opt = optimizer(lr=lr, decay=lr_decay)
	model.compile(optimizer=opt, loss=[loss] * len(outs), metrics=[metrics.cc])
	print('M4i3o')
	return model


def M3i2o_2(merge=concatenate, max_filters=64, activation='relu', kernel_size=3,
			optimizer=optimizers.adam, loss=losses.sse, lr=0.0001, lr_decay=None, l2=None,
            pp=3, sigma=0.5, **kwargs):
	"""
	CNN with 3 inputs and 2 outputs. Model type 2 (autoencoder before and after merge).
	Args:
		merge: Branch merge type (Keras function).
		max_filters: Number of maximum filters on the thickest layer.
		activation: Keras activation function.
		kernel_size: Convolutional kernel size.
		optimizer: Keras optimizer.
		loss: Type of loss. See sanspy.losses.
		lr: Learning rate.
		lr_decay: Learning rate decay.
		l2: Regularization factor.
		pp: Output layer processing (0, 1, 2, 3). See sanspy.model.Model for more details.
		sigma: When pp=2 or pp=3, select the sigma value of the Gaussian function to filter high freqs in output.
		**kwargs:
	Returns: Keras model
	"""
	if l2 is None or l2 == 0:
		reg = None
	else:
		reg = regularizers.l2(l2)

	# Column 1
	i1 = Input(shape=sh, name='x1')
	x1 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)
	x1 = MaxPooling2D(pool_size=2)(x1)

	x1 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)
	x1 = MaxPooling2D(pool_size=2)(x1)

	x1 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = UpSampling2D(size=2)(x1)
	x1 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = UpSampling2D(size=2)(x1)
	x1 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	# Column 2
	i2 = Input(shape=sh, name='x2')
	x2 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)
	x2 = MaxPooling2D(pool_size=2)(x2)

	x2 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)
	x2 = MaxPooling2D(pool_size=2)(x2)

	x2 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = UpSampling2D(size=2)(x2)
	x2 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = UpSampling2D(size=2)(x2)
	x2 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	# Column 3
	i3 = Input(shape=sh, name='x3')
	x3 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)
	x3 = MaxPooling2D(pool_size=2)(x3)

	x3 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)
	x3 = MaxPooling2D(pool_size=2)(x3)

	x3 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	x3 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	x3 = UpSampling2D(size=2)(x3)
	x3 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	x3 = UpSampling2D(size=2)(x3)
	x3 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	# Merge
	x = merge([x1, x2, x3])
	x = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)
	x = MaxPooling2D(pool_size=2)(x)

	x = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)
	x = MaxPooling2D(pool_size=2)(x)

	x = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	x = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	x = UpSampling2D(size=2)(x)
	x = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	x = UpSampling2D(size=2)(x)
	x = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	# Outputs
	if pp == 1:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y1 = Lambda(clipFunc, name='y1')([y1, i2, i3])
		y2 = Lambda(clipFunc, name='y2')([y2, i2, i3])
	elif pp == 2:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y1 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y1')(y1)
		y2 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y2')(y2)
	elif pp == 3:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y1 = Lambda(clipFunc)([y1, i2, i3])
		y2 = Lambda(clipFunc)([y2, i2, i3])
		y1 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y1')(y1)
		y2 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y2')(y2)
	else:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear', name='y1')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear', name='y2')(x)

	# Compile model
	ins = [i1, i2, i3]
	outs = [y1, y2]
	model = K_model(inputs=ins, outputs=outs)
	opt = optimizer(lr=lr, decay=lr_decay)
	model.compile(optimizer=opt, loss=[loss] * len(outs), metrics=[metrics.cc])
	print('M3i2o')
	return model


def M4i2o_2(merge=concatenate, max_filters=64, activation='relu', kernel_size=3,
			optimizer=optimizers.adam, loss=losses.sse, lr=0.0001, lr_decay=None, l2=None,
            pp=3, sigma=0.5, **kwargs):
	"""
	CNN with 4 inputs and 2 outputs. The 4th input is used as the wake mask computed in k_generators.Gww.
	Model type 2 (autoencoder before and after merge).
	Args:
		merge: Branch merge type (Keras function).
		max_filters: Number of maximum filters on the thickest layer.
		activation: Keras activation function.
		kernel_size: Convolutional kernel size.
		optimizer: Keras optimizer.
		loss: Type of loss. See sanspy.losses.
		lr: Learning rate.
		lr_decay: Learning rate decay.
		l2: Regularization factor.
		pp: Output layer processing (0, 1, 2, 3). See sanspy.model.Model for more details.
		sigma: When pp=2 or pp=3, select the sigma value of the Gaussian function to filter high freqs in output.
		**kwargs:
	Returns: Keras model
	"""
	if l2 is None or l2 == 0:
		reg = None
	else:
		reg = regularizers.l2(l2)

	# Column 1
	i1 = Input(shape=sh, name='x1')
	x1 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)
	x1 = MaxPooling2D(pool_size=2)(x1)

	x1 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)
	x1 = MaxPooling2D(pool_size=2)(x1)

	x1 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = UpSampling2D(size=2)(x1)
	x1 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = UpSampling2D(size=2)(x1)
	x1 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	# Column 2
	i2 = Input(shape=sh, name='x2')
	x2 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)
	x2 = MaxPooling2D(pool_size=2)(x2)

	x2 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)
	x2 = MaxPooling2D(pool_size=2)(x2)

	x2 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = UpSampling2D(size=2)(x2)
	x2 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = UpSampling2D(size=2)(x2)
	x2 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	# Column 3
	i3 = Input(shape=sh, name='x3')
	x3 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)
	x3 = MaxPooling2D(pool_size=2)(x3)

	x3 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)
	x3 = MaxPooling2D(pool_size=2)(x3)

	x3 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	x3 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	x3 = UpSampling2D(size=2)(x3)
	x3 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	x3 = UpSampling2D(size=2)(x3)
	x3 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	# Merge
	x = merge([x1, x2, x3])
	x = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)
	x = MaxPooling2D(pool_size=2)(x)

	x = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)
	x = MaxPooling2D(pool_size=2)(x)

	x = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	x = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	x = UpSampling2D(size=2)(x)
	x = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	x = UpSampling2D(size=2)(x)
	x = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	# Outputs
	i4 = Input(shape=sh, name='x4')
	if pp == 1:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y1 = Lambda(clip_wake, name='y1')([y1, i4])
		y2 = Lambda(clip_wake, name='y2')([y2, i4])
	elif pp == 2:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y1 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y1')(y1)
		y2 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y2')(y2)
	elif pp == 3:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y1 = Lambda(clip_wake)([y1, i4])
		y2 = Lambda(clip_wake)([y2, i4])
		y1 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y1')(y1)
		y2 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y2')(y2)
	else:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear', name='y1')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear', name='y2')(x)

	# Compile model
	ins = [i1, i2, i3, i4]
	outs = [y1, y2]
	model = K_model(inputs=ins, outputs=outs)
	opt = optimizer(lr=lr, decay=lr_decay)
	model.compile(optimizer=opt, loss=[loss] * len(outs), metrics=[metrics.cc])
	print('M4i2o')
	return model


def M3i2o_2_stack(merge=None, max_filters=64, activation='relu', kernel_size=3,
				  optimizer=optimizers.adam, loss=losses.sse, lr=0.0001, lr_decay=None, l2=None,
                  pp=0, sigma=1, **kwargs):
	"""
	CNN with 3 inputs stacked in a single column and 2 outputs. Model type 2 (autoencoder before and after merge).
	Args:
		merge: Branch merge type (Keras function).
		max_filters: Number of maximum filters on the thickest layer.
		activation: Keras activation function.
		kernel_size: Convolutional kernel size.
		optimizer: Keras optimizer.
		loss: Type of loss. See sanspy.losses.
		lr: Learning rate.
		lr_decay: Learning rate decay.
		l2: Regularization factor.
		pp: Output layer processing (0, 2). See sanspy.model.Model for more details.
		sigma: When pp=2, select the sigma value of the Gaussian function to filter high freqs in output.
		**kwargs:
	Returns: Keras model
	"""
	if l2 is None or l2 == 0:
		reg = None
	else:
		reg = regularizers.l2(l2)

	# Encode
	i = Input(shape=(1216, 540, 3), name='x1')
	x = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)
	x = MaxPooling2D(pool_size=2)(x)

	x = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)
	x = MaxPooling2D(pool_size=2)(x)

	x = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	# Decode
	x = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	x = UpSampling2D(size=2)(x)
	x = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	x = UpSampling2D(size=2)(x)
	x = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	# Outputs
	if pp == 2:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y1 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y1')(y1)
		y2 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y2')(y2)
	else:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear', name='y1')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear', name='y2')(x)

	outs = [y1, y2]
	model = K_model(inputs=i, outputs=outs)
	opt = optimizer(lr=lr, decay=lr_decay)

	model.compile(optimizer=opt, loss=[loss] * len(outs), metrics=[metrics.cc])
	return model


def M6i2o_2(merge=concatenate, max_filters=64, activation='relu', kernel_size=3,
			optimizer=optimizers.adam, loss=losses.sse, lr=0.0001, lr_decay=None, l2=None,
            pp=0, sigma=1, **kwargs):
	"""
	CNN with 6 inputs and 2 outputs. Model type 2 (autoencoder before and after merge).
	Args:
		merge: Branch merge type (Keras function).
		max_filters: Number of maximum filters on the thickest layer.
		activation: Keras activation function.
		kernel_size: Convolutional kernel size.
		optimizer: Keras optimizer.
		loss: Type of loss. See sanspy.losses.
		lr: Learning rate.
		lr_decay: Learning rate decay.
		l2: Regularization factor.
		pp: Output layer processing (0, 2). See sanspy.model.Model for more details.
		sigma: When pp=2, select the sigma value of the Gaussian function to filter high freqs in output.
		**kwargs:
	Returns: Keras model
	"""
	if l2 is None or l2 == 0:
		reg = None
	else:
		reg = regularizers.l2(l2)

	# Column 1
	i1 = Input(shape=sh, name='x1')
	x1 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)
	x1 = MaxPooling2D(pool_size=2)(x1)

	x1 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)
	x1 = MaxPooling2D(pool_size=2)(x1)

	x1 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = UpSampling2D(size=2)(x1)
	x1 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = UpSampling2D(size=2)(x1)
	x1 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	# Column 2
	i2 = Input(shape=sh, name='x2')
	x2 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)
	x2 = MaxPooling2D(pool_size=2)(x2)

	x2 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)
	x2 = MaxPooling2D(pool_size=2)(x2)

	x2 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = UpSampling2D(size=2)(x2)
	x2 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = UpSampling2D(size=2)(x2)
	x2 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	# Column 3
	i3 = Input(shape=sh, name='x3')
	x3 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)
	x3 = MaxPooling2D(pool_size=2)(x3)

	x3 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)
	x3 = MaxPooling2D(pool_size=2)(x3)

	x3 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	x3 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	x3 = UpSampling2D(size=2)(x3)
	x3 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	x3 = UpSampling2D(size=2)(x3)
	x3 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	# Column 4
	i4 = Input(shape=sh, name='x4')
	x4 = Conv2D(filters=int(max_filters / 4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
				kernel_regularizer=reg)(i4)
	x4 = BatchNormalization()(x4)
	x4 = Activation(activation)(x4)
	x4 = MaxPooling2D(pool_size=2)(x4)

	x4 = Conv2D(filters=int(max_filters / 2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
				kernel_regularizer=reg)(x4)
	x4 = BatchNormalization()(x4)
	x4 = Activation(activation)(x4)
	x4 = MaxPooling2D(pool_size=2)(x4)

	x4 = Conv2D(filters=int(max_filters / 4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
				kernel_regularizer=reg)(x4)
	x4 = BatchNormalization()(x4)
	x4 = Activation(activation)(x4)

	x4 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
				kernel_regularizer=reg)(x4)
	x4 = BatchNormalization()(x4)
	x4 = Activation(activation)(x4)

	x4 = UpSampling2D(size=2)(x4)
	x4 = Conv2D(filters=int(max_filters / 2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
				kernel_regularizer=reg)(x4)
	x4 = BatchNormalization()(x4)
	x4 = Activation(activation)(x4)

	x4 = UpSampling2D(size=2)(x4)
	x4 = Conv2D(filters=int(max_filters / 4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
				kernel_regularizer=reg)(x4)
	x4 = BatchNormalization()(x4)
	x4 = Activation(activation)(x4)

	# Column 5
	i5 = Input(shape=sh, name='x5')
	x5 = Conv2D(filters=int(max_filters / 4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
				kernel_regularizer=reg)(i5)
	x5 = BatchNormalization()(x5)
	x5 = Activation(activation)(x5)
	x5 = MaxPooling2D(pool_size=2)(x5)

	x5 = Conv2D(filters=int(max_filters / 2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
				kernel_regularizer=reg)(x5)
	x5 = BatchNormalization()(x5)
	x5 = Activation(activation)(x5)
	x5 = MaxPooling2D(pool_size=2)(x5)

	x5 = Conv2D(filters=int(max_filters / 4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
				kernel_regularizer=reg)(x5)
	x5 = BatchNormalization()(x5)
	x5 = Activation(activation)(x5)

	x5 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
				kernel_regularizer=reg)(x5)
	x5 = BatchNormalization()(x5)
	x5 = Activation(activation)(x5)

	x5 = UpSampling2D(size=2)(x5)
	x5 = Conv2D(filters=int(max_filters / 2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
				kernel_regularizer=reg)(x5)
	x5 = BatchNormalization()(x5)
	x5 = Activation(activation)(x5)

	x5 = UpSampling2D(size=2)(x5)
	x5 = Conv2D(filters=int(max_filters / 4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
				kernel_regularizer=reg)(x5)
	x5 = BatchNormalization()(x5)
	x5 = Activation(activation)(x5)

	# Column 6
	i6 = Input(shape=sh, name='x6')
	x6 = Conv2D(filters=int(max_filters / 4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
				kernel_regularizer=reg)(i6)
	x6 = BatchNormalization()(x6)
	x6 = Activation(activation)(x6)
	x6 = MaxPooling2D(pool_size=2)(x6)

	x6 = Conv2D(filters=int(max_filters / 2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
				kernel_regularizer=reg)(x6)
	x6 = BatchNormalization()(x6)
	x6 = Activation(activation)(x6)
	x6 = MaxPooling2D(pool_size=2)(x6)

	x6 = Conv2D(filters=int(max_filters / 4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
				kernel_regularizer=reg)(x6)
	x6 = BatchNormalization()(x6)
	x6 = Activation(activation)(x6)

	x6 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
				kernel_regularizer=reg)(x6)
	x6 = BatchNormalization()(x6)
	x6 = Activation(activation)(x6)

	x6 = UpSampling2D(size=2)(x6)
	x6 = Conv2D(filters=int(max_filters / 2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
				kernel_regularizer=reg)(x6)
	x6 = BatchNormalization()(x6)
	x6 = Activation(activation)(x6)

	x6 = UpSampling2D(size=2)(x6)
	x6 = Conv2D(filters=int(max_filters / 4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
				kernel_regularizer=reg)(x6)
	x6 = BatchNormalization()(x6)
	x6 = Activation(activation)(x6)

	# Merge
	x = merge([x1, x2, x3, x4, x5, x6])
	x = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)
	x = MaxPooling2D(pool_size=2)(x)

	x = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)
	x = MaxPooling2D(pool_size=2)(x)

	x = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	x = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	x = UpSampling2D(size=2)(x)
	x = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	x = UpSampling2D(size=2)(x)
	x = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x = BatchNormalization()(x)
	x = Activation(activation)(x)

	# Outputs
	if pp == 2:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear')(x)
		y1 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y1')(y1)
		y2 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y2')(y2)
	else:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear', name='y1')(x)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear', name='y2')(x)

	# Compile model
	ins = [i1, i2, i3, i4, i5, i6]
	outs = [y1, y2]
	model = K_model(inputs=ins, outputs=outs)
	opt = optimizer(lr=lr, decay=lr_decay)
	model.compile(optimizer=opt, loss=[loss] * len(outs), metrics=[metrics.cc])
	return model


def M3i2o_3(merge=concatenate, max_filters=64, activation='relu', kernel_size=3,
			optimizer=optimizers.adam, loss=losses.sse, lr=0.0001, lr_decay=None, l2=None,
            pp=0, sigma=1, **kwargs):
	"""
	CNN with 3 inputs and 2 outputs. Model type 3 (encode, merge, decode).
	Args:
		merge: Branch merge type (Keras function).
		max_filters: Number of maximum filters on the thickest layer.
		activation: Keras activation function.
		kernel_size: Convolutional kernel size.
		optimizer: Keras optimizer.
		loss: Type of loss. See sanspy.losses.
		lr: Learning rate.
		lr_decay: Learning rate decay.
		l2: Regularization factor.
		pp: Output layer processing (0, 1, 2, 3). See sanspy.model.Model for more details.
		sigma: When pp=2 or pp=3, select the sigma value of the Gaussian function to filter high freqs in output.
		**kwargs:
	Returns: Keras model
	"""
	if l2 is None or l2 == 0:
		reg = None
	else:
		reg = regularizers.l2(l2)

	# ENCODE
	# column 1
	i1 = Input(shape=sh, name='x1')
	x1 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)
	x1 = MaxPooling2D(pool_size=2)(x1)

	x1 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)
	x1 = MaxPooling2D(pool_size=2)(x1)

	x1 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	# column 2
	i2 = Input(shape=sh, name='x2')
	x2 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)
	x2 = MaxPooling2D(pool_size=2)(x2)

	x2 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)
	x2 = MaxPooling2D(pool_size=2)(x2)

	x2 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	# column 3
	i3 = Input(shape=sh, name='x3')
	x3 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)
	x3 = MaxPooling2D(pool_size=2)(x3)

	x3 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)
	x3 = MaxPooling2D(pool_size=2)(x3)

	x3 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x3)
	x3 = BatchNormalization()(x3)
	x3 = Activation(activation)(x3)

	# MERGE
	x = merge([x1, x2, x3])

	# DECODE
	# column 1
	x1 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = UpSampling2D(size=2)(x1)
	x1 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = UpSampling2D(size=2)(x1)
	x1 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	# column 2
	x2 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = UpSampling2D(size=2)(x2)
	x2 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = UpSampling2D(size=2)(x2)
	x2 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	# Outputs
	if pp == 1:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear')(x1)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear')(x2)
		y1 = Lambda(clipFunc, name='y1')([y1, i2, i3])
		y2 = Lambda(clipFunc, name='y2')([y2, i2, i3])
	elif pp == 2:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear')(x1)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear')(x2)
		y1 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y1')(y1)
		y2 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y2')(y2)
	elif pp == 3:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear')(x1)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear')(x2)
		y1 = Lambda(clipFunc)([y1, i2, i3])
		y2 = Lambda(clipFunc)([y2, i2, i3])
		y1 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y1')(y1)
		y2 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y2')(y2)
	else:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear', name='y1')(x1)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear', name='y2')(x2)

	# Compile model
	ins = [i1, i2, i3]
	outs = [y1, y2]
	model = K_model(inputs=ins, outputs=outs)
	opt = optimizer(lr=lr, decay=lr_decay)
	model.compile(optimizer=opt, loss=[loss] * len(outs), metrics=[metrics.cc])
	return model


def M2i2o_3(merge=concatenate, max_filters=64, activation='relu', kernel_size=3,
			optimizer=optimizers.adam, loss=losses.sse, lr=0.0001, lr_decay=None, l2=None,
			pp=0, sigma=1, **kwargs):
	"""
	CNN with 2 inputs and 2 outputs. Model type 3 (encode, merge, decode).
	Args:
		merge: Branch merge type (Keras function).
		max_filters: Number of maximum filters on the thickest layer.
		activation: Keras activation function.
		kernel_size: Convolutional kernel size.
		optimizer: Keras optimizer.
		loss: Type of loss. See sanspy.losses.
		lr: Learning rate.
		lr_decay: Learning rate decay.
		l2: Regularization factor.
		pp: Output layer processing (0, 2). See sanspy.model.Model for more details.
		sigma: When pp=2, select the sigma value of the Gaussian function to filter high freqs in output.
		**kwargs:
	Returns: Keras model
	"""
	if l2 is None or l2 == 0:
		reg = None
	else:
		reg = regularizers.l2(l2)

	# ENCODE
	# column 1
	i1 = Input(shape=sh, name='x1')
	x1 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)
	x1 = MaxPooling2D(pool_size=2)(x1)

	x1 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)
	x1 = MaxPooling2D(pool_size=2)(x1)

	x1 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	# column 2
	i2 = Input(shape=sh, name='x2')
	x2 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(i2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)
	x2 = MaxPooling2D(pool_size=2)(x2)

	x2 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)
	x2 = MaxPooling2D(pool_size=2)(x2)

	x2 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	# MERGE
	x = merge([x1, x2])

	# DECODE
	# column 1
	x1 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = UpSampling2D(size=2)(x1)
	x1 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	x1 = UpSampling2D(size=2)(x1)
	x1 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x1)
	x1 = BatchNormalization()(x1)
	x1 = Activation(activation)(x1)

	# column 2
	x2 = Conv2D(filters=max_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = UpSampling2D(size=2)(x2)
	x2 = Conv2D(filters=int(max_filters/2), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	x2 = UpSampling2D(size=2)(x2)
	x2 = Conv2D(filters=int(max_filters/4), kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=reg)(x2)
	x2 = BatchNormalization()(x2)
	x2 = Activation(activation)(x2)

	# Outputs
	if pp == 2:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear')(x1)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear')(x2)
		y1 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y1')(y1)
		y2 = GaussianConv2D(filters=1, sigma=sigma, padding='same', name='y2')(y2)
	else:
		y1 = Conv2D(filters=1, kernel_size=1, activation='linear', name='y1')(x1)
		y2 = Conv2D(filters=1, kernel_size=1, activation='linear', name='y2')(x2)

	# Compile model
	ins = [i1, i2]
	outs = [y1, y2]
	model = K_model(inputs=ins, outputs=outs)
	opt = optimizer(lr=lr, decay=lr_decay)
	model.compile(optimizer=opt, loss=[loss] * len(outs), metrics=[metrics.cc])
	return model


# Auxiliar functions for last processing layer.

def clipFunc(x):
	"""
	Function to mask the output according to `x` input (field of 0s and 1s arising from eg. some value of vorticity).
	Instead of using k_generators.Gww, the mask field is created here from vorticity).
	Args:
		x: x[0] Unmasked output
		x: x[1] Streamwise velocity (U)
		x: x[2] Crossflow velocity (V)
	Returns: Masked output
	"""
	y = x[0]
	U = x[1]
	V = x[2]

	dUdy = U[:, :, 1:, :] - U[:, :, :-1, :]
	dUdy = K.concatenate((dUdy, K.expand_dims(dUdy[:, :, -1, ...], 2)), axis=2)
	dVdx = V[:, 1:, :, :] - V[:, :-1, :, :]
	dVdx = K.concatenate((dVdx, K.expand_dims(dVdx[:, -1, ...], 1)), axis=1)
	vort = dVdx - dUdy

	xi = np.arange(sh[0], dtype='float32')
	yi = np.arange(sh[1], dtype='float32')
	zi = np.arange(1, dtype='float32')
	_, Xi, _ = tf.meshgrid(yi, xi, zi)
	Xi = K.expand_dims(Xi, axis=0)
	mask_boolean_tensor1 = K.greater_equal(Xi, 1.5 * D)
	mask_tensor1 = K.cast(mask_boolean_tensor1, dtype=K.floatx())
	mask_boolean_tensor2 = K.greater_equal(K.abs(vort), 0.0035)
	mask_tensor2 = K.cast(mask_boolean_tensor2, dtype=K.floatx())
	mask_tensor = mask_tensor1 * mask_tensor2
	return y * mask_tensor


def clipFunc_and_positive(x):
	"""
	Function to mask the output according to `x` input (field of 0s and 1s arising from eg. some value of vorticity).
	Instead of using k_generators.Gww, the mask field is created here from vorticity).
	It only takes positive value regions as well (useful for normal stresses)
	Args:
		x: x[0] Unmasked output
		x: x[1] Streamwise velocity (U)
		x: x[2] Crossflow velocity (V)
	Returns: Masked output
	"""
	y = x[0]
	U = x[1]
	V = x[2]

	dUdy = U[:, :, 1:, :] - U[:, :, :-1, :]
	dUdy = K.concatenate((dUdy, K.expand_dims(dUdy[:, :, -1, ...], 2)), axis=2)
	dVdx = V[:, 1:, :, :] - V[:, :-1, :, :]
	dVdx = K.concatenate((dVdx, K.expand_dims(dVdx[:, -1, ...], 1)), axis=1)
	vort = dVdx - dUdy

	xi = np.arange(sh[0], dtype='float32')
	yi = np.arange(sh[1], dtype='float32')
	zi = np.arange(1, dtype='float32')
	_, Xi, _ = tf.meshgrid(yi, xi, zi)
	Xi = K.expand_dims(Xi, axis=0)
	mask_boolean_tensor1 = K.greater_equal(Xi, 1.5 * D)
	mask_tensor1 = K.cast(mask_boolean_tensor1, dtype=K.floatx())
	mask_boolean_tensor2 = K.greater_equal(K.abs(vort), 0.0035)
	mask_tensor2 = K.cast(mask_boolean_tensor2, dtype=K.floatx())
	mask_boolean_tensor3 = K.greater_equal(y, 0)
	mask_tensor3 = K.cast(mask_boolean_tensor3, dtype=K.floatx())
	return y * mask_tensor1 * mask_tensor2 * mask_tensor3


def clip_wake(x):
	"""
	Function to mask the output according to `x` input (field of 0s and 1s arising from eg. some value of vorticity).
	Args:
		x: x[0] Unmasked output
		x: x[1] Mask field (can be generated with k_generators.Gww).
	Returns: Masked output
	"""
	x_unmasked = x[0]
	wake_mask = x[1]
	return x_unmasked * wake_mask


def clip_wake_and_positive(x):
	"""
	Function to mask the output according to `x` input (field of 0s and 1s arising from eg. some value of vorticity).
	It only takes positive value regions as well (useful for normal stresses)
	Args:
		x: x[0] Unmasked output field
		x: x[1] Mask field (can be generated with k_generators.Gww).
	Returns: Masked output
	"""
	x_unmasked = x[0]
	wake_mask = x[1]
	positive_mask_bool = K.greater_equal(x_unmasked, 0)
	positive_mask = K.cast(positive_mask_bool, dtype=K.floatx())
	return x_unmasked * wake_mask * positive_mask

class GaussianConv2D(Conv2D):
	"""
	Custom convolutional layer with Gaussian function to filter high freqs.
	"""
	def __init__(self, filters, sigma=0.5, **kwargs):
		def k_size(sigma):
			delta = np.zeros((99, 99))
			delta[48, 48] = 1
			k = gaussian_filter(delta, sigma=sigma)
			n = int(np.sqrt(k[k != 0].shape[0]))
			return n, k[k != 0].reshape(n, n)

		n, k_sliced = k_size(sigma)
		self.sigma = sigma
		self.kernel_size = n
		self.k_sliced = k_sliced
		super(GaussianConv2D, self).__init__(filters, self.kernel_size, **kwargs)

	def build(self, input_shape):
		if self.data_format == 'channels_first':
			channel_axis = 1
		else:
			channel_axis = -1
		if input_shape[channel_axis] is None:
			raise ValueError('The channel dimension of the inputs '
							 'should be defined. Found `None`.')
		input_dim = input_shape[channel_axis]


		self.kernel = K.constant(value=self.k_sliced[..., np.newaxis, np.newaxis])
		self.input_spec = InputSpec(ndim=self.rank + 2,	axes={channel_axis: input_dim})
		self.bias = None
		self.trainable = False
		self.built = True

	def call(self, inputs):
		return K.conv2d(inputs, kernel=self.kernel, padding=self.padding)