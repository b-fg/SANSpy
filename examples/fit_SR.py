"""
Training script using type 2 model with 4 inputs and 2 outputs: M4i2o_2
"""
import tensorflow as tf
from sanspy.model import Model
from sanspy.history import History


# Constants
params = {'data_path': '/dataset/', # Absolute path of dataset
	  'weights_save': 'weights/weights_SR_1', # Relative weights path
	  'history_path': 'history_SR.json', # Relative history path
	  'M_type': 'M4i2o_2',
	  'n_in': 4,
	  'n_out': 2,
	  'skip': ['uu', 'uv', 'vv'],
      'n_epochs': 100,
	  'patience': 5,
	  'batch_size': 4,
	  'queue_size': 1,
	  'loss': 'sse',
	  'max_filters': 64,
	  'activation': 'relu',
	  'kernel_size': 5,
	  'norm_in': True,
	  'pp': 3,
	  'ww': 'length',
	  'sigma': 0.5,
	  'l2': 0,}


# Main
def main():
	sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	model = Model(**params)
	history = model.fit()
	h = History(history_dict=history.history)
	h.print_info()


if __name__ == '__main__':
	main()