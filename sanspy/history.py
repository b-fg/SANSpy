import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from postproc.plotter import makeSquare

class History(object):
	def __init__(self, history_dict=None, history_path='history.json'):
		self.history_dict = history_dict
		self.history_path = history_path
		if history_dict is None: self.__load_history()
		self.n_epochs = len(self.history_dict.get('loss', None))
		self.best_epoch = np.argmin(self.history_dict['val_loss'])+1

	def __load_history(self):
		with open(self.history_path, 'r') as f:
			self.history_dict = json.loads(f.read())

	def __str__(self):
		return json.dumps(self.history_dict, indent=4, sort_keys=True)

	def print_info(self, n=20):
		"""
		Prints training history.
		Args:
			n: number of epochs.
		Returns: -
		"""
		if n is None or n >= self.n_epochs: n = self.n_epochs
		print('\nLoss [last ' + str(n) + ' iterations] (fit, val):')
		for i, v in enumerate(zip(self.history_dict['loss'][-n:], self.history_dict['val_loss'][-n:])):
			print('{:2d}: {:.4e}, {:.4e}'.format(self.n_epochs-n+i+1,v[0],v[1]))

		all_corr_keys = [k for k in self.history_dict if k.endswith('cc')]
		fit_corr_keys = [k for k in all_corr_keys if k.startswith('y')]
		val_corr_keys = [k for k in all_corr_keys if k.startswith('val')]

		prnt = '{:2d}:' + ' {:.4f}'*len(all_corr_keys)
		dicts = [self.history_dict[k][-n:] for k in fit_corr_keys] + \
		        [self.history_dict[k][-n:] for k in val_corr_keys]
		print('\nCorrelation coefficients [last ' + str(n) + ' iterations] (fit, val):')
		for i,v in enumerate(zip(*dicts)):
			print(prnt.format(self.n_epochs-n+i+1,*v))
		return

	def save_info(self, n=20):
		"""
		Save training history
		Args:
			n: number of epochs.
		Returns: Text file with training history.
		"""
		if n is None or n >= self.n_epochs: n = self.n_epochs
		with open(self.history_path.split('.')[0]+'.txt', 'w') as file:
			file.write('Loss [last ' + str(n) + ' iterations] (fit, val):')
			file.write('\n')
			for i, v in enumerate(zip(self.history_dict['loss'][-n:], self.history_dict['val_loss'][-n:])):
				file.write('{:2d}: {:.4e}, {:.4e}'.format(self.n_epochs - n + i + 1, v[0], v[1]))
				file.write('\n')
			file.write('\n')
			file.write('Correlation coefficients [last ' + str(n) + ' iterations] (fit, val):')
			file.write('\n')

			all_corr_keys = [k for k in self.history_dict if k.endswith('cc')]
			fit_corr_keys = [k for k in all_corr_keys if k.startswith('y')]
			val_corr_keys = [k for k in all_corr_keys if k.startswith('val')]

			prnt = '{:2d}:' + ' {:.4f}' * len(all_corr_keys)
			dicts = [self.history_dict[k][-n:] for k in fit_corr_keys] + \
			        [self.history_dict[k][-n:] for k in val_corr_keys]
			for i,v in enumerate(zip(*dicts)):
				file.write(prnt.format(self.n_epochs-n+i+1,*v))
				file.write('\n')
		return

	def plot(self, file='history.pdf', max_epochs=None):
		"""
		Plots training history of correlation coefficient for training and validation data.
		Args:
			file: output file.
			max_epochs: number of epochs to plot. If `None`, all are plotted.

		Returns: Outputs pdf plot of training history.
		"""
		plt.rc('text', usetex=True)
		plt.rc('font', size=16)  # use 13 for squared double columns figures
		mpl.rc('xtick', labelsize=16)
		mpl.rc('ytick', labelsize=16)
		mpl.rc('figure', max_open_warning=0)
		mpl.rcParams['axes.linewidth'] = 0.5
		mpl.rcParams['axes.unicode_minus'] = False
		plt.switch_backend('PDF')

		fig, ax1 = plt.subplots()

		if max_epochs is not None and max_epochs > self.n_epochs: max_epochs = self.n_epochs
		elif max_epochs is None: max_epochs = self.n_epochs

		x = range(1,self.n_epochs+1)
		ax1.plot(x, self.history_dict['loss'], color='grey', linewidth=1)
		ax1.plot(x, self.history_dict['val_loss'], color='black', linewidth=1)

		min_y1_1 = np.floor(np.log10(self.history_dict['loss'][max_epochs-1]))
		max_y1_1 = np.ceil(np.log10(self.history_dict['loss'][0]))
		min_y1_2 = np.floor(np.log10(self.history_dict['val_loss'][max_epochs-1]))
		max_y1_2 = np.ceil(np.log10(self.history_dict['val_loss'][0]))

		ax1.set_xlim(min(x), max_epochs)
		ax1.set_ylim(10**(np.min([min_y1_1,min_y1_2])), 10**(np.max([max_y1_1,max_y1_2])))
		ax1.set_yscale('log')
		ax1.set_xlabel('$\mathrm{Epoch}$')
		ax1.set_ylabel('$\mathrm{Loss}$')
		ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
		ax2.set_ylabel('$\mathcal{CC}$')
		ax2.set_ylim(0, 1)
		ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
		# ax1.set_xticks(np.arange(0,max_epochs+1,10)[1:])
		# np.arange(1,)
		# ax1.grid(axis='x', alpha=0.5)

		all_corr_keys = [k for k in self.history_dict if k.endswith('cc')]
		val_corr_keys = [k for k in all_corr_keys if k.startswith('val')]

		colors = ['blue', 'purple', 'cyan'] # Increase if more than 3 outputs.
		for i,k in enumerate(val_corr_keys):
			ax2.plot(x, self.history_dict[k], color=colors[i], linewidth=1)

		if self.best_epoch < max_epochs: ax2.vlines(x=self.best_epoch, ymin=0, ymax=1, linewidth=1, color='r', ls='--')

		ax1.tick_params(bottom=True, top=True, right=False, which='both', direction='in', length=2)
		ax2.tick_params(bottom=False, top=False, right=True, which='both', direction='in', length=2)

		ax2.tick_params(axis='y')

		fig, ax1 = makeSquare(fig, ax1)
		plt.savefig(file, transparent=True, bbox_inches='tight')
		plt.clf()