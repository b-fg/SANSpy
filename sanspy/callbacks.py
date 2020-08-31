import keras
import json
import os
from sanspy.history import History

class SaveHistoryEpochEnd(keras.callbacks.Callback):
	"""
	Callback to output history in `history_path` at the end of every training step.
	"""
	def __init__(self, history_path='history.json'):
		super(SaveHistoryEpochEnd, self).__init__()
		self.history_path = history_path

	def on_epoch_end(self, batch, logs=None):
		history = None
		if (os.path.isfile(self.history_path)):
			with open(self.history_path, 'r') as f:
				history = json.loads(f.read())

		if history is not None and history.keys() == logs.keys():
			for k, _ in logs.items():
				history[k].append(logs[k])
			with open(self.history_path, 'w') as f:
				json.dump(history, f)
		elif history is None or history.keys() != logs.keys():
			history = {}
			for k, v in logs.items():
				history[k] = [v]
			with open(self.history_path, 'w') as f:
				json.dump(history, f)

		h = History(history_path=self.history_path)
		h.save_info(n=None)
		return