from keras import backend as K

def mse(a, b):
	return K.mean(K.square(a - b), axis=-1)


def mae(a, b):
	return K.mean(K.abs(a - b), axis=-1)


def sse(a, b):
	return K.sum(K.square(a - b), axis=-1)


def sae(a, b):
	return K.sum(K.abs(a - b), axis=-1)


def cc(a, b): # correlation factor
	am = K.mean(a)
	bm = K.mean(b)
	ac = a - am  # a centered
	bc = b - bm  # b centered

	cov_ab = K.mean(ac * bc)
	cov_aa = K.mean(ac ** 2)
	cov_bb = K.mean(bc ** 2)

	return 1 - cov_ab / K.sqrt(cov_aa * cov_bb)


def ccp(a, b): # correlation factor penalised to match field intensity as well
	am = K.mean(a)
	bm = K.mean(b)
	c = (am + bm) / 2
	ac = a - c  # a centered
	bc = b - c  # b centered

	cov_ab = K.mean(ac * bc)
	cov_aa = K.mean(ac ** 2)
	cov_bb = K.mean(bc ** 2)

	return 1 - 2 * cov_ab / (cov_aa + cov_bb)

