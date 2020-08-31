from keras import backend as K

def cc(a, b): # correlation factor
	am = K.mean(a)
	bm = K.mean(b)
	ac = a - am  # a centered
	bc = b - bm  # b centered

	cov_ab = K.mean(ac * bc)
	cov_aa = K.mean(ac ** 2)
	cov_bb = K.mean(bc ** 2)

	return cov_ab / K.sqrt(cov_aa * cov_bb)


def ccp(a, b): # correlation factor penalised mean
	am = K.mean(a)
	bm = K.mean(b)
	c = (am + bm) / 2
	ac = a - c  # a centered
	bc = b - c  # b centered

	cov_ab = K.mean(ac * bc)
	cov_aa = K.mean(ac ** 2)
	cov_bb = K.mean(bc ** 2)

	return 2 * cov_ab / (cov_aa + cov_bb)

