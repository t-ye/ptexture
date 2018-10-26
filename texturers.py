from ptexture import ptexture

def noise_generator(**kwargs) :
	import numpy as np

	return np.random.randint(256, size=(kwargs['R'], kwargs['C']), dtype=np.uint8)

def wood_generator(base, **kwargs) :
	import numpy as np

	period, power, size = kwargs['period'], kwargs['power'], kwargs['size']
	R,C = base.shape
	x,y = np.indices((R, C))

	# center at origin
	x = (x - R / 2) / R
	y = (y - C / 2) / C
	d = np.sqrt(x**2 + y**2) + power * turbulence(base, size) / 256
	return 128 * np.abs(np.sin(2 * period * d * np.pi))


def zoomed_smooth_noise(base, zoom) :

	import numpy as np

	# base assumed 2D
	R,C = base.shape

	#idxs = np.arange(R, dtype=np.float).repeat(C).reshape(R,C)
	idxs = np.indices(base.shape)

	# get ranges corresonding to the top left (1/zoom)th
	# portion of the matrix
	f, i = np.modf(idxs / zoom)
	x, y = i.astype(np.int)
	xf, yf = f


	# up, left (negative indices allowed!)
	u = (x-1)
	l = (y-1)

	v  =    xf  *    yf  * base[x,y]
	v += (1-xf) *    yf  * base[u,y]
	v +=    xf  * (1-yf) * base[x,l]
	v += (1-xf) * (1-yf) * base[u,l]

	return v

def turbulence(base, size) :

	import numpy as np

	v = np.zeros_like(base,dtype=np.float)
	isize = size

	while size >= 1 :
		v += zoomed_smooth_noise(base, size) * size
		size /= 2

	return v / (2*isize)

noise = ptexture(noise_generator)
wood = ptexture(wood_generator, noise)
