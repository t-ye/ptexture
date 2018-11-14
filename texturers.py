from __future__ import annotations # delayed type hint evaluation

def noise_generator(**kwargs) :
	import numpy as np

def partial_ext(f, *args1, **kwargs1) :
	def	ff(*args2, **kwargs2) :
		return partial(f, *args1, *args2, **kwargs1, **kwargs2)
	return ff


import typing
import color

class ptexture() :

	def __init__(self, texturefun :
	                   	typing.Callable[...,(np.ndarray,color.colorformat)],
										 reqd_kwargs :
										 	set = set(),
										 default_kwargs :
										 	dict = dict(),
										 base :
											ptexture = None) :
		self.texturefun = texturefun
		self.reqd_kwargs = reqd_kwargs
		self.default_kwargs = default_kwargs
		self.base = base

		if self.base is not None :
			self.reqd_kwargs.update(self.base.reqd_kwargs)
			self.default_kwargs = dict(base.default_kwargs, **self.default_kwargs)

	def __call__(self, **kwargs) :
		if self.base is not None :
			base_arr, base_fmt = self.base(**kwargs)
			if 'fmt' not in kwargs :
				kwargs['fmt'] = fmt
			kwargs['base'] = base_arr

		if not self.reqd_kwargs.issubset(kwargs) :
			raise ValueError('Not enough kwargs: ' + \
				str(self.reqd_kwargs - kwargs.keys()) + ' missing')

		# defaults updated with override
		kwargs = dict(self.default_kwargs, **kwargs)

		return self.texturefun(**kwargs)




def noisefun(**kwargs) :

	import numpy as np
	from PyQt5.QtGui import QImage
	import color

	R,C = kwargs['R'], kwargs['C']
	fmt = kwargs.get('fmt')
	if fmt is None :
		fmt = color.gray8

	arr = np.random.randint(256, \
		size=(kwargs['R'], kwargs['C'], len(fmt.channels)), dtype=np.uint8)

	return (arr, fmt)

noise = ptexture(noisefun, {'R', 'C'}, {'fmt' : color.gray8})


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

wood = ptexture(wood_generator, {'period', 'power', 'size'}, base=noise)

def zoomed_smooth_noise(**kwargs) :

	import numpy as np

	zoom = kwargs['zoom']
	base = kwargs['base']
	zoom = kwargs['zoom']
	fmt = kwargs['fmt']
	# m assumed 2D

	from time import time

	t = time()

	# get ranges corresonding to the top left (1/zoom)th
	# portion of the matrix
	(xf, yf), i = np.modf(np.indices(base.shape[:2]) / zoom)
	x, y = i.astype(np.int)

	# up, left (negative indices allowed!)
	u = (x-1)
	l = (y-1)

	depth = 1 if base.ndim == 2 else base.shape[2]

	t = time()
	v  = (   xf  *    yf ).repeat(depth).reshape(base.shape) * base[x,y] \
	   + ((1-xf) *    yf ).repeat(depth).reshape(base.shape) * base[u,y] \
	   + (   xf  * (1-yf)).repeat(depth).reshape(base.shape) * base[x,l] \
	   + ((1-xf) * (1-yf)).repeat(depth).reshape(base.shape) * base[u,l]

	return (v, fmt)

#def zoomed_smooth_noise(**kwargs) :
#
#	import numpy as np
#
#	base = kwargs['base']
#	R,C = kwargs['R'], kwargs['C']
#	zoom = kwargs['zoom']
#
#	# base assumed 2D
#
#	#idxs = np.arange(R, dtype=np.float).repeat(C).reshape(R,C)
#	idxs = np.indices(base.shape[:2]) # (2, R, C, 1)
#
#	# get ranges corresonding to the top left (1/zoom)th
#	# portion of the matrix
#	f, i = np.modf(idxs / zoom) # each (2, R, C, 1)
#	x, y = i.astype(np.int)
#	xf, yf = f
#
#
#	# up, left (negative indices allowed!)
#	u = (x-1)
#	l = (y-1)
#
#	v  =    xf  *    yf  * base[x,y]
#	v += (1-xf) *    yf  * base[u,y]
#	v +=    xf  * (1-yf) * base[x,l]
#	v += (1-xf) * (1-yf) * base[u,l]
#
#	return v

zsn = ptexture(zoomed_smooth_noise, set(), {'zoom':8}, base=noise)

def turbulence(base, size) :

	import numpy as np

	v = np.zeros_like(base,dtype=np.float)
	isize = size

	while size >= 1 :
		v += zoomed_smooth_noise(base, size) * size
		size /= 2

	return v / (2*isize)

turb = ptexture(turbulence, {'size'}, {'size':64}, noise)

#noise = ptexture(texturefun(noise, ))
#wood = ptexture(wood_generator, noise)

