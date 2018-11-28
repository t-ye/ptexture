from __future__ import annotations # delayed type hint evaluation

import typing
import color
import dataclasses



# all ptextures
ptextures = dict()







class ptexture() :

	def __new__(cls, name, texturefun = None, kwargs = None) :
		if name in ptextures :
			return ptextures[name]
		else :
			return super(ptexture, cls).__new__(cls)

	def __init__(self, name :
	                    str,
		                 texturefun :
	                   	typing.Callable[...,(np.ndarray,color.colorformat)] = None,
		                 params :
										 	typing.List[ptexture_param] = None) :
		if name in ptextures :
			return
			#raise ValueError('Ptexture already exists: ' + self.name)
		self.name = name
		ptextures[self.name] = self
		self.texturefun = texturefun
		#self.params, self.types, self.defaults = zip(*kwargs)
		self.params = list(map(lambda params : typed_param(*params), params))
		self.params_dict = {param.name : param for param in self.params}


	def partial(self, **kwargs) :
		import copy
		partialled = copy.deepcopy(self)
		for param,value in kwargs :
			if param in partialled.params :
				try :
					i = partialled.params.index(param)
					partialled.params.pop(i)
					types.pop(i)
					params.pop(i)

				except ValueError :
					pass


		texture.params.update(kwargs)
		return texture

	def __call__(self, **kwargs) :

		kwargs = kwargs[self.name]
		for k,v in kwargs.items() :
			if isinstance(v, dict) :
				inner_name, inner_kwargs = v.popitem()
				inner_kwargs = {inner_name:inner_kwargs}
				kwargs[k] = ptexture(inner_name)(**inner_kwargs)

		return self.texturefun(**kwargs)

	def __str__(self) :
		return f'ptexture {self.name}'

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


def get_noise() :
	import main
	screen_width, screen_height = 1920, 1080
	return ptexture('noise', noisefun,
	[('C', int, screen_width),
	 ('R', int, screen_height),
	 ('fmt', color.colorformat, 'gray8')])

#colornoise = ptexture('colornoise', noisefun, {'R', 'C'}, {'fmt' : color.rgb888})


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

#wood = ptexture('wood', wood_generator, {'period', 'power', 'size'}, base=noise)

def zoomed_smooth(**kwargs) :

	import numpy as np

	zoom = kwargs['zoom']
	base, fmt = kwargs['base']

	from time import time

	t = time()

	# get ranges corresonding to the top left (1/zoom)th
	# portion of the matrix
	(xf, yf), i = np.modf(np.indices(base.shape[:2]) / zoom)
	x, y = i.astype(np.int)

	# up, left (negative indices allowed!)
	u = (x-1)
	l = (y-1)

	# If base is 2D, do entrywise multiplication;
	# else, multiply each depth vector in the 2nd matrix with
	# each scalar entry in the 1st matrix
	einsum_str = 'ij,ij->ij' if base.ndim == 2 else 'ij,ijk->ijk'

	v = np.einsum(einsum_str,    xf  *    yf , base[x,y]) \
	  + np.einsum(einsum_str, (1-xf) *    yf , base[u,y]) \
    + np.einsum(einsum_str,    xf  * (1-yf), base[x,l]) \
	  + np.einsum(einsum_str, (1-xf) * (1-yf), base[u,l])

	return (v, fmt)

def get_zs(base) :
	return ptexture('zs', zoomed_smooth,
		[('zoom', int, 8),
		 ('base', ptexture, base.name) # has to go last
		])

def turbulence(base, size) :

	import numpy as np

	v = np.zeros_like(base,dtype=np.float)
	isize = size

	while size >= 1 :
		v += zoomed_smooth(base, size) * size
		size /= 2

	return v / (2*isize)

#turb = ptexture('turbulence', turbulence, {'size'}, {'size':64}, noise)

#noise = ptexture(texturefun(noise, ))
#wood = ptexture(wood_generator, noise)

