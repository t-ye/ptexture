from __future__ import annotations # delayed type hint evaluation

import typing
import color
import dataclasses
import functional
import functools
import operator
import numpy as np

@dataclasses.dataclass(frozen=True)
class texture() :
	arr : 'np.ndarray'
	fmt : color.colorformat

	def __iter__(self) :
		return iter((self.arr, self.fmt))


class ptexture() :
	instances = dict()

	def __new__(cls, texfun, *params : typing.Tuple[functional.StringParameter]) :
		if texfun.__name__ in cls.instances :
			raise ValueError('Attempt to create ptexture with same name as ' + \
					             cls.instances[texfun.__name__])
		return super().__new__(cls)

	def __init__(self, texfun, *params : typing.Tuple[functional.StringParameter]) :
		functools.update_wrapper(self, texfun)
		self.texfun = texfun
		self.params = params
		type(self).instances[texfun.__name__] = self

	def __call__(self, **kwargs) -> typing.Tuple['np.ndarray', color.colorformat] :
		for param in self.params :
			# add laziness here - store result of converter?
			if isinstance(kwargs[param.name], dict) :
				kwargs[param.name] = kwargs[param.name].name(**kwargs[param.name])
			setattr(self.texfun, param.name, kwargs[param.name])
		return self.texfun(self.texfun)


def make_ptexture(*params : typing.Tuple[functional.StringParameter]) :
	# TODO : function decorator!
	def decorator(texfun :
	              typing.Callable[[], typing.Tuple['np.ndarray',
								color.colorformat]]) :
		return ptexture(texfun, *params)
	return decorator


@make_ptexture(
	functional.StringParameter('R', parse=int),
	functional.StringParameter('C', parse=int),
	functional.StringParameter('fmt',
		choices    = tuple(color.colorformat.instances.keys()),
		parse      = lambda choice : color.colorformat.instances[choice])
)
def noisefun(self) :

	arr = np.random.randint(256, \
		size=(self.R, self.C, len(self.fmt.channels)), dtype='uint8')

	return texture(arr, self.fmt)



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

@make_ptexture(
	functional.StringParameter('base',
		choices    = list(ptexture.instances.keys()),
		parse      = lambda choice : ptexture.instances[choice],
		get_params = lambda ptex : ptex.params),
	functional.StringParameter('zoom',
		parse = int)
)
def zoomed_smooth(self) :

	import numpy as np

	#zoom = kwargs['zoom']
	#base, fmt = kwargs['base']


	# get ranges corresonding to the top left (1/zoom)th
	# portion of the matrix
	(xf, yf), i = np.modf(np.indices(self.base.arr.shape[:2]) / self.zoom)
	x, y = i.astype(np.int)

	# up, left (negative indices allowed!)
	u = (x-1)
	l = (y-1)

	# If base is 2D, do entrywise multiplication;
	# else, multiply each depth vector in the 2nd matrix with
	# each scalar entry in the 1st matrix
	einsum_str = 'ij,ij->ij' if self.base.arr.ndim == 2 else 'ij,ijk->ijk'

	arr = np.einsum(einsum_str,    xf  *    yf , self.base.arr[x,y]) \
	    + np.einsum(einsum_str, (1-xf) *    yf , self.base.arr[u,y]) \
      + np.einsum(einsum_str,    xf  * (1-yf), self.base.arr[x,l]) \
	    + np.einsum(einsum_str, (1-xf) * (1-yf), self.base.arr[u,l])

	return (arr, self.base.fmt)

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

def textureToImage(tex : texture) :

	from functools import partial
	import PyQt5.QtGui as QtGui
	import numpy as np

	arr, fmt = tex
	if arr.dtype != np.uint8 :
		arr = arr.astype(np.uint8)

	imgen = partial(QtGui.QImage, arr, arr.shape[1], arr.shape[0])

	# hmm
	bytesPerPixel = (sum(fmt.channels)+7) // 8

	return imgen(arr.shape[1] * bytesPerPixel, fmt.wrapped)
