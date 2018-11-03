#class texturefun() :
#
#	"""
#	Functor for procedural textures.
#
#	"""
#
#	def __init__(self, f, name=None) :
#		self.f = f
#		self.name = self.f.__name__ if name is None else name
#
#	def __call__(self, *args, **kwargs) :
#		"""
#		Generate a texture with the given arguments.
#		"""
#		return self.f(*args, **kwargs)
#
#class ptexture() :
#
#	def __init__(self, generator, dependency=None) :
#		self.generator = generator
#		self.updated = dict()
#		self.data = None
#		self.last_kwargs = {}
#
#		self.dependency = None
#		self.set_dependency(dependency)
#
#	def set_dependency(self, dependency) :
#
#		if self.dependency is not None :
#			del self.dependency.updated[self]
#
#		self.dependency = dependency
#
#		if self.dependency is not None :
#			self.dependency.updated[self] = True
#
#
#	def __call__(self, **kwargs) :
#
#		need_update = (self.dependency is not None
#		               and self.dependency.updated[self]) \
#									 or len(kwargs) > 0
#
#		if need_update :
#			nkwargs = self.last_kwargs
#			nkwargs.update(kwargs)
#
#			call = lambda : self.generator(**nkwargs) if self.dependency is None \
#			           else self.generator(self.dependency(), **nkwargs)
#			self.data = call()
#
#			# self now ahead of all dependents
#			for dependent in self.updated.keys() :
#				self.updated[dependent] = True
#
#		self.last_kwargs = nkwargs
#		# dependency is no longer ahead of self
#		self.dependency.updated[self] = False
#
#
#		return self.data
#
#	def as_image() :
#		pass

class filter() :

	def __init__(self, *channelmaps) :
		self.channelmaps = channelmaps

	def __call__(self, arr) :
		if arr.ndim not in (2,3) :
			raise ValueError()

		if len(self.channelmaps) == 1 :
			return self.channelmaps[0](arr)

		if arr.ndim == 2 :
			return np.dstack(map(lambda f : f(arr), self.channelmaps))

		if arr_depth == len(self.channelmaps) :
			np.apply_along_axis(lambda x : [fi(xi) for fi,xi in zip(channelmaps, x)],
			2, arr)

