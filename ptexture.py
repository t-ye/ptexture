class texturefun() :

	"""
	Functor for procedural textures.

	"""

	def __init__(self, f, name=None) :
		self.f = f
		self.name = self.f.__name__ if name is None else name

	def __call__(self, *args, **kwargs) :
		"""
		Generate a texture with the given arguments.
		"""
		return self.f(*args, **kwargs)

class ptexture() :

	def __init__(self, generator, dependency=None) :
		self.generator = generator
		self.updated = dict()
		self.data = None
		self.last_kwargs = {}

		self.dependency = None
		self.set_dependency(dependency)

	def set_dependency(self, dependency) :

		if self.dependency is not None :
			del self.dependency.updated[self]

		self.dependency = dependency

		if self.dependency is not None :
			self.dependency.updated[self] = True


	def __call__(self, **kwargs) :

		need_update = (self.dependency is not None
		               and self.dependency.updated[self]) \
									 or len(kwargs) > 0

		if need_update :
			nkwargs = self.last_kwargs
			nkwargs.update(kwargs)

			call = lambda : self.generator(**nkwargs) if self.dependency is None \
			           else self.generator(self.dependency(), **nkwargs)
			self.data = call()

			# self now ahead of all dependents
			for dependent in self.updated.keys() :
				self.updated[dependent] = True

		self.last_kwargs = nkwargs
		# dependency is no longer ahead of self
		self.dependency.updated[self] = False


		return self.data

	def as_image() :
		pass
