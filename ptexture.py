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

