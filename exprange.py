class exprange() :

	"""
	exprange(base, stop)
	exprange(base, start, stop[, step])

	Yields a geometric sequence, analogous to how range() yields an
	arithmetic sequence.

	This function is designed for integer inputs *and* integer outputs only.
	Floating point inputs will produce undefined behavior. If a floating point


	base - geometric ratio
	start - initial exponent
	stop - ending exponent (non-inclusive)
	step - how many times base is applied between every yielded value

	"""

	def __init__(self, base : int, start : int, stop : int = None, step : int = 1) :

		#from functools import partial

		if stop is None : # then defer to default arguments
			stop = start
			start = 0

		#self.__dict__.update(locals())
		#del self.self
		self.base = base

		self.range = range(start, stop, step)
		#self.__dict__.update((x,partial(getattr(range, x), self.range)) for x in dir(range) if
		#             callable(getattr(range, x)))
		#del partial



	def __iter__(self) :

		from itertools import islice

		if self.range.start >= self.range.stop and self.range.step > 0 or \
		   self.range.start <= self.range.stop and self.range.step < 0 : # check bounds
			return

		exp = self.base ** self.range.start # exponentiation result
		base = self.base ** abs(self.range.step) # geometric ratio

		yield exp

		for i in islice(self.range, 1, None) :
			if self.range.step > 0 :
				exp *= base
			else :
				exp //= base
			yield exp

	def __eq__(self, other) :
		return len(self) == len(other) and \
			(len(self) == 0 or self[0] == other[0] and self[1] == other[1])



	def __getitem__(self, key) :
		return self.base ** key

	def __reversed__(self) :
		base, start, stop, step = self.base, self.range.start, self.range.stop, self.range.step
		sgn = bool(step > 0) - bool(step < 0) # 1, 0,  -1 - sign(self.step)
		return exprange(base, stop-step+(stop-start)%step, start-sgn, -step)

	def __str__(self) :
		return f'exprange({self.base}, {self.range.start}, {self.range.stop}, {self.range.step})'

	def __repr__(self) :
		return str(self)


	def count(self, el) :
		from primefac import ilog

		if self.base == 1 :
			return len(self) if el == 1 else 0
		#if ilog(el, self.base) in self.range :

	def index(self, el, start=0, stop=None) :
		import bisect
		from primefac import ilog

		# normalize index bounds
		if stop is None :
			stop = len(self)
		# make index bounds positive, if not already
		# very negative indices are treated as wrapping around the list multiple
		# times, unlike with [], but identical to list.index's behavior.
		if start < 0 :
			start %= len(self)
		if stop < 0 :
			stop %= len(self)

		# check index bounds
		if start >= stop :
			raise ValueError(f'{el} is not in range') # not found

		lb, lbi = self[start], start # lower bound
		ub, ubi = self[stop], stop # upper bound

		if self.range.step < 0 : # adjust bounds as necessary
			lb, lbi = ub, ubi

		# check lower bound
		if lb >= el :
			if lb == el :
				return lbi
			raise ValueError(f'{el} is not in range ')

		# check upper bound
		if ub <= el :
			if ub == el :
				return ubi
			raise ValueError(f'{el} is not in range')

		# search manually
		idx = ilog(abs(el), abs(self.base))
		idx = self.range.start + self.range.step * idx

		if self[idx] == el :
			return idx

		raise ValueError(f'{el} is not in range')




# from inspect import signature


#from functools import partial
#exprange.__len__ = lambda self : 7
#setattr(exprange, '__len__', lambda self, *args, **kwargs : getattr(self.range,
#	'__len__')(*args, **kwargs))
#setattr(exprange, 'count', lambda self, *args, **kwargs : getattr(self.range,
#	'count')(*args, **kwargs))


for attrname in dir(range) :
	if not callable(getattr(range, attrname)) :
		continue
	if attrname == '__class__' :
		continue
	if hasattr(exprange, attrname) :
		continue


	setattr(exprange, attrname,
		(lambda attrname :
	  lambda self, *args, **kwargs : getattr(self.range, attrname)(*args,
			**kwargs))(attrname))

	getattr(exprange, attrname).__doc__ = getattr(range, attrname).__doc__

#exprange.__dict__.update((x,partial(getattr(range, x), self.range)) for x in dir(range) if
#		             callable(getattr(range, x)))

#sig = signature(getattr(range, attrname))

def main() :
	a = exprange(10, 5)
	print(a.count(4))
	print(len(a))


if __name__ == '__main__' :
	main()



