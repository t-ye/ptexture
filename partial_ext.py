def partial_ext(f, *args1, **kwargs1) :
	def	ff(*args2, **kwargs2) :
		return f(*args1, *args2, **kwargs1, **kwargs2)
	return ff
