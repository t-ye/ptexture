def f() :
	a = 4
	def g() :
		return a
	return g()

def F(kwargs) :
	def g() :
		return a
	return g()


def dec(a=5) :
	def dec_inner(f) :
		def wrapper(*args, **kwargs) :
			return f(a)
		return wrapper
	return dec_inner

def partial(*outer_args, **outer_kwargs) :
	def decorator(f) :
		def wrapper(*args, **kwargs) :
			all_kwargs = {**outer_kwargs, **kwargs}
			return f(*outer_args, *args, all_kwargs)
		return wrapper
	return decorator


def test() :
	def decorator(f) :
		def wrapper() :
			return f()
		return wrapper
	return decorator

@dec(4)
def inside(a) :
	return 2*a

@test()
def f() :
	return f.a

print(f())
