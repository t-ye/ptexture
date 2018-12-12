import typing
import functools

def default_parse(choices : typing.Dict[str, typing.Any], choice :
		typing.AnyStr) :
	return choices.get(choice) if choices else None


_empty = ()

def empty(*args, **kwargs) -> typing.Tuple :
	"""
	Returns an empty sequence.
	"""
	return _empty

def read_only(cls) :
	"""
	Wrapper to make class attributes write-once.
	"""

	class ReadOnlyWrapper(cls) :

		def __init__(self, *args, **kwargs) :
			super().__init__(*args, **kwargs)
			functools.update_wrapper(self, super())

		def __setattr__(self, name, value) :
			if name in self.__dict__ :
				raise AttributeError(f'Can\'t modify {name}')

			super().__setattr__(name, value)

	#functools.update_wrapper(ReadOnlyWrapper, cls)
	#print(ReadOnlyWrapper)

	return ReadOnlyWrapper
