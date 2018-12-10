from __future__ import annotations # delayed type evaluation

import dataclasses
import typing
import functools

def identity(o : typing.Any) :
	"""
	Psuedoidentity function. Objects that are "falsy" are converted to
	None.
	"""
	return o or None

def default_parse(choices : typing.Dict[str, typing.Any],
                 choice : typing.AnyStr) :
	return choices.get(choice) if choices else None


def empty(*args, **kwargs) :
	"""
	Returns an empty sequence.
	"""
	return ()

def read_only(cls) :
	"""
	Wrapper to make class attributes write-once.
	"""

	class ReadOnlyWrapper(cls) :

		def __setattr__(self, name, value) :
			if name in self.__dict__ :
				raise AttributeError(f'Can\'t modify {name}')

			super().__setattr__(name, value)

	return ReadOnlyWrapper

@read_only
class StringParameter() :

	def __init__(self, name, choices = None, parse = None, get_params = None) :

		self.name = name
		self.choices = choices
		self.parse = parse if parse else functools.partial(default_parse, choices)
		self.get_params = get_params if get_params else empty


del dataclasses
del typing
del annotations
del identity
