from __future__ import annotations # delayed type evaluation

import dataclasses
import typing

def identity(o : typing.Any) :
	"""
	Psuedoidentity function. Objects that are "falsy" are converted to
	None.
	"""
	return o or None

def empty(*args, **kwargs) :
	"""
	Returns an empty sequence.
	"""
	return ()

@dataclasses.dataclass(frozen=True)
class StringParameter() :
	"""
	A parameter that can be inputted by the user, as a string.

	If StringParameter.choices is empty, then the user is permitted to input any
	string. Else, the user must choose fromf StringParameter.choices.
	"""
	name : str
	choices : typing.List[str]  = None
	parse : typing.Callable[[str], typing.Any] = identity
	get_params : typing.Callable[[typing.Any], typing.List[StringParameter]] = empty




del dataclasses
del typing
del annotations
del identity
