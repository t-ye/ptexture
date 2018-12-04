from __future__ import annotations # delayed type evaluation

import dataclasses
import typing


@dataclasses.dataclass(frozen=True)
class StringParameter() :
	name : str
	choices : typing.Tuple[str] = ()
	default : str = ''
	children : typing.Tuple[Parameter] = ()

def identity(s : typing.AnyStr) :
	"""
	Psuedoidentity function on strings. The empty string is treated as None.
	"""
	return s or None

@dataclasses.dataclass(frozen=True)
class TypedParameter(StringParameter) :
	converter : typing.Callable[[str], typing.Any] = identity
	type_hint : type = type(None)


del dataclasses
del typing
del annotations
del identity
