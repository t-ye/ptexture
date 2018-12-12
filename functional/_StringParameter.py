from ._internal import read_only, default_parse, empty
import functools
import typing


@read_only
class StringParameter() :

	def __init__(self, name : typing.AnyStr,
			               choices : typing.Dict[str, typing.Any] = None,
										 parse = None,
										 get_params = None) :

		self.name = name
		self.choices = choices
		self.default = ''
		self.parse = parse if parse else \
								 functools.partial(default_parse, self.choices)
		self.get_params = get_params if get_params else empty
