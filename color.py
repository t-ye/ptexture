def channelsplit(splitter, arr) :
	"""
	Converts a single (color) channel to multiple channels.

	The type of the original channel is preserved.
	"""
	import numpy as np

	return np.dstack(splitter(arr))

import numpy as np
from npqt import arr_to_image
import PyQt5.QtGui as QtGui

def hsv2rgb(hsv) :

	"""
	(0,360), (0, 1), (0, 1) -> (0, 255), (0, 255), (0, 255)
	"""

	import numpy as np

	h,s,v = hsv.T

	rgb = np.empty(hsv.shape)
	r,g,b = rgb.T # views

	f, i = np.modf(h / 60)
	i = i.astype(np.int)

	p = v * (1 - s)
	q = v * (1 - (s * f))
	t = v * (1 - (s * (1 - f)))

	i0 = i == 0
	i1 = i == 1
	i2 = i == 2
	i3 = i == 3
	i4 = i == 4
	i5 = i == 5

	r[i0] = v[i0]
	g[i0] = t[i0]
	b[i0] = p[i0]

	r[i1] = q[i1]
	g[i1] = v[i1]
	b[i1] = p[i1]

	r[i2] = p[i2]
	g[i2] = v[i2]
	b[i2] = t[i2]

	r[i3] = p[i3]
	g[i3] = q[i3]
	b[i3] = v[i3]

	r[i4] = t[i4]
	g[i4] = p[i4]
	b[i4] = v[i4]

	r[i5] = v[i5]
	g[i5] = p[i5]
	b[i5] = q[i5]

	slt = s <= 0
	vs = v[slt]
	r[slt] = vs
	g[slt] = vs
	b[slt] = vs

	rgb *= 256
	rgb[rgb >= 256] -= 1
	return rgb
