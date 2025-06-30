import numpy as np

class IntensityProfile():
	'''
	Class for describing emission line intensity profile.

	Args:
		amp (numpy.ndarray): The amplitude of the intensity profile.

	Attributes:
		amp (numpy.ndarray): The amplitude of the intensity profile.

	Methods:
		get_intensity_profile(ndim): Returns the 2D intensity profile.

	'''

	def __init__(self, amp):
		self.amp = amp


	def get_intensity_profile(self, ndim):
		'''
		Returns the 2D intensity profile.

		Args:
			ndim (tuple): The dimensions of the intensity profile.

		Returns:
			list: A list of 2D intensity profiles.

		'''
		return self._build_line_intensity(self.amp, ndim)


	def _build_line_intensity(self, intensity, ndim):
		'''
		Builds the 2D intensity profile.

		Args:
			intensity (numpy.ndarray): The amplitude of the intensity profile.
			ndim (tuple): The dimensions of the intensity profile.

		Returns:
			list: A list of 2D intensity profiles.

		'''
		assert intensity.shape == (ndim[0], 2)

		intensity_2D = []
		for i in range(2):
			## The intensity profile estimated from the obs. captures the variation along the slit
			## The following allows us to have a similar variation across the slit
			A = np.array([intensity[:, i]]).T@np.array([intensity[:, i]])
			## A will have shape (nx, nx)
			## Now pad the other direction axis=1 with zeros
			pad_width =  ndim[1] - A.shape[1]
			assert pad_width%2 == 0.
			A_pad = np.apply_along_axis(np.pad, axis=1, arr=A, pad_width=int(pad_width/2), mode='constant', constant_values=0.)

			intensity_2D.append(A_pad)

		return intensity_2D