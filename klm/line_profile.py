import numpy as np

class LineProfile:
	'''
	Class representing a line profile.

	Args:
		sigma1 (float): The linewidth parameter for the first half Gaussian.
		sigma2 (float): The linewidth parameter for the second half Gaussian.
		use_Gaussian (bool): Flag indicating whether to use a Gaussian line profile.

	Methods:
		get_linewidth_profile(ndim): Returns the linewidth profile for the given dimensions.

	'''

	def __init__(self, sigma1, sigma2, use_Gaussian=False):
		self.sigma1 = sigma1
		self.sigma2 = sigma2

	def get_linewidth_profile(self, ndim):
		'''
		Returns the linewidth profile for the given dimensions.

		Args:
			ndim (tuple): A tuple containing the dimensions of the profile.

		Returns:
			tuple: A tuple containing the linewidth profiles for the two components.

		'''
		return self._build_linewidth_profile(self.sigma1, ndim), self._build_linewidth_profile(self.sigma2, ndim)

	def _build_linewidth_profile(self, sigma_arr, ndim):
		'''
		Builds the linewidth profile for a given sigma array and dimensions.

		Args:
			sigma_arr (ndarray): An array containing the linewidth values.
			ndim (tuple): A tuple containing the dimensions of the profile.

		Returns:
			ndarray: The linewidth profile.

		'''
		# Expects linewidth as a function of x
		# Should have two columns, the second column is used only for doublets
		assert sigma_arr.shape == (ndim[0], 2)

		sigma_3D = []
		for i in range(2):
			# First we take the linewidth measured along the slit and repeat it in the other direction to create a 2D grid
			sigma_2Dgrid = np.repeat(sigma_arr[:, i][:, np.newaxis], ndim[1], axis=-1)

			# Now we repeat the 2D grid nLambda times
			sigma_3Dcube = np.repeat(sigma_2Dgrid[:, :, np.newaxis], ndim[2], axis=-1)

			sigma_3D.append(sigma_3Dcube)

		return sigma_3D