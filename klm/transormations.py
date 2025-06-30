import numpy as np

class Transformations():
	def __init__(self):
		'''Here are the definitions of the different planes
			- slit frame: x/y axis aligned with slit
			- obs frame: lensed (x/y axis aligned with RA/Dec axis)
			- source frame: object has inclination, intr. PA
			- gal frame: object has inclination, x/y axis aligned with major/minor axis
			- disk frame: object is a face-on disk
		'''
		pass
	
	@classmethod
	def transform_frame(cls, X, Y, params, start, end, return_all=False):
		sequence = ('slit', 'obs', 'source', 'gal', 'disk')

		start_id = sequence.index(start)
		end_id = sequence.index(end)

		if start_id > end_id:
			sequence = sequence[::-1]
			start_id = sequence.index(start)
			end_id = sequence.index(end)

		# We will store all grid in this list
		current_frame = [X, Y]
		grids = {}
		# Iterate over transformations
		for i in range(start_id, end_id):
			plane_transform = sequence[i]+'2'+sequence[i+1]
			X_new, Y_new = cls._transform_frame(current_frame[0], current_frame[1], params, plane_transform)
			grids[i+1] = [X_new, Y_new]
			current_frame = [X_new, Y_new]

		if return_all is True:
			output = {}
			output[sequence[start_id]] = {'X': X, 'Y': Y}
			for i in range(start_id, end_id):
				this_plane = sequence[i+1]
				output[this_plane] = {'X': grids[i+1][0], 'Y': grids[i+1][1]}

			return output

		else:
			return grids[end_id][0], grids[end_id][1]

	@classmethod
	def _transform_frame(cls, X, Y, params, plane):
		'''Lower level function that transforms between planes that are linked 
		by a single transformation

		Args:
			X (float/array): _description_
			Y (float/array): _description_
			g1 (float): Cartesian shear 1st component
			g2 (float): Cartesian shear 2nd component
			PA_int (float): Intrinsic position angle (in radians)
			cosi (float): Inclination

		Returns:
			_type_: _description_
		'''

		## Backward transformations
		if plane == 'slit2obs':
			slitPA = params['slitLPA']
			return cls._slit2obs(X, Y, slitPA)
		
		elif plane == 'obs2source':
			g1 = params['g1']
			g2 = params['g2']
			return cls._obs2source(X, Y, g1=g1, g2=g2)

		elif plane == 'source2gal':
			PA_int = params['theta_int']
			return cls._source2gal(X, Y, PA_int=PA_int)

		elif plane == 'gal2disk':
			cosi = params['cosi']
			return cls._gal2disk(X, Y, cosi=cosi)

		## Forward transformations
		elif plane == 'disk2gal':
			cosi = params['cosi']
			return cls._disk2gal(X, Y, cosi=cosi)

		elif plane == 'gal2source':
			PA_int = params['theta_int']
			return cls._gal2source(X, Y, PA_int=PA_int)

		elif plane == 'source2obs':
			g1 = params['g1']
			g2 = params['g2']
			return cls._source2obs(X, Y, g1=g1, g2=g2)

		elif plane == 'obs2slit':
			slitPA = params['slitLPA']
			return cls._obs2slit(X, Y, slitPA)

	@classmethod
	def _slit2obs(cls, X, Y, slitPA):
		'''To go from slit to sky frame we need to rotate
		anti-clockwise by the slit P.A.
		Make sure that the slit P.A. is measured from the +RA-axis

		Args:
			X (_type_): _description_
			Y (_type_): _description_
			slitPA (_type_): _description_

		Returns:
			_type_: _description_
		'''
		return cls._apply_rotation(X, Y, slitPA)

	@classmethod
	def _obs2slit(cls, X, Y, slitPA):
		'''To go from slit to sky frame we need to rotate
		anti-clockwise by the slit P.A.
		Make sure that the slit P.A. is measured from the +RA-axis

		Args:
			X (_type_): _description_
			Y (_type_): _description_
			slitPA (_type_): _description_

		Returns:
			_type_: _description_
		'''
		return cls._apply_rotation(X, Y, -slitPA)

	@classmethod
	def _obs2source(cls, X, Y, g1, g2):
		'''The object has to be unsheared in the sky frame

		Args:
			X (_type_): _description_
			Y (_type_): _description_

		Returns:
			_type_: _description_
		'''
		return cls._apply_unshear(X, Y, g1, g2)

	@classmethod
	def _source2obs(cls, X, Y, g1, g2):
		'''The object has to be unsheared in the sky frame

		Args:
			X (_type_): _description_
			Y (_type_): _description_

		Returns:
			_type_: _description_
		'''
		return cls._apply_shear(X, Y, g1, g2)

	@classmethod
	def _source2gal(cls, X, Y, PA_int):
		'''

		Args:
			X (_type_): _description_
			Y (_type_): _description_

		Returns:
			_type_: _description_
		'''
		return cls._apply_rotation(X, Y, -PA_int)

	@classmethod
	def _gal2source(cls, X, Y, PA_int):
		'''

		Args:
			X (_type_): _description_
			Y (_type_): _description_

		Returns:
			_type_: _description_
		'''
		return cls._apply_rotation(X, Y, PA_int)

	@classmethod
	def _gal2disk(cls, X, Y, cosi):
		return cls._apply_uninclination(X, Y, cosi)

	@classmethod
	def _disk2gal(cls, X, Y, cosi):
		return cls._apply_inclination(X, Y, cosi)

	@classmethod
	def _apply_rotation(cls, X, Y, theta):
		'''Performs a counter-clockwise rotation by angle theta

		Args:
			X (_type_): _description_
			Y (_type_): _description_
			theta (_type_): _description_

		Returns:
			_type_: _description_
		'''		
		transform = cls._get_rotation_transform(theta)

		return cls._apply_transform(X, Y, transform)

	@classmethod
	def _apply_shear(cls, X, Y, g1, g2):
		'''Applies a shear effect for Cartesian shears g1, g2

		Args:
			X (_type_): _description_
			Y (_type_): _description_
			g1 (_type_): _description_
			g2 (_type_): _description_

		Returns:
			_type_: _description_
		'''
		transform = cls._get_shear_transform(g1, g2)

		return cls._apply_transform(X, Y, transform)


	@classmethod
	def _apply_unshear(cls, X, Y, g1, g2):
		'''Unshears the positions for Cartesian shears g1, g2

		Args:
			X (_type_): _description_
			Y (_type_): _description_
			g1 (_type_): _description_
			g2 (_type_): _description_

		Returns:
			_type_: _description_
		'''
		transform = cls._get_unshear_transform(g1, g2)

		return cls._apply_transform(X, Y, transform)


	@classmethod
	def _apply_inclination(cls, X, Y, cosi):
		'''Applies inclination

		Args:
			X (_type_): _description_
			Y (_type_): _description_
			cosi (_type_): _description_

		Returns:
			_type_: _description_
		'''
		transform = cls._get_inclination_transform(cosi)

		return cls._apply_transform(X, Y, transform)

	@classmethod
	def _apply_uninclination(cls, X, Y, cosi):
		''' `Uninclines` coordinates

		Args:
			X (_type_): _description_
			Y (_type_): _description_
			cosi (_type_): _description_

		Returns:
			_type_: _description_
		'''
		transform = cls._get_uninclination_transform(cosi)

		return cls._apply_transform(X, Y, transform)

	@classmethod
	def _apply_transform(cls, X, Y, transform):
		'''Effectively does a matrix multiplication

			Requires that (X, Y) be passed along with the transform to apply

		Args:
			X (_type_): _description_
			Y (_type_): _description_
			transform (_type_): _description_

		Returns:
			_type_: _description_
		'''
		X_prime = X*transform[0, 0] + Y*transform[0, 1]
		Y_prime = X*transform[1, 0] + Y*transform[1, 1]

		return X_prime, Y_prime

	@classmethod
	def _get_shear_transform(cls, g1, g2):
		'''This is the transform used to go from unlensed to 
		lensed coordinates.
		So the function returns the inverse of `A` so that
		(x_l, y_l)^T = A^-1 @ (x_ul, y_ul)^T
		(x_l, y_l): Lensed coordinates
		(x_ul, y_ul): Unlensed coordinates
		Args:
			g1 (float): _description_
			g2 (float): _description_

		Returns:
			2d array: transform matrix
		'''
		transform = np.linalg.inv(cls._get_unshear_transform(g1, g2))
		
		return transform

	@classmethod
	def _get_unshear_transform(cls, g1, g2):
		''' Returns the lensing matrix `A`, which is traditionally defined
		to go from lensed to unlensed coordinates.

		Args:
			g1 (float): _description_
			g2 (float): _description_

		Returns:
			2d array: transform matrix
		'''		
		transform = np.array(
						[[1.-g1, -g2],
						[-g2, 1.+g1]]
						)

		return transform

	@classmethod
	def _get_rotation_transform(cls, theta):
		'''Transform for performing a counter-clockwise rotation by angle theta

		Args:
			theta (float): Angle in radians

		Returns:
			2d array: transform matrix
		'''		
		cos, sin = np.cos(theta), np.sin(theta)
		transform = np.array([
							[cos, -sin],
							[sin, cos]]
							)
		
		return transform

	@classmethod
	def _get_inclination_transform(cls, cosi):
		'''Transform for applying inclination

		Args:
			cosi (float): _description_

		Returns:
			2d array: transform matrix
		'''		
		transform = np.array(
							[[1., 0.],
							[0., cosi]]
							)
		
		return transform

	@classmethod
	def _get_uninclination_transform(cls, cosi):
		'''Transform for undoing the effect of inclination

		Args:
			cosi (float): _description_

		Returns:
			2d array: transform matrix
		'''		
		transform = cls._get_inclination_transform(cosi)

		return np.linalg.inv(transform)