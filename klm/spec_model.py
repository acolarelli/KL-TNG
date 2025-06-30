from abc import abstractmethod
from matplotlib.patches import Polygon
import numpy as np
import scipy.signal

import galsim
import astropy.units as u

from klm.utils import build_map_grid
from klm.parameters import Parameters
from klm.transormations import Transformations
from klm.velocity import VelocityModel
from klm.emission_line import EmissionLine


class KinematicModel():
	def __init__(self) -> None:
		pass

	# Each subclass must have a method that initializes the observable
	@abstractmethod
	def _init_observable():
		'''
		Initializes the observable for the spec model
		Setup grid and necessary class attributes here
		'''
		pass

	# Each subclass must have a method that returns the observable
	@abstractmethod
	def get_observable():
		'''
		Method for returning the observable Spectrum/velocity field
		'''
		pass


class IFUModel(KinematicModel):
	def __init__(self, meta_param=None, rc_type='arctan'):
		self._init_observable(meta_param)
		self.velocity_model = VelocityModel(rc_type)

	def _init_observable(self, meta_param):
		self.obs_xx = meta_param['RA_grid']
		self.obs_yy = meta_param['Dec_grid']
		self.psf = meta_param.get('psf', None)
		self.Halpha_flux = meta_param.get('Halpha_flux', None)
		self.Halpha_flux_conv = meta_param.get('Halpha_flux_conv', None)

	def get_observable(self, params):
		disk_xx, disk_yy = Transformations.transform_frame(self.obs_xx, self.obs_yy, params=params, start='obs', end='disk')

		velocity_map = self.velocity_model.build_vfield(params, disk_xx, disk_yy)
		if self.psf is not None:
			velocity_map = scipy.signal.convolve2d(self.Halpha_flux*velocity_map, self.psf, mode='same')/self.Halpha_flux_conv
			
			# Halpha flux and Halpha_flux_conv contain zero entries which gives rise to NaN values
			# These will be masked in the likelihood, so replace them by zero
			velocity_map[np.isnan(velocity_map)] = 0.

		return velocity_map


class SlitModel(KinematicModel):
	'''
	Class for creating model spectrum
	'''
	def __init__(self, obj_param=None, meta_param=None, line_profile_path=None, rc_type='arctan'):
		self.velocity_model = VelocityModel(rc_type)
		self._init_observable(obj_param, meta_param)

		self.sigma_intr = 0.01*u.nm
		self.line_name, self.line_wav = Parameters()._get_species(meta_param['line_species'], obj_param['redshift'])

				## sigma1 & sigma2 are the line widths of the two half-Gaussians

		if line_profile_path is not None:
			self.emission_line_model = EmissionLine(line_species=meta_param['line_species'], line_profile_path=line_profile_path)

		elif 'rhl' in meta_param.keys():
			self.emission_line_model = EmissionLine(use_analytic=True, line_species=meta_param['line_species'], rhl=meta_param['rhl'], slit_grid=self.slit_x)

		else:
			raise ValueError('Either need line profile from obs or rhl to instantiate analytic line profile')

		self.profile_sigma1, self.profile_sigma2 = self.emission_line_model.line_profile.get_linewidth_profile(self.ndim)
		self.profile_one_over_sigma1 = [1/p for p in self.profile_sigma1]
		self.profile_one_over_sigma2 = [1/p for p in self.profile_sigma2]

		self.profile_A = self.emission_line_model.intensity.get_intensity_profile(self.ndim)

	def _init_observable(self, obj_param, meta_param):
		'''Initialize grid and slit mask for spec calculation
		Assumes ngrid[0] corresponds to the spatial axis

		Parameters
		----------
		obj_param : dict
			Object parameters including RA, Dec
		meta_param : dict
			Spectrum meta parameters
		'''

		RA_obj, Dec_obj = obj_param['RA'].to(u.arcsec), obj_param['Dec'].to(u.arcsec)  # convert degrees to arcsec
		RA_slit, Dec_slit = meta_param['slitRA'].to(u.arcsec), meta_param['slitDec'].to(u.arcsec)

		self.lambda_grid = meta_param['lambda_grid'].to(u.nm).value  ##lambda should run along rows & spatial position along columns
		#Shape of 3D data-cube
		# Indexed as (x, y, lambda)
		self.ndim = (meta_param['ngrid'][0], meta_param['ngrid'][0]+10, meta_param['ngrid'][1])

		slit_xx, slit_yy =  build_map_grid(Nx=self.ndim[0], Ny=self.ndim[1], pix_scale=meta_param['pixScale'])
		self.slit_x = slit_xx[:,0]

		self.slit_patch = self._get_sky_xaxis_aligned_slit_patch(RA_slit.value-RA_obj.value, Dec_slit.value-Dec_obj.value, meta_param['slitLen'],
											meta_param['slitWidth'], meta_param['slitLPA'], meta_param['slitWPA'])
		self.slit_mask = self._get_slit_mask(self.slit_patch, slit_xx, slit_yy)

		self.obs_xx, self.obs_yy = Transformations.transform_frame(slit_xx, slit_yy, start='slit', end='obs', params={'slitLPA':meta_param['slitLPA']})

		# Grid for creating image cutout from which the brightness profile is extracted
		self.cutout_grid = galsim.Image(nrow=meta_param['ngrid'][0], ncol=meta_param['ngrid'][0], scale=meta_param['pixScale'])


	def get_observable(self, params):
		# self.gal_image_cutout = self._get_image_cutout(params, slit_angle)
		self.gal_image_cutout = 0.
		spectra = self._build_spectrum(params, self.gal_image_cutout)

		return spectra


	def _build_spectrum(self, params, gal_image_cutout):
		'''
		Generates 2D spectrum

		Args:
			params (dict): Parameter dictionary

		Returns:
			array: 2D spectrum
		'''
		disk_xx, disk_yy = Transformations.transform_frame(self.obs_xx, self.obs_yy, params=params, start='obs', end='disk')

		vfields = self.build_line_vfield(params, disk_xx, disk_yy)

		self.vfield = vfields

		galaxy_emission = self._add_emission_line(params, vfields)

		spectrum3D = galaxy_emission * (~self.slit_mask)[:, :, np.newaxis] #* gal_image_cutout[:, :, np.newaxis]
		# spectrum3D = scipy.ndimage.gaussian_filter(spectrum3D, (self.psf_sigma, self.psf_sigma, 0), mode='constant')
		spectrum2D = np.sum(spectrum3D, axis=1)

		return spectrum2D + params['bkg_level']

	def build_line_vfield(self, params, Xgrid, Ygrid):
		''' Evaluates the velocity field on a 2D grid. Computes two separate rotation curves
		if the emission line is a doublet.
		Actual computation is done in a lower level function.
		Args:
			params (_type_): _description_
			Xgrid (_type_): _description_
			Ygrid (_type_): _description_

		Returns:
			_type_: _description_
		'''
		Vfield1, Vfield2 = 0., 0.

		params_vfield = {'v_0': params['v_0'],
						'vcirc': params['vcirc'],
						'cosi': params['cosi'],
						'vscale': params['vscale'],
						'v_outer': params['v_outer'],
						'dx_vel': params['dx_vel'],
						'dy_vel': params['dy_vel'],
						'r_hl_disk': params['r_hl_disk']}
		Vfield1 = self.velocity_model.build_vfield(params_vfield, Xgrid, Ygrid)

		if self.line_name in ['OIIa', 'OII']:
			params_vfield2 = {'v_0': params['v_0_2'],
							'vcirc': params['vcirc'],
							'cosi': params['cosi'],
							'vscale': params['vscale'],
							'v_outer': params['v_outer'],
							'dx_vel': params['dx_vel_2'],
							'dy_vel': params['dy_vel_2'],
							'r_hl_disk': params['r_hl_disk']}
			Vfield2 = self.velocity_model.build_vfield(params_vfield2, Xgrid, Ygrid)

		return Vfield1, Vfield2

	def _add_emission_line(self, params, velocity_fields):
		'''Returns a delta wavelength grid
		'''
		c = 299792.45  # Speed of light in km/s
		cube = 0.0

		for i, this_line in enumerate(self.line_wav):
			vfield = velocity_fields[i]
			line_center = (1 + vfield/ c) * (this_line.value)
			delta_wavelength = self.lambda_grid[:, np.newaxis, :] - line_center[:, :, np.newaxis]
			# delta_wavelength = self.lambda_grid[np.newaxis, :, :] - line_center[:, :, np.newaxis]
			self.delta_wav = delta_wavelength

			exponent = np.zeros_like(delta_wavelength)
			idx = delta_wavelength<0
			exponent[idx] = delta_wavelength[idx] * self.profile_one_over_sigma1[i][idx] * params[f'f{i+1}_1']
			exponent[~idx] = delta_wavelength[~idx] * self.profile_one_over_sigma2[i][~idx] * params[f'f{i+1}_2']

			## Now create an emission profile
			# A = np.tile(self.profile_A[:, i], (delta_wavelength.shape[0], 1))
			# sigma = self.profile_sigma[:, i][np.newaxis, :, np.newaxis]


			# sigma = self.profile_sigma[:, i][np.newaxis, :, np.newaxis]
			# exponent = delta_wavelength/sigma

			cube += params[f'I0{i+1}'] * (np.exp(-exponent**2 / 2.)) * self.profile_A[i][:, :, np.newaxis]

		return cube

	def _get_image_cutout(self, params, slitAngle):
		'''
		Rotates the galaxy image based on the slit angle and generates image cutout after PSF convolution
		Args:
			params (dict): Parameter dictionary

		Returns:
			array: Galaxy image
		'''
		half_light_radius = params['r_hl_spec']
		theta_int = params['theta_int']
		inclination = np.arccos(params['cosi'])
		g1, g2 = params['g1'], params['g2']
		slitAngle = slitAngle.to(u.radian).value

		disk = galsim.inclined.InclinedSersic(n=params['sersic_spec'], inclination=inclination*galsim.radians, half_light_radius=half_light_radius,
												scale_h_over_r=params['aspect'], trunc=4*half_light_radius)

		disk = disk.rotate(theta_int * galsim.radians)
		disk = disk.shear(g1=g1, g2=g2)
		disk = disk.rotate(-slitAngle*galsim.radians)

		galObj = galsim.Convolution([disk, self.psf_spec])

		newImage = galObj.drawImage(image=self.cutout_grid)
		newImage_cutout = np.ma.array(newImage.array, mask=self.slit_mask)
		return newImage_cutout


	def _get_slit_patch(self, RA, Dec, LEN, WID, LPA, WPA):
		'''_summary_

		Parameters
		----------
		RA : float
			RA (in arcsec)
		Dec : float
			Dec (in arcsec)
		LEN : float
			Slit length (in arcsec)
		WID : float
			slit width (in arcsec)
		LPA : float
			Position angle of the long side of the slit (in degrees)
		WPA : float
			Position angle of the short side of the slit (in degrees)

		Returns
		-------
		_type_
			_description_
		'''
		LPA, WPA = LPA.to(u.radian).value, WPA.to(u.radian).value
		v_center = np.array([RA, Dec])

		# vectors from center to mid-points of the four sides
		vec_N = WID/2*np.array([np.cos(WPA), np.sin(WPA)])  # North
		vec_S = WID/2*np.array([np.cos(np.pi+WPA), np.sin(np.pi+WPA)])  # South
		vec_E = LEN/2*np.array([np.cos(LPA), np.sin(LPA)])  # East
		vec_W = LEN/2*np.array([np.cos(np.pi+LPA), np.sin(np.pi+LPA)])  # West

		# The four corners of the slit mask
		vec_NW = v_center + vec_N + vec_W
		vec_NE = v_center + vec_N + vec_E
		vec_SE = v_center + vec_S + vec_E
		vec_SW = v_center + vec_S + vec_W

		vec_slit = [vec_NW, vec_NE, vec_SE, vec_SW]
		self._assert_slit_PA(vec_slit, LPA, WPA)
		print('Creating slit mask...')
		slit_patch = Polygon(vec_slit)
		slit_patch.set_closed(True)

		return slit_patch

	def _get_sky_xaxis_aligned_slit_patch(self, RA, Dec, LEN, WID, LPA, WPA):
		original_slit_patch = self._get_slit_patch(RA, Dec, LEN, WID, LPA, WPA)

		vertices = original_slit_patch.get_xy()
		xp, yp = Transformations._apply_rotation(vertices[:, 0], vertices[:, 1], -LPA)
		aligned_patch = Polygon(np.column_stack((xp, yp)))
		aligned_patch.set_closed(True)

		return aligned_patch


	def _assert_slit_PA(self, vectors, LPA, WPA):
		''' Checks if the slit mask has the correct PA given coordinates of the
		four slit corners
		'''
		vec_NW, vec_NE, vec_SE, vec_SW = vectors
		unit_x = np.array([1, 0])

		vec_left = (vec_NW-vec_SW)/np.linalg.norm(vec_NW-vec_SW)
		vec_right = (vec_NE-vec_SE)/np.linalg.norm(vec_NE-vec_SE)
		vec_top = (vec_NE-vec_NW)/np.linalg.norm(vec_NE-vec_NW)
		vec_bot = (vec_SE-vec_SW)/np.linalg.norm(vec_SE-vec_SW)

		angle_left = np.arccos(np.dot(vec_left, unit_x))
		angle_right = np.arccos(np.dot(vec_right, unit_x))
		angle_top = np.arccos(np.dot(vec_top, unit_x))
		angle_bot = np.arccos(np.dot(vec_bot, unit_x))

		assert np.isclose(np.cos(angle_left), np.cos(WPA)), f'Angle of left segement is {angle_left*180/np.pi:.2f} but PA is {WPA*180/np.pi}!'
		assert np.isclose(np.cos(angle_right), np.cos(WPA)), f'Angle of right segement is {angle_right*180/np.pi:.2f} but PA is {WPA*180/np.pi}!'
		assert np.isclose(np.cos(angle_top), np.cos(LPA)), f'Angle of top segement is {angle_top*180/np.pi:.2f} but PA is {LPA*180/np.pi}!'
		assert np.isclose(np.cos(angle_bot), np.cos(LPA)), f'Angle of bot segement is {angle_bot*180/np.pi:.2f} but PA is {LPA*180/np.pi}!'


	def _get_slit_mask(self, slit_patch, x_grid, y_grid):
		'''Checks if each pair of (x, y) is in the slit patch and returns
		a mask for masking the x, y grids
		The points where the mask is True will be masked out
		'''
		mask = np.ones(x_grid.shape, dtype=bool)
		for i, (these_x, these_y) in enumerate(zip(x_grid, y_grid)):
			mask[i] = ~slit_patch.contains_points(np.column_stack((these_x, these_y)))

		return mask
