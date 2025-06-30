import glob
import h5py
import joblib
import json
import numpy as np
import re
from scipy.optimize import curve_fit
import scipy.signal

import astropy as ap
from astropy.coordinates import FK5, SkyCoord
from astropy.cosmology import Planck18
from astropy.convolution import convolve
from astropy.io import fits
import astropy.units as u
from astroquery.sdss import SDSS
from photutils.segmentation import make_2dgaussian_kernel, detect_sources, SourceCatalog
from photutils.background import Background2D, MedianBackground

import galsim
import kcorrect.kcorrect
import photutils

import klm.utils as utils

class Instrument():
	"""Base class for reading spectroscopic data from different instruments
	"""
	def __init__(self):
		pass


class Imaging():
	"""Base class for handling imaging data
	Currently only works with Subaru data
	"""
	def __init__(self, cam='subaru_suprimecam', base_path='../../KL_DEIMOS_data/klens_data/'):
		self.a2261_RA = 260.612917*u.deg
		self.a2261_Dec = 32.133889*u.deg

		if cam == 'subaru_suprimecam':
			# From https://archive.stsci.edu/missions/hlsp/clash/a2261/catalogs/subaru/hlsp_clash_subaru_suprimecam_a2261_catalog_readme.txt
			self.cam = cam
			self.pixscale = 0.2  # arcsec/pix
			self.psfFWHM = 0.57  # arcsec
			self.image_file_path = base_path + 'images/hlsp_clash_subaru_suprimecam_a2261-orient1_rc_2004-v20110514_drz.fits'
			self.weight_file_path = self.image_file_path.replace('drz', 'drz-weight')
			self.wcs = ap.wcs.WCS(fits.open(self.image_file_path)[0].header)
			self.responses = ['subaru_suprimecam_B', 'subaru_suprimecam_V',
			 'subaru_suprimecam_Rc', 'subaru_suprimecam_Ic', 'subaru_suprimecam_z']  # For k-corrections

	@staticmethod
	def _get_abs_magnitude(m, z, cosmology=Planck18):
		"""M = m - 5*log10(d_L/10pc))

		Parameters
		----------
		m : float
			apparent magnitude
		z : float
			redshift
		cosmology : Astropy cosmology object, optional
			Cosmology to be used for computing distance modulus by default Planck18

		Returns
		-------
		_type_
			_description_
		"""
		return m - Planck18.distmod(z).value


	def get_flux(self, RA, Dec, r, coord_units=(u.deg, u.deg)):
		"""Computes the flux of the object at (RA, Dec) in
		a circular aperture of radius r.

		Parameters
		----------
		RA : float or astropy.units object
			Right Ascension
		Dec : _type_
			Declination
		r : float
			Radius of circular aperture (in arcsec)
		"""
		image = fits.open(self.image_file_path)[0].data
		center = ap.coordinates.SkyCoord(RA, Dec, unit=coord_units, frame=FK5, equinox='J2000')
		aperture = photutils.SkyCircularAperture(center, r*u.arcsec)

		photon_table = photutils.aperture_photometry(image, aperture, wcs=self.wcs)
		count = photon_table['aperture_sum'][0]

		return count


	def guessHLR(self, RA, Dec):
		im = self.get_cutout(RA, Dec)
		gal_image = galsim.image.Image(im, scale=self.pixscale)
		HLR = gal_image.calculateHLR()

		return HLR

	def get_cutout(self, RA, Dec, boxsizex=50, boxsizey=52, file='image', return_wcs=False):
		"""From Eric's make_images.py
		Generates image cutout
		To Do: Add file path/filter + instrument specification as argument
		"""
		data = galsim.fits.read(self.__getattribute__(f'{file}_file_path'))
		ap_wcs = ap.wcs.WCS(fits.open(self.__getattribute__(f'{file}_file_path'))[0].header)
		# cut it out.
		coord = galsim.CelestialCoord(RA*galsim.degrees, Dec*galsim.degrees )
		objpos = data.wcs.toImage(coord)
		objbounds = galsim.BoundsI(int(objpos.x - boxsizex/2),
									int(objpos.x + boxsizex/2),
									int(objpos.y - boxsizey/2),
									int(objpos.y + boxsizey/2))

		if file == 'weight':
			image = self.get_cutout(RA, Dec, boxsizex, boxsizey, file='image')
			masked_weights = self.mask_contaminants(image, data[objbounds].array)
			return masked_weights

		elif return_wcs is True:
			return data[objbounds].array, data.wcs, ap_wcs

		else:
			return data[objbounds].array


	def mask_contaminants(self, data, weight):
		'''Detects objects in the image cutout using `photutils.segmentation`.
		The detected sources are used to create a mask, the variance for the masked pixels to a large value.

		Parameters
		----------
		data : 2D-array
			Image cutout

		weight : 2D-array
			Image weight
		'''
		## Subtract median background
		bkg_estimator = MedianBackground()
		bkg = Background2D(data, (10, 10), filter_size=(3, 3), bkg_estimator=bkg_estimator)
		data -= bkg.background  # subtract the background

		## Convolve the image
		kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
		convolved_data = convolve(data, kernel)

		## Threshold for source detection
		threshold = 1.5 * bkg.background_rms

		## Detect sources
		segment_map = detect_sources(convolved_data, threshold, npixels=10)

		cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)

		tbl = cat.to_table()

		## Find the label corresponding to the target
		## Cutout is always centered on the target
		center_x, center_y = data.shape[1]/2, data.shape[0]/2
		ind = np.where(np.isclose(tbl['xcentroid'], center_x, atol=1) & np.isclose(tbl['ycentroid'], center_y, atol=1))[0][0]

		## All other segments are contaminants- mask them
		label_to_mask = np.delete(segment_map.labels, ind)
		cat_subset = cat.get_labels(label_to_mask)  # New catalog from labels to be masked
		sub_tbl = cat_subset.to_table()

		xmin, xmax = sub_tbl['bbox_xmin'], sub_tbl['bbox_xmax']
		ymin, ymax = sub_tbl['bbox_ymin'], sub_tbl['bbox_ymax']

		for i in range(len(label_to_mask)):
			this_xmin, this_xmax = xmin[i], xmax[i]
			this_ymin, this_ymax = ymin[i], ymax[i]

			weight[this_ymin:this_ymax, this_xmin:this_xmax] = 1e10

		return weight

	def get_obj_PA(self, RA, Dec):
		'''Returns the polar angle of the object w.r.t. the cluster center

		Parameters
		----------
		RA : float
			RA
		Dec : float
			Dec
		'''
		RA_s, Dec_s = RA.to(u.radian).value, Dec.to(u.radian).value
		RA_l, Dec_l = (self.a2261_RA).to(u.radian).value, (self.a2261_Dec).to(u.radian).value

		numerator = (Dec_s - Dec_l)/np.cos((Dec_s+Dec_l)/2)
		denominator = (RA_s - RA_l)

		return np.arctan2(numerator, denominator)*u.radian


	def _get_obj_appmag(self, RA, Dec, phot_band, return_error=False):
		'''Finds the closest object in the Subaru Catalog and returns
		the app. magnitude in the desired filter

		Parameters
		----------
		RA : float
			RA (in deg.)
		Dec : float
			Declination (in deg.)
		filter : string
			_description_
		'''
		catalog = np.loadtxt(self.catalog_file_path)
		obj_coord = SkyCoord(ra=RA*u.deg, dec=Dec*u.deg)
		ap_catalog = SkyCoord(ra=catalog[:, 1]*u.deg, dec=catalog[:, 2]*u.deg)
		idx, d2d, _ = obj_coord.match_to_catalog_sky(ap_catalog)
		print(f'Distance to closest object is {d2d[0].to(u.arcsec).value:.4f} arcsec')

		if type(phot_band) is str:
			app_mag = catalog[idx, self.catalog_bands[phot_band]]
			error = catalog[idx, self.catalog_bands[phot_band]+1]

			if return_error is True: return 	app_mag, error
			else: return app_mag

		elif type(phot_band) is list:
			app_mag = [catalog[idx, self.catalog_bands[this_band]] for this_band in phot_band]
			error = [catalog[idx, self.catalog_bands[this_band]+1] for this_band in phot_band]

			if return_error is True: return app_mag, error
			else: return app_mag


	def get_obj_absmag(self, RA, Dec, phot_band, z):
		'''Converts apparant to absolute magnitude
		To Do: Apply K-correction and account for dust extinction

		Parameters
		----------
		RA : float
			RA
		Dec : float
			Dec
		phot_band : str
			photometric band

		Returns
		-------
		_type_
			_description_
		'''
		app_mag, error = self._get_obj_appmag(RA, Dec, list(self.catalog_bands.keys()), return_error=True)
		abs_mag = self._get_abs_magnitude(app_mag, z)
		k_correct = self._get_k_correction(app_mag, error, z)

		abs_mag_k_corrected = abs_mag - k_correct
		index = self.responses.index(f'{self.cam}_{phot_band}')
		return abs_mag_k_corrected[index]

	def _get_k_correction(self, app_mag, error, z):
		self.kc = kcorrect.kcorrect.Kcorrect(responses=self.responses)

		maggies = 10**(-np.array(app_mag)/2.5)
		maggies_var = (np.array(error)*maggies*np.log(10)/2.5)**2
		ivar = 1/maggies_var

		coeffs = self.kc.fit_coeffs(redshift=z, maggies=maggies, ivar=ivar)
		k = self.kc.kcorrect(redshift=z, coeffs=coeffs)

		return k


class Deimos(Instrument):
	def __init__(self, base_path='../../KL_DEIMOS_data/klens_data/'):
		'''Currently, all the object and redshift catalogs are loaded and saved
		when the object is instantiated. Might need to be changed later.

		Parameters
		----------
		zcat_base_path : str, optional
			_description_, by default '../../KL_DEIMOS_data/klens_data/'
		'''
		Instrument.__init__(self)
		self.masks = ['a2261aB', 'a2261b', 'a2261c', 'a2261d']
		self.load_zcat(base_path)
		self.load_allobjcat(base_path)
		self.base_path = base_path

		self.Image = Imaging(base_path=base_path)

		self.FWHM_resolution = 1.7*u.Angstrom  # arcsec per pix (for DEIMOS)
		self.pix_scale = 0.1185

	@staticmethod
	def _assert_specify_obj(mask, slit_name, obj_coords):
		assert (mask is not None and slit_name is not None) or obj_coords is not None, \
		'Must specify either object coordinates or mask and slit name'


	def load_zcat(self, base_path):
		self.zcat = {}
		for mask in self.masks:
			dtypes = [('MASK','<U15'),('SLITNO','<U15'),
						('OBJID','<U15'),('z',float),
						('CONFIDENCE',float), ('OBJTYPE', '<U15')]

			with open(f'{base_path}/{mask}/specpro_{mask}_zinfo.dat') as file:
				contents = file.readlines()

			# Catalog file has inconsistent column delimiter
			# Replace all spaces >4 with 4 spaces, then split to get columns
			modified_content = []
			for line in contents:
				modified_line = re.sub(r'\s{4,}', '    ', line)
				modified_content.append(modified_line.split('   '))

			modified_content = np.array(modified_content)
			modified_content = np.column_stack((modified_content[:, 0:2], modified_content[:, 4:]))

			# Create record array
			modified_content = np.core.records.fromarrays(modified_content.T, dtype=dtypes)

			self.zcat[mask] = modified_content


	def load_allobjcat(self, base_path):
		self.objcat = {}
		for mask in self.masks:
			self.objcat[mask] = fits.open(f'{base_path}/{mask}/{mask}.bintabs.fits')


	def _get_objindex_from_slitname(self, mask, slit_name):
		'''Returns the index of the object in the mask catalog given the slit name
		Need to map slit_name -> slit_id -> object_id -> index
		'''
		if isinstance(slit_name, str):
			this_cat = self.objcat[mask]
			slit_id = this_cat[3].data['DSLITID'][this_cat[3].data['SLITNAME']==slit_name][0]
			obj_id = this_cat[4].data['OBJECTID'][this_cat[4].data['DSLITID']==slit_id][0]
			index = np.where(this_cat[1].data['OBJECTID']==obj_id)[0][0]

			return index

		if isinstance(slit_name, list) or isinstance(slit_name, np.ndarray):
			index = []
			for name in slit_name:
				this_cat = self.objcat[mask]
				slit_id = this_cat[3].data['DSLITID'][this_cat[3].data['SLITNAME']==name][0]
				obj_id = this_cat[4].data['OBJECTID'][this_cat[4].data['DSLITID']==slit_id][0]
				index.append(np.where(this_cat[1].data['OBJECTID']==obj_id)[0][0])

			return index



	def _load_obj_spec(self, mask, slit_name):
		'''Loads processed spectra already saved on disk
		'''
		try:
			return joblib.load(f'{self.base_path}{mask}.{slit_name}.pkl')

		except FileNotFoundError:
			pass

		try:
			return h5py.File(f'{self.base_path}{mask}.{slit_name}.h5')

		except :
			raise FileNotFoundError(f'''Spectra for slit name:{slit_name} in mask:{mask} does not exist! \n

Did not find {f'{self.base_path}{mask}.{slit_name}.pkl'} or {f'{self.base_path}{mask}.{slit_name}.h5'}!''')


	def getCoords_from_slit_name(self, mask, slit_name):
		'''Returns RA and DEC of the object from catalog given slit mask and slit name

		Parameters
		----------
		mask : str
			mask name e.g, a2261aB
		slit_name : str
			slit # e.g, '024'

		Returns
		-------
		np.array
			[RA, DEC]
		'''
		index = self._get_objindex_from_slitname(mask, slit_name)

		ra, dec = self.objcat[mask][1].data['RA_OBJ'][index], self.objcat[mask][1].data['DEC_OBJ'][index]

		if isinstance(ra, float):
			return np.array([ra, dec])

		else:
			return np.column_stack((ra, dec))

	def get_redshift_from_slit_name(self, mask, slit_name):
		this_zcat = self.zcat[mask]
		this_objcat = self.objcat[mask]

		# Get redshift based on slit no. of the object
		assert this_zcat['CONFIDENCE'][this_zcat['SLITNO']==slit_name] > 2, 'Redshift for object not known precisely!'
		redshift = this_zcat['z'][this_zcat['SLITNO']==slit_name][0]

		return redshift


	def _isMultipleSlitObs(self, mask=None, slit_name=None, obj_coords=None):
		'''Checks if an object is present in multiple slit masks
			Returns a dict of masks (and slit name) in which the object is present
		'''
		self._assert_specify_obj(mask, slit_name, obj_coords)

		if obj_coords is not None:
			ra, dec = obj_coords[0], obj_coords[1]

		elif mask is not None and slit_name is not None:
			ra, dec = self.getCoords_from_slit_name(mask, slit_name)


		index = {}
		for this_mask, this_cat in self.objcat.items():
			these_ra, these_dec = this_cat[1].data['RA_OBJ'], this_cat[1].data['DEC_OBJ']
			this_index = np.where((np.isclose(ra, these_ra)) & (np.isclose(dec, these_dec)))[0]

			if len(this_index)>0:
				obj_id = this_cat[1].data['OBJECTID'][this_index[0]]
				dslit_id = this_cat[4].data['DSLITID'][this_cat[4].data['OBJECTID']==obj_id][0]
				slit_index = np.where(this_cat[3].data['DSLITID']==dslit_id)[0][0]

				## Check if this slit exists in the redshift catalog
				## If not then probably doesn't have emission lines anyway
				if this_cat[3].data['SLITNAME'][slit_index] in self.zcat[this_mask]['SLITNO']:
					index[this_mask] = this_cat[3].data['SLITNAME'][slit_index]  # Object cannot be present >1 time in a mask!

		if len(index.keys())==1:
			print(f'Only one spectra for object at RA:{ra}, Dec:{dec}')
			return False, index

		else:
			print(index)
			return True, index

	def subtract_continuum(self, spec):
		'''Estimate continuum by fitting a Gaussian profile.
		axis=0 of the spectrum should correspond to the spatial dimension
		Note: Currently uses a median estimator, might be more robust to sky lines

		Parameters
		----------
		spec : 2D array
			Spectrum

		Returns
		-------
		2D array
			Continuum subtracted spectra, continuum model
		'''
		column_index = int(spec.shape[0]/4)  # Use first and last n columns to fit Gaussian
		profiles = np.column_stack((spec[:,:column_index], spec[:, -column_index:]))

		y = np.median(profiles, axis=1)
		x = utils.build_1d_grid(len(y), self.pix_scale)

		# Initial parameter guess
		initial_guess = [12, 4, 3]  # amplitude, mu, sigma
		bounds = [(0, -10, 0), (100, 10, 5)]
		fit_params, _ = curve_fit(gaussian, x, y, p0=initial_guess, bounds=bounds)

		cont1D = gaussian(x, *fit_params)
		new_spec = spec-cont1D[:, np.newaxis]

		return new_spec, cont1D


	def get_mean_psfFWHM_mask(self, mask, pixScale=0.1185):
		'''Computes the mean psf FWHM from alignment calibration stars
		Assumes star and slit are perfectly aligned, currently does not take the
		spectrum covariance into account when fitting Gaussian profile

		Note: Not clear how to account for spatially varying psf
		Parameters
		----------
		mask : str
			Slit mask for which to compute psf
		'''

		calib_star_files = sorted(glob.glob(f'{self.base_path}Star_{mask}.*h5'))
		sigma = []

		for f in calib_star_files:
			this_calib_star = h5py.File(f)
			for line in list(this_calib_star.keys())[1:]:
				y = np.sum(this_calib_star[line]['flux'], axis=0)
				x = utils.build_1d_grid(len(y), pixScale)
				initial_guess = [1e5, 0, 0.5]  # Initial parameter guess
				fit_params, _ = curve_fit(gaussian, x, y, p0=initial_guess)
				sigma.append(fit_params[-1])

			this_calib_star.close()

		FWHM = 2*np.sqrt(2*np.log(2))*np.mean(sigma)
		return FWHM


	def get_data_info(self, mask=None, slit_name=None, obj_coords=None, spec_inds=[1]):
		'''Reads specified spectra given the file path.
  		Packs the spectra, image, variance and relevant meta data
		into a dictionary.
		Currently only reads the first emission line for a given object i.e, index=0
		May need to be changed later

		To Do: Update image pix scale, currently set to Keck

		Parameters
		----------
		file_path : str
			Path to the object spectra, spectra to be computed from raw data using
			make_images.py

		Returns
		-------
		dict
			_description_
		'''
		self._assert_specify_obj(mask, slit_name, obj_coords)
		# _, observation_dict = self._isMultipleSlitObs(mask, slit_name, obj_coords)

		all_data = {}
		all_data['spec'] = []
		# for this_mask, this_slit in observation_dict.items():
		this_mask, this_slit = mask, slit_name
		for i in spec_inds:
			data_info_spec, par_obj = self._get_data_info_single_mask(this_mask, this_slit, i)
			all_data['spec'].append(data_info_spec)


		# Image dict
		RA_obj, Dec_obj = self.getCoords_from_slit_name(this_mask, this_slit)
		phot_band = 'B'
		image, galsim_wcs, ap_wcs = self.Image.get_cutout(RA_obj, Dec_obj, 60, return_wcs=True)
		bkg_var = self.Image.get_cutout(RA_obj, Dec_obj, 60, file='weight')

		### Convert from ADU to electron counts
		GAIN = 2250  # e-/ADU : From Subaru image header
		image = image*GAIN
		bkg_var = bkg_var*GAIN**2
		image_var = bkg_var + image  # Variance = background variance + Poisson noise

		all_data['image'] = {}
		all_data['image']['data'] =	image
		all_data['image']['var'] = image_var
		all_data['image']['sky_var'] = bkg_var
		all_data['image']['par_meta'] = {'ngrid': all_data['image']['data'].shape,
											'pixScale': self.Image.pixscale,
											'psfFWHM': self.Image.psfFWHM,
											'wcs': galsim_wcs,
											'ap_wcs': ap_wcs,
											'RA': par_obj['RA'].value,
											'Dec': par_obj['Dec'].value}

		all_data['galaxy'] = {}
		all_data['galaxy']['RA'] = par_obj['RA']
		all_data['galaxy']['Dec'] = par_obj['Dec']
		all_data['galaxy']['redshift'] = par_obj['redshift']
		# all_data['galaxy']['Mag'] = self.Image.get_obj_absmag(RA_obj, Dec_obj, phot_band, par_obj['redshift'])
		# all_data['galaxy']['phot_band'] = phot_band
		Mstar, log10_Mstar_err = get_stellar_mass(this_mask, this_slit)
		all_data['galaxy']['log10_Mstar'] = np.log10(Mstar)
		all_data['galaxy']['log10_Mstar_err'] = log10_Mstar_err
		all_data['galaxy']['beta'] = self.Image.get_obj_PA(par_obj['RA'], par_obj['Dec'])

		return all_data


	def _get_data_info_single_mask(self, mask, slit_name, i):
		this_data = self._load_obj_spec(mask, slit_name)

		this_zcat = self.zcat[mask]
		this_objcat = self.objcat[mask]

		# Get redshift based on slit no. of the object
		assert this_zcat['CONFIDENCE'][this_zcat['SLITNO']==slit_name] > 2, 'Redshift for object not known precisely!'
		redshift = this_zcat['z'][this_zcat['SLITNO']==slit_name][0]


		# Now get object/slit data from object catalog
		obj_index = self._get_objindex_from_slitname(mask, slit_name)
		RA_obj = this_objcat[1].data[obj_index]['RA_OBJ']*u.deg
		Dec_obj = this_objcat[1].data[obj_index]['DEC_OBJ']*u.deg

		# app_mag = this_objcat[1].data[index]['MAG']
		# mag_band = this_objcat[1].data[index]['PBAND']
		slit_index = np.where(this_objcat[3].data['SLITNAME']==slit_name)[0][0]
		slit_RA = this_objcat[3].data[slit_index]['SLITRA']*u.deg
		slit_Dec = this_objcat[3].data[slit_index]['SLITDec']*u.deg
		slit_len = this_objcat[3].data[slit_index]['SLITLEN']
		slit_width = this_objcat[3].data[slit_index]['SLITWID']
		slit_LPA = this_objcat[3].data[slit_index]['SLITLPA']*u.deg
		slit_WPA = this_objcat[3].data[slit_index]['SLITWPA']*u.deg

		# slit_LPA is degrees East of North
		# We want to w.r.t. E
		slit_LPA = 90*u.deg - slit_LPA

		# Now get image cutout
		# The [3:-3, :] gets ride of artefacts near the slit edge
		spec, cont_model = self.subtract_continuum(np.array(this_data[f'spec{i}']['flux']).T[3:-3, :])
		lambda_grid = np.array(this_data[f'spec{i}']['lambda']).T[3:-3, :]
		var = 1/np.array(this_data[f'spec{i}']['ivar']).T[3:-3, :]

		data_info = {}
		par_obj = {}
		par_obj = {'redshift': redshift,
					'RA': RA_obj,
					'Dec': Dec_obj
					}

		data_info['par_meta'] = {'line_species':this_data['lines'][i-1][0].decode("utf-8"),  # need i-1 since i is 1-indexed and lines is 0-indexed
								'lambda_grid': lambda_grid*u.Angstrom,
								'pixScale': self.pix_scale,
								'ngrid': spec.shape,
								'FWHMresolution': self.FWHM_resolution,  # in Angstrom
								'psfFWHM': self.get_mean_psfFWHM_mask(mask),
								'slitRA': slit_RA,
								'slitDec': slit_Dec,
								'slitWidth': slit_width,
								'slitLen': slit_len,
								'slitLPA': slit_LPA,
								'slitWPA': slit_LPA + 90*u.deg  # Assume rectangular slit
								}

		data_info['data'] = spec
		data_info['cont_model'] = cont_model
		data_info['var'] = var

		return data_info, par_obj

class Manga(Instrument):
	def __init__(self, cat_path='../', base_path='../../KL_DEIMOS_data/klens_data/'):
		Instrument.__init__(self)
		with fits.open(cat_path+'/MaNGA_DR17_Kinematic_Ristea23_v1fits') as f:
			self.catalog = f[1].data

		self.drpall = fits.open(cat_path+'drpall-v3_1_1.fits')
		self.base_path = base_path
		self.base_path_MAP  =  base_path + 'sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/HYB10-MILESHC-MASTARSSP/'
		self.base_path_cube  =  base_path + 'sas/dr17/v3_1_1/Cube/'
		self.sdss_pixscale = 0.39  # arcsec/pix (used for image cutouts)
		# Typical detector gain; from https://data.sdss.org//datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html
		self.gain_dict = {'r': {1: 4.71, 2: 4.6, 3: 4.72, 4: 4.76, 5: 4.725, 6: 4.895}}

	def get_image(self, RA, Dec, band='r', boxsize=None, n=6, filename=None, use_cache=True):
		pos = SkyCoord(ra=RA, dec=Dec,unit='deg')

		if filename is None:
			filename = f'{RA}_{Dec}_{band}'

		download = True
		if use_cache is True:
			try:
				HDUlist = [fits.open(f'{self.base_path}/sas/sdss/{filename}.fits')]
				download = False
				# print(f'Getting cutout for mangaid {filename} from cache...')

			except FileNotFoundError:
				print('Tried to use cache but file not found...downloading image...')

		if download is True:
			result = SDSS.query_region(pos)
			HDUlist = SDSS.get_images(matches=result,band=band)

		# Determine size of image cutout
		if boxsize is None:
			if n<4:
				return
			idx = self.drpall[1].data['mangaid']==filename
			# out of NSA_ELPETRO_TH50_R and NSA_PETRO_TH50, select smaller one
			R_elpetro = self.drpall[1].data[idx]['NSA_ELPETRO_TH50_R']
			if R_elpetro.size>1:
				print(f'Multiple objects with same mangaid: {filename}\n Using mean R')

			R = np.mean(R_elpetro)

			# By default creates a box 6x times the HLR
			boxsize = [int(R/self.sdss_pixscale)*n, int(R/self.sdss_pixscale)*n]

		# Create cutouts
		for i in range(len(HDUlist)):
			data = galsim.fits.read(hdu_list=HDUlist[i])

			coord = galsim.CelestialCoord(RA*galsim.degrees, Dec*galsim.degrees)
			objpos = data.wcs.toImage(coord)

			try:
				objbounds = galsim.BoundsI(int(objpos.x - boxsize[0]/2),
										int(objpos.x + boxsize[0]/2),
										int(objpos.y - boxsize[1]/2),
										int(objpos.y + boxsize[1]/2))

				# We also need astropy WCS to determine
				# pixel scale
				ap_wcs = ap.wcs.WCS(HDUlist[i][0].header)

				# Convert to electron counts
				NMGY = HDUlist[i][0].header['NMGY']  # used to convert flux to ADU
				CAMCOL = HDUlist[i][0].header['CAMCOL']
				gain = self.gain_dict[band][CAMCOL]
				flux_to_electron_factor = 1/NMGY * gain

				mean_bkg_ADU = np.mean(HDUlist[i][2].data['ALLSKY'])
				mean_bkg_electron = mean_bkg_ADU * gain

				meta_dict = {'NMGY': NMGY,
				 			'gain': gain,
							'flux_to_electron_factor': flux_to_electron_factor,
							'mean_bkg_electron': mean_bkg_electron}

				image = data[objbounds].array*flux_to_electron_factor
				if download is True:
					HDUlist[i].writeto(f'/Users/pranjalrs/sas/sdss/{filename}.fits', overwrite=True)

				# import ipdb; ipdb.set_trace()
				return image, data.wcs, ap_wcs, meta_dict

			except galsim.errors.GalSimBoundsError:
				print('Object close to frame edge...trying next image frame...')
				continue

		print(f'Trying smaller bounding box...{n-1}x HLR')
		return self.get_image(RA=RA, Dec=Dec, band=band, n=n-1, filename=filename, use_cache=use_cache)

	@staticmethod
	def mask_contaminants(data, weight):
		''' Copied from `Image` class
		Detects objects in the image cutout using `photutils.segmentation`.
		The detected sources are used to create a mask, the variance for the masked pixels to a large value.

		Parameters
		----------
		data : 2D-array
			Image cutout

		weight : 2D-array
			Image weight
		'''
		## Subtract median background
		bkg_estimator = MedianBackground()
		bkg = Background2D(data, (10, 10), filter_size=(3, 3), bkg_estimator=bkg_estimator)
		# data -= bkg.background  # subtract the background

		## Convolve the image
		kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
		convolved_data = convolve(data, kernel)

		## Threshold for source detection
		cat = 0.
		factor_threshold = 20
		while cat == 0:
			if factor_threshold < 0:
				print('Factor threshold negative...')
				raise AssertionError
			try:
				threshold = factor_threshold * np.mean(bkg.background_rms)
				segment_map = detect_sources(convolved_data, threshold, npixels=8)

				cat = SourceCatalog(data, segment_map, convolved_data=convolved_data)
				tbl = cat.to_table()

			except TypeError:
				# Probably the threshold is too high
				factor_threshold -= 1


		## Find the label corresponding to the target
		## Cutout is always centered on the target
		center_x, center_y = data.shape[1]/2, data.shape[0]/2

		#------- Only the part below is different from the implementation in Image class
		ind = np.where(np.isclose(tbl['xcentroid']/center_x-1, 0, atol=0.2) & np.isclose(tbl['ycentroid']/center_y-1, 0, atol=0.2))[0]

		if len(ind)==0:
			print(('No object detected in the cutout...'))
			raise AssertionError

		if len(ind)>1:
			print(('Multiple objects detected in the cutout...'))
			raise AssertionError

		ind = ind[0]

		#-------

		## All other segments are contaminants- mask them
		label_to_mask = np.delete(segment_map.labels, ind)
		cat_subset = cat.get_labels(label_to_mask)  # New catalog from labels to be masked
		sub_tbl = cat_subset.to_table()

		xmin, xmax = sub_tbl['bbox_xmin'], sub_tbl['bbox_xmax']
		ymin, ymax = sub_tbl['bbox_ymin'], sub_tbl['bbox_ymax']

		for i in range(len(label_to_mask)):
			this_xmin, this_xmax = xmin[i], xmax[i]
			this_ymin, this_ymax = ymin[i], ymax[i]

			if data.shape[0]-np.round(this_xmax)<2:
				this_xmax = None

			if data.shape[1]-np.round(this_ymax)<2:
				this_ymax = None
			weight[this_ymin:this_ymax, this_xmin:this_xmax] = 1e14

		return weight

	def get_data_info(self, mangaid):
		'''The output 2D velocity maps are such that +RA is +x axis
		and +Dec is +y axis.
		The image array is not changed- use wcs to get the correct orientation

		Parameters
		----------
		mangaid : int
			Manga ID of the object

		Returns
		-------
		dict
			Dictionary containing image, galaxy and velocity map information
		'''
		idx = self.catalog['mangaid'] == mangaid
		this_obj = self.catalog[idx][0]
		plateID, ifuID = this_obj['plateifu'].split('-')

		# 2D MAP for velocity field
		filename = self.base_path_MAP+ f'{plateID}/{ifuID}/manga-{plateID}-{ifuID}-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz'
		MAP_data = fits.open(filename)

		# 3D cube for PSF
		filename = self.base_path_cube + f'/manga-{plateID}-{ifuID}-LOGCUBE.fits.gz'
		cube_data = fits.open(filename)

		all_info = {}
		all_info['image'] = {}
		# (Nspec, Nx, Ny)
		# flux = cube_data[1].data
		# ivar = cube_data[2].data

		# mean_flux = np.sum(flux*ivar, axis=0)/np.sum(ivar, axis=0)
		# new_var = 1/np.sum(ivar, axis=0)

		# mean_flux[~np.isfinite(mean_flux)] = 0.
		try:
			image, galsim_wcs, ap_wcs, header_info = self.get_image(this_obj['RA'], this_obj['DEC'], filename=mangaid)
		except TypeError:
			print('Failed to get image for mangaid:', mangaid)
			print('Data info not saved...')
			return
		pix_scale = ap.wcs.utils.proj_plane_pixel_scales(ap_wcs)
		NMGY = header_info['NMGY']
		gain = header_info['gain']
		flux_to_electron_factor = header_info['flux_to_electron_factor']
		## Convert pixel scale to arcsec/pix
		pix_scale[0] = pix_scale[0]*ap_wcs.wcs.cunit[0].to(u.arcsec)
		pix_scale[1] = pix_scale[1]*ap_wcs.wcs.cunit[1].to(u.arcsec)

		# Compute image variance: this is Poisson noise from object + background counts
		image_variance = image + header_info['mean_bkg_electron']**2

		try:
			image_variance = self.mask_contaminants(image, image_variance)

		except AssertionError:
			print(f'Failed to mask contaminants for {mangaid}')

		all_info['image']['data'] =	image
		all_info['image']['var'] = image_variance
		all_info['image']['par_meta'] = {'ngrid': image.shape,
										'pixScale': pix_scale,
		 								'psfFWHM': 1.32,  #Median seeing in the r-band https://www.sdss4.org/dr17/imaging/other_info/
										'wcs': galsim_wcs,
										'ap_wcs': ap_wcs,
										'RA': this_obj['RA'],
										'Dec': this_obj['Dec'],
										'NMGY': NMGY,
										'gain': gain,
										'flux_to_electron_factor': flux_to_electron_factor,
										'mean_bkg': header_info['mean_bkg_electron']}


		all_info['galaxy'] = {}
		all_info['galaxy']['RA'] = this_obj['RA']
		all_info['galaxy']['Dec'] = this_obj['DEC']
		all_info['galaxy']['redshift'] = MAP_data[0].header['SCINPVEL']/299792
		all_info['galaxy']['log10_Mstar'] = this_obj['log_Mstar']
		all_info['galaxy']['Rmax_ST'] = this_obj['Rmax_ST']
		all_info['galaxy']['Rmax_G'] = this_obj['Rmax_G']
		all_info['vmap'] = {}

		#------------------ Masking Spaxels ------------------#
		# Make channel dictionary
		# From: https://www.sdss4.org/dr17/manga/manga-tutorials/dap-tutorial/dap-python-tutorial/#Makinganemission-linemapandapplyingmasks
		emline = {}
		for k, v in MAP_data['EMLINE_GFLUX'].header.items():
			if k[0] == 'C':
				try:
					i = int(k[1:])-1
				except ValueError:
					continue
				emline[v] = i
		# Mask spaxels with low S/N
		Halpha_flux = np.flip(MAP_data['EMLINE_GFLUX'].data[emline['Ha-6564']].copy())
		flux_mask_extension = MAP_data['EMLINE_GFLUX'].header['QUALDATA']
		flux_ivar_extension = MAP_data['EMLINE_GFLUX'].header['ERRDATA']
		Halpha_ivar = np.flip(MAP_data[flux_ivar_extension].data[emline['Ha-6564']].copy())
		Halpha_mask = -1/~np.flip(MAP_data[flux_mask_extension].data[emline['Ha-6564']].copy())  # Only pixels with zero mask value are good
		Halpha_anr = np.flip(MAP_data['EMLINE_GANR'].data[emline['Ha-6564']].copy())
		# Halpha_snr = Halpha_flux*Halpha_ivar**0.5

		Halpha_snr_mask = np.ones_like(Halpha_mask)
		Halpha_snr_mask[Halpha_anr<5] = 0.

		# Velocity error mask for gas
		vmap_gas_data = np.flip(MAP_data['EMLINE_GVEL'].data[emline['Ha-6564']])
		vmap_gas_var = 1/np.flip(MAP_data['EMLINE_GVEL_IVAR'].data[emline['Ha-6564']]) + 25  # Add error of 5km/s in quadrature
		vmap_gas_mask = -1/~np.flip(MAP_data['EMLINE_GVEL_MASK'].data[emline['Ha-6564']])
		max_vmap_err = 0.5 * np.abs(vmap_gas_data) + 15

		vmap_gas_err_mask = np.ones_like(vmap_gas_mask)
		vmap_gas_err_mask[vmap_gas_var**0.5>max_vmap_err] = 0.0
		
		PSF_vmap = np.flip(cube_data['RPSF'].data)
		#------------------ Save data info: Stellar Vmap ------------------#

		# The raw velocity map arrays are such that +RA is -x axis and -Dec is +y axis
		# Below we flip so that +RA is +x axis and +Dec is +y axis
		## Stellar kinematics
		vmap_stellar = {}
		vmap_stellar['par_meta'] = {'RA_grid': np.flip(MAP_data[1].data[0, :, :]),
								'Dec_grid': np.flip(MAP_data[1].data[1, :, :]),
								'r': np.flip(MAP_data[2].data[0, :, :]),
								'theta': np.flip(MAP_data[2].data[3, :, :])}

		vmap_stellar['data'] = np.flip(MAP_data['STELLAR_VEL'].data)
		vmap_stellar['var'] = 1/np.flip(MAP_data['STELLAR_VEL_IVAR'].data) + 25  # Add error of 5km/s in quadrature
		vmap_stellar['mask'] = -1/~np.flip(MAP_data['STELLAR_VEL_MASK'].data)  # Invert bit mask so that masked regions have weight zero

		all_info['vmap']['stellar'] = vmap_stellar


		#------------------ Save data info: Gas Vmap ------------------#
		vmap_gas = {}
		vmap_gas['par_meta'] = {'RA_grid': np.flip(MAP_data[1].data[0, :, :]),
								'Dec_grid': np.flip(MAP_data[1].data[1, :, :]),
								'r': np.flip(MAP_data[2].data[0, :, :]),
								'theta': np.flip(MAP_data[2].data[3, :, :]),
								'psf': PSF_vmap,
								'Halpha_flux': Halpha_flux,
								'Halpha_flux_conv': scipy.signal.convolve2d(Halpha_flux, PSF_vmap, mode='same')}

		vmap_gas['data'] = vmap_gas_data
		vmap_gas['var'] = vmap_gas_var

		# Update masks
		vmap_gas['default_mask'] = vmap_gas_mask # Invert bit mask so that masked regions have weight zero
		vmap_gas['Halpha_snr_mask'] = Halpha_snr_mask
		vmap_gas['verr_mask'] = vmap_gas_err_mask
		vmap_gas['mask'] = vmap_gas_mask * Halpha_snr_mask * vmap_gas['verr_mask']
		all_info['vmap']['gas'] = vmap_gas

		return all_info


def gaussian(x, amplitude, mean, std_dev):
	return amplitude * np.exp(-(x - mean)**2 / (2 * std_dev**2))

def get_stellar_mass(mask, slit, imf=1, dust=4):
	'''
	Retrieve the stellar mass for a given mask and slit.

	Parameters:
	- mask (str): The mask identifier.
	- slit (int): The slit number.

	Returns:
	- list: A list containing the stellar mass and error in log10 stellar mass.

	'''
	with open(f'../data/stellar_mass_imf{imf}_dust{dust}.json') as f:
		stellar_mass_dict = json.load(f)

	try:
		return stellar_mass_dict[mask][slit]

	except KeyError:
		print(f'No stellar mass found for mask {mask} and slit {slit}.')
		return 0., None
