'''functions for read in photometric data from NED query
'''

import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.coordinates import FK5, SkyCoord
import astropy.units as u
import pyphot


### Filters used for SED fitting
subaru_filters = ['subaru_suprimecam_B', 'subaru_suprimecam_V', 'subaru_suprimecam_Rc']
subaru_filters += ['sdss_{0}0'.format(b) for b in 'iz']  # Mayall

wfc3_uvis = ['f225w', 'f275w', 'f336w', 'f390w', 'f475w', 'f606w', 'f814w']
wfc3_ir = ['f105w', 'f110w', 'f125w', 'f140w', 'f160w']
acs_wfc = ['f435w', 'f625w', 'f775w', 'f850lp']

HST_filters = [f'wfc3_uvis_{f}' for f in wfc3_uvis]
HST_filters += [f'wfc3_ir_{f}' for f in wfc3_ir]
HST_filters += [f'acs_wfc_{f}' for f in acs_wfc]

IRAC_filters = ['spitzer_irac_ch1', 'spitzer_irac_ch2', 'spitzer_irac_ch3', 'spitzer_irac_ch4']
WISE_filters = ['wise_w1', 'wise_w2', 'wise_w3']

filterset = {'B': 'subaru_suprimecam_B',
			'V': 'subaru_suprimecam_V',
			'Rc': 'subaru_suprimecam_Rc',
			'i': 'sdss_i0',
			'z': 'sdss_z0',
			'IRAC_3.6': 'spitzer_irac_ch1', 
			'IRAC_4.5': 'spitzer_irac_ch2',
			'IRAC_5.8': 'spitzer_irac_ch3',
			'IRAC_8.0': 'spitzer_irac_ch4',
			'WISE_W1': 'wise_w1',
			'WISE_W2': 'wise_w2',
			'WISE_W3': 'wise_w3',
			'WISE_W4': 'wise_w4'}

#### Filters used for converting infrared flux to AB magnitude
lib = pyphot.get_library()

pyphot_bands = {'IRAC_3.6': 'SPITZER_IRAC_36', 
				'IRAC_4.5': 'SPITZER_IRAC_45',
				'IRAC_5.8': 'SPITZER_IRAC_58',
				'IRAC_8.0': 'SPITZER_IRAC_80',
				'WISE_W1': 'WISE_RSR_W1',
				'WISE_W2': 'WISE_RSR_W2',
				'WISE_W3': 'WISE_RSR_W3',
				'WISE_W4': 'WISE_RSR_W4'}


def get_photometry_infrared(mask, slit,band):
	all_mag, all_error = [], []
	all_filter = []

	flux = _get_flux_infrared(mask, slit,band=band)

	for instrument in flux.keys():
		for i, channel in enumerate(flux[instrument][:, 0]):
			filter_for_flux_to_mag = pyphot_bands[f'{instrument}_{channel}']
			this_channel_flux = float(flux[instrument][i, 1])
			this_channel_flux_error = float(flux[instrument][i, 2])

			f0 = lib[filter_for_flux_to_mag].AB_zero_Jy.value  # channel zero_point

			mag = -2.5*np.log10(this_channel_flux/f0)
			error = 2.5*this_channel_flux_error/this_channel_flux/np.log(10)

			if np.isfinite(mag) and np.isfinite(error):
				all_filter.append(filterset[f'{instrument}_{channel}'])
				all_mag.append(mag)
				all_error.append(error)

	return all_filter, all_mag, all_error

def get_photometry_Subaru(RA, Dec, base_path):
		"""Finds the closest object in the Subaru Catalog and returns
		the app. magnitude in the desired filter

		Parameters
		----------
		RA : float
			RA (in deg.)
		Dec : float
			Declination (in deg.)
		filter : string
			_description_
		"""
		catalog_file_path = base_path + 'images/hlsp_clash_subaru_suprimecam_a2261_cat.txt'
		passbands = {'B': 6, 'V': 8, 'Rc': 10, 'i': 12, 'z':14}

		catalog = np.loadtxt(catalog_file_path)
		obj_coord = SkyCoord(ra=RA*u.deg, dec=Dec*u.deg)
		ap_catalog = SkyCoord(ra=catalog[:, 1]*u.deg, dec=catalog[:, 2]*u.deg)
		idx, d2d, _ = obj_coord.match_to_catalog_sky(ap_catalog)
		print(f'Distance to closest object is {d2d[0].to(u.arcsec).value:.4f} arcsec')
		
		app_mag = [catalog[idx, passbands[this_band]] for this_band in passbands.keys()]
		error = [catalog[idx, passbands[this_band]+1] for this_band in passbands.keys()]

		return subaru_filters, app_mag, error


def get_photometry_HST(RA, Dec):
	HST_cat_data = fits.open('/xdisk/timeifler/pranjalrs/CLASH_HST_catalog.fits')
	HST_catalog = SkyCoord(HST_cat_data[1].data['RAdeg']*u.deg, HST_cat_data[1].data['DEdeg']*u.deg)
	target = SkyCoord(RA*u.deg, Dec*u.deg)
	idx, d2d, _ = target.match_to_catalog_sky(HST_catalog)
	d2d = d2d[0]
	if d2d.to(u.arcsec).value>1: raise Exception(f'Closest object is at a separation of {d2d.to(u.arcsec).value:.4f} arcsec!')

	this_row_in_HST_cat = HST_cat_data[1].data[idx]

	mag, error = [], []
	filters = []
	for this_filter in HST_filters:
		f = this_filter.split('_')[-1]
		if this_row_in_HST_cat[f.upper()+'mag']>0.0:
			mag.append(this_row_in_HST_cat[f.upper()+'mag'])
			error.append(this_row_in_HST_cat['e_'+f.upper()+'mag'])
			filters.append(this_filter)
	
	return filters, mag, error


def _get_flux_infrared(mask, slit, skiprows=16, band='all'):
	name = str(slit)+mask[5:]
	try:
		path = f'../data/NED_query/{name}.txt'
		data = pd.read_table(path, delimiter='|', skiprows=skiprows)
	except:
		raise FileNotFoundError(f'File at {path} does not exist!')

	passbands = np.unique(np.array(data['Observed Passband']))  # Unique bandpasses

	# IRAC data
	flux_irac = []
	error_irac = []
	channel_irac = []
	if np.any(['IRAC' in b for b in passbands]):
		print('Gathering Spitzer (IRAC) data...')
		for channel in ['3.6', '4.5', '5.8', '8.0']:
			channel_idx = get_band(channel, data['Observed Passband'])

			if len(channel_idx)==0: continue

			elif len(channel_idx)==1:
				flux_irac.append(data['Flux Density'][channel_idx].values[0])
				error_irac.append(data['NED Uncertainty'][channel_idx].values[0])

			else:  # If multiple fluxes choose
				if channel == '24':
					this_flux_id = is_in('PSF fit', data['Qualifiers'][channel_idx], return_idx=True)

				else:
					this_flux_id = is_in('3.8" aperture', data['Qualifiers'][channel_idx], return_idx=True)

				flux_irac.append(data['Flux Density'][channel_idx].values[this_flux_id])
				error_irac.append(data['NED Uncertainty'][channel_idx].values[this_flux_id][3:])
				assert data['NED Units'][channel_idx].values[this_flux_id] == 'Jy', 'Flux not in Jy!'
			
			channel_irac.append(channel)
	

	# WISE data
	flux_wise = []
	error_wise = []
	channel_wise = []
	if np.any(['WISE' in b for b in passbands]):
		print('Gathering WISE data...')

		for channel in ['W1', 'W2', 'W3', 'W4']:
			channel_idx = get_band(channel, data['Observed Passband'])

			if len(channel_idx)==0: continue

			elif len(channel_idx)==1:
				flux_wise.append(data['Flux Density'][channel_idx].values[0])
				error_wise.append(data['NED Uncertainty'][channel_idx].values[0])

			else:  # If multiple fluxes choose profile-fit
				this_flux_id = is_in('Profile-fit', data['Qualifiers'][channel_idx], return_idx=True)

				flux_wise.append(data['Flux Density'][channel_idx].values[this_flux_id])
				error_wise.append(data['NED Uncertainty'][channel_idx].values[this_flux_id][3:])  #exclude the +/- with [3:]

				assert data['NED Units'][channel_idx].values[this_flux_id] == 'Jy', 'Flux not in Jy!'

			channel_wise.append(channel)
	
	if band=='Spitzer':
		flux  = {'IRAC':np.column_stack((channel_irac, flux_irac, error_irac))}
		
	elif band=='WISE':
		flux  = {'WISE': np.column_stack([channel_wise, flux_wise, error_wise])}

	else:
		flux  = {'IRAC':np.column_stack((channel_irac, flux_irac, error_irac)),
				'WISE': np.column_stack([channel_wise, flux_wise, error_wise])}

	return flux
			

def is_in(this, these, return_idx=False):
	for i, item in enumerate(these):
		if this in item:
			if return_idx is False:
				return True
			else:
				return i

	return False

def get_band(this_band, all_bands):
	id_list = []
	for i, item in enumerate(all_bands):
		if this_band in item:
			id_list.append(i)
	
	return id_list
