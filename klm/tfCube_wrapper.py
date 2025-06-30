import numpy as np
import sys

import astropy.units as u

sys.path.append('../../')
sys.path.append('../../BinnedFit')

from binnedFit_utilities import cal_e_int, cal_theta_obs
from KLtool import find_flux_norm
from tfCube2 import TFCube


def gen_mock_data(pars, line_species='Halpha', axis='both', delta=0.0, inoise=0, return_vmap=False):
	pars['flux'] = find_flux_norm(pars, R=1.5)

	eint_thy = cal_e_int(sini=pars['sini'], q_z=pars['aspect'])
	theta_obs = cal_theta_obs(g2=pars['g2'], e_int=eint_thy, theta_int=pars['theta_int'])

	slitAng_major_p = theta_obs
	slitAng_minor_p = theta_obs + np.pi / 2.
	
	if axis == 'major':
		pars['slitAngles'] = np.array([slitAng_major_p + delta])

	elif axis == 'minor':
		pars['slitAngles'] = np.array([slitAng_minor_p + delta])

	elif axis == 'both':
		print(f'Original slit angles are (in deg): {slitAng_major_p*180/np.pi:.2f}, {slitAng_minor_p*180/np.pi:.2f}')
		print(f'Offset is (in deg): {delta*180/np.pi:.2f}')
		pars['slitAngles'] = np.array([slitAng_major_p+delta, slitAng_minor_p+delta])


	# ------ generate mock data ------
	TF = TFCube(pars=pars, line_species=line_species, sky_norm=1.)

	dataInfo = TF.gen_mock_data(noise_mode=inoise)
	if return_vmap is True:
		vmap = TF.getVmap(pars['vcirc'], pars['sini'], pars['g1'], pars['g2'], 
		pars['vscale'], pars['v_0'], 0., pars['theta_int'])
		dataInfo['vmap'] = vmap
	return dataInfo


def gen_deimos_mock_data(pars, line_species, axis, delta=0., inoise=0., return_vmap=False):
	data_info = gen_mock_data(pars, line_species, axis, delta, inoise, return_vmap)
	par_meta = data_info['par_meta']
	all_data = {}
	all_data['spec'] = []
	all_data['image'] = {}
	if return_vmap is True:
		all_data['vmap'] = data_info['vmap']

	wav_obs = (min(data_info['lambdaGrid']) + max(data_info['lambdaGrid']))/2

	for i in range(len(data_info['spec'])):
		lambda_1d = data_info['lambdaGrid']
		spatial_1d = data_info['spaceGrid']

		lambdaGrid, _ = np.meshgrid(lambda_1d, spatial_1d)

		temp = {}
		temp['data'] = data_info['spec'][i]
		temp['var'] = data_info['spec_variance'][i]

		temp['par_meta'] = {'line_species': line_species,
							'lambda_grid': lambdaGrid*u.nm,
							'pixScale': data_info['par_meta']['pixScale'],
							'ngrid': data_info['spec'][i].shape,
							'FWHMresolution': wav_obs/data_info['par_meta']['Resolution']*u.nm,
							'psfFWHM': data_info['par_meta']['psfFWHM'],
							'slitRA': 0.*u.degree,
							'slitDec': 0.*u.degree,
							'slitWidth': data_info['par_meta']['slitWidth'],
							'slitLen': 12.,
							'slitLPA': data_info['par_meta']['slitAngles'][i]*u.radian,
							'slitWPA': data_info['par_meta']['slitAngles'][i]*u.radian+np.pi/2*u.radian
						}

		temp['cont_model'] = np.zeros(data_info['spec'][i].shape[0])
		all_data['spec'].append(temp)
	
	all_data['galaxy'] = {'Mstar': 0,
			 			'Mag': 0,
						'mag_band': None,
						'redshift': data_info['par_fid']['redshift'],
						'RA': 0.*u.deg,
						'Dec': 0.*u.deg,
						'beta': 0*u.radian
						}

	all_data['image']['data'] = data_info['image']
	all_data['image']['var'] = data_info['image_variance']
	all_data['image']['par_meta'] = {'ngrid': data_info['image'].shape,
				  					'pixScale': data_info['par_meta']['pixScale'],
									'psfFWHM': data_info['par_meta']['psfFWHM'],
									'flux': data_info['par_meta']['flux']}

	all_data['par_fid'] = data_info['par_fid']
	return all_data
