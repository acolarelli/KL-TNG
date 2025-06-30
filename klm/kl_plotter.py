import matplotlib.pyplot as plt
import numpy as np
import scipy

import getdist

from kl_model import Model

import sys
sys.path.append('../KLens/')
from tfCube2 import gen_grid

class KLPlotter():
	def __init__(self):
		pass
		
	def _plot(self, spec1, spec2, extent, label1, label2):

		fig, ax = plt.subplots(1, 3, figsize=(15, 8))
		fig.subplots_adjust(wspace=0.01, hspace=0, top=1.)

		im1 = ax[0].imshow(spec1, extent=extent, origin='lower', aspect='auto')
		im2 = ax[1].imshow(spec2, extent=extent, origin='lower', aspect='auto')
		im3 = ax[2].imshow((spec1-spec2), extent=extent, origin='lower', aspect='auto')

		fig.colorbar(im1, ax=ax[0], location='top')
		fig.colorbar(im2, ax=ax[1], location='top')
		fig.colorbar(im3, ax=ax[2], location='top')

		ax[0].set(xlabel='wavelength [nm]', ylabel='slit position [arcsec]', title=label1)
		ax[1].set(xlabel='wavelength [nm]', title=label2)
		ax[2].set(xlabel='wavelength [nm]', title='Residual')

		ax[1].tick_params(axis='y', which='both', left=False, labelleft=False)
		ax[2].tick_params(axis='y', which='both', left=False, labelleft=False)
		
		return fig, ax

	def _plot_spectra(self, spec1, spec2, axis, extent, slit_width, title, label1, label2):

		if axis == 'major':
			norm=1
# 			norm = np.sum(spec1[0]*spec2[0])/np.sum(spec1[0]*spec1[0])
			
			fig, ax = self._plot(spec1[0], spec2[0]/norm, extent, label1, label2)	
			fig.suptitle(f'{title}: Major axis (slitWidth={slit_width})')
		
		if axis == 'minor':
			norm=1
# 			norm = np.sum(spec1[0]*spec2[0])/np.sum(spec1[0]*spec1[0])
			
			fig, ax = self._plot(spec1[1], spec2[1]/norm, extent, label1, label2)	
			fig.suptitle(f'{title}: Minor axis (slitWidth={slit_width})')

		if axis == 'both':
			norm=1
# 			norm = np.sum(spec1[0]*spec2[0])/np.sum(spec1[0]*spec1[0])

			fig, ax = self._plot(spec1[0], spec2[0], extent, label1, label2)
			fig.suptitle(f'{title}: Major axis (slitWidth={slit_width})')
			
			norm = np.sum(spec1[0]*spec2[0])/np.sum(spec1[0]*spec1[0])
			fig, ax = self._plot(spec1[1], spec2[1]/norm, extent, label1, label2)
			fig.suptitle(f'{title}: Minor axis (slitWidth={slit_width})')
		
		return fig, ax

	def plot_spectra_dm(self, data_info, parameter_dict, axis='both', title='', **kwargs):
		"""[summary]

		Args:
			pars ([type]): [description]

		Returns:
			[type]: [description]
		"""
		meta_pars = data_info['par_meta']
		slit_angles = meta_pars['slitAngles']
		slit_width = meta_pars['slitWidth']
		TFspec = data_info['spec']

		model = Model(parameter_dict, meta_pars, **kwargs)
		model_spec = model.get_spectrum(parameter_dict, slit_angles)['spec']

		x_min, x_max = min(model.spatial_x), max(model.spatial_x)
		lambda_min, lambda_max = min(model.X), max(model.X)
		
		extent = [lambda_min, lambda_max, x_min, x_max]

		self._plot_spectra(TFspec, model_spec, axis, extent, slit_width, title, 'TFcube spectrum', 'Model spectrum')



	def plot_spectra_mm(self, parameter_dict1, parameter_dict2, meta_par1, meta_par2, axis='both', title='', **kwargs):
		"""[summary]

		Args:
			pars ([type]): [description]

		Returns:
			[type]: [description]
		"""
		slit_angles1 = meta_par1['slitAngles']
		slit_width1 = meta_par1['slitWidth']
		model1 = Model(parameter_dict1, meta_par1, **kwargs)
		model_spec1 = model1.get_spectrum(parameter_dict1, slit_angles1)['spec']

		slit_angles2 = meta_par2['slitAngles']
		model2 = Model(parameter_dict2, meta_par2, **kwargs)
		model_spec2 = model2.get_spectrum(parameter_dict2, slit_angles2)['spec']

		x_min, x_max = min(model1.spatial_x), max(model1.spatial_x)
		lambda_min, lambda_max = min(model1.X), max(model1.X)
		
		extent = [lambda_min, lambda_max, x_min, x_max]

		fig, ax = self._plot_spectra(model_spec1, model_spec2, axis, extent, slit_width1, title, 'Model spec 1', 'Model spec 2')
		
		return fig, ax
	
	def plot_spectra_dd(self, data_info1, data_info2, axis='both', title=''):
		TFspec1 = data_info1['spec']
		TFspec2 = data_info2['spec']

		
		slitwidth = data_info1['par_meta']['slitWidth']
		model = Model(data_info1['par_fid'], data_info1['par_meta'])

		x_min, x_max = min(model.spatial_x), max(model.spatial_x)
		lambda_min, lambda_max = min(model.X), max(model.X)
		
		extent = [lambda_min, lambda_max, x_min, x_max]

		fig, ax = self._plot_spectra(TFspec1, TFspec2, axis, extent, slitwidth, title, 'TFcube spectrum 1', 'TFcube spectrum2')
		
		return fig, ax