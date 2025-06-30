#import os
#os.environ["OMP_NUM_THREADS"] = "1"

import json
import numpy as np
import scipy
import joblib

import getdist
from getdist import chains
import pymultinest

from klm.kl_inference import KLInference
from klm.utils import check_dir

chains.print_load_details = False  # Disables 'No burn in steps' message when using getdist


class MultinestSampler(KLInference):
	'''
	Sub-class for parameter inference using nested sampling
	'''

	def __init__(self, data_info=None, fit_par_names=None, isFitImage=True, isFitSpec=True, **kwargs):
		KLInference.__init__(self, data_info, fit_par_names, isFitImage, isFitSpec, **kwargs)
		self.n_params = len(self.fit_params.names)
		self.chain_info['sampler'] = 'multinest'
        
	def prior(self, cube, ndim, nparams):
		'''
		Transform unit cube into parameter cube

		Args:
			cube (nd array): Nd unit dimensional cube
			ndim (int): Number of dimensions
			nparams (int): Number of fit parameters
		'''
		for i, key in enumerate(self.fit_params.names):
			if key == 'vcirc':  # Change implentation
				cube[i] = scipy.stats.norm(loc=self.log10_vTF, scale=self.sigmaTF).ppf(cube[i])      # Gaussian prior on log10(vcirc)

			else:
				# Get lower and upper bounds
				low_lim, up_lim = self.fit_params.par_limits[key][0], self.fit_params.par_limits[key][1]

				# Uniform prior
				cube[i] = cube[i] * (up_lim - low_lim) + low_lim

	def calc_joint_loglike(self, cube, ndim, nparams):
		'''
		Computes the joint likelihood of image and spectra

		Args:
			fit_par_values (list): list of fit parameter values from sampler
			ndim (int): Number of dimensions
			nparams (int): Number of fit parameters

		Returns:
			float: log likelihood
		'''

		if 'vcirc' in self.fit_params.names:
			index = self.fit_params.names.index('vcirc')
			cube[index] = 10**cube[index]

		# Get a dictionary of updated fit parameter values
		pars = self.par_obj.gen_param_dict(self.fit_params.names, cube)

		image_loglike, spec_loglike = 0., 0.
		# Get image log likelihood
		if self._isFitImage is True:
			image_loglike = self.calc_image_loglike(pars)

		# Get spectrum log likelihood
		if self._isFitSpec is True:
			spec_loglike = self.calc_spectrum_loglike(pars)

		# Compute joint likelihood
		joint_loglike = -0.5 * (spec_loglike + image_loglike)

		return joint_loglike

	def run(self, outputfiles_basename='./multinest_output/', **kwargs):
		'''
		Runs a nested sampler using MultiNest

		Args:
			outputfiles_basename (str, optional): Basename for output files

		Returns:
			object: MultiNest analyzer object
		'''

		check_dir(outputfiles_basename)

		assert self.n_params != 0, 'No fit parameters entered'

		pymultinest.run(self.calc_joint_loglike, self.prior, self.n_params, evidence_tolerance=0.5,
						outputfiles_basename=outputfiles_basename, resume=False, **kwargs)
		json.dump(self.fit_params.names, open(outputfiles_basename + 'params.json', 'w'))  # save parameter names

		analyzer = pymultinest.Analyzer(outputfiles_basename=outputfiles_basename, n_params=self.n_params)
		
		self.chain_info['chain'] = analyzer.get_equal_weighted_posterior()
		joblib.dump(self.chain_info, outputfiles_basename+'chain_info.pkl', 3)

		return analyzer

	def get_estimates(self, chain, names=None, **kwargs):
		'''
		Return means and standard deviations for marginal distributions of fit parameters using getdist

		Args:
			chain (array): Array of samples

		Returns:
			array: Structured array of name, mean, sigma
		'''
		if names is None:
			names = self.fit_params.names

		samples = getdist.MCSamples(samples=chain, names=names, sampler='nested', **kwargs)
		marginal_estimates = np.empty(len(names), dtype=[('parameter', 'U20'),
									('mean', 'f4'),
									('sigma', 'f4')])

		means = samples.getMeans()
		covariance = samples.getCov()
		sigma = [covariance[i, i]**0.5 for i in range(len(covariance))]

		marginal_estimates['parameter'] = names
		marginal_estimates['mean'] = means
		marginal_estimates['sigma'] = sigma

		return marginal_estimates

	def get_vsini_estimates(self, chain, **kwargs):
		'''
		Return means and standard deviations for marginal distributions of vsini, vcirc, sini

		Args:
			chain (array): Array of samples

		Returns:
			array: Structured array of name, mean, sigma
		'''
		ind_vcirc, ind_sini = self.fit_params.names.index('vcirc'), self.fit_params.names.index('sini')

		vcirc, sini = chain[:, ind_vcirc], chain[:, ind_sini]
		vsini = vcirc * sini

		new_chain = np.column_stack((vsini, vcirc, sini))

		names = ['vsini', 'vcirc', 'sini']
		samples = getdist.MCSamples(samples=new_chain, names=names, sampler='nested', **kwargs)

		marginal_estimates = np.empty(len(names), dtype=[('parameter', 'U20'),
									('mean', 'f4'),
									('sigma', 'f4')])

		means = samples.getMeans()
		covariance = samples.getCov()
		sigma = [covariance[i, i]**0.5 for i in range(len(covariance))]

		marginal_estimates['parameter'] = names
		marginal_estimates['mean'] = means
		marginal_estimates['sigma'] = sigma

		return marginal_estimates

	def get_best_fit_dict(self, chain):
		
		best_fit_pars = self.get_estimates(chain=chain)
		best_fit_dict = self.galaxy_model.Pars.gen_param_dict(self.fit_params.names, best_fit_pars['mean'])

		return best_fit_dict

	def _get_chain(self, **kwargs):
		'''
		Wrapper for MultiNestSampler.run() used in CalibrateBias class

		Returns:
			array: array of samples
		'''
		analyzer = self.run(outputfiles_basename=self.outputfiles_basename, **kwargs)
		samples = analyzer.get_equal_weighted_posterior()

		return samples

	def get_analyzer(self):
		'''
		Returns MultiNest Analyzer

		Returns:
			object: MultiNest Analyzer
		'''
		pass
