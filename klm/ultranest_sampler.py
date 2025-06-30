import numpy as np
import scipy
import joblib

import getdist
from getdist import chains
import ultranest.stepsampler

from klm.kl_inference import KLInference
from klm.utils import check_dir

chains.print_load_details = False  # Disables 'No burn in steps' message when using getdist


class UltranestSampler(KLInference):
	'''
	Sub-class for parameter inference using nested sampling
	'''

	def __init__(self, data_info=None, config=None):
		KLInference.__init__(self, data_info, config)
		self.n_params = len(self.config.params.names)

		# For deltax_vel/deltay_vel
		a, b = (-1 - 0.)/0.5, (1 - 0.)/0.5
		self.truncnormprior = scipy.stats.truncnorm(a, b)
		if not self.config.TFprior.use_TFprior:
			print('Warning: Not using using TF prior')
		self._init_prior(self.config.likelihood.set_non_analytic_prior)


	def _get_wrapped_params(self):
		'''Returns a list of bools, True for wrapped(circular) params
		otherwise False
		'''
		master_wrapped_params = ['theta_int', 'phi0']
		wrapped_params = [False]*len(self.config.params.names)

		for item in master_wrapped_params:
			for i, key in enumerate(self.config.params.names):
				if item in key:
					wrapped_params[i] = True

		return wrapped_params


	def _init_prior(self, set_prior):
		self.prior_ppf = {}
		if set_prior is not None:
			for par in set_prior:
				prior_samples = set_prior[par]
				hist, bin_edges = np.histogram(prior_samples, bins=100)
				hist_cumulative = np.cumsum(hist / hist.sum())
				bin_middle = (bin_edges[:-1] + bin_edges[1:]) / 2

				self.prior_ppf[par] = scipy.interpolate.interp1d(hist_cumulative, bin_middle,
						     bounds_error=False, fill_value=(bin_middle[0], bin_middle[-1]))
		return


	def prior(self, cube):
		'''
		Transform unit cube into parameter cube

		Args:
			cube (nd array): Nd unit dimensional cube
			ndim (int): Number of dimensions
			nparams (int): Number of fit parameters
		'''
		params = np.zeros_like(cube)

		for i, key in enumerate(self.config.params.names):
			if key == 'shared_params-vcirc' and self.config.TFprior.use_TFprior:  # Change implentation
				params[i] = 10**self.config.params.prior[key].ppf(cube[i])      # Gaussian prior on log10(vcirc)

			#elif 'dx_vel' in key or 'dy_vel' in key:
			#	params[i] = 10**self.truncnormprior.ppf(cube[i])

			else:
				# Get lower and upper bounds
				low_lim, up_lim = self.config.params.prior[key][0], self.config.params.prior[key][1]

				# Uniform prior
				params[i] = cube[i] * (up_lim - low_lim) + low_lim


		# If we have a prior from a previous chain sample from that
		for key in self.prior_ppf:
			idx = self.config.params.names.index(key)
			params[idx] = self.prior_ppf[key](cube[idx])

		return params

	def calc_joint_loglike(self, cube):
		'''
		Computes the joint likelihood of image and spectra

		Args:
			fit_par_values (list): list of fit parameter values from sampler
			ndim (int): Number of dimensions
			nparams (int): Number of fit parameters

		Returns:
			float: log likelihood
		'''
		temp = cube.copy()

		# Get a dictionary of updated fit parameter values
		pars = self.params.gen_param_dict(self.config.params.names, cube)

		image_loglike, spec_loglike = 0., 0.
		# Get image log likelihood
		if self.config.likelihood.isFitImage is True:
			image_loglike = self.calc_image_loglike(pars)

		# Get spectrum log likelihood
		if self.config.likelihood.isFitSpec is True:
			spec_loglike = self.calc_spectrum_loglike(pars)

		# Compute joint likelihood
		joint_loglike = -0.5 * (spec_loglike + image_loglike)

		if self.config.likelihood.apply_rhl_constraint:
			constraint = pars['shared_params']['r_hl_disk'] - pars['shared_params']['r_hl_bulge']
			constraint2 = pars['shared_params']['flux'] - pars['shared_params']['flux_bulge']

			if constraint < 0.:
				return -1e100 * (1 - constraint)

			if constraint2 < 0.:
				return -1e100 * (1 - constraint2)

		return joint_loglike

	def run(self, output_dir='./ultranest_output/', test_run=False, run_num=1, **kwargs):
		"""Run a nested sampler using UltraNest


		Parameters
		----------
		output_dir : str, optional
			_description_, by default './ultranest_output/'
		test_run : bool, optional
			Perform a few likelihood evaluations, by default False

		Returns
		-------
		_type_
			_description_
		"""
		assert self.n_params != 0, 'No fit parameters entered'

		sampler = ultranest.ReactiveNestedSampler(self.config.params.names, self.calc_joint_loglike, self.prior,
					    wrapped_params=self._get_wrapped_params(), log_dir=output_dir, run_num=run_num, **kwargs)
		sampler.stepsampler = ultranest.stepsampler.SliceSampler(nsteps=2*self.n_params,
					       generate_direction=ultranest.stepsampler.generate_mixture_random_direction)

		if test_run is False:
			sampler.run(min_num_live_points=400, max_ncalls=8e6)

		elif test_run is True:
			print('Testing likelihood evaluations...')
			sampler.run(min_num_live_points=400, max_iters=10, show_status=True)
			print('Finished..no issues encountered!')

		# joblib.dump(self.chain_info, outputfiles_basename+'chain_info.pkl', 3)

		return sampler

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
