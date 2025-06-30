import os
os.environ["OMP_NUM_THREADS"] = "1"

from multiprocessing import Pool
import numpy as np
from schwimmbad import MPIPool
import time
import sys

import emcee
import getdist
import joblib

from klm.kl_inference import KLInference
from klm.utils import check_dir


class EmceeSampler(KLInference):
	'''
	Sub-class for parameter inference using MCMC
	'''

	def __init__(self, data_info=None, config=None):
		KLInference.__init__(self, data_info, config)
# 		self._init_prior(self.config.likelihood.set_non_analytic_prior)

	def _init_walkers(self, nwalkers):

		starting_point = list(self.fit_params.params.values())
		std = list(self.fit_params.param_std.values())

		p0_walkers = emcee.utils.sample_ball(starting_point, std, size=nwalkers)

		for i, key in enumerate(self.fit_params.names):
			low_lim, up_lim = self.fit_params.param_limits[key][0], self.fit_params.param_limits[key][1]

			for walker in range(nwalkers):
				while p0_walkers[walker, i] < low_lim or p0_walkers[walker, i] > up_lim:
					p0_walkers[walker, i] = np.random.rand()*std[i] + starting_point[i]

		return p0_walkers


	def get_prior_vcirc(self, vcirc):
		'''
		Tully-Fisher prior on log10(vcirc)
				log(prior) = 0.5*(log10(vcirc)-log10(v_TF))**2/sigma_TF**2
		Args:
			vcirc (float): Circular velocity (km/s)

		Returns:
			float: log(prior)
		'''
		log_prior = 0.5 * ((np.log10(vcirc) - self.config.TFprior.log10_vTF) / self.config.TFprior.sigmaTF)**2
		return log_prior


	def calc_joint_loglike(self, sample, return_chi2=False):
		'''
		Computes the joint likelihood of image and spectra

		Args:
			fit_par_values (list): list of fit parameter values from sampler

		Attributes:
			par_limits (dict): Dicitionary of fit parameter limits
			isFitIimage (bool): Fit image if True
			isFitSpec (bool): Fit spectrum if True

		Returns:
				float: log likelihood
		'''
		params = self.params.gen_param_dict(self.config.params.names, sample)

		for i, key in enumerate(self.config.params.names):
			if 'shared_params-vcirc' == key:
				low_lim, up_lim = 0, 1000

			else:
				low_lim, up_lim = self.config.params.prior[key][0], self.config.params.prior[key][1]

			if self.params._flatten(params, level=1)[key] < low_lim or self.params._flatten(params, level=1)[key] > up_lim:
				return -np.inf, -np.inf, -np.inf

		spec_loglike, image_loglike = 0., 0.

		# Get image log likelihood
		if self.config.likelihood.isFitImage is True:
			image_loglike = self.calc_image_loglike(params)

		# Get spectrum log likelihood
		if self.config.likelihood.isFitSpec is True:
			spec_loglike = self.calc_spectrum_loglike(params)

		# Get log TF prior
		prior_vcirc = 0.0
		if 'shared_params-vcirc' in params.keys():
			prior_vcirc = self.get_prior_vcirc(self.params._flatten(params, level=1)['shared_params-vcirc'])

		# Additonal prior
		add_prior = 0.0
# 		if self.set_prior is not None:
# 			x = np.array([params[name] for name in self.set_prior['par']])
# 			mu = self.set_prior['mean']
# 			cov = self.set_prior['covariance']
# 			add_prior = 0.5*((x-mu).T@np.linalg.inv(cov)@(x-mu))

		# Compute joint likelihood
		log_likelihood = -0.5 * (spec_loglike + image_loglike)
		log_posterior =  log_likelihood - prior_vcirc - add_prior

		if return_chi2 is True:
			return -np.sum(log_posterior)#, log_posterior, log_likelihood#, spec_loglike, image_loglike)

		return log_posterior#, log_posterior, log_likelihood#, spec_loglike, image_loglike)


	def run(self, nwalkers, nsteps, outputfile_name='./chain_info.pkl', **kwargs):
		'''
		Runs MCMC chain using the emcee sampler

		Args:
			nwalkers (int): Number of walkers
			nsteps (int): Number of steps

		Returns:
				nd array: array of samples
		'''
		Ndim = len(self.fit_params.names)

		check_dir(outputfile_name)
		assert Ndim != 0, 'No fit parameters entered'

		p0_walkers = self._init_walkers(nwalkers)

		dtype = [('log_post', float), ('log_like', float)]
		start_time = time.time()
		with Pool() as pool:
			sampler = emcee.EnsembleSampler(nwalkers, Ndim, self.calc_joint_loglike, blobs_dtype=dtype, pool=pool)

			sampler.run_mcmc(p0_walkers, nsteps=nsteps, progress=True, **kwargs)

		blobs = sampler.get_blobs()
		self.chain_info = {}
		self.chain_info['chain'] = sampler.get_chain(flat=True)
		self.chain_info['walkers'] = sampler.get_chain()
		self.chain_info['log_post'] = blobs['log_post']
		self.chain_info['log_like'] = blobs['log_like']
		self.chain_info['runtime'] = (time.time() - start_time)/60
		self.chain_info['nwalkers'] = nwalkers
		self.chain_info['nsteps'] = nsteps

		joblib.dump(self.chain_info, outputfile_name, 3)

		return sampler


	def run_mpi(self, nwalkers, nsteps, init_walkers=None, outputfile_name='./chain_info.pkl', **kwargs):
		'''
		Runs MCMC chain using the emcee sampler

		Args:
			nwalkers (int): Number of walkers
			nsteps (int): Number of steps

		Returns:
				nd array: array of samples
		'''
		ndim = len(self.fit_params.names)

		check_dir(outputfile_name)
		assert ndim != 0, 'No fit parameters entered'

		p0_walkers = self._init_walkers(nwalkers)

		if init_walkers is not None:
			p0_walkers = init_walkers


		dtype = [('log_post', float), ('log_like', float)]
		start_time = time.time()
		with MPIPool() as pool:
			if not pool.is_master():
				pool.wait()
				sys.exit(0)
			sampler = emcee.EnsembleSampler(nwalkers,ndim, self.calc_joint_loglike, blobs_dtype=dtype, pool=pool)

			sampler.run_mcmc(p0_walkers, nsteps=nsteps, progress=True, **kwargs)
			pool.close()

		blobs = sampler.get_blobs()
		self.chain_info['chain'] = sampler.get_chain(flat=True)
		self.chain_info['walkers'] = sampler.get_chain()
		self.chain_info['log_post'] = blobs['log_post']
		self.chain_info['log_like'] = blobs['log_like']
		self.chain_info['runtime'] = (time.time() - start_time)/60
		self.chain_info['nwalkers'] = nwalkers
		self.chain_info['nsteps'] = nsteps

		joblib.dump(self.chain_info, outputfile_name, 3)
		return sampler


	def get_estimates(self, chain, names=None, **kwargs):
		'''
		Uses GetDist to compute estimates from parameter chains
		Discard burn-in steps before passing chain

		Args:
			chain (array): Array of sampled parameter values

		Returns:
			array: Structured array (parameter, mean, sigma)
		'''
		if names is None:
			names = self.fit_params.names

		samples = getdist.MCSamples(samples=chain, names=names, **kwargs)
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
		Computes estimates on vsini, vcirc, sini given parameter chains
		Needs the complete chain and not just vcirc and sini
		Discard burn-in steps before passing chain

		Args:
			chain (array): Array of sampled

		Returns:
			array: Structured array (parameter, mean, sigma)
		'''
		ind_vcirc, ind_sini = self.fit_params.names.index('vcirc'), self.fit_params.names.index('sini')

		vcirc, sini = chain[:, ind_vcirc], chain[:, ind_sini]
		vsini = vcirc * sini

		new_chain = np.column_stack((vsini, vcirc, sini))
		names = ['vsini', 'vcirc', 'sini']

		samples = getdist.MCSamples(samples=new_chain, names=names, **kwargs)

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


	def _get_chain(self, **kwargs):
		'''
		Wrapper for EmceeSampler.run() used in CalibrateBias class

		Returns:
			array: array of samples
		'''
		samples = self.run(**kwargs)
		return samples.get_chain(flat=True)
