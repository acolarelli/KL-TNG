import joblib
import numpy as np
import scipy.signal
from scipy.optimize import curve_fit, differential_evolution
import warnings

import astropy.units as u

from klm.intensity import IntensityProfile
from klm.line_profile import LineProfile

class EmissionLine:
	def __init__(self,  use_analytic=False, **kwargs) -> None:
		if use_analytic:
			Amp, sigma1, sigma2 = self._init_analytic_profile(**kwargs)

		else:
			Amp, sigma1, sigma2 = self._init_profile_from_obs(**kwargs)

		self.line_profile = LineProfile(sigma1.to(u.nm).value, sigma2.to(u.nm).value)
		self.intensity = IntensityProfile(Amp)


	@staticmethod
	def _init_profile_from_obs(line_species, line_profile_path):
		f = open(line_profile_path, 'rb')
		line_profile = joblib.load(f)[line_species]
		f.close()
		Amp, sigma1, sigma2 = line_profile[0], line_profile[2], line_profile[3]

		return Amp, sigma1, sigma2

	@staticmethod
	def _init_analytic_profile(line_species, rhl, slit_grid):
		''' Simple Emission line model. Exponential intensity profile and
		constant line width across the slit.
		NOTE: Only meant for mock analysis; not for real data!!
		'''
		sigma1 = np.ones((len(slit_grid), 2))*1.7*u.Angstrom
		sigma2 = np.ones((len(slit_grid), 2))*1.7*u.Angstrom


		intensity_profile = np.exp(-np.abs(slit_grid)/rhl)
		Amp = np.column_stack((intensity_profile, intensity_profile))

		if line_species not in ['OII', 'OIIa', 'OIIb']:
			Amp[:, 1] = 0.0

		return Amp, sigma1, sigma2

	@staticmethod
	def extract_emission(this_spec, return_fit=False):
		"""Fits a Gaussian line profile across wavelengths at each spatial position and
		extracts line center, amplitude and width

		Parameters
		----------
		spec : 2D array
			2D spectrum, axis=0 is assumed to be the spatial axis
		lambda_grid : 2D array
			wavelength grid with associate astropy units
		this_line: str
			Only needed to check if it is a doublet (and instead fit a double Gaussian)
			has no other effect
		return_fit : bool, optional
			"estimated" spectrum for visual inspection, by default False

		Returns
		-------
		amplitude: 1D array
			Gaussian amplitude
		sigma: 1D array
			Gaussian standard deviation
		"""
		def remove_spurious_rows(arr):
			count_forward = count_backward = 0

			# Forward iteration
			for i in range(np.argmax(arr), len(arr)):
				if arr[i] == 0.: count_forward += 1
				else: count_forward = 0

				if count_forward > 3:
					arr[i:] = 0.
					break

			# Backward iteration
			for i in range(np.argmax(arr), -1, -1):
				if arr[i] == 0.: count_backward += 1
				else: count_backward = 0

				if count_backward > 3:
					arr[:i] = 0.
					break

			return arr

		def Gaussian(x, amp, mu, sigma):
			return amp * np.exp(-(x - mu)**2/2/sigma**2)


		def Gaussian_doublet(x, amp1, mu1, sigma1, amp2, mu2, sigma2):
			if mu1>mu2:
				return 1e9

			return Gaussian(x, amp1, mu1, sigma1) + Gaussian(x, amp2, mu2, sigma2)

		def two_half_Gaussians(x, A, mu, sigma1, sigma2):
			G1 = lambda x: A * np.exp(-(mu-x)**2/2/sigma1**2)
			G2 = lambda x: A * np.exp(-(mu-x)**2/2/sigma2**2)

			return np.where(x<mu, G1(x), G2(x))

		def two_half_Gaussian_doublet(x, Amp1, mu1, sigma1_1, sigma1_2, Amp2, mu2, sigma2_1, sigma2_2):
			if mu1>mu2:
				return 1e9

			return two_half_Gaussians(x, Amp1, mu1, sigma1_1, sigma1_2) + two_half_Gaussians(x, Amp2, mu2, sigma2_1, sigma2_2)


		def likelihood(cube, x, y, std, profile_func):
			t = profile_func(x, *cube)

			return np.sum((t-y)**2/std**2)

		def run_fit(x, y, this_std, prof, x0):
			sol = differential_evolution(likelihood, bounds=np.array(bounds).T, x0=x0, args=(x, y, this_std, prof))

			# D.E. is a gradient free method, so we use
			# curve_fit to estimate the covariance
			curve_fit_sol, curve_fit_cov = curve_fit(prof, x, y, p0=sol.x, bounds=bounds, sigma=this_std)
			err = curve_fit_cov.diagonal()**0.5
			diff = np.abs(sol.x-curve_fit_sol)/err

			# if the solution differs by 0.01 sigma raise warning
			if not np.all(diff<1e-2):
				warnings.warn(f'D.E. and curve fit solutions differ by {np.array2string(diff, precision=4)} sigma for row index {i}')

			fit_params = sol.x
			spec_estimate[i] = prof(x, *fit_params)
			amplitude[i][0] = fit_params[0]
			mu[i][0] = fit_params[1]
			sigma1[i][0] = fit_params[2]
			sigma2[i][0] = fit_params[3]

			amplitude_err[i][0] = err[0]
			mu_err[i][0] = err[1]
			sigma1_err[i][0] = err[2]
			sigma2_err[i][0] = err[3]

			if is_doublet:
				amplitude[i][1] = fit_params[4]
				mu[i][1] = fit_params[5]
				sigma1[i][1] = fit_params[6]
				sigma2[i][1] = fit_params[7]

				amplitude_err[i][1] = err[4]
				mu_err[i][1] = err[5]
				sigma1_err[i][1] = err[6]
				sigma2_err[i][1] = err[7]

		doublets = ['OII', 'OIIa', 'OIIb']

		## Unpack required data
		this_line = this_spec['par_meta']['line_species']
		data = this_spec['data'].copy()
		std = this_spec['var']**0.5
		psfFWHM = this_spec['par_meta']['psfFWHM']/this_spec['par_meta']['pixScale']  # Convert to pixels
		psfsigma = psfFWHM/(2*np.sqrt(2*np.log(2)))
		FWHMresolution = this_spec['par_meta']['FWHMresolution'].to(u.Angstrom).value
		lambda_grid = this_spec['par_meta']['lambda_grid'].to(u.Angstrom).value
		lambda_min, lambda_max = lambda_grid.min(), lambda_grid.max()
		center = (lambda_min+lambda_max)/2

		#Fit gaussian
		amplitude = np.zeros((data.shape[0], 2))
		amplitude_err = np.zeros((data.shape[0], 2))
		mu = np.zeros((data.shape[0], 2))
		mu_err = np.zeros((data.shape[0], 2))
		sigma1 = np.zeros((data.shape[0], 2))
		sigma2 = np.zeros((data.shape[0], 2))
		sigma1_err = np.zeros((data.shape[0], 2))
		sigma2_err = np.zeros((data.shape[0], 2))
		spec_estimate = np.zeros_like(data)

		prof = two_half_Gaussians
		bounds = [[0, center-10, 0.5, 0.5], [200, center+10, 3., 3.]]
		initial_guess = [30, center, 1, 1]  # amplitude, mu, sigma

		is_doublet = False

		data[std**2>400] = 0.0
		data = scipy.signal.medfilt2d(data, (9, 5))

		if this_line in doublets:
			is_doublet = True
			prof = two_half_Gaussian_doublet
			bounds = [bounds[0]*2, bounds[1]*2]
			initial_guess = initial_guess*2

		for i in range(data.shape[0]):
			y = data[i, :]
			x = lambda_grid[i, :]
			this_std = std[i, :]

			if np.mean(y)>0.03:
				try:
					run_fit(x, y, this_std, prof, initial_guess)

				except RuntimeError:
					pass

		# This part essentially identifies the blue-shifted and redshifted
		# parts of the rotation curve and checks that all line centers in
		# blue(red)-shifted part are smaller(larger) than the mean line center
		# identify +delta lambda and -delta lambda directions
		# identify central pixel
		mask0 = mu[:, 0]>0.  # Ignore x-row where there is no signal
		mask_ind0 = np.where(mask0)[0]
		mu0 = mu[:, 0].copy()[mask0]
		mean0 = np.median(mu0)  # This is the mean line center
		idx0 = np.argmin(np.abs(mu0-mean0))  # line center index

		# we will update the default bounds now
		bounds_mu0 = np.ones((data.shape[0], 2))*(lambda_min, lambda_max)

		if np.median(mu0[:idx0])>mean0:
			bounds_mu0[:mask_ind0[idx0]-2] = (mean0-0.3, lambda_max) #red_shifted
			bounds_mu0[mask_ind0[idx0]+2:] = (lambda_min, mean0+0.3) #blue_shifted

		else :
			bounds_mu0[:mask_ind0[idx0]-2] =  (lambda_min, mean0+0.3) #blue_shifted
			bounds_mu0[mask_ind0[idx0]+2:] = (mean0-0.3, lambda_max) #red_shifted

		if is_doublet:
			mask1 = mu[:, 1]>0.
			mask_ind1 = np.where(mask1)[0]
			mu1 = mu[:, 1].copy()[mask1]
			mean1 = np.median(mu1)
			idx1 = np.argmin(np.abs(mu1-mean1))

			bounds_mu1 = np.ones((data.shape[0], 2))*(lambda_min, lambda_max)

			if np.median(mu1[:idx1])>mean1:
				bounds_mu1[:mask_ind1[idx1]-2] = (mean1-0.3, lambda_max) #red_shifted
				bounds_mu1[mask_ind1[idx1]+2:] = (lambda_min, mean1+0.3) #blue_shifted

			else :
				bounds_mu1[:mask_ind1[idx1]-2] =  (lambda_min, mean1+0.3) #blue_shifted
				bounds_mu1[mask_ind1[idx1]+2:] = (mean1-0.3, lambda_max) #red_shifted

		# ipdb.set_trace()
		# Need to change this
		for i in range(data.shape[0]):
			y = data[i, :]
			x = lambda_grid[i, :]
			this_std = std[i, :]

			# Update min. and max. lambda for emission line
			bounds[0][1] = bounds_mu0[i][0]
			bounds[1][1] = bounds_mu0[i][1]

			if is_doublet:
				bounds[0][5] = bounds_mu1[i][0]
				bounds[1][5] = bounds_mu1[i][1]

				initial_guess = [30, (bounds[0][1]+ bounds[1][1])/2, 2, 2, 30, (bounds[0][5]+ bounds[1][5])/2, 2, 2]

			else:
				initial_guess = [30, (bounds[0][1]+ bounds[1][1])/2, 2, 2]

			if np.mean(y)>0.03:
				try:
					run_fit(x, y, this_std, prof, initial_guess)

				except RuntimeError:
					pass


		spec_estimate = np.array(spec_estimate)

		sigma1 = np.array(sigma1)
		sigma2 = np.array(sigma2)


		# Now remove rows with spurious extraction
		# Suffices to just set the amplitude to zero
		amplitude[:, 0] = remove_spurious_rows(amplitude[:, 0])
		if is_doublet: amplitude[:, 1] = remove_spurious_rows(amplitude[:, 1])

		for i in range(data.shape[0]):
			x = lambda_grid[i, :]
			if is_doublet is False:
				spec_estimate[i] = two_half_Gaussians(x, amplitude[i, 0], mu[i, 0], sigma1[i, 0], sigma2[i, 0])

			else:
				spec_estimate[i] = two_half_Gaussian_doublet(x, amplitude[i, 0], mu[i, 0], sigma1[i, 0], sigma2[i, 0],
															amplitude[i, 1], mu[i, 1], sigma1[i, 1], sigma2[i, 1])

		if return_fit is True:
			results = (np.array(amplitude)/1e2, np.array(mu)*u.Angstrom, np.array(sigma1)*u.Angstrom, np.array(sigma2)*u.Angstrom, data, np.array(spec_estimate))
			errors = (np.array(amplitude_err)/1e2, np.array(mu_err)*u.Angstrom, np.array(sigma1_err)*u.Angstrom, np.array(sigma2_err)*u.Angstrom)
			return results, errors

		return np.array(amplitude)/1e2, np.array(sigma1)*u.Angstrom, np.array(sigma2)*u.Angstrom


