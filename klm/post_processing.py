from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

import getdist
from getdist import plots
from getdist import chains

from kl_model import FitParameters
chains.print_load_details = False  # Disables 'No burn in steps' message when using getdist
analysis_settings = {'smooth_scale_1D':0.45,'smooth_scale_2D':0.45, 'fine_bins_2D': 500,'num_bins_2D': u'1000', 'fine_bins': u'1000','num_bins': u'1000'}

class PostProcessing():
	def __init__(self) -> None:
		self.g1_range = np.linspace(-0.2, 0.2, 600)
		self.g2_range = np.linspace(-0.2, 0.2, 600)
		self.vcirc_range = np.linspace(0, 1000, 600)
		self.sini_range = np.linspace(-1, 1, 600)

		self.range_list = np.column_stack((self.g1_range, self.g2_range, self.vcirc_range, self.sini_range))
		
	def _get_MAP(self, chain_info):
		walkers = chain_info['walkers']
		log_posterior = chain_info['log_post']

		x, y = np.where(log_posterior==np.max(log_posterior))
		pars = walkers[x, y][0]

		return pars

	def make_traceplots(self, chain_info, npar=None, thin=10):
		sampler = chain_info['sampler']
		assert sampler=='emcee' or sampler=='zeus', 'Can only make traceplots for MCMC samplers'

		ndim = len(chain_info['fit_par'])
		nwalkers = chain_info['nwalkers']
		nsteps = chain_info['nsteps']

		fit_par = chain_info['fit_par']
		walkers = chain_info['walkers']

		steps_range = np.arange(1, nsteps+1)

		if npar is None: npar=ndim
		plt.figure(figsize=(10,1.5*npar))

		for n in range(npar):
			plt.subplot2grid((npar, 1), (n, 0))
			for i in range(nwalkers):
				plt.plot(steps_range[::thin], walkers[::thin,i,n], '--', alpha=0.5)
			plt.ylabel(fit_par[n])
		plt.tight_layout()
		plt.savefig('traceplots.pdf')


	def make_traceplots_weighted(self, chain_info, npar=None, thin=10, savefile='triangle_plot.pdf'):
		sampler = chain_info['sampler']
		assert sampler=='emcee' or sampler=='zeus', 'Can only make traceplots for MCMC samplers'

		ndim = len(chain_info['fit_par'])
		nwalkers = chain_info['nwalkers']
		nsteps = chain_info['nsteps']

		fit_par = chain_info['fit_par']
		walkers = chain_info['walkers']
		log_like = chain_info['log_post']

		truths = self._get_truths(chain_info)

		if npar is None: npar=ndim
		steps_range = np.arange(1, nsteps+1)
		fig = plt.figure(figsize=(10,1.5*npar))

		for n in range(npar):
			plt.subplot2grid((npar, 1), (n, 0))
			l_max = np.max(log_like)
			norm = plt.Normalize(log_like.max()-100, log_like.max())

			for i in range(nwalkers):
				x = steps_range[::thin]
				y = walkers[::thin,i,n]
				points = np.array([x, y]).T.reshape(-1, 1, 2)
				segments = np.concatenate([points[:-1], points[1:]], axis=1)

				lc = LineCollection(segments, cmap='viridis', norm=norm)
				lc.set_array(log_like[::thin, i])
				line = plt.gca().add_collection(lc)
				if (l_max-log_like[-1, i])>10:
					plt.text(nsteps+100, y[-1], '%.2f'%(l_max-log_like[-1, i]))
				plt.gca().autoscale()
			fig.colorbar(line, ax=plt.gca())
            
			if n<len(truths):           
				plt.axhline(truths[n], ls='--', c='k')           
            
			plt.ylabel(fit_par[n])
		plt.tight_layout()
		plt.savefig(savefile)

	
	def make_triangle_plot_convergence(self, chain_info, burn_in=0):
		'''
		Divides chain into sub-samples
		'''
		
		ndim = len(chain_info['fit_par'])
		nwalkers = chain_info['nwalkers']
		nsteps = chain_info['nsteps']
		
		samples = chain_info['walkers']
# 		samples = np.reshape(samples, (nsteps, nwalkers, ndim))[burn_in:, :, :]

		fit_par = chain_info['fit_par']
		truths = [chain_info['fid_par'][name] for name in fit_par[:-2]]

		GD_samples = []
		legend_labels = []
		nsub_samples = 10000
		for i in range(int(nsteps/nsub_samples)):
			sub_samples = samples[i*nsub_samples:(i+1)*nsub_samples, 2, :]

			GD_samples.append(getdist.MCSamples(samples=sub_samples, names=fit_par))
			legend_labels.append(str(i*nsub_samples+1)+'-'+str((i+1)*nsub_samples+1))

		g = plots.get_subplot_plotter()
		g.triangle_plot(GD_samples, markers=truths,alpha2=0.5, legend_labels=legend_labels)
		
		plt.savefig('convergence.pdf')
		return GD_samples

    
	def _get_gd_samples(self, chain_info, npar=None, burn_in=0):
		'''
		Get GD_samples object from MCMC chain
		'''

		ndim = len(chain_info['fit_par'])
		nwalkers = chain_info['nwalkers']
		nsteps = chain_info['nsteps']

		if npar is None: npar=ndim
		walkers = chain_info['walkers']
		samples = walkers[burn_in:, :, :npar].reshape((nsteps-burn_in)*nwalkers, npar)

		fit_par = chain_info['fit_par']
        
		truths = self._get_truths(chain_info)

		latex_names = [FitParameters(fit_par)._get_latex_names()[name] for name in fit_par]

		GD_samples = getdist.MCSamples(samples=samples, names=fit_par[:npar], labels=latex_names, settings=analysis_settings)
		
		return GD_samples, truths


	def _get_gd_samples_vsini(self, chain_info, npar=None, burn_in=0):
		'''
		Get GD_samples object from MCMC chain
		'''

		ndim = len(chain_info['fit_par'])
		nwalkers = chain_info['nwalkers']
		nsteps = chain_info['nsteps']

		if npar is None: npar=ndim
		walkers = chain_info['walkers']
		samples = walkers[burn_in:, :, :npar].reshape((nsteps-burn_in)*nwalkers, npar)

		fit_par = ['vsini']#chain_info['fit_par']
		truths = [chain_info['fid_par']['vcirc']*chain_info['fid_par']['sini']]

		latex_names = ['v\sin i']

		GD_samples = getdist.MCSamples(samples=samples[:,2]*samples[:,3], names=fit_par, labels=latex_names, settings=analysis_settings)
		
		return GD_samples, truths

	def make_triangle_plot(self, chains, labels, npar=None, burn_in=0, samples_1d=None, isMAP=False, savefig=None):
		
		g = plots.get_subplot_plotter()

		g.settings.lw_contour = 1.2
		g.settings.legend_rect_border = False
		g.settings.figure_legend_frame = False
		g.settings.axes_fontsize = 18#9.5
		g.settings.legend_fontsize = 15.5
		g.settings.alpha_filled_add = 0.7
		g.settings.lab_fontsize = 22
		g.legend_labels = False

		GD_samples = []
		truths_list = []
		if isinstance(chains, list):
			for i, chain in enumerate(chains):
				max_log_post = np.max(chain['log_post'])
				labels[i] = labels[i] + ' best fit log_post:%.2f' %(max_log_post)
				samples, truths = self._get_gd_samples(chain, npar=npar, burn_in=burn_in[i])
				GD_samples.append(samples)
				if len(truths_list)<len(truths):
					truths_list = truths


		else:
			samples, truths = self._get_gd_samples(chains, npar=npar, burn_in=burn_in)
			GD_samples.append(samples)

			truths_list = truths
			map_pars = self._get_MAP(chains)

		line_args = [{'lw':2., 'color':'royalblue'}, {'lw':2., 'color':'red'}, {'lw':1., 'color':'k', 'ls':'--'}]
		contour_args = [{'lw':1, 'color':'royalblue'}, {'lw':1., 'color':'red'}, {'lw':1., 'color':'k', 'ls':'--'}]


		g.triangle_plot(GD_samples, line_args=line_args, contour_args=contour_args, norm_1d_density=True, legend_labels=labels)
        
		g.subplots[2][2].legend([Line2D([0], [0], color='k', ls='--', lw=1)], ['TF_prior'])

		for i in range(npar):
			for ax in g.subplots:
				if ax[i] is not None:
					ax[i].axvline(x=truths_list[i], ls='--', color='gray', lw=2.5)

		if isMAP is True:
			for i, ax in enumerate(g.subplots):
				ax[i].axvline(x=map_pars[i], ls='--', color='b', label='MAP')
			legend = g.subplots[0][0].legend()
			
			for legobj in legend.legendHandles:
				legobj.set_linewidth(1.5)

		if savefig is not None:
			plt.savefig(savefig)
		
		else:
			plt.savefig('triangleplot.pdf')
			plt.close()


	def get_mean_posterior(self, chains, npar=4, burn_in=0):

		joint = np.ones((len(self.g1_range), 4))
		for chain in chains:
			samples, truths = self._get_gd_samples(chain, npar, burn_in)

			g1_interp = samples.get1DDensity('g1').Prob(self.g1_range)
			g2_interp = samples.get1DDensity('g2').Prob(self.g2_range)
			vcirc_interp = samples.get1DDensity('vcirc').Prob(self.vcirc_range)
			sini_interp = samples.get1DDensity('sini').Prob(self.sini_range)

			interp_stacked = np.column_stack((g1_interp, g2_interp, vcirc_interp, sini_interp))

			joint[:, 0] *= g1_interp
			joint[:, 1] *= g2_interp
			joint[:, 2] *= vcirc_interp
			joint[:, 3] *= sini_interp

		joint[np.where(np.isclose(joint, 0.0) & (joint<0.0))] = 0.0
		joint = joint**(1/len(chains))

		return joint


	def get_mean_posterior_vsini(self, chains, npar=4, burn_in=0):

		joint = np.ones((len(self.vcirc_range), 1))
		for chain in chains:
			samples, truths = self._get_gd_samples_vsini(chain, npar, burn_in)

			vsini_interp = samples.get1DDensity('vsini').Prob(self.vcirc_range*self.sini_range)

			joint[:, 0] *= vsini_interp

		joint[np.where(np.isclose(joint, 0.0) & (joint<0.0))] = 0.0
		joint = joint**(1/len(chains))

		return joint


	def plot_mean_posterior(self, mean_posteriors, chains, ax, isave=False, savefig='./mean_posterior.pdf'):
		

		truths = [0.05, 0.05, 200, chains[0]['fid_par']['sini']]
		names = ['g1', 'g2', 'vcirc', 'sini']

		limits = [[-0.15, 0.15], [-0.15, 0.15], [50, 300], [0.1, 1]]

		for i in range(4):
			ax[i].plot(self.range_list[:, i], mean_posteriors[:, i], label='sini: %.2f'%truths[i])
			ax[i].axvline(x=truths[i], ls='--', c='k')
			ax[i].set_xlabel(names[i])
			ax[i].set_xlim(limits[i])

			ax[i].minorticks_on()
			ax[i].tick_params(axis='both', which='minor', length=3, direction='in')
			ax[i].tick_params(axis='both', which='major', length=7, direction='in')
			ax[i].get_yaxis().set_visible(False)

		ax[3].legend()	

		if isave is True:
			plt.subplots_adjust(wspace=0)
			plt.savefig(savefig)
			plt.close()
	

	def make_triangle_plot_samples(self, samples, labels, savefig=False):

		g = plots.get_subplot_plotter()

		g.settings.lw_contour = 1.2
		g.settings.legend_rect_border = False
		g.settings.figure_legend_frame = False
		g.settings.axes_fontsize = 15#9.5
		g.settings.legend_fontsize = 15.5
		g.settings.alpha_filled_add = 0.7
		g.settings.lab_fontsize = 15
		g.legend_labels = False
		g.triangle_plot(samples, alpha2=0.5, line_args={'lw':2.}, contour_args={'lw':2, 'alpha':0.6}, legend_labels=labels)
		if savefig is not None:
			plt.savefig(savefig)

	def _calc_mean(self, f_x, x):
		"""[summary]

		Args:
			f_x (array): Probability density
			x (array): Grid

		Returns:
			[type]: [description]
		"""		
		dx = x[1] - x[0]
		norm = np.sum(f_x*dx)
		f_x = f_x/norm
		
		avg_x = np.sum(f_x*x*dx)
		return avg_x


	def _calc_std(self, f_x, x):
		dx = x[1] - x[0]
		norm = np.sum(f_x*dx)
		f_x = f_x/norm
		
		avg_x = np.sum(f_x*x*dx)
		avg_x2 = np.sum(f_x*x**2*dx)
		
		return (avg_x2 - avg_x**2)**0.5


	def calc_stats(self, f_x, x):
		mean = []
		sigma = []
		
		for i in range(len(x)):
			mean.append(self._calc_mean(f_x[:, i], x[i]))
		
		for i in range(len(x)):
			sigma.append(self._calc_std(f_x[:, i], x[i]))
		
		return np.concatenate((mean, sigma))

	def _get_truths(self, chain_info):
		try:
			truths = [chain_info['fid_par'][name] for name in fit_par if name in chain_info['fid_par']]
		except:
			truths = []
			print('No fiducial parameters found!')

		return truths
