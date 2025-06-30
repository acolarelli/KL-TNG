import numpy as np

from klm.parameters import Parameters, FitParameters
from klm.spec_model import SlitModel, IFUModel
from klm.image_model import ImageModel
from klm.config import Config

class KLInference():
	'''
	Base class for KL inference
	'''

	def __init__(self, data_info, config):
		'''
		Initializes the KLInference object.

		Args:
			data_info (str): Information about the data.
			config (dict): Configuration parameters.

		'''
		self.config = Config(config)

		self._init_data(data_info)

		if self.config.likelihood.fid_params is not None:
			fid_params = self.config.likelihood.fid_params
			self.params._update_params({'shared_params':{**fid_params['shared_params']}, **fid_params['line_params']})


	def _init_data(self, data_info):
		self.meta_gal = data_info['galaxy']
		if self.config.likelihood.isFitImage:
			self.meta_image = data_info['image']['par_meta']
			self.data_image = data_info['image']['data']
			self.var_image = data_info['image']['var']
			self.image_model = ImageModel(self.meta_image)


		if self.config.galaxy_params.obs_type == 'slit':
			self.params = Parameters({'shared_params':{'beta':self.meta_gal['beta']}}, line_species=self.config.galaxy_params.line_species)

			self._spectrum_loglike = self._loglike_slit
			self._init_slit_data(data_info)
			# self.fit_params = FitParameters(self.config.fit_param_names, line_species=self.config.line_species)

		elif self.config.galaxy_params.obs_type == 'IFU':
			self.params = Parameters(line_species=self.config.galaxy_params.line_species)

			self._spectrum_loglike = self._loglike_ifu
			self._init_ifu_data(data_info)
			# self.fit_params = FitParameters(self.config.fit_param_names, line_species=[self.config.vmap_type])


	def _init_slit_data(self, data_info):
		'''
		Initializes slit data.

		Parameters:
		- data_info (dict): A dictionary containing information about the spectral data.

		'''

		self.meta_spec = []
		self.data_spec = []
		self.var_spec = []
		self.cont_model = []

		self.spec_model = []

		for i in range(len(data_info['spec'])):
			this_spec = data_info['spec'][i]

			if this_spec['par_meta']['line_species'] not in self.config.galaxy_params.line_species:
				continue

			self.meta_spec.append(this_spec['par_meta'])
			self.data_spec.append(this_spec['data'])
			self.var_spec.append(this_spec['var'])
			self.cont_model.append(this_spec['cont_model'])

			self.spec_model.append(SlitModel(self.meta_gal, this_spec['par_meta'], self.config.galaxy_params.line_profile_path,
									self.config.galaxy_params.rc_type))

			if self.config.verbose:
				print(f'[INFO]: Finished initializing model for {self.config.line_species[i]}')

		if len(self.config.galaxy_params.line_species) > len(self.spec_model):
			print(f'[WARNING]: Specified {self.config.galaxy_params.line_species} line specie(s) but only {len(self.spec_model)} emission lines in data.')

		if len(self.spec_model) > len(self.config.galaxy_params.line_species):
			raise Exception(f'Data has {len(self.spec_model)} emission lines but only {self.config.galaxy_params.line_species} line species are specified in config file.')


	def _init_ifu_data(self, data_info):
		'''
		Initializes IFU data.

		Parameters:
		- data_info (dict): A dictionary containing information about the data.

		'''
		this_vmap = data_info['vmap'][self.config.vmap_type]
		self.meta_2Dmap = this_vmap['par_meta']
		self.data_2Dmap = this_vmap['data']
		self.var_2Dmap = this_vmap['var']
		self.mask_2Dmap = this_vmap['mask']

		self.IFU_model = IFUModel(this_vmap['par_meta'], self.config.galaxy_params.rc_type)

		if self.config.verbose:
			print('Initializing IFU data...')
			print(f'Finished initializing model for {self.config.vmap_type} vmap')


	def _print_TF_relation_table(self, a, b, sigmaTF_intr):
		'''
		Prints a table of the Tully-Fisher relation parameters.

		Parameters:
		- a (float): The slope of the Tully-Fisher relation.
		- b (float): The intercept of the Tully-Fisher relation.
		- sigmaTF_intr (float): The intrinsic scatter of the Tully-Fisher relation.
		'''
		print('Tully-Fisher Relation:')
		print('----------------------')
		print(f'Slope (a): {a}')
		print(f'Intercept (b): {b}')
		print(f'Intrinsic Scatter: {sigmaTF_intr}')
		print('\n')


	def calc_spectrum_loglike(self, pars):
		'''
		Computes log likelihood for the spectrum fit

		Args:
			pars (dict): Dictionary of model parameters

		Attributes:
			meta_pars (float): Slit angle
			data_info (2d array): Spectrum and spectrum variance

		Returns:
				float: log likelihood spectrum
		'''
		return self._spectrum_loglike(pars)


	def _loglike_slit(self, pars):
		'''
		Calculate the log-likelihood of the slit data.

		Parameters:
		- pars (dict): A dictionary containing the parameter values for the emission lines.

		Returns:
		- log_like (float): The calculated log-likelihood of the slit data.
		'''
		log_like = 0.

		for i in range(len(self.spec_model)):
			line = self.config.galaxy_params.line_species[i]
			this_line_dict = {**pars['shared_params'], **pars[f'{line}_params']}
			model_spec = self.spec_model[i].get_observable(this_line_dict)
			log_like += np.sum((self.data_spec[i] - model_spec)**2 / (self.data_spec[i] + self.var_spec[i] + self.cont_model[i][:, np.newaxis]))

		return log_like


	def _loglike_ifu(self, pars):
		'''
		Calculate the log-likelihood of the IFU data given the model parameters.

		Parameters:
		- pars (dict): A dictionary containing the model parameters.

		Returns:
		- log_like (float): The log-likelihood of the IFU data.

		'''
		this_line_dict = {**pars['shared_params'], **pars[f'{self.config.vmap_type}_params']}
		model_2Dmap = self.IFU_model.get_observable(this_line_dict)
		log_like = np.sum((self.data_2Dmap - model_2Dmap)**2 * self.mask_2Dmap/ (self.var_2Dmap))

		return log_like


	def calc_image_loglike(self, pars):
		'''
		Computes log likelihood for the image fit

		Args:
			pars (dict): Dictionary of model parameters

		Attributes:
			data_info (2d array): Image and image variance

		Returns:
			float: log likehood of image
		'''
		model_image = self.image_model.get_image(pars['shared_params'])

		log_like = np.sum((self.data_image - model_image)**2 / (self.var_image))

		return log_like
