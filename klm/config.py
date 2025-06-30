from functools import reduce
import numpy as np
import scipy.stats

from klm.parameters import Parameters
class Config():
	def __init__(self, config_dict=None):
		if config_dict is None or config_dict is {}:
			print('[WARNING]: No config file provided. Using default values.')
			config_dict = {}
		self._set_config(config_dict)

	def _set_config(self, config_dict):
		self.init_galaxy_params(config_dict)
		self.init_TFprior(config_dict)
		self.init_params(config_dict)
		self.init_likelihood(config_dict)

		# Misc
		self.verbose = config_dict.get('verbose', False)

	def init_likelihood(self, config_dict):
		likelihood_params = config_dict['likelihood']
		self.likelihood = Container()
		# General options
		self.likelihood.isFitImage = likelihood_params.get('fit_image', True)
		self.likelihood.isFitSpec = likelihood_params.get('fit_spec', True)
		self.likelihood.fid_params = likelihood_params.get('fid_params', None)
		self.likelihood.set_non_analytic_prior = likelihood_params.get('set_non_analytic_prior', None)  # should be a dict of prior for each parameter

		if all(f'shared_params-{p}' in self.params.names for p in ['r_hl_disk', 'r_hl_bulge']):
			self.likelihood.apply_rhl_constraint = True

		else:
			self.likelihood.apply_rhl_constraint = False
			print('[WARNING]: r_hl_disk and r_hl_bulge not in shared_params. r_hl_bulge< r_hl_disk constraint not enforced! \n')


	def init_galaxy_params(self, config_dict):
		'''
		Initializes galaxy parameters based on the observation type.
		'''
		self.galaxy_params = Container()
		galaxy_params = config_dict.get('galaxy_params', {})

		self.galaxy_params.obs_type = galaxy_params.get('obs_type', 'slit')
		self.galaxy_params.rc_type = galaxy_params.get('rc_type', 'arctan')

		if self.galaxy_params.obs_type == 'slit':
			line_species = galaxy_params.get('line_species', 'Halpha')
			if isinstance(line_species, str):
				self.galaxy_params.line_species = [line_species]

			elif isinstance(line_species, list):
				if len(line_species)>1:
					assert all(x==line_species[0] for x in line_species), 'Can only fit multi-slit obs. of same emission line'
				self.galaxy_params.line_species = line_species

			self.galaxy_params.line_profile_path = galaxy_params.get('line_profile_path', None)
			assert config_dict.get('vmap_type', None) is None, 'vmap_type should not be set for slit data'
			self.vmap_type = None

		elif self.galaxy_params.obs_type == 'IFU':
			self.galaxy_params.Rmax_G = get_key_dtype(galaxy_params, 'Rmax_G')
			self.galaxy_params.Rmax_ST = get_key_dtype(galaxy_params, 'Rmax_ST')

			self.vmap_type = galaxy_params.get('vmap_type', 'gas')
			assert config_dict.get('line_species', None) is None, 'line_species should not be set for IFU data'
			self.galaxy_params.line_species = [self.vmap_type]

		self.galaxy_params.log10_Mstar = get_key_dtype(galaxy_params, 'log10_Mstar')
		self.galaxy_params.log10_Mstar_err = get_key_dtype(galaxy_params, 'log10_Mstar_err', 0.)

		#If any attribute is None, raise an error
		for attr in self.galaxy_params.__dict__:
			if getattr(self.galaxy_params, attr) is None:
				if attr == 'line_profile_path':
					print('[WARNING]: line_profile_path is set to None')
					continue

# 				raise ValueError(f'{attr} is not set in galaxy_params')
				print(f'[WARNING]: {attr} is not set in galaxy_params')


	def init_TFprior(self, config_dict):
		'''
		Initializes the Tully-Fisher prior based on the observation type.
		'''
		self.TFprior = Container()
		TFprior = config_dict.get('TFprior', {})
		self.TFprior.use_TFprior = TFprior.get('use_TFprior', True)

		if self.TFprior.use_TFprior is False:
			return

		# First check if log10vTF is set
		if TFprior.get('log10_vTF', None) is not None:
			self.TFprior.log10_vTF = TFprior['log10_vTF']
			self.set_sigmaTF(TFprior)
			self.TFprior.a = None
			self.TFprior.b = None


		# Next we check if the relation is supplied
		elif TFprior.get('relation', None) is not None:
			a = TFprior.get('a', None)
			b = TFprior.get('b', None)
			self.set_sigmaTF(TFprior)

			log10_Mstar = self.galaxy_params.log10_Mstar
			log10_Mstar_err = self.galaxy_params.log10_Mstar_err

			self.TFprior.log10_vTF = eval(TFprior['relation'])
			self.TFprior.a = a
			self.TFprior.b = b

		else:
			print('[INFO]: TF prior not specified. Using default values.')
			self._init_TF_relation()


	def init_params(self, config_dict):
		self.params = Container()
		params = config_dict.get('params', {})

		shared_params_dict = params.get('shared_params', {})
		shared_params_dict = {} if shared_params_dict is None else shared_params_dict

		line_params_dict = params.get('line_params', {})
		line_params_dict = {} if line_params_dict is None else line_params_dict

		shared_params = Parameters._flatten(shared_params_dict, level=0)
		line_params = Parameters._flatten(line_params_dict, level=0)

		self.params.names = []
		self.params.latex_names = {}

		self.params.prior = {}

		# Iterate over shared params
		for p in shared_params.keys():
			this_prior = shared_params[p].get('prior', None)
			latex_name = shared_params[p].get('latex_name', p)
			if this_prior is None:
				raise ValueError(f'Prior not set for {p}')

			p = 'shared_params-'+p
			self._init_prior(p, this_prior)

			self.params.names.append(p)
			self.params.latex_names[p] = latex_name

		# Iterate over line params
		doublet_params = ['v_0_2', 'dx_vel_2', 'dy_vel_2', 'v_0_2', 'I02', 'f2_1', 'f2_2']
		for p in line_params.keys():
			this_prior = line_params[p].get('prior', None)
			latex_name = line_params[p].get('latex_name', p)

			# Iterate over all lines
			for line in self.galaxy_params.line_species[:1]:
				# Remove doublet params for singlet lines
				if any([name in p for name in doublet_params]) and line not in ['OII', 'OIIa', 'OIIb']:
					print(f'Removed {p} from {line} fit parameters...')
					continue

				# Now add line name to the parameter and latex name
				p = f'{line}_params-{p}'
				latex_name = latex_name# + ' ' + line

				if this_prior is None:
					raise ValueError(f'Prior not set for {p}')

				self._init_prior(p, this_prior)
				self.params.names.append(p)
				self.params.latex_names[p] = latex_name

	def _init_prior(self, param_name, prior_dict):
		'''
		Initialize the prior distribution for a given parameter.

		Args:
			param_name (str): The name of the parameter.
			prior_dict (dict): A dictionary containing the prior information.

		Returns:
			None
		'''
		if 'min' in prior_dict.keys():
			min_val, max_val = prior_dict.get('min', None), prior_dict.get('max', None)
			if type(min_val) is str:
				min_val = eval(min_val)
			if type(max_val) is str:
				max_val = eval(max_val)
			self.params.prior[param_name] = [min_val, max_val]

		if 'norm' in prior_dict.keys():
			loc = reduce(getattr, [self]+prior_dict['norm']['loc'].split('.'))  # Hack to get attribute of attribute
			scale = reduce(getattr, [self]+prior_dict['norm']['scale'].split('.')) # From https://stackoverflow.com/questions/4247036/python-recursively-getattribute
			self.params.prior[param_name] = scipy.stats.norm(loc=loc, scale=scale)

	def _init_TF_relation(self):
		'''
		Initializes the Tully-Fisher relation based on the observation type and velocity map type.
		For slit data uses relation from Miller et al. 2011: https://arxiv.org/pdf/1102.3911.pdf
		For IFU data uses relation from Ristea et al. 2023: https://arxiv.org/pdf/2311.13251.pdf
		'''
		log10_Mstar = float(self.galaxy_params.log10_Mstar)
		log10_Mstar_err = float(self.galaxy_params.log10_Mstar_err)

		if self.galaxy_params.obs_type == 'IFU':
			if self.vmap_type == 'stellar':

				if self.galaxy_params.Rmax_ST == 1:
					a = 0.282
					b = -0.78
					sigmaTF_intr = 0.07

				if self.galaxy_params.Rmax_ST == 1.3:
					a = 0.279
					b = -0.73
					sigmaTF_intr = 0.06

				if self.galaxy_params.Rmax_ST == 2:
					a = 0.27
					b = -0.60
					sigmaTF_intr = 0.05

				print(f'[INFO]: Using TFR for Rmax_ST = {self.galaxy_params.Rmax_ST}')

			elif self.vmap_type == 'gas':

				if self.galaxy_params.Rmax_G == 1:
					a = 0.282
					b = -0.75
					sigmaTF_intr = 0.06

				if self.galaxy_params.Rmax_G == 1.3:
					a = 0.275
					b = -0.65
					sigmaTF_intr = 0.06

				if self.galaxy_params.Rmax_G == 2:
					a = 0.26
					b = -0.48
					sigmaTF_intr = 0.04

				print(f'[INFO]: Using TFR for Rmax_G = {self.galaxy_params.Rmax_G}')

			TF_relation_str = 'log10_Mstar * a + b'
			log10_vTF = eval(TF_relation_str)
			self.TFprior.sigmaTF = sigmaTF_intr

		elif self.galaxy_params.obs_type == 'slit':
			# Stellar mass - TF relation
			a = 1.718
			b = 3.869
			sigmaTF_intr = 0.058
			TF_relation_str = '(log10_Mstar - a) / b'
			log10_vTF = eval(TF_relation_str)
			self.TFprior.sigmaTF = (sigmaTF_intr**2 + (log10_Mstar_err/b)**2)**0.5

		self.TFprior.a = a
		self.TFprior.b = b
		self.TFprior.sigmaTF_intr = sigmaTF_intr
		self.TFprior.log10_vTF = log10_vTF
		self.TFprior.TF_relation_str = TF_relation_str

		# if self.config.verbose:
		# 	print('\n')
		# 	print('Initializing TFR...')
		# 	print(f'Stellar Mass is 10^{log10_Mstar:.2f}')
		# 	print(f'log10_v prior at {self.TFprior.log10_vTF:0.2f} dex')

	def __repr__(self):
		config_str = repr(self)
		# return the string
		return config_str

	def set_sigmaTF(self, TFprior):
		# Check if sigmaTF is set
		if TFprior.get('sigmaTF', None) is None:
			if  TFprior.get('sigmaTF_intr', None) is None:
				raise ValueError('sigmaTF or sigmaTF_intr not set in TFprior')
			else:
				self.TFprior.sigmaTF = TFprior['sigmaTF_intr']
				print('[INFO]: Using intrinsic scatter as sigmaTF')

		else:
			self.TFprior.sigmaTF = TFprior['sigmaTF']


class Container():
	def __init__(self):
		pass

	def __repr__(self):
		config_str = repr(self)
		# return the string
		return config_str

def repr(self):
	# Get all the attributes of the class
	attributes = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]

	# Initialize an empty string
	config_str = '\n'

	# Loop over all the attributes
	for attr in attributes:
		# Get the value of the attribute
		value = getattr(self, attr)
		print(attr)
		# Add the attribute name and value to the string
		if isinstance(value, dict):
			config_str += f'{attr}:\n'
			for k, v in value.items():
				v = isclassinstance(v)
				config_str += f'    {k}: {v}\n'

		elif isinstance(value, list):
			config_str += f'    {attr}:\n'
			for v in value:
				v = isclassinstance(v)
				config_str += f'    {v}\n'

		# Determine if value object is any class

		else:
			config_str += f'{attr}: {value}\n'

	return config_str

def isclassinstance(obj):
	if hasattr(obj, '__dict__'):
		return obj.__class__.__name__
	else:
		return obj

def get_key_dtype(d, key, fill_value=None, dtype=float):
    value = d.get(key, fill_value)
    if value is None:
        return value
    else:
        return dtype(value)
        
        
