import joblib
import numpy as np
import scipy.optimize

from klm.parameters import Parameters
from klm.spec_model import SlitModel
from klm.image_model import ImageModel
import klm.utils as utils

class Mock:
	def __init__(self, data_info, line_species):
		pass

	@classmethod
	def build_mock(cls, settings):
		data_info = settings['data_info']
		line_species = settings['line_species']
		fid_params = settings['fid_params']
		line_profile_path = settings['line_profile_path']


		if type(line_species) is list:
			assert len(line_species) == 1, 'Only one line species allowed for now'
			line_species = line_species[0]

		# Create parameter object and update fiducial values
		meta_gal = data_info['galaxy']
		params = Parameters({'shared_params':{'beta':meta_gal['beta']}}, line_species=[line_species])
		updated_dict = params.gen_param_dict(fid_params.keys(), fid_params.values())

		## Now we start building mock data

		image_snr = settings.get('image_snr', None)
		spec_snr = settings.get('spec_snr', None)

		# First create mock image
		meta_image = data_info['image']['par_meta']
		image_var = data_info['image']['var']
		image_model = ImageModel(meta_image=meta_image)
		image_data = image_model.get_image(updated_dict['shared_params'])

		if image_snr is not None:
			image_var = cls._set_snr(image_data, image_var, image_snr, 'image')

		# Now create mock spectrum
		for i, spec_info in enumerate(data_info['spec']):
			if spec_info['par_meta']['line_species'] == line_species:
				idx = i
				break

		meta_spec = data_info['spec'][idx]['par_meta']
		spec_var = data_info['spec'][idx]['var']
		cont_model_spec = data_info['spec'][idx]['cont_model']

		meta_spec = cls._set_line_profile(meta_spec, line_species, line_profile_path)
		spec_model = SlitModel(obj_param=meta_gal, meta_param=meta_spec)
		this_line_dict = {**updated_dict['shared_params'], **updated_dict[f'{line_species}_params']}
		spec_data = spec_model.get_observable(this_line_dict)

		if spec_snr is not None:
			spec_var = cls._set_snr(spec_data, spec_var, spec_snr, 'spec')

		spec_data_info = {'data': spec_data,
    				'var': spec_var,
				'cont_model': cont_model_spec,
				'par_meta': meta_spec}

		image_data_info = {'data': image_data,
				'var': image_var,
				'par_meta': meta_image}


		mock_data_info = {'spec': [spec_data_info],
			'image': image_data_info,
			'galaxy': meta_gal,
			'par_fit': Parameters._flatten(updated_dict),
			'image_snr': cls._calculate_snr(image_data, image_var, 'image'),
			'spec_snr': cls._calculate_snr(spec_data, spec_var, 'spec')}

		return mock_data_info

	@classmethod
	def _set_snr(cls, data, var, snr, data_type, verbose=True):
		idx = np.where(~np.isfinite(var))
		var[idx] = 0.
		data[idx] = 0.

		if verbose:
			print('Initial SNR:', cls._calculate_snr(data, var, data_type))
		# we want to scale the variance such that the SNR is equal to the desired SNR
		# use scipy optimize to find the scaling factor
		scaled_snr = lambda _, scale: cls._calculate_snr(data, var*scale, data_type)

		if data_type=='spec':
			scale = scipy.optimize.curve_fit(scaled_snr, [1], [snr], bounds=(0, 100))[0][0]

		elif data_type=='image':
			scale = (scaled_snr(1,1)/snr)**2  # (Old/desired)**2

		if verbose:
			print(f'Scaling {data_type} variance by {scale}\n')

		new_var = var*scale

		if verbose:
			print('Final SNR:', cls._calculate_snr(data, new_var, data_type))

		return new_var

	@staticmethod
	def _set_line_profile(meta_param, line_species, line_profile_path):
		f = open(line_profile_path, 'rb')
		line_profile = joblib.load(line_profile_path)[line_species]
		# f.close()
		Amp, sigma1, sigma2 = line_profile[0], line_profile[2], line_profile[3]

		meta_param['Amp'] = Amp
		meta_param['sigma1'] = sigma1
		meta_param['sigma2'] = sigma2

		return meta_param

	@staticmethod
	def _calculate_snr(data, var, data_type):
		if data_type == 'image':
			calculate_snr = utils.calculate_image_snr

		elif data_type == 'spec':
			calculate_snr = utils.calculate_spec_snr

		return calculate_snr(data, var)
