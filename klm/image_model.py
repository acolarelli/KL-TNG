import numpy as np

import galsim

class ImageModel():
	def __init__(self, meta_image):
		self.meta_image = meta_image
		self.psf_image = galsim.Gaussian(fwhm=meta_image['psfFWHM'])
		self._gal_image_grid = galsim.Image(nrow=meta_image['ngrid'][0], ncol=meta_image['ngrid'][1],
											wcs=meta_image['wcs'])

		self.world_pos = galsim.CelestialCoord(meta_image['RA']*galsim.degrees, meta_image['Dec']*galsim.degrees)
		self.galsim_wcs = meta_image['wcs']

	def get_image(self, pars, return_array=True):
		'''
		Applies PSF and generates galaxy image from Galsim object

		Args:
			pars (dict): Parameter dictionary

		Returns:
			array: Galaxy image
		'''

		# disk parameters
		n_sersic = pars['sersic_image']
		rhl = pars['r_hl_disk']
		theta_int = np.pi - pars['theta_int']
		flux = 10**pars['flux']
		dx_disk = pars['dx_disk'] * rhl
		dy_disk = pars['dy_disk'] * rhl

		inclination = np.arccos(pars['cosi'])
		g1, g2 = pars['g1'], pars['g2']
		qz = pars['aspect']

		# Bulge parameters
		rhl_bulge = pars['r_hl_bulge']
		flux_bulge = 10**pars['flux_bulge']
		dx_bulge = pars['dx_bulge'] * rhl
		dy_bulge = pars['dy_bulge'] * rhl

		disk = galsim.InclinedSersic(n=n_sersic, inclination=inclination*galsim.radians, half_light_radius=rhl,
									trunc=4*rhl, scale_h_over_r=qz, flux=flux)
		disk = disk.shift(dx_disk, dy_disk)

		bulge = galsim.DeVaucouleurs(half_light_radius=rhl_bulge, trunc=4*rhl_bulge, flux=flux_bulge)
		bulge = bulge.shift(dx_bulge, dy_bulge)

		# Disk operations
		disk = disk.rotate(theta_int * galsim.radians)

		# Flip g2 since galsim follows different convention
		disk = disk.shear(g1=g1, g2=-g2)
		bulge = bulge.shear(g1=g1, g2=-g2)

		# Bulge operations

		gal = bulge + disk

		galObj = galsim.Convolution([gal, self.psf_image])

		newImage = galObj.drawImage(image=self._gal_image_grid)

		if return_array is False:
			return newImage

		return newImage.array.copy()
