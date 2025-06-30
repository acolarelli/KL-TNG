from kl_tools.intensity import build_intensity_map, BasisIntensityMap

intensity_type = 'basis'


# Specific to exp shapelets
basis_kwargs = {'nx': 100,
                'ny': 100,
                'pix_scale': 1.,
                'plane': 'obs',
                # 'beta':,
                # 'Nmax':,
                # 'psf': , ## Galsim GSObject
                }

# Specific to Imap
imap_kwargs = {'basis_type': 'exp_shapelets',
                'basis_kwargs': basis_kwargs}

datavector = 

class DataVector():
    __init__()
        nx, ny, pix_scale, 

    get_psf(self, psf=None):

        if psf is not None:
            return self.psf


imap = build_intensity_map(datavector, imap_kwargs)