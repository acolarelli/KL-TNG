import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm, ticker
import numpy as np
import os

import astropy.units as u
from astropy import wcs
from astropy.io import fits
from pafit import fit_kinematic_pa as fitpa
import joblib
import galsim

from klm.parameters import Parameters
from klm.spec_model import IFUModel
from klm.image_model import ImageModel
from klm.mock import Mock
import klm.utils
from klm.ultranest_sampler import UltranestSampler

import argparse
from mpl_toolkits.axes_grid1 import make_axes_locatable
import yaml

import corner
import getdist

'''
PAFit and citation can be found here: https://pypi.org/project/pafit/
'''
'''
To Do:
A lot of the time shear parameters either hit one prior or both, find out what's causing this
Find a way/write code to estimate velocity scale radius/Rmax_ST if necessary
Maybe create a function to make priors narrower for known values like cosi
Go back to rotator_sample and bugfix more thoroughly

'''

#iMaNGA_VAC.fits can be downloaded from here: https://www.tng-project.org/data/docs/specifications/#sec5_4
hdu_lst = fits.open('/home/acolarelli/path/to/venv/bin/kl_measurement-manga/iMaNGA_vac.fits')

with open('rotator_indices.txt', 'r') as file:
    rotator_indices = file.readlines()
rotator_indices = list(map(int, rotator_indices))

gen_info = hdu_lst[1].data
image_morphology = hdu_lst[2].data
spatial_info = hdu_lst[3].data #position of each spaxel, [0] for arcsec and [1] for r_eff
kinematics = hdu_lst[4].data #kinematics[0]: Stellar velocity along the LOS (km/s) and kinematics[1]: Stellar velocity dispersion along the LOS (km/s)

inclinations = image_morphology['inc_kin'] #Kinematics-dependent galaxy inclination, assuming the kinematics provided by TNG50 (deg)
r_effs = image_morphology['sersic_reff_arsec'] #Sersic effective radius (arcsec)
sersics = image_morphology['sersic_n'] #Sersic index
t_morph = image_morphology['T_morph'] #T-value, late type T=3
snr = image_morphology['snr_2Dsersic_fit']
st_mass = gen_info['TNG_tot_stellar_mass'] #total stellar mass in units of 10^10 solar masses
zshift = gen_info['obs_redshift']

''''''

def get_FOV_mask(gal):
    imanga_FOV = spatial_info[3,gal,:,:]
    FOV_mask = np.where(imanga_FOV>-1, True, False)
    return FOV_mask #this had a use in an older draft, keeping for now just in case

def get_log10_m_star(gal):
    #iMaNGA gives mass in units of 10^10 solar masses
    return float(np.log10(float(st_mass[gal]))+10)

def get_redshift(gal):
    return float(zshift[gal])

def get_vmap_data(gal):
    kin = np.ma.masked_invalid(kinematics[0,gal,:,:])
    return kin

''''''

def get_mock_params(gal):
    cosi = float(np.cos(np.radians(float(inclinations[gal])))) 
    spax_coords = np.nonzero((spatial_info[1,gal,:,:]>1.0) & (kinematics[0,gal,:,:] == kinematics[0,gal,:,:] ))

    #subtracting 75 to make (0,0) the galactic center
    xbin = spax_coords[0] - 75
    ybin = spax_coords[1] - 75
    vels = kinematics[0,gal,spax_coords[0],spax_coords[1]] - np.nanmedian(kinematics[0,gal,spax_coords[0],spax_coords[1]]) #subtract estimate of systemic velocity
    disps = kinematics[1,gal,spax_coords[0],spax_coords[1]]
    pos_angle, pa_err, vsys_correction = fitpa.fit_kinematic_pa(xbin,ybin,vels,nsteps=361, quiet=False, plot=False, dvel=disps)
    theta = np.arctan(ybin/xbin) - np.radians(pos_angle)

    vels = (vels + vsys_correction) * (np.pi/2) / (np.cos(theta) * np.sin(np.radians(float(inclinations[gal]))))
    r_hl = r_effs[gal]
    vsys = vsys_correction + np.nanmedian(kinematics[0,gal,spax_coords[0],spax_coords[1]])
    n = sersics[gal]

    #Making a rough estimate for v_circ (unsure how to get vscale radius)
    vcirc_est = np.nanmedian(np.abs(vels)) / np.arctan(1.)


    #Printing some known or given values for later reference
    print("Theta_int (rad): " + str(np.radians(pos_angle)))
    print("i (deg): "+str(inclinations[gal]))
    print("cosi: " + str(cosi))
    print("r_hl: "+str(r_hl))
    print("PAFit vsys: "+str(vsys))

    mocks = {
    'shared_params-g1': 0.,
    'shared_params-g2': 0.,
    'shared_params-vcirc': vcirc_est,
    'shared_params-cosi': cosi,
    'shared_params-theta_int': float(np.radians(pos_angle)),  # in radians
    'shared_params-sersic_image': n,
    'shared_params-r_hl_disk': r_hl,  # in arcsec
    'shared_params-dx_disk': 0.,  # fraction of r_hl_disk
    'shared_params-dy_disk': 0.,  # fraction of r_hl_disk
    'shared_params-flux': 4,  # arbitrary units
    'shared_params-r_hl_bulge': 0.,  # in arcsec
    'shared_params-flux_bulge': 1,  # arbitrary units
    'shared_params-dx_bulge': 0.,  # fraction of r_hl_disk
    'shared_params-dy_bulge': 0.,  # fraction of r_hl_disk
    'shared_params-vscale': r_hl,  # in arcsec #not always a great estimate, fix later
    'stellar_params-v_0': vsys,  # in km/s
    'stellar_params-dx_vel': 0.,  # fraction of r_hl_disk
    'stellar_params-dx_vel': 0.,  # fraction of r_hl_disk
    }

    return mocks

def param_to_dict(mock_params):
    params=Parameters(line_species=["stellar"])
    updated_dict = params.gen_param_dict(par_names=mock_params.keys(), par_values=mock_params.values())
    return updated_dict

''''''

def write_data_info(gal_index, run_num, save_path = "/home/acolarelli/test_chain/"):
    
    mock_params = get_mock_params(gal_index)
    updated_dict = param_to_dict(mock_params)
    #FOV_mask = get_FOV_mask(gal_index)

    #Create directory to save data
    folder_name = "galaxy" + str(gal_index)
    save_path1 = save_path+folder_name
    save_path2 = save_path + folder_name + "/run"+ str(run_num)
    os.makedirs(save_path2, exist_ok=True)

    REDSHIFT = get_redshift(gal_index)
    LOG10_MSTAR = get_log10_m_star(gal_index) #float(np.log10(st_mass[gal_index])) + 10.
    LOG10_MSTAR_ERR = 0.0
    RA_OBJ, DEC_OBJ = 180.0*u.deg, 32.0*u.deg

    #OUTPUT SETTINGS
    IMAGE_SNR = 80 #snr[gal_index]?
    IMAGE_SHAPE = (150, 150) # (x, y)
    SKY_VAR_IMAGE = np.ones(IMAGE_SHAPE)*1500
    IMAGE_PIX_SCALE = 0.5 # arcsec/pix
    IMAGE_PSF_FWHM = 2.5 # arcsec
    '''
    https://arxiv.org/pdf/2203.11575
    iMaNGA "paper I"

    "The typical fibre-convolved point spread function 
    (PSF) has full width at half-maximum (FWHM) of 2.5 arcsec (Law 
    et al. 2015)"
    '''

    # Create WCS
    # Required so that the model image is rendered on the same grid as the image
    AP_WCS = wcs.WCS(naxis=2)
    AP_WCS.wcs.crpix = [np.round(IMAGE_SHAPE[0]/2), np.round(IMAGE_SHAPE[1]/2)]  # Central pixel
    AP_WCS.wcs.crval = [180, 32]  # RA, Dec at central pixel
    AP_WCS.wcs.ctype = ['RA---TAN' , 'DEC--TAN']  # Projection, see: https://docs.astropy.org/en/stable/wcs/supported_projections.html
    AP_WCS.wcs.cdelt = [1, 1]
    AP_WCS.wcs.pc = np.array([[-IMAGE_PIX_SCALE/3600, 0],
                            [0, IMAGE_PIX_SCALE/3600]]) # convert arcsec/pix -> deg/pix
    GALSIM_WCS = galsim.AstropyWCS(wcs=AP_WCS)


    meta_image = {'ngrid': IMAGE_SHAPE,
                'pixScale': IMAGE_PIX_SCALE,
                'psfFWHM': IMAGE_PSF_FWHM,
                'wcs': GALSIM_WCS,
                'ap_wcs': AP_WCS,
                'RA': RA_OBJ.value, # not required, but should they be changed anyway?
                'Dec': DEC_OBJ.value}

    image_model = ImageModel(meta_image=meta_image)
    image_data = image_model.get_image(updated_dict['shared_params']) #Create galaxy image from input data
    image_var = image_data + SKY_VAR_IMAGE

    # Set variance to match SNR
    image_var = Mock._set_snr(image_data, image_var, IMAGE_SNR, 'image', verbose=False)

    fig = plt.figure(figsize=(12, 6))

    ax = fig.add_subplot(121)
    ax.imshow(image_data)
    ax.set_title('Simulated Image', fontsize=12)

    ax = fig.add_subplot(122, projection=AP_WCS)
    lon = ax.coords[0]
    lat = ax.coords[1]
    lon.display_minor_ticks(True)
    lat.display_minor_ticks(True)
    ax.grid(color='white', ls='solid')

    # Show galaxy image
    ax.imshow(image_data)
    ax.set_title('with RA & Dec grid', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{save_path2}/simulated_image.png')

    kin = get_vmap_data(gal_index)

    VMAP_SHAPE = (150, 150)  # Dec, RA
    VMAP_PIX_SCALE = 0.5  # arcsec/pix
    VMAP_SNR = 100 #snr[gal_index]?
    VMAP_VAR = np.ones(kin.shape)*20 #pranjal says 20km/s, overplot zero shear with current fit to investigate degeneracy structure


    # Grids must be zero centered
    Dec_grid, RA_grid = klm.utils.build_map_grid(Nx=VMAP_SHAPE[0], Ny=VMAP_SHAPE[1], pix_scale=VMAP_PIX_SCALE)
    Dec_grid = np.flip(Dec_grid)

    meta_vmap = {'RA_grid': RA_grid,
                'Dec_grid': Dec_grid}
    vmap_data = kin
    vmap_var = VMAP_VAR

    # Set variance to match SNR
    vmap_var = Mock._set_snr(vmap_data, vmap_var, VMAP_SNR, 'image', verbose=False)

    plt.figure(figsize=(6, 5))

    plt.pcolormesh(RA_grid, Dec_grid, vmap_data)
    plt.xlabel('$\Delta$ RA [arcsec]')
    plt.ylabel('$\Delta$ Dec [arcsec]')

    plt.colorbar(label='V [km/s]')
    plt.savefig(f'{save_path2}/vel_map.png')

    '''
    plt.clf()

    plt.imshow(kinematics[0,gal_index,:,:])
    plt.colorbar()
    plt.savefig(f"{save_path2}/original_vmap.png")

    plt.clf()

    plt.imshow(spatial_info[3,gal_index,:,:])
    plt.colorbar()
    plt.savefig(f"{save_path2}/fov_mask.png")
    '''

    plt.clf()

    plt.imshow(vmap_var)
    plt.colorbar(label='Vmap variance [km/s]')
    plt.savefig(f"{save_path2}/vmap_var.png")

    data_info = {}

    data_info['image'] = {}
    data_info['image']['data'] = image_data
    data_info['image']['var'] = image_var
    data_info['image']['par_meta'] = meta_image


    data_info['galaxy'] = {}
    data_info['galaxy']['ID'] = '000'
    data_info['galaxy']['RA'] = RA_OBJ
    data_info['galaxy']['Dec'] = DEC_OBJ
    data_info['galaxy']['redshift'] = REDSHIFT
    data_info['galaxy']['log10_Mstar'] = LOG10_MSTAR
    data_info['galaxy']['Rmax_ST'] = 2 #Need to find a way to get this
    data_info['vmap'] = {}


    vmap_stellar = {}
    vmap_stellar['par_meta'] = meta_vmap

    vmap_stellar['data'] = vmap_data
    vmap_stellar['var'] = vmap_var
    vmap_stellar['mask'] = np.ones_like(vmap_data)


    data_info['vmap']['stellar'] = vmap_stellar
    del data_info['image']['par_meta']['wcs']
    data_info['mock_params'] = mock_params

    joblib.dump(data_info, f'{save_path1}/mock_data.pkl')







def sampler(gal_index, run_num, save_path = "/home/acolarelli/test_chain/", 
            config_path = '/home/acolarelli/path/to/venv/bin/kl_measurement-manga/config/iMaNGA_config.yaml'):
    
    folder_name = "galaxy" + str(gal_index)
    save_path1 = save_path+folder_name
    save_path2 = save_path+folder_name+"/run"+ str(run_num)

    matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['xtick.bottom'] = True
    matplotlib.rcParams['xtick.top'] = False
    matplotlib.rcParams['ytick.right'] = False
    # Axes options
    matplotlib.rcParams['axes.titlesize'] = 'x-large'
    matplotlib.rcParams['axes.labelsize'] = 'x-large'
    matplotlib.rcParams['axes.edgecolor'] = 'black'
    matplotlib.rcParams['axes.linewidth'] = '1.0'
    matplotlib.rcParams['axes.grid'] = False
    matplotlib.rcParams['legend.fontsize'] = 'large'
    matplotlib.rcParams['legend.labelspacing'] = 0.77
    matplotlib.rcParams['savefig.bbox'] = 'tight'
    matplotlib.rcParams['savefig.format'] = 'pdf'
    matplotlib.rcParams['savefig.dpi'] = 300

    plt.rc('figure', dpi=300)

    data_info = joblib.load(f'{save_path1}/mock_data.pkl')

    # Re-instantiate the galsim WCS object using the astropy wcs
    ap_wcs = data_info['image']['par_meta']['ap_wcs']
    data_info['image']['par_meta']['wcs'] = galsim.AstropyWCS(wcs=ap_wcs)

    # Load YAML file for config
    with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)

    config['TFprior']['log10_vTF'] = np.log10(data_info['mock_params']['shared_params-vcirc'])
    config['TFprior']['sigmaTF_intr'] = 0.05
    config['galaxy_params']['vmap_type'] = 'stellar'
    
    config['galaxy_params']['log10_Mstar'] = data_info['galaxy']['log10_Mstar']
    config['galaxy_params']['log10_Mstar_err'] = 0.

    inference = UltranestSampler(data_info=data_info, config=config)
    inference.config
    inference.__dict__.keys()

    sampler = inference.run(output_dir=save_path1, test_run=False, run_num=run_num)
    sampler.print_results()
    sampler.plot()
    sampler.plot_trace()

    samples = np.array(sampler.results['weighted_samples']['points'])
    weights = np.array(sampler.results['weighted_samples']['weights'])

    latex_names = [f'${p}$' for p in inference.config.params.latex_names.values()]
    true_values = [data_info['mock_params'][p] for p in inference.config.params.names]
    MAP_values = sampler.results['maximum_likelihood']['point']

    ndim = len(latex_names)

    figure = corner.corner(samples, weights=weights, labels=latex_names, color='dodgerblue',
                show_titles=True, title_fmt='.2f', bins=20,
                plot_datapoints=False, range=np.repeat(0.999, ndim));

    axes = np.array(figure.axes).reshape((ndim, ndim))

    # Loop over the diagonal
    for i in range(ndim):
        ax = axes[i, i]
        ax.axvline(true_values[i], color="r")

    # Loop over the histograms
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(true_values[xi], color="r")
            ax.axhline(true_values[yi], color="r")
            ax.plot(true_values[xi], true_values[yi], "sr")
            ax.plot(MAP_values[xi], MAP_values[yi], marker='*', c="purple", ms=12)

    axes[0, 1].plot([], [], 'sr', label='True value')
    axes[0, 1].plot([], [], marker='*', c="purple", ms=12, label='MAP')
    axes[0, 1].legend()
    plt.savefig(f'{save_path2}/corner.png')

    best_fit_dict = inference.params.gen_param_dict(inference.config.params.names, MAP_values)

    chi2 = 0.5*inference.calc_spectrum_loglike(best_fit_dict)
    chi2_red = chi2/np.sum(inference.mask_2Dmap)

    image_chi2 = 0.5*inference.calc_image_loglike(best_fit_dict)

    this_line_dict = {**best_fit_dict['shared_params'], **best_fit_dict['stellar_params']}
    spec = inference.IFU_model.get_observable(this_line_dict)
    image = inference.image_model.get_image(best_fit_dict['shared_params'])


    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    data = inference.data_2Dmap
    mask = inference.mask_2Dmap

    im1 = ax[0, 0].imshow(data*mask, cmap=plt.cm.RdBu, vmin=-300, vmax=300, aspect='auto')
    im2 = ax[0, 1].imshow(spec*mask, cmap=plt.cm.RdBu, vmin=-300, vmax=300, aspect='auto')
    im3 = ax[0, 2].imshow((data-spec)*mask, cmap=plt.cm.RdBu, aspect='auto')


    divider = make_axes_locatable(ax[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im1, cax=cax)
    plt.sca(ax[0, 0])

    divider = make_axes_locatable(ax[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im2, cax=cax)
    plt.sca(ax[0, 1])

    divider = make_axes_locatable(ax[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im3, cax=cax)
    plt.sca(ax[0, 2])

    ax[0, 0].set(title='TNG Data')
    ax[0, 1].set(title='Best Fit')
    ax[0, 2].set(title='Residual')

    ax[0, 2].text(0.1, 0.8, f'$\chi^2$={chi2:.2f}', c='k', transform=ax[0,2].transAxes)
    ax[0, 2].text(0.1, 0.7, f'$\chi^2_\\nu$={chi2_red:.2f}', c='k', transform=ax[0,2].transAxes)

    ### Image 
    x_min, x_max = inference.IFU_model.obs_xx.min(), inference.IFU_model.obs_xx.max()
    y_min, y_max = inference.IFU_model.obs_yy.min(), inference.IFU_model.obs_yy.max()

    data = inference.data_image
    var = inference.var_image

    im1 = ax[1, 0].contourf(data/var, vmin=(data/var).min(), vmax=(data/var).max(), aspect='auto')
    im2 = ax[1, 1].contourf(image/var, vmin=(data/var).min(), vmax=(data/var).max(), aspect='auto')
    im3 = ax[1, 2].contourf((image-data)/var, aspect='auto')

    divider = make_axes_locatable(ax[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im2, cax=cax)
    plt.sca(ax[1, 1])

    divider = make_axes_locatable(ax[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im3, cax=cax)
    plt.sca(ax[1, 2])

    ax[0, 0].text(0.1, 0.8, "Velocity Map", c='k', fontsize=20, transform=ax[0, 0].transAxes)
    ax[1, 0].text(0.1, 0.8, "Image", c='white', fontsize=20, transform=ax[1,0].transAxes)
    ax[1, 2].text(0.1, 0.8, "$\chi^2=$"+f"{image_chi2:.2f}", c='white', fontsize=20, transform=ax[1,2].transAxes)
    plt.tight_layout()
    plt.savefig(f'{save_path2}/best_fit_image.png')
    with open( f'{save_path2}/config.yaml', 'w') as file:
        yaml.dump(inference.config.__repr__, file)

def run_pipeline(gal_index, run_num=0, save_path = "/home/acolarelli/test_chain/", 
                 config_path = '/home/acolarelli/path/to/venv/bin/kl_measurement-manga/config/iMaNGA_config.yaml', preserve_old_runs = False):
    print("Galaxy#: "+str(gal_index))
    if preserve_old_runs:
        new_run_num = 0
        while new_run_num <= run_num:
            new_run_num += 1
        run_num = new_run_num
    write_data_info(gal_index,run_num=run_num,save_path=save_path)
    sampler(gal_index,run_num=run_num,save_path=save_path,config_path=config_path)


''''''
#Note that the file path for the iMaNGA VAC at the start of the code needs to be manually changed for the script to run. 
#Save paths and config file path can be changed below.
''''''
#Change the list index to change which galaxy is being put in, or manually choose a galaxy index from rotator_indices.txt
run_pipeline(rotator_indices[14], preserve_old_runs=True)
hdu_lst.close()