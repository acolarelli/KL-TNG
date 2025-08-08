import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm, ticker
import numpy as np
import os

import astropy.units as u
from astropy import wcs
from astropy.io import fits
import astropy.stats
from astropy.convolution import convolve
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

import requests

from iMANGA_functions import apply_FOV

'''
PAFit and citation can be found here: https://pypi.org/project/pafit/

ePSF, SDSS filters, and example code for convolution (along with citation instructions) can be found here: https://github.com/lonanni/iMaNGA/blob/main/iMANGA_mockMaNGAdatacubes.ipynb

'''

'''
To Do: 
Determine if there is a better way to "flatten" the datacube than just crudely summing over the frequency axis
A lot of the time shear parameters either hit one prior or both, find out what's causing this
Find a way/write code to estimate velocity scale radius/Rmax_ST if necessary
Maybe create a function to make priors narrower for known values like cosi
Go back to rotator_sample and bugfix more thoroughly
Maybe change functions to take a "file and save path directory" dict to reduce the number of parameters

'''
np.set_printoptions(threshold=np.inf)

np.random.seed(42)

#iMaNGA_VAC.fits can be downloaded from here: https://www.tng-project.org/data/docs/specifications/#sec5_4
hdu_lst = fits.open('/home/acolarelli/path/to/venv/bin/kl_measurement-manga/iMaNGA_vac.fits')

baseUrl = 'http://www.tng-project.org/api/'
headers = {"api-key":"f419ad81a31806df57104c83a5bfa2d5"}

with open('rotator_indices.txt', 'r') as file:
    rotator_indices = file.readlines()
rotator_indices = list(map(int, rotator_indices))


#VAC Info
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

hdu_psf, headerPSF = fits.getdata("/home/acolarelli/path/to/venv/bin/kl_measurement-manga/ePSF.fits.gz", header=True)

psfg = hdu_psf[:,:, 0]
psfr = hdu_psf[:,:, 1]
psfi = hdu_psf[:,:, 2]
psfz = hdu_psf[:,:, 3]

catalog = fits.open("/home/acolarelli/path/to/venv/bin/kl_measurement-manga/iMaNGA_catalog.fits")
wave_rest = np.float64(catalog[2].data)

snr_file = np.genfromtxt("/home/acolarelli/path/to/venv/bin/kl_measurement-manga/snr_average.dat")
snr_avg = snr_file[:,1]

''''''

def get_FOV_mask(gal):
    imanga_FOV = spatial_info[3,gal,:,:]
    FOV_mask = np.where(imanga_FOV>-1, False, True)
    return FOV_mask #this had a use in an older draft, keeping for now just in case

def get_log10_m_star(gal):
    #iMaNGA gives mass in units of 10^10 solar masses
    return float(np.log10(float(st_mass[gal]))+10)

def get_redshift(gal):
    return float(zshift[gal])

def get_vmap_data(gal):
    kin = np.ma.masked_invalid(kinematics[0,gal,:,:])
    return kin

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

''''''

def get(path, params=None):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically

    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename # return the filename string

    return r

''''''

def flux_to_photon(waves, image):
    im = image.copy()
    for wv in range(len(waves)-1):
        e_photon = 1.98644568327e-08/waves[wv] #E = hc/lambda, hc in erg*angstrom
        im[wv,:,:] = image[wv,:,:]/e_photon
    return im

''''''

#This was borrowed from an iManga notebook, see link above for source and citation instructions
def convolve_psf(gal_index, datacube):
    redshift = get_redshift(gal_index)
    wave = wave_rest*(1+redshift) * 10000 #convert micron to angstrom
    wave = np.delete(wave,0)
    grid_ = get_FOV_mask(gal_index)
    grid, FoV_datacube = apply_FOV(grid_, datacube)

    grid_zero = grid.copy()
    grid_zero[grid==True] = np.nanmin(FoV_datacube[:, (grid==True)],axis=0)>0

    bkg_x = np.where((grid == True)&(grid_zero==True))[0][-1]
    bkg_y = np.where((grid == True)&(grid_zero==True))[1][-1]
    bkg_  = FoV_datacube[:, bkg_x, bkg_y]

    pflux = np.zeros((FoV_datacube.shape))
    newnoise = np.zeros((FoV_datacube.shape))

    newnoise[:,:,:] = abs(np.sqrt(FoV_datacube[:,:,:])*(np.sqrt(bkg_.reshape(-1,1,1))/snr_avg.reshape(-1,1,1)))
    pflux[:,:,:] = np.random.normal(FoV_datacube[:,:,:], scale = newnoise[:,:,:])
    #print(wave)
    #print(pflux.shape)

    #to avoid unphysical noise produced by the random process 
    pflux[np.less(pflux, 0., where=~np.isnan(pflux))] = np.min(pflux[np.greater(pflux, 0., where=~np.isnan(pflux))])
    newnoise[np.less(newnoise, 0., where=~np.isnan(newnoise))] = np.min(newnoise[np.greater(newnoise, 0., where=~np.isnan(newnoise))])


    filer_g = np.genfromtxt('/home/acolarelli/path/to/venv/bin/kl_measurement-manga/sdss/SLOAN_SDSS.g.dat')
    wave_g = filer_g[:,0]
    response_g = filer_g[:,1]

    filer_r = np.genfromtxt('/home/acolarelli/path/to/venv/bin/kl_measurement-manga/sdss/SLOAN_SDSS.r.dat')
    wave_r = filer_r[:,0]
    response_r = filer_r[:,1]

    filer_i = np.genfromtxt('/home/acolarelli/path/to/venv/bin/kl_measurement-manga/sdss/SLOAN_SDSS.i.dat')
    wave_i = filer_i[:,0]
    response_i = filer_i[:,1]

    filer_z = np.genfromtxt('/home/acolarelli/path/to/venv/bin/kl_measurement-manga/sdss/SLOAN_SDSS.z.dat')
    wave_z = filer_z[:,0]
    response_z = filer_z[:,1]
    
    pflux_300 = pflux[:,~np.isnan(pflux[300,:,:]).all((1))]
    pflux_300 = pflux_300[:,:,~np.isnan(pflux[300,:,:]).all((1))]

    newnoise_300 = newnoise[:,~np.isnan(pflux[300,:,:]).all((1))]
    newnoise_300 = newnoise_300[:,:,~np.isnan(pflux[300,:,:]).all((1))]

    grid = np.where(grid==0, np.nan,grid)
    grid_300 = grid[:,~np.isnan(grid).all(1)]
    grid_300 = grid_300[~np.isnan(grid).all(1)]

    convolve_img = np.zeros((np.shape(pflux_300)))
    convolve_noise = np.zeros((np.shape(newnoise_300)))
    print("Convolving PSF... (this may take a few minutes)")
    for k in range(len(wave)):
        if wave[k]<=wave_r.min():
            convolve_img[k, :,:] = np.where(grid_300==True, convolve(pflux_300[k, :,:], psfg), float("Nan"))
            convolve_noise[k,:,:] = np.where(grid_300==True, convolve(newnoise_300[k, :,:], psfg), float("Nan"))

        if wave_r.min()<wave[k]<=wave_g.max():
            convolve_img[k, :,:] = np.where(grid_300==True, convolve(pflux_300[k, :,:], np.mean((psfr, psfg), axis=0)), float("Nan"))
            convolve_noise[k, :,:] = np.where(grid_300==True, convolve(newnoise_300[k, :,:], np.mean((psfr, psfg), axis=0)), float("Nan"))

        if wave_g.max()<wave[k]<=wave_i.min():
            convolve_img[k, :,:] = np.where(grid_300==True, convolve(pflux_300[k, :,:], psfr), float("Nan"))
            convolve_noise[k, :,:] = np.where(grid_300==True, convolve(newnoise_300[k, :,:], psfr), float("Nan"))

        if wave_i.min()<wave[k]<=wave_r.max():
            convolve_img[k, :,:] = np.where(grid_300==True, convolve(pflux_300[k, :,:], np.mean((psfr, psfi), axis=0)), float("Nan"))
            convolve_noise[k, :,:] = np.where(grid_300==True, convolve(newnoise_300[k, :,:], np.mean((psfr, psfi), axis=0)), float("Nan"))

        if wave_r.max()<wave[k]<=wave_z.min():
            convolve_img[k, :,:] = np.where(grid_300==True, convolve(pflux_300[k, :,:],psfi), float("Nan"))
            convolve_noise[k, :,:] = np.where(grid_300==True, convolve(newnoise_300[k, :,:],psfi), float("Nan"))

        if wave_z.min()<wave[k]<=wave_i.max():
            convolve_img[k, :,:] = np.where(grid_300==True, convolve(pflux_300[k, :,:],np.mean((psfi, psfz), axis=0)), float("Nan"))
            convolve_noise[k, :,:] = np.where(grid_300==True, convolve(newnoise_300[k, :,:],np.mean((psfi, psfz), axis=0)), float("Nan"))

        if wave_i.max()<wave[k]<=wave_z.max():
            convolve_img[k, :,:] = np.where(grid_300==True, convolve(pflux_300[k, :,:],psfz), float("Nan"))
            convolve_noise[k, :,:] = np.where(grid_300==True, convolve(newnoise_300[k, :,:],psfz), float("Nan"))

        if wave[k]>wave_z.max():
            convolve_img[k, :,:] = np.where(grid_300==True, convolve(pflux_300[k, :,:],psfz), float("Nan"))
            convolve_noise[k, :,:] = np.where(grid_300==True, convolve(newnoise_300[k, :,:],psfz), float("Nan"))

    print("Converting flux density to photon count...")
    print("Convolving finished")
    convolve_img = flux_to_photon(wave, convolve_img)
    convolve_noise = flux_to_photon(wave, convolve_noise)
    convolve_flat = np.nansum(convolve_img, axis=0)
    noise_flat = np.nansum(convolve_noise,axis=0)
    return noise_flat, convolve_flat    


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

    #Making a rough estimate for v_circ (unsure if this is right or how to get vscale radius, but curve should flatten out at r = r_vscale ?)
    vcirc_est = np.nanmedian(np.abs(vels)) / np.arctan(1.)


    #Printing some known or given values for later reference
    print("Theta_int (rad): " + str(np.radians(pos_angle)))
    print("i (deg): "+str(inclinations[gal]))
    print("cosi: " + str(cosi))
    print("r_hl (arcsec): "+str(r_hl))
    print("PAFit vsys (km/s): "+str(vsys))
    print("Sersic: "+str(n))


    #As a sanity check because rotator code is acting up
    print("T-morph: "+str(t_morph[gal]))

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
    'shared_params-flux': 4e-12,  # arbitrary units
    'shared_params-r_hl_bulge': 0.,  # in arcsec
    'shared_params-flux_bulge': 1e-12,  # arbitrary units
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

def write_data_info(gal_index, run_num, file_path = "/home/acolarelli/", save_path = "/home/acolarelli/test_chain/"):

    mock_params = get_mock_params(gal_index)
    #updated_dict = param_to_dict(mock_params)
    FOV_mask = get_FOV_mask(gal_index)
    #FOV_mask_img = np.pad(FOV_mask, 75, pad_with, padder=True)
    #print(FOV_mask_img.shape)

    snap_id = gen_info['TNG_snap_id'][gal_index]
    snap = snap_id[:2]
    gal_id = snap_id[3:]
    #print(snap)
    #print(gal_id)
    #print(snap_id)



    #Check if desired datacube is already downloaded, and if not download from TNG
    url = "http://www.tng-project.org/api/TNG50-1/snapshots/"+str(snap)+"/subhalos/"+str(gal_id)+"/imanga.fits"
    if os.path.exists(file_path+"/"+str(snap)+"_"+str(gal_id)+".fits"):
        print("Existing IFU datacube found")
        data0 = fits.open(file_path+"/"+str(snap)+"_"+str(gal_id)+".fits")[0]
    else:
        print("IFU datacube not found, pulling from TNG site... (this may take a few minutes)\nIf there is a crash after downloading, try running the code again.\n") 
        r = get(url)
        data0 = fits.getdata(r)
        print("Datacube downloaded")

    

    #Sum over the wavelength/frequency axis to "flatten" the datacube and produce a 2D image
    datacube = data0.data
    grid, datacube2 = apply_FOV(FOV_mask, datacube)
    flat_data = np.nansum(datacube2, axis=0)
    
    #Do the same, but with the image convolved with the ePSF
    noise, image = convolve_psf(gal_index, datacube)


    #Create directory to save data
    folder_name = "galaxy" + str(gal_index)
    save_path1 = save_path+folder_name
    save_path2 = save_path + folder_name + "/run"+ str(run_num)
    os.makedirs(save_path2, exist_ok=True)


    #Basic info
    REDSHIFT = get_redshift(gal_index)
    LOG10_MSTAR = get_log10_m_star(gal_index) #float(np.log10(st_mass[gal_index])) + 10.
    LOG10_MSTAR_ERR = 0.0
    RA_OBJ, DEC_OBJ = 180.0*u.deg, 32.0*u.deg

    #OUTPUT SETTINGS
    IMAGE_SNR = snr[gal_index]
    IMAGE_SHAPE = image.shape


    #Noise from equation 4: https://academic.oup.com/mnras/article/515/1/320/6603844
    SKY_VAR_IMAGE = noise + 1e-18 #adding a very small value to prevent divide by zero errors, later see if there's a better way to fix this or find a better value for background sky variance
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



    #image_model = ImageModel(meta_image=meta_image)
    image_data = image #image_model.get_image(updated_dict['shared_params']) #Create galaxy image from input data
    image_var =  SKY_VAR_IMAGE #+ image_data
    #flat_data_mask = np.where(FOV_mask_img==False, image_data, np.nan)
    #image_var_mask = np.where(FOV_mask_img==False,image_var, np.nan)

    # Set variance to match SNR
    image_var = Mock._set_snr(image_data, image_var, IMAGE_SNR, 'image', verbose=True)

    '''
    plt.clf()

    plt.imshow(flat_data_mask)
    plt.colorbar(label='Image Data')
    plt.savefig(f"{save_path2}/image_masked.png")


    plt.clf()
    plt.imshow(image_var_mask)
    plt.colorbar(label='Image Variance')
    plt.savefig(f"{save_path2}/img_var_masked.png")
    image_var = Mock._set_snr(flat_data, image_var, IMAGE_SNR, 'image', verbose=True)
    #print(image_data)
    '''

    fig = plt.figure(figsize=(12, 6))

    ax = fig.add_subplot(121)
    ax.imshow(image_data)
    fig.colorbar(mappable=ax.imshow(image_data), ax=ax, label="photons/s/m^2")
    ax.set_title('TNG Image Data', fontsize=12)

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

    VMAP_SHAPE = kin.shape  # Dec, RA
    VMAP_PIX_SCALE = 0.5  # arcsec/pix
    VMAP_SNR = snr[gal_index] #?
    VMAP_VAR = np.ones(kin.shape)*20 #pranjal says 20km/s, overplot zero shear with current fit to investigate degeneracy structure


    # Grids must be zero centered
    Dec_grid, RA_grid = klm.utils.build_map_grid(Nx=VMAP_SHAPE[0], Ny=VMAP_SHAPE[1], pix_scale=VMAP_PIX_SCALE)
    Dec_grid = np.flip(Dec_grid)

    meta_vmap = {'RA_grid': RA_grid,
                'Dec_grid': Dec_grid}
    vmap_data = kin
    vmap_var = VMAP_VAR

    # Set variance to match SNR
    #vmap_var = Mock._set_snr(vmap_data, vmap_var, VMAP_SNR, 'image', verbose=True) #Is this still needed? Is the velocity map SNR the same as the image SNR?


    print(np.nansum(image_data))

    #Plot velocity map
    plt.figure(figsize=(6, 5))

    plt.pcolormesh(RA_grid, Dec_grid, vmap_data)
    plt.xlabel('$\Delta$ RA [arcsec]')
    plt.ylabel('$\Delta$ Dec [arcsec]')

    plt.colorbar(label='V [km/s]')
    plt.savefig(f'{save_path2}/vel_map.png')

    plt.clf()

    #Plot image variance
    plt.imshow(image_var)
    plt.colorbar(label="Image Variance (photons/s/m^2)")
    plt.savefig(f"{save_path2}/image_var.png")

    plt.clf()

    #Plot non-convolved image (as a sanity check)
    plt.imshow(flat_data)
    plt.colorbar(label="Pre-convolution data (photons/s/m^2)")
    plt.savefig(f"{save_path2}/flat_data.png")

    plt.clf()

    #Plot velocity map variance
    plt.imshow(vmap_var)
    plt.colorbar(label='Vmap variance [km/s]')
    plt.savefig(f"{save_path2}/vmap_var.png")


    #Write data info
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

    joblib.dump(data_info, f'{save_path1}/data_info.pkl')

''''''

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

    #Load data info
    data_info = joblib.load(f'{save_path1}/data_info.pkl')

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
    

    #Leftover "bandaid fix" from when r_hl was not being properly fitted
    '''
    if float(data_info['mock_params']['shared_params-r_hl_disk']) <5.:
        config['params']['shared_params']['r_hl_disk']['prior']['min'] = 0.1
    else:
        config['params']['shared_params']['r_hl_disk']['prior']['min'] = float(data_info['mock_params']['shared_params-r_hl_disk']) - 5.
    
    config['params']['shared_params']['r_hl_disk']['prior']['max'] = float(data_info['mock_params']['shared_params-r_hl_disk'])+5.
    '''

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
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im2, cax=cax1)
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

''''''

def run_pipeline(gal_index, run_num=0, file_path="/home/acolarelli", save_path = "/home/acolarelli/test_chain/", 
                 config_path = '/home/acolarelli/path/to/venv/bin/kl_measurement-manga/config/iMaNGA_config.yaml', preserve_old_runs = False):
    
    #Preserve old runs doesn't work right now, leave as false and just change run num until fixed
    print("Galaxy#: "+str(gal_index))
    if preserve_old_runs:
        new_run_num = 0
        while new_run_num <= run_num:
            new_run_num += 1
        run_num = new_run_num
    else:
        new_run_num = run_num
    write_data_info(gal_index,run_num=new_run_num,file_path=file_path, save_path=save_path)
    sampler(gal_index,run_num=new_run_num,save_path=save_path,config_path=config_path)


''''''

#Note that the file path for the iMaNGA VAC and other necessary files at the start of the code needs to be manually changed for the script to run. 
#Save paths and config file path can be changed below.

''''''

#Change the list index to change which galaxy is being put in, or manually choose a galaxy index from rotator_indices.txt
run_pipeline(rotator_indices[45], run_num=1)
file.close()
hdu_lst.close()