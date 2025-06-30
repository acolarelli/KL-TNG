import numpy as np
import math
import matplotlib.pyplot as plt
from astropy.io import fits
from pafit import fit_kinematic_pa as fitpa
from plotbin import symmetrize_velfield as vsym
from scipy.optimize import least_squares

'''
PAFit and citation can be found here: https://pypi.org/project/pafit/
'''

imanga_vac = "/home/acolarelli/path/to/venv/bin/kl_measurement-manga/iMaNGA_vac.fits"

hdu_lst = fits.open(imanga_vac)
#hdu_lst.info()
#hdr = hdu_lst[4].header

gen_info = hdu_lst[1].data
TNG_snap_IDs = gen_info['TNG_snap_id']
env_density = gen_info['environment']


image_morphology = hdu_lst[2].data
inclinations = image_morphology['inc_kin']
#r_effs = image_morphology['sersic_reff_kpc']
#ellips = image_morphology['sersic_ellip']
t_morph = image_morphology['T_morph']
snr = image_morphology['snr_2Dsersic_fit']


spatial_info = hdu_lst[3].data
kinematics = hdu_lst[4].data

rot_dom_indices = []
rot_dom_IDs = []

##################################################################################################################

def is_late_type(gal_index):
    return True if t_morph[gal_index] == 3 else False

##################################################################################################################

def med_v_over_sigma(gal_index):
    try:
        vels_over_sigs=[]
        spax_coords = np.nonzero((spatial_info[1,gal_index,:,:]>1.0) & (kinematics[0,gal_index,:,:] == kinematics[0,gal_index,:,:] ))

        #subtracting 75 to make (0,0) the galactic center
        xbin = spax_coords[0] - 75
        ybin = spax_coords[1] - 75
        vels = kinematics[0,gal_index,spax_coords[0],spax_coords[1]] - np.nanmedian(kinematics[0,gal_index,spax_coords[0],spax_coords[1]]) #subtract estimate of systemic velocity
        disps = kinematics[1,gal_index,spax_coords[0],spax_coords[1]]

        pos_angle, pa_err, vsys = fitpa.fit_kinematic_pa(xbin,ybin,vels,nsteps=361,
                        quiet=False, plot=False, dvel=20)
        theta = np.arctan(ybin/xbin) - np.radians(pos_angle)
        sigma_correct = np.sin(np.radians(inclinations[gal_index]))*np.cos(theta)*disps
        vels_over_sigs = np.abs(vels/sigma_correct)

        return np.nanmedian(vels_over_sigs)
    except ValueError:
        pass

def is_rotationally_dominated(gal_index):
    try:
        v_over_sig = np.abs(med_v_over_sigma(gal_index))
        return True if v_over_sig > 0.56 else False
    except TypeError:
        return False

def is_isolated(gal_index):
    return True if env_density[gal_index]<1 else False


##################################################################################################################

#Old attempt at creating a more efficient method of getting PA, ignore
'''
def fit_kpa(gal, guess, plot=False):
    spax_coords = np.nonzero((spatial_info[1,gal,:,:]>1.0) & (kinematics[0,gal,:,:] == kinematics[0,gal,:,:] ))
    xbin = spax_coords[0] - 75
    ybin = spax_coords[1] - 75
    vels = kinematics[0,gal,spax_coords[0],spax_coords[1]]
    disps = kinematics[1,gal,spax_coords[0],spax_coords[1]]

    ang = least_squares(kpa.vel_residual, float(guess), args=(vels,xbin,ybin,disps))
    cost = float(ang['cost'])

    while cost>50.:
        guess -= 1.
        ang = least_squares(kpa.vel_residual, float(guess), args=(vels,xbin,ybin,disps))
        cost = ang['cost']
        if guess<0:
            guess=90.
        print(cost)

    print(ang)
    if plot:
        theta = float(ang['x'][0])
        xend1, yend1, xend2, yend2 = kpa.line_coords(75,75,theta+90.,150,150)

        plt.plot([xend1,xend2], [yend1,yend2], color="black", linewidth=2)
        #plt.plot([-xend,75], [-yend,75], color="black", linewidth=2) 
        plt.imshow(kinematics[0,gal,:,:])
        plt.colorbar()
        plt.show()
    return ang
'''

##################################################################################################################
'''
IMPORTANT: iMaNGA already discards spaxels with poor S/N. 

Minimum SNR ~ 10 
'''
spirals = []

def is_rotator(gal):
    if not is_late_type(gal):
        return False
    elif not is_isolated(gal):
        return False
    elif float(inclinations[gal]) < 36.:
        return False
    elif not is_rotationally_dominated(gal):
        return False
    else:
        return True

indices = open("rotator_indices.txt", "w")
snap_ids = open("rotator_snap_IDs.txt","w")

for i in range(len(gen_info)):
    if is_rotator(i):
        rot_dom_indices.append(i)
        rot_dom_IDs.append(str(TNG_snap_IDs[i]))
        indices.write(str(i)+"\n")
        snap_ids.write(str(TNG_snap_IDs[i])+"\n")

print(len(rot_dom_indices))

snap_ids.close()
indices.close()

##################################################################################################################
'''
Old plot test, ignore

for i in range(len(gen_info)):
    if is_rotationally_dominated(i):
        rot_dom_indices.append(i)
        rot_dom_IDs.append(str(TNG_snap_IDs[i]))

print(len(rot_dom_indices))
print(rot_dom_indices)
print(" ")
print(rot_dom_IDs)
'''
'''
n = 4
fig, axs = plt.subplots(n,n)
images = []
img_num = len(axs.flat)
#max_channel = len(spirals)-1
wvs = random.sample(spirals, img_num)

for ax in axs.flat:
    ax.set_xlabel('Galaxy num: ' + str(int(wvs[img_num-1])))
    images.append(ax.imshow(kinematics[0, int(wvs[img_num-1]), :, :]))
    img_num -= 1

fig.colorbar(images[0], ax = axs)
plt.show()

#print(str(spatial_info[3,50,50,100]))


plt.imshow(spatial_info[2,50,:,:])
plt.imshow(kinematics[1,50,:,:])
plt.colorbar()
plt.show()
'''

hdu_lst.close()
