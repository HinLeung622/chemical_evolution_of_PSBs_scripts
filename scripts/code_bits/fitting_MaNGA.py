import bagpipes as pipes
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PyAstronomy import pyasl
import pandas as pd
import os
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)
from scipy.optimize import curve_fit

class fitting:
    """ Bundles of setup functions used when fitting with Bagpipes.

    Parameters
    ----------

    skylines_path : str
        Path to the place where "skyline.txt" is stored
    run_params : dict
        Dictionary that contains the fitting configs
    data : str or bagpipes.model_galaxy object
        only passed if not passing run_params
        if str: Path to the place where the data (in .csv)
        is loaded in
        if object: A mock bagpipes galaxy with a galaxy.spectrum
        array used as the input spectrum to be fitted
    z : float
        only passed if not passing run_params, observed
        redshift of the (mock) galaxy, used to scale rest-frame
        masks and spectral limits
    binby: int
        only passed if not passing run_params, default 1
        binby N = to bin input spectrum by every N bins, reducing
        the spectral resolution by N folds
    sky_masking : bool
        whether to include skyline masks or not
    full : bool
        if full == False: limits to MILES range (max 7500 in
        rest-frame)
    model_galaxy_SNR : float
        only passed if data is a bagpipes.model_galaxy object.
        Used to create the observational noise column of the
        input spectrum, without perturbation of the mock spectrum
    model_data_path : str
        only passed if data is a bagpipes.model_galaxy object.
        Path to the folder that would normally hold the spectrum
        .csv file, used to find the files that detail additional
        masking, balmer infilling masking, etc
    """
    def __init__(self, skylines_path, run_params=None, data=None, z=0, binby=1, sky_masking=True,
                 full=True, model_galaxy_SNR=None, model_data_path=None):
        self.skylines_path = skylines_path
        if run_params is None:
            if type(data) is str:
                self.data_path = data
                self.loadfromfile = True
            else:
                self.model_galaxy = data
                self.data_path = model_data_path
                self.loadfromfile = False
            self.z = z
            self.binby = binby
            self.model_galaxy_SNR = model_galaxy_SNR
            self.extra_masking = False
        else:
            self.run_params = run_params
            self.loadfromfile = True
            self.data_path = run_params['spectrum_dir']
            self.z = run_params['z']
            self.binby = run_params['bin_by']
            self.extra_masking = True
        self.sky_masking = sky_masking
        self.full = full
        # OII, OII, NeIII, Hdelta, Hgamma, Hbeta, OIII, OIII, OI, ? : in vaccuum
        self.mask_em_vals = [3727.092,3729.875,3870.,4102.892,
                             4341.692,4862.683,4960.295,5008.24,
                             6302.046, 6918.6]
        # these are in air
        self.mask_NaD_range = [5885, 5904]
        # doublet
        self.mask_SII_range = [6705.17, 6744.2]
        # Halpha + NII
        self.mask_Halpha_range = [6539.849, 6595.2784]
        self.load_sky()
        
    def bin(self, spectrum, binn):
        """ Bins up two or three column spectral data by a specified factor. """
    
        binn = int(binn)
        nbins = len(spectrum)/binn
        binspec = np.zeros((int(nbins), spectrum.shape[1]))
    
        for i in range(binspec.shape[0]):
            spec_slice = spectrum[i*binn:(i+1)*binn, :]
            binspec[i, 0] = np.mean(spec_slice[:, 0])
            binspec[i, 1] = np.mean(spec_slice[:, 1])
    
            if spectrum.shape[1] == 3:
                binspec[i,2] = (1./float(binn)
                                *np.sqrt(np.sum(spec_slice[:, 2]**2)))
    
        return binspec
        
    def load_sky(self):
        """ Loads in the skyline array, selects those with flux>5 """
        np_arr = np.loadtxt(self.skylines_path+'/skylines.txt')
        lines_air_df = pd.DataFrame(np_arr, columns=['wavelength', 'width', 'flux'])
        self.lines_vac_sky = pyasl.airtovac2(lines_air_df[lines_air_df['flux']>=5]['wavelength'])

    def mask_sky(self, wave):
        """ Masks strong night sky emission lines that are often not removed
        properly in the data processing. """

        masksize=10
        lines_vac = self.lines_vac_sky #np.array([5578.5486,4279.2039,6301.7423,6365.7595])
        lines = pyasl.vactoair2(lines_vac)
        # Manually specified extra skylines
        if "extra_skylines.txt" in os.listdir(self.data_path):
            lines_extra = np.loadtxt(fname=self.data_path+
                                  "/extra_skylines.txt", skiprows=1)
            lines = np.concatenate([lines,lines_extra])
        
        mask = []
        for i in range(lines.shape[0]):
            ind = np.where((wave>lines[i]-masksize) & (wave<lines[i]+masksize))
            mask.extend(ind[0])

        return mask

    def mask_em(self, wave):
        """ from maskem.pro OII, Hgama, Hbeta, OIII, OIII: Vacuum """
    
        lines_vac = np.array(self.mask_em_vals)
        lines = pyasl.vactoair2(lines_vac)
        
        mask = []
        for i in range(lines.shape[0]):
            masksize = 5

            ind = np.where((wave>lines[i]-masksize) & (wave<lines[i]+masksize))
            mask.extend(ind[0])
            
        # Na D
        ind = np.where((wave>self.mask_NaD_range[0]) & (wave<self.mask_NaD_range[1]))
        mask.extend(ind[0])
            
        # Halpha
        ind = np.where((wave>self.mask_Halpha_range[0]) & (wave<self.mask_Halpha_range[1]))
        mask.extend(ind[0])
        
        # SII
        ind = np.where((wave>self.mask_SII_range[0]) & (wave<self.mask_SII_range[1]))
        mask.extend(ind[0])
        
        # balmer line infilling masks
        if "balmer_infilling_masking.txt" in os.listdir(self.data_path):
            lines_balmer = pd.read_fwf(self.data_path+
                            "/balmer_infilling_masking.txt")
            for i,row in lines_balmer.iterrows():
                if lines_balmer.columns[2] == "lower_wave":
                    ind = np.where((wave>row.iloc[2]) & (wave<row.iloc[3]))
                elif lines_balmer.columns[2] == "wave":
                    ind = np.where((wave>row.iloc[2]-row.iloc[3]/2) & (wave<row.iloc[2]+row.iloc[3]/2))
                mask.extend(ind[0])
    
        # MgII  2796.352 ,2803.531
        #ind = np.where((wave>2766.4) & (wave<2833.5))
    
        # remove everything bluewards of 3000A
        #ind = np.where(wave<3000)
        #mask.extend(ind[0])
    
        return mask
    
    def mask_extra(self, wave):
        """ hand picked extra masking based on prelim fit residuals """
        mask = []
        if "extra_masking.txt" in os.listdir(self.data_path):
            lines_extra = pd.read_fwf(self.data_path+
                                  "/extra_masking.txt")
            lines_gal_extra = lines_extra[lines_extra.iloc[:,0]==self.run_params['ID_plateifu']]
            
            for i,row in lines_gal_extra.iterrows():
                ind = np.where((wave>row.iloc[1]-row.iloc[2]) & (wave<row.iloc[1]+row.iloc[2]))
                mask.extend(ind[0])

        return mask
    
    def load_manga_spec(self, ID):
    
        # load spectral data
        if self.loadfromfile:
            # load from save, csv file
            spectrum = np.loadtxt(fname=self.data_path+'/Spectrum_'+
                                  ID+".csv", delimiter=',', skiprows=1)
            
            spectrum[:,1] *= 10**-16
            spectrum[:,2] *= 10**-16
        
        else:
            # load from existing galaxy object
            spectrum = self.model_galaxy.spectrum.copy()
            spectrum_noise = spectrum[:,1]/self.model_galaxy_SNR
            spectrum = np.hstack([spectrum, np.expand_dims(spectrum_noise, axis=1)])
    
        # blow up the errors associated with any bad points in the spectrum and photometry
        for i in range(len(spectrum)):
            if spectrum[i,1] == 0 or spectrum[i,2] <= 0:
                spectrum[i,1] = 0.
                spectrum[i,2] = 9.9*10**99.
        
        # nebular emission lines and interstellar absorption lines
        mask = self.mask_em(spectrum[:,0]/(1+self.z))
        spectrum[mask, 2] = 9.9*10**99.
        
        # skylines
        if self.sky_masking:
            linemask = self.mask_sky(spectrum[:,0])
            spectrum[linemask, 2] = 9.9*10**99.
        
        # extra manual masking
        if self.extra_masking:
            linemask = self.mask_extra(spectrum[:,0])
            spectrum[linemask, 2] = 9.9*10**99.
        
        for j in range(len(spectrum)):
            if (spectrum[j, 1] == 0) or (spectrum[j, 2] <= 0):
                spectrum[j, 2] = 9.9*10**99.
        
        # O2 telluric
        #mask = ((spectrum[:,0] > 7580.) & (spectrum[:,0] < 7650.))
        #spectrum[mask, 2] = 9.9*10**99.
        
        if self.full == False:
            endmask = (spectrum[:,0]/(1+self.z) < 7500) # just miles range
        else:
            endmask = (spectrum[:,0]>0)
    
        if self.binby > 1:
            return self.bin(spectrum[endmask], self.binby)
        else:
            return spectrum[endmask]
            
    def load_manga_photo(self, ID):
        """ Reads spectrum csv and bins up the portion that corresponds to 7500-9500AA into one band """
        
        # load spectral data
        if self.loadfromfile:
            # load from save, csv file
            spectrum = np.loadtxt(fname=self.data_path+'/Spectrum_'+
                                  ID+".csv", delimiter=',', skiprows=1)

            spectrum = spectrum[np.where((spectrum[:,0]>7500) & (spectrum[:,0]<9500))]
        
            spectrum[:,1] *= 10**-16
            spectrum[:,2] *= 10**-16
            
        else:
            # load from existing galaxy object
            spectrum = self.model_galaxy.spectrum.copy()
            spectrum_noise = spectrum[:,1]/self.model_galaxy_SNR
            spectrum = np.hstack([spectrum, np.expand_dims(spectrum_noise, axis=1)])
            spectrum = spectrum[np.where((spectrum[:,0]>7500) & (spectrum[:,0]<9500))]
        
        dlambda = pipes.utils.make_bins(spectrum[:,0])[1]
        tp = np.ones(len(spectrum[:,0]))
        weights = dlambda*tp
        eff_wave = np.sqrt(np.sum(weights)/np.sum(weights/spectrum[:,0]**2))
        #print(eff_wave)
        
        phot_flux = (np.sum(spectrum[:,0]*spectrum[:,1]*dlambda) /
                     np.sum(spectrum[:,0]*dlambda)) / (10**-29*2.9979*10**18/eff_wave**2)
        phot_err = (np.sqrt(np.sum((spectrum[:,0]*dlambda*spectrum[:,2])**2)) /
                    np.sum(spectrum[:,0]*dlambda)) / (10**-29*2.9979*10**18/eff_wave**2)
        if phot_flux/phot_err>20:
            phot_err = phot_flux/15
        #print(phot_flux)
        #print(phot_err)
        
        return [[phot_flux, phot_err]]
            
    def load_manga_photo_plot(self, ID):
        """
        difference: returns flux in erg/s/cm^2/AA, instead of converting it into mJy, also returns effective
        wavelength, used for plotting
        """
        
        # load spectral data
        if self.loadfromfile:
            # load from save, csv file
            spectrum = np.loadtxt(fname=self.data_path+'/Spectrum_'+
                                  ID+".csv", delimiter=',', skiprows=1)
            
            spectrum = spectrum[np.where((spectrum[:,0]>7500) & (spectrum[:,0]<9500))]
        
            spectrum[:,1] *= 10**-16
            spectrum[:,2] *= 10**-16
            
        else:
            # load from existing galaxy object
            spectrum = self.model_galaxy.spectrum.copy()
            spectrum_noise = spectrum[:,1]/self.model_galaxy_SNR
            spectrum = np.hstack([spectrum, np.expand_dims(spectrum_noise, axis=1)])
            spectrum = spectrum[np.where((spectrum[:,0]>7500) & (spectrum[:,0]<9500))]
        
        dlambda = pipes.utils.make_bins(spectrum[:,0])[1]
        tp = np.ones(len(spectrum[:,0]))
        weights = dlambda*tp
        eff_wave = np.sqrt(np.sum(weights)/np.sum(weights/spectrum[:,0]**2))
        #print(eff_wave)
        
        phot_flux = (np.sum(spectrum[:,0]*spectrum[:,1]*dlambda) /
                     np.sum(spectrum[:,0]*dlambda))
        phot_err = (np.sqrt(np.sum((spectrum[:,0]*dlambda*spectrum[:,2])**2)) /
                    np.sum(spectrum[:,0]*dlambda))
        if phot_flux/phot_err>20:
            phot_err = phot_flux/15
        
        #print(phot_flux)
        #print(phot_err)
        
        return eff_wave, [phot_flux, phot_err]

    def load_both(self, ID):
        spectrum = self.load_manga_spec(ID)
        phot = self.load_manga_photo(ID)

        return spectrum, phot
        
        
class get_ceh_array:
    """
    Evaluates the metallicity values at a list of ages (in lb time) given the
    metallicity model choice and model parameters.
    """
    def delta(ages, sfh_dict):
        return np.ones(len(ages))*sfh_dict['metallicity']
    
    def two_step(ages, sfh_dict):
        pre_step_ind = np.where(ages > sfh_dict['metallicity_step_age'])
        post_step_ind = np.isin(np.arange(len(ages)), pre_step_ind, invert=True)
        ceh = np.zeros(len(ages))
        ceh[pre_step_ind] = sfh_dict['metallicity_old']
        ceh[post_step_ind] = sfh_dict['metallicity_new']
        return ceh
    
    def psb_two_step(ages, sfh_dict):
        pre_step_ind = np.where(ages > sfh_dict['burstage'])
        post_step_ind = np.isin(np.arange(len(ages)), pre_step_ind, invert=True)
        ceh = np.zeros(len(ages))
        ceh[pre_step_ind] = sfh_dict['metallicity_old']
        ceh[post_step_ind] = sfh_dict['metallicity_burst']
        return ceh

# plotting functions
# extracted from bagpipes.models.star_formation_history.py, with a bit of tweaking
def psb_wild2020(age_list, age, tau, burstage, alpha, beta, fburst, Mstar):
    age_lhs = pipes.utils.make_bins(np.log10(age_list)+9, make_rhs=True)[0]
    age_list = age_list*10**9
    age_lhs = 10**age_lhs
    age_lhs[0] = 0.
    age_lhs[-1] = 10**9*pipes.utils.age_at_z[pipes.utils.z_array == 0.]
    age_widths = age_lhs[1:] - age_lhs[:-1]
    sfr = np.zeros(len(age_list))
    
    age_of_universe = 10**9*np.interp(0, pipes.utils.z_array,
                                               pipes.utils.age_at_z)
    
    age = age*10**9
    tau = tau*10**9
    burstage = burstage*10**9

    ind = (np.where((age_list < age) & (age_list > burstage)))[0]
    texp = age - age_list[ind]
    sfr_exp = np.exp(-texp/tau)
    sfr_exp_tot = np.sum(sfr_exp*age_widths[ind])

    mask = age_list < age_of_universe
    tburst = age_of_universe - age_list[mask]
    tau_plaw = age_of_universe - burstage
    sfr_burst = ((tburst/tau_plaw)**alpha + (tburst/tau_plaw)**-beta)**-1
    sfr_burst_tot = np.sum(sfr_burst*age_widths[mask])

    sfr[ind] = (1-fburst) * np.exp(-texp/tau) / sfr_exp_tot

    dpl_form = ((tburst/tau_plaw)**alpha + (tburst/tau_plaw)**-beta)**-1
    sfr[mask] += fburst * dpl_form / sfr_burst_tot
    
    return sfr*10**Mstar

# a copy of the function, with a bit of tweaking
def psb_twin_(age_list, age, alpha1, beta1, burstage, alpha2, beta2, fburst, Mstar):
    age_lhs = pipes.utils.make_bins(np.log10(age_list)+9, make_rhs=True)[0]
    age_list = age_list*10**9
    age_lhs = 10**age_lhs
    age_lhs[0] = 0.
    age_lhs[-1] = 10**9*pipes.utils.age_at_z[pipes.utils.z_array == 0.]
    age_widths = age_lhs[1:] - age_lhs[:-1]
    sfr = np.zeros(len(age_list))
    
    age_of_universe = 10**9*np.interp(0, pipes.utils.z_array,
                                               pipes.utils.age_at_z)
    
    age = age*10**9
    burstage = burstage*10**9

    ind = (np.where((age_list < age_of_universe) & (age_list > burstage)))[0]
    told = age_of_universe - age_list[ind]
    tau_old = age_of_universe - age
    sfr_old = ((told/tau_old)**alpha1 + (told/tau_old)**-beta1)**-1
    sfr_old_tot = np.sum(sfr_old*age_widths[ind])

    mask = age_list < age_of_universe
    tburst = age_of_universe - age_list[mask]
    tau_plaw = age_of_universe - burstage
    sfr_burst = ((tburst/tau_plaw)**alpha2 + (tburst/tau_plaw)**-beta2)**-1
    sfr_burst_tot = np.sum(sfr_burst*age_widths[mask])

    old_dpl_form = ((told/tau_old)**alpha1 + (told/tau_old)**-beta1)**-1
    sfr[ind] = (1-fburst) * old_dpl_form / sfr_old_tot

    burst_dpl_form = ((tburst/tau_plaw)**alpha2 + (tburst/tau_plaw)**-beta2)**-1
    sfr[mask] += fburst * burst_dpl_form / sfr_burst_tot
    
    return sfr*10**Mstar

def load_model_sfh(filepath):
    # load in true SFH of a SPH sim
    #age_at_z = pipes.utils.cosmo.age(0).value
    sim_data = np.loadtxt(filepath)
    model_sfh = sim_data[:,2]
    model_ages = sim_data[:,0]
    mask = model_ages > 0
    model_ages = model_ages[mask].copy()
    model_sfh = model_sfh[mask].copy()
    return model_ages, model_sfh

def get_advanced_quantities(fit, save=True):
    """ a workaround of having to recalculate the advanced quantities upon every re-loading of results, saves some time
    But each saved full sample instead occupies much more space than the raw samples .h5 file, so if you don't want this to happen, replace all get_advanced_quantities with bagpipes' own posterior.get_advanced_quantities function """
    # a workaround of having to recalculate the advanced
    # quantities upon every re-loading of results
    import os
    import deepdish as dd
    if "spectrum_full" in list(fit.posterior.samples):
        return
    elif os.path.exists(fit.fname + "full_samp.h5"):
        # load and replace samples from file
        fit.posterior.samples = dd.io.load(fit.fname + "full_samp.h5")
        fit.posterior.fitted_model._update_model_components(fit.posterior.samples2d[0, :])
        fit.posterior.model_galaxy = pipes.models.model_galaxy(
            fit.posterior.fitted_model.model_components,
            filt_list=fit.posterior.galaxy.filt_list,
            spec_wavs=fit.posterior.galaxy.spec_wavs,
            index_list=fit.posterior.galaxy.index_list
        )
    else:
        fit.posterior.get_advanced_quantities()
        # save it, path is pipes/[runID]/[galID]_full_samp.h5
        if save:
            dd.io.save(fit.fname + "full_samp.h5", fit.posterior.samples)
            print(f'Advanced quantities saved in {fit.fname + "full_samp.h5"}.')

def hide_obs_noise(spectrum):
    """ sets all non-infinite obs noise to 0, returns the original obs noise array to undo to original obs noise after function """
    obs_noise_ = spectrum[:,2].copy()
    obs_noise = obs_noise_.copy()
    obs_noise[obs_noise>9e10] = np.nan
    mask = spectrum[:, 2] < 1.
    masked_spec = np.where(spectrum[:,2]>1)[0]
    spectrum[mask, 2] = 0.
    
    return masked_spec, obs_noise, obs_noise_
    
def get_mask_edges(masked_spec):
    # get the edges in wavelength for each mask in the spectrum,
    # used for making vertical gray bands in panels
    mask_edges = [[masked_spec[0]],[]]
    for i,indi in enumerate(masked_spec[:-1]):
        if masked_spec[i+1] - indi > 1:
            mask_edges[1].append(indi)
            mask_edges[0].append(masked_spec[i+1])
    mask_edges[1].append(masked_spec[-1])
    mask_edges = np.array(mask_edges).T
    
    return mask_edges
    
def draw_vertical_mask_regions(ax, spectrum, mask_edges, limits=[-1,1]):
    # draw the vertical gray bands in panels for spec masks
    for [mask_min, mask_max] in mask_edges:
        ax.fill_between([spectrum[:,0][mask_min], spectrum[:,0][mask_max]],
                         [limits[0]]*2, [limits[1]]*2, color='lightgray', zorder=2)
                         
def get_residual_spec(fit, y_scale):
    # calculate the residual spectrum
    if 'noise' in fit.posterior.samples.keys():
        post_median = np.median(fit.posterior.samples["spectrum"]+fit.posterior.samples["noise"], axis=0)
    else:
        post_median = np.median(fit.posterior.samples["spectrum"], axis=0)

    residuals = (fit.galaxy.spectrum[:,1] - post_median)*10**-y_scale
    
    return residuals

def add_obs_unc(fit, ax, obs_noise, y_scale, freeze_ylims=False):
    """ adds observational uncertainty lines to the residual and noise panels. Also checks if it has GP:scaling free parameter, if it has, also adds in a scaled obs unc version """
    ax.plot(fit.galaxy.spectrum[:,0], obs_noise*10**-y_scale,
            color='steelblue', lw=1, zorder=4, label='obs noise')
    ylims = ax.get_ylim()
    if 'noise:scaling' in fit.posterior.samples.keys():
        median_noise_scale = np.median(fit.posterior.samples['noise:scaling'])
        ax.plot(fit.galaxy.spectrum[:,0], obs_noise*median_noise_scale*10**-y_scale,
                color='cyan', lw=1, zorder=4, ls='--', label='scaled obs noise')
    if freeze_ylims:
        ax.set_ylim(ylims)

def add_phot_and_full_spec(fit, ax, y_scale, fit_obj, full_spec_label, gal_ID, plot_red_phot=True):
    """ Extends the spectrum to the right and adds in the red photometric point converted from the portion of the spectrum redward of 7500AA. Also plots the full input spectrum and the (before veldisp adjusted) full model fitted spectrum"""
    # Calculate posterior median redshift.
    if "redshift" in fit.fitted_model.params:
        redshift = np.median(fit.posterior.samples["redshift"])
    else:
        redshift = fit.fitted_model.model_components["redshift"]

    # Plot the posterior photometry and full spectrum.
    full_wavs = fit.posterior.model_galaxy.wavelengths*(1.+redshift)

    spec_post = np.percentile(fit.posterior.samples["spectrum_full"],
                              (16, 84), axis=0).T*10**-y_scale

    spec_post = spec_post.astype(float)  # fixes weird isfinite error

    ax.plot(full_wavs, spec_post[:, 0], color="navajowhite",
            zorder=-1)

    ax.plot(full_wavs, spec_post[:, 1], color="navajowhite",
            zorder=-1)

    ax.fill_between(full_wavs, spec_post[:, 0], spec_post[:, 1],
                    zorder=-1, color="navajowhite", linewidth=0,
                     label=full_spec_label)

    fit_obj_full_init = fit_obj.full
    fit_obj.full = True
    full_spectrum = fit_obj.load_manga_spec(gal_ID)
    ax.plot(full_spectrum[:,0], full_spectrum[:,1]*10**-y_scale, color='b', alpha=0.2, zorder=-2,
             label='non-fitted full obs spec')
    if plot_red_phot:
        eff_wave, phot = fit_obj.load_manga_photo_plot(gal_ID)
        fit_obj.full = fit_obj_full_init
        ax.errorbar(eff_wave, phot[0]*10**-y_scale,
                    yerr=phot[1]*10**-y_scale, lw=1,
                    linestyle=" ", capsize=3, capthick=1, zorder=0,
                    color="black")
        #print(eff_wave, phot[0], phot[0]*10**-y_scale)

        ax.scatter(eff_wave, phot[0]*10**-y_scale, color="blue",
                   s=40, zorder=1, linewidth=1, facecolor="blue",
                   edgecolor="black", marker="o", label='converted red-end photometry')
               
    return full_spectrum[:,0]

def plot_spec(fit, fit_obj, figsize=(15, 9.), save=True, save_aq=True, plot_red_phot=True):
    """ Plots the fitted spectrum plot, including the input and posterior fitted spectrum, residuals and fitted GP noise (if applicable)

    Parameters
    ----------
    fit : object
        The bagpipes.fitting.fit object
    fit_obj : object
        The fitting object of this script
    figsize : tuple
        Size of the figure
    save : bool
        Whether to save the resulting figure. Save path is ./pipes/plots/[runID]/[galID]_fit.pdf
    save_aq : bool
        Whether to save the advanced quantities sample dictionary, passed to get_advanced_quantities
    """

    # sort out latex labels
    tex_on = pipes.plotting.tex_on
    if tex_on:
        full_spec_label = r'post full spectrum(no noise) $\pm 1 \sigma$'
        wavelength_label = "$\\lambda / \\mathrm{\\AA}$"
        matplotlib.rcParams['text.usetex'] = True
        
    else:
        full_spec_label = 'post full spectrum(no noise) +- 1sigma'
        wavelength_label = "lambda / A"
        matplotlib.rcParams['text.usetex'] = False

    # Make the figure
    matplotlib.rcParams.update({'font.size': 16})
    params = {'legend.fontsize': 16,
              'legend.handlelength': 1}
    matplotlib.rcParams.update(params)
    get_advanced_quantities(fit, save=save_aq)
    
    gal_ID = fit.fname.split('/')[-1][:-1]
    print(gal_ID)

    fig = plt.figure(figsize=figsize)

    if 'noise' in fit.posterior.samples.keys():
        gs_rows = 5
    else:
        gs_rows = 4
    gs1 = matplotlib.gridspec.GridSpec(gs_rows, 1, hspace=0., wspace=0.)
    ax1 = plt.subplot(gs1[:3])
    ax3 = plt.subplot(gs1[3])
    if 'noise' in fit.posterior.samples.keys():
        # GP noise panel
        ax4 = plt.subplot(gs1[4])
    
    # limit to only the first 3 columns
    if fit.galaxy.spectrum.shape[1] > 3:
        fit.galaxy.spectrum = fit.galaxy.spectrum[:,:3]

    masked_spec, obs_noise, obs_noise_ = hide_obs_noise(fit.galaxy.spectrum)
    
    # plot observed spec line
    y_scale = pipes.plotting.add_spectrum(fit.galaxy.spectrum, ax1, label='fitted obs spec')
    # plot main median fitted spec line
    pipes.plotting.add_spectrum_posterior(fit, ax1, y_scale=y_scale)
    
    # non masked fluxes to adjust y limits
    non_masked_obs_spec = np.delete(fit.galaxy.spectrum, masked_spec, axis=0)
    # fix y limits
    if ax1.get_ylim()[0] < 0.9*min(non_masked_obs_spec[:,1])*10**-y_scale:
        ax1.set_ylim(bottom=0.9*min(non_masked_obs_spec[:,1])*10**-y_scale)
    if ax1.get_ylim()[1] > 1.1*max(non_masked_obs_spec[:,1])*10**-y_scale:
        ax1.set_ylim(top=1.1*max(non_masked_obs_spec[:,1])*10**-y_scale)

    #recover masks on spectrum and plot them as gray bands in residual plot
    mask_edges = get_mask_edges(masked_spec)
    draw_vertical_mask_regions(ax3, fit.galaxy.spectrum, mask_edges, limits=[-10,10])

    # calculate residuals
    residuals = get_residual_spec(fit, y_scale)
    # plot residuals
    non_masked_res = np.delete(residuals, masked_spec)
    ax3.axhline(0, color="black", ls="--", lw=1)
    ax3.plot(np.delete(fit.galaxy.spectrum[:,0], masked_spec), non_masked_res, color="sandybrown", zorder=1)
             
    # plot observational uncertainty along residuals
    add_obs_unc(fit, ax3, obs_noise, y_scale)
    ax3.set_ylabel('residual')
    ax3.set_ylim([1.1*min(non_masked_res), 1.1*max(non_masked_res)])

    #extend posterior spectrum, add red photometric point, etc
    full_spectrum_waves = add_phot_and_full_spec(fit, ax1, y_scale, fit_obj, full_spec_label, gal_ID, plot_red_phot=plot_red_phot)
    
    # indicate if photometric point was used
    if plot_red_phot:
        if fit.galaxy.photometry_exists:
            text = 'photometric point fitted'
        else:
            text = 'photometric point NOT fitted'
        ax1.text(0.72,0.56, text, transform=ax1.transAxes)
        ax1.legend(loc='upper right')
    ax1.set_xlim([min(fit.galaxy.spectrum[:,0]),max(full_spectrum_waves)])
    ax3.set_xlim(ax1.get_xlim())

    # Plot the noise factor
    if 'noise' in fit.posterior.samples.keys():
        ax4.axhline(0, color="black", ls="--", lw=1)
        
        noise_percentiles = np.percentile(fit.posterior.samples['noise'],(16,50,84),axis=0)*10**-y_scale
        ax4.plot(fit.galaxy.spectrum[:,0], noise_percentiles[1],color="sandybrown", zorder=1)
        ax4.fill_between(fit.galaxy.spectrum[:,0], noise_percentiles[0], noise_percentiles[2],
                         color='navajowhite', zorder=-1)
        # plot observational uncertainty along GP noise
        add_obs_unc(fit, ax4, obs_noise, y_scale, freeze_ylims=True)

        ylim_ax4 = ax4.get_ylim()
        draw_vertical_mask_regions(ax4, fit.galaxy.spectrum, mask_edges, limits=[-10,10])
        ax4.set_xlim(ax1.get_xlim())
        ax4.set_ylim(ylim_ax4)
        pipes.plotting.auto_x_ticks(ax4)
        ax4.set_xlabel(wavelength_label)
        ax4.set_ylabel('noise')
    
    if save:
        fname_parts = fit.fname.split('/')
        fig.savefig('pipes/plots/'+fname_parts[2]+'/'+fname_parts[3]+'fit.pdf')
    plt.show()
    
    # return obs noise to normal
    fit.galaxy.spectrum[:, 2] = obs_noise_
    
    if 'noise' in fit.posterior.samples.keys():
        return fig, [ax1,ax3,ax4]
    else:
        return fig, [ax1,ax3]
    
def plot_spec_lite(fit, fit_obj, figsize=(15, 9.), save=True, save_aq=True):
    """ Lite version of plot_spec. Plots the fitted spectrum plot, does not plot raw model spectrum
    and red photometric point

    Parameters
    ----------
    fit : object
        The bagpipes.fitting.fit object
    fit_obj : object
        The fitting object of this script
    figsize : tuple
        Size of the figure
    save : bool
        Whether to save the resulting figure. Save path is ./pipes/plots/[runID]/[galID]_fit.pdf
    save_aq : bool
        Whether to save the advanced quantities sample dictionary, passed to get_advanced_quantities
    """

    # sort out latex labels
    tex_on = pipes.plotting.tex_on
    if tex_on:
        full_spec_label = r'post full spectrum(no noise) $\pm 1 \sigma$'
        wavelength_label = "$\\lambda / \\mathrm{\\AA}$"
        matplotlib.rcParams['text.usetex'] = True
        
    else:
        full_spec_label = 'post full spectrum(no noise) +- 1sigma'
        wavelength_label = "lambda / A"
        matplotlib.rcParams['text.usetex'] = False

    # Make the figure
    matplotlib.rcParams.update({'font.size': 16})
    params = {'legend.fontsize': 16,
              'legend.handlelength': 1}
    matplotlib.rcParams.update(params)
    get_advanced_quantities(fit, save=save_aq)
    
    gal_ID = fit.fname.split('/')[-1][:-1]
    print(gal_ID)

    fig = plt.figure(figsize=figsize)

    if 'noise' in fit.posterior.samples.keys():
        gs_rows = 5
    else:
        gs_rows = 4
    gs1 = matplotlib.gridspec.GridSpec(gs_rows, 1, hspace=0., wspace=0.)
    ax1 = plt.subplot(gs1[:3])
    ax3 = plt.subplot(gs1[3])
    if 'noise' in fit.posterior.samples.keys():
        # GP noise panel
        ax4 = plt.subplot(gs1[4])
    
    # limit to only the first 3 columns
    if fit.galaxy.spectrum.shape[1] > 3:
        fit.galaxy.spectrum = fit.galaxy.spectrum[:,:3]

    masked_spec, obs_noise, obs_noise_ = hide_obs_noise(fit.galaxy.spectrum)

    # plot observed spec line
    y_scale = pipes.plotting.add_spectrum(fit.galaxy.spectrum, ax1, label='fitted obs spec')
    # plot main median fitted spec line
    pipes.plotting.add_spectrum_posterior(fit, ax1, y_scale=y_scale)
    
    # non masked fluxes to adjust y limits
    non_masked_obs_spec = np.delete(fit.galaxy.spectrum, masked_spec, axis=0)
    # fix y limits
    if ax1.get_ylim()[0] < 0.9*min(non_masked_obs_spec[:,1])*10**-y_scale:
        ax1.set_ylim(bottom=0.9*min(non_masked_obs_spec[:,1])*10**-y_scale)
    if ax1.get_ylim()[1] > 1.1*max(non_masked_obs_spec[:,1])*10**-y_scale:
        ax1.set_ylim(top=1.1*max(non_masked_obs_spec[:,1])*10**-y_scale)

        #recover masks on spectrum and plot them as gray bands in residual plot
    mask_edges = get_mask_edges(masked_spec)
    draw_vertical_mask_regions(ax3, fit.galaxy.spectrum, mask_edges, limits=[-10,10])

    # calculate residuals
    residuals = get_residual_spec(fit, y_scale)
    # plot residuals
    non_masked_res = np.delete(residuals, masked_spec)
    ax3.axhline(0, color="black", ls="--", lw=1)
    ax3.plot(np.delete(fit.galaxy.spectrum[:,0], masked_spec), non_masked_res, color="sandybrown", zorder=1)
             
    # plot observational uncertainty along residuals
    add_obs_unc(fit, ax3, obs_noise, y_scale)
    ax3.set_ylabel('residual')
    ax3.set_ylim([1.1*min(non_masked_res), 1.1*max(non_masked_res)])
    ax3.set_xlim(ax1.get_xlim())

    # Plot the noise factor
    if 'noise' in fit.posterior.samples.keys():
        ax4.axhline(0, color="black", ls="--", lw=1)
        
        noise_percentiles = np.percentile(fit.posterior.samples['noise'],(16,50,84),axis=0)*10**-y_scale
        ax4.plot(fit.galaxy.spectrum[:,0], noise_percentiles[1],color="sandybrown", zorder=1)
        ax4.fill_between(fit.galaxy.spectrum[:,0], noise_percentiles[0], noise_percentiles[2],
                         color='navajowhite', zorder=-1)
        # plot observational uncertainty along GP noise
        add_obs_unc(fit, ax4, obs_noise, y_scale, freeze_ylims=True)

        ylim_ax4 = ax4.get_ylim()
        draw_vertical_mask_regions(ax4, fit.galaxy.spectrum, mask_edges, limits=[-10,10])
        ax4.set_xlim(ax1.get_xlim())
        ax4.set_ylim(ylim_ax4)
        pipes.plotting.auto_x_ticks(ax4)
        ax4.set_xlabel(wavelength_label)
        ax4.set_ylabel('noise')
    
    if save:
        fname_parts = fit.fname.split('/')
        fig.savefig('pipes/plots/'+fname_parts[2]+'/'+fname_parts[3]+'fit_lite.pdf')
    plt.show()
    
    # return obs noise to normal
    fit.galaxy.spectrum[:, 2] = obs_noise_
    
    if 'noise' in fit.posterior.samples.keys():
        return fig, [ax1,ax3,ax4]
    else:
        return fig, [ax1,ax3]
    
def _cal_residual_indicators(gal_spec, model_spec_post):
    """ The calculations """
    # 1. measure abs residual between predicted spectrum vs obs spec, mask away inf obs noise
    model_spec_abs_obserr_multiple = np.abs(model_spec_post - gal_spec[:,1])/gal_spec[:,2]
    mask = gal_spec[:,2] < 1
    model_spec_abs_obserr_multiple = model_spec_abs_obserr_multiple[:,mask]

    # 2. get width of wavelength bins, remove those that are masked by inf obs noise
    wave_bin_diffs = np.diff(gal_spec[:,0])
    wave_bin_widths = (wave_bin_diffs[:-1]+wave_bin_diffs[1:])/2
    wave_bin_widths = np.insert(wave_bin_widths, [0,-1], [wave_bin_diffs[0], wave_bin_diffs[-1]])
    wave_bin_widths = wave_bin_widths[mask]

    # 3. Integrate obs residual divided by sigma by wave widths
    total_prediction_err = np.sum(model_spec_abs_obserr_multiple*wave_bin_widths, axis=1)

    # 4. divide by summed bin widths
    total_prediction_err /= np.sum(wave_bin_widths)
    
    return total_prediction_err

def cal_residual_indicators(fit):
    """
    Calculates the spectrum-averaged number of times of observational uncertainties is each model
    predicted spectrum deviating away from the input galaxy spectrum, in absolute values. Masked wavelengths
    are ignored.
    The exact formula for one sample's value is:
    sum(abs(obs-predicted)/(sigma*delta_lambda)) / sum(delta_lambda)
    where
    obs = input observed fluxes
    predicted = model predicted fluxes, can be from the physical model only, or with contributions from
                noise components
    sigma = observational uncertainty array
    delta_lambda = width of each wavelength bin
    """
    model_spec_post = fit.posterior.samples["spectrum"]
    model_prediction_err = _cal_residual_indicators(fit.galaxy.spectrum, model_spec_post)
    fit.posterior.samples['model_prediction_err'] = model_prediction_err
    
    if "noise" in fit.posterior.samples.keys():
        model_noise_spec_post = fit.posterior.samples["spectrum"] + fit.posterior.samples["noise"]
        model_noise_prediction_err = _cal_residual_indicators(fit.galaxy.spectrum, model_noise_spec_post)
        fit.posterior.samples['model_noise_prediction_err'] = model_noise_prediction_err
    
def integrate_sfh(ages, sfh, Mstar=None):
    """
    takes a sfh and integrates it to return a cumulative SFH (normalized to run from 0 to 1) fraction of
    mass formed
    """
    if Mstar is None:
        Mstar = np.trapz(y=sfh,x=ages)
    c_sfh = np.zeros(len(sfh))
    for i,sfhi in enumerate(sfh):
        c_sfh[i] = np.trapz(sfh[:i+1],x=ages[:i+1]/Mstar)
    return c_sfh

def fit_f_burst(ages, sfh, age_at_z, SFH_comp):
    # using scipy curve fit to get a fit to the true SFH from a sim
    if SFH_comp == "psb2" or SFH_comp == "psb_wild2020":
        popt,pcov = curve_fit(psb_wild2020, ages, sfh,
                              bounds=([10,1,0,10,10,0,10],[13,10,2,1000,1000,1,12]))
        [age, tau, burstage, alpha, beta, fburst, Mstar] = popt
    elif SFH_comp == "psb_twin":
        popt,pcov = curve_fit(psb_twin_, ages, sfh,
                              bounds=([10,0.01,100,0,10,10,0,10],[13,1000,10000,2,1000,1000,1,12]))
        [age, alpha1, beta1, burstage, alpha2, beta2, fburst, Mstar] = popt
    #tform = age_at_z - age
    tburst = age_at_z - burstage
    return fburst, tburst

def plot_sfh(fit, model_sfh=None, plot_mean=False, model_f_burst=None,
             model_burstage=None, ninty_region=False, samples=0, figsize=[15,13], save=True):
    """
    Plots the regular SFH (SFR vs age of universe) plot on the top, cumulative SFH plot on the bottom.
    If a non-constant metallicity model is used, adds in a third plot to show metallicity trends at the bottom
    
    Parameters
    ----------
    fit : object
        The bagpipes.fitting.fit object
    model_sfh : 2D array
        True/known SFH, plotted in blue.
        Column 0 = lookback times in years
        Column 1 = SFRs in Msuns/yr
    plot_mean : bool
        Whether to plot the posterior mean as a dashed black line.
        Default only plots the median as a solid line.
    model_f_burst : float
        The true (or best fit) fburst value of the true SFH. If not passed, will attempt to fit the currently used SFH model to passed model_sfh
    model_burstage : float
        The true (or best fit) burstage value of the true SFH, same as above
    ninty_region : bool
        Whether to plot the posterior 90% highest confidence region in the SFH as a lighter shade of gray
    samples : int
        Number of individual SFH and cumulative SFH samples to plot as dotted gray lines
    save : bool
        Whether to save the resulting figure. Save path is ./pipes/plots/[runID]/[galID]_combined_SFH.pdf
    """
    
    # sort out latex labels
    tex_on = pipes.plotting.tex_on
    if tex_on:
        Mstar_label = r"formed $\log_{10}M_*$="
        true_Mstar_label = r"true formed $\log_{10}M_*$="
        metallicity_label = "$\\mathrm{Z_{*}}/Z_{\\odot}$"
        
    else:
        Mstar_label = "formed log10 M\_*="
        true_Mstar_label = "true formed log10 M\_*="
        metallicity_label = "Z\_*/Z\_sun"
    
    if 'redshift' in fit.posterior.samples.keys():
        post_z = np.median(fit.posterior.samples['redshift'])
    else: post_z = 0.04
    age_at_z = pipes.utils.cosmo.age(post_z).value
    
    #identify SFH component used
    if "psb2" in fit.fit_instructions.keys():
        SFH_comp = "psb2"
    elif "psb_wild2020" in fit.fit_instructions.keys():
        SFH_comp = "psb_wild2020"
    elif "psb_twin" in fit.fit_instructions.keys():
        SFH_comp = "psb_twin"

    #posterior sfh
    post_sfh = fit.posterior.samples['sfh']
    #median_sfh = np.median(post_sfh,axis=0)
    mean_sfh = np.mean(post_sfh,axis=0)
    age_of_universe = np.interp(post_z, pipes.utils.z_array, pipes.utils.age_at_z)
    post_ages = age_of_universe - fit.posterior.sfh.ages*10**-9
    post_ages_int = post_ages.copy()[::-1]*10**9

    #model sfh
    plot_model_sfh = False
    if model_sfh is not None:
        plot_model_sfh = True
        model_lookbacktime = model_sfh[:,0]
        model_sfh_val = model_sfh[:,1]
        model_ages = age_at_z-model_lookbacktime.copy()
        model_ages_int = model_ages.copy()[::-1]*10**9
        model_m_total = np.trapz(y=model_sfh_val[::-1], x=model_ages_int)
        # integrate to get cumulative of model sfh
        c_model_sfh_val = integrate_sfh(model_ages_int, model_sfh_val[::-1], Mstar=model_m_total)
        
        print('only recovered',10**np.median(fit.posterior.samples[SFH_comp+":massformed"])
              /model_m_total,'of total mass formed.')
        print(np.median(fit.posterior.samples[SFH_comp+":massformed"]), np.log10(model_m_total))

    #calculating posterior tx and their uncertainties
    mass_percentiles = np.linspace(0,1,5)[1:-1]
    if 'c_sfh' not in fit.posterior.samples.keys():
        txs = np.zeros([len(mass_percentiles), fit.posterior.n_samples])
        c_sfh_samples = []
        for i,sfh_sample in enumerate(fit.posterior.samples['sfh']):
            sfh_ = sfh_sample[::-1]
            c_sfh_ = integrate_sfh(post_ages_int, sfh_)
            c_sfh_samples.append(c_sfh_)
            txs[:,i] = np.interp(mass_percentiles, c_sfh_, post_ages_int)
        txs = txs/10**9
        fit.posterior.samples['c_sfh'] = c_sfh_samples
        fit.posterior.samples['tx'] = txs
    else:
        print('loaded from samples')
        txs = fit.posterior.samples['tx']
        c_sfh_samples = fit.posterior.samples['c_sfh']
    tx_percentiles = []
    for i,txi in enumerate(txs):
        tx_percentiles.append(np.percentile(txi, (16,50,84)))
    tx_percentiles = np.array(tx_percentiles)
    c_sfh_percentiles = np.percentile(c_sfh_samples, (16,50,84), axis=0)
    c_sfh_mean = np.mean(c_sfh_samples, axis=0)
    
    # check if using complex CEH models
    plot_metallicity = False
    if "metallicity_type" in fit.fit_instructions[SFH_comp].keys():
        if fit.fit_instructions[SFH_comp]["metallicity_type"] != 'delta':
            plot_metallicity = True
            zmet_evo = np.zeros([fit.posterior.n_samples, len(fit.posterior.sfh.ages)])
            for i in range(fit.posterior.n_samples):
                sfh_dict = {}
                for sfh_key in fit.fit_instructions[SFH_comp]:
                    try:
                        sfh_dict[sfh_key] = fit.posterior.samples[f'{SFH_comp}:{sfh_key}'][i]
                    except KeyError:
                        pass
                zmet_evo[i] = getattr(get_ceh_array,
                               fit.fit_instructions[SFH_comp]["metallicity_type"])(
                               fit.posterior.sfh.ages/10**9, sfh_dict)
            zmet_evo_percentiles = np.percentile(zmet_evo, (16,50,84), axis=0)
    
    ################# plotting
    
    if plot_metallicity:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(5,1, hspace=0.4)
        ax1 = plt.subplot(gs[:2])
        ax2 = plt.subplot(gs[2:4])
        ax3 = plt.subplot(gs[4])
        ax = [ax1, ax2, ax3]
    else:
        fig, ax = plt.subplots(2,1, figsize=figsize)
    pipes.plotting.add_sfh_posterior(fit, ax[0], z_axis=False, zorder=9)
    if plot_mean:
        ax[0].plot(post_ages, mean_sfh, color='k', ls='--', zorder=7)
    if ninty_region:
        ninty_sfh = np.percentile(post_sfh, (5,95), axis=0)
        ax[0].fill_between(post_ages, ninty_sfh[0], ninty_sfh[1], color='gray',
                           alpha=0.3, zorder=6)
    if plot_model_sfh:
        ax[0].plot(model_ages, model_sfh_val, zorder=10)
    ylim = ax[0].get_ylim()

    if plot_model_sfh:
        #calculate model burst fraction
        if model_f_burst is None and model_burstage is None:
            model_f_burst, model_t_burst = fit_f_burst(
                model_lookbacktime.copy(), model_sfh_val, age_at_z, SFH_comp)
        else:
            model_t_burst = age_at_z - model_burstage
        print('model f_burst and t_burst:',model_f_burst,model_t_burst)
        ax[0].vlines(model_t_burst, 0, ylim[1], color='red', ls='--', zorder=8)
        ax[0].arrow(age_at_z,ylim[1]*0.8,-(age_at_z-model_t_burst),0.0,color='red',head_width=np.max(ylim)/20.,
                 head_length=0.1,length_includes_head=True, zorder=8)

    #use psb2's built in fburst and tburst posteriors to plot arrows
    post_f_burst = np.percentile(fit.posterior.samples[SFH_comp+":fburst"], (16,50,84))
    post_t_burst = age_of_universe-np.percentile(fit.posterior.samples[SFH_comp+":burstage"], (84,50,16))
    post_Mstar = np.percentile(fit.posterior.samples['formed_mass'], (16,50,84))
    
    print('posterior f_burst and t_burst:',post_f_burst,post_t_burst)
    ax[0].vlines(post_t_burst[1], 0, ylim[1], color='sandybrown', ls='--', zorder=8)
    ax[0].arrow(age_of_universe,ylim[1]*0.9,-(age_of_universe-post_t_burst[1]),0.0,color='sandybrown',
             head_width=np.max(ylim)/20., head_length=0.1,length_includes_head=True, zorder=8)

    #plot vertical bands of tx percentiles
    for i,[l,m,u] in enumerate(tx_percentiles):
        ax[0].vlines(m, 0, 10*ylim[1], color = 'k', ls='--', alpha=0.5, zorder=1)
        ax[0].fill_betweenx([0,10*ylim[1]], l, u, facecolor='royalblue', alpha=(1.5-(i+1)/len(txs))/2.5,
                           zorder=1)
    
    ax[0].set_ylim(ylim)
    #add text about z, age at z, poster f_burst and t_burst
    f_burst_r = [np.round(post_f_burst[1],2),np.round(post_f_burst[2]-post_f_burst[1],2),
                 np.round(post_f_burst[1]-post_f_burst[0],2)]
    f_burst_text = f'f\_burst={f_burst_r[0]}+{f_burst_r[1]}-{f_burst_r[2]}\n '
    t_burst_r = [np.round(post_t_burst[1],2),np.round(post_t_burst[2]-post_t_burst[1],2),
                 np.round(post_t_burst[1]-post_t_burst[0],2)]
    t_burst_text = f't\_burst={t_burst_r[0]}+{t_burst_r[1]}-{t_burst_r[2]}Gyr \n '
    Mstar_r = [np.round(post_Mstar[1],2),np.round(post_Mstar[2]-post_Mstar[1],2),
                 np.round(post_Mstar[1]-post_Mstar[0],2)]
    Mstar_text = Mstar_label + \
                f'{Mstar_r[0]}+{Mstar_r[1]}-{Mstar_r[2]}'
    if plot_model_sfh:
        ax[0].text(0.03,0.95,
                f'redshift={np.round(post_z,3)}\n ' +
                f'age at z={np.round(age_at_z,2)}Gyr\n ' +
                f_burst_text +
                f'true f\_burst={np.round(model_f_burst,2)}\n ' +
                t_burst_text +
                f'true t\_burst={np.round(model_t_burst,2)}Gyr\n' +
                Mstar_text + '\n' +
                true_Mstar_label + str(np.round(np.log10(model_m_total),2))
                ,
                fontsize=14, transform=ax[0].transAxes, bbox=dict(boxstyle='round', facecolor='white'), zorder=20, va='top')
    else:
        ax[0].text(0.03,0.95,
            f'redshift={np.round(post_z,3)}\n ' +
            f'age at z={np.round(age_at_z,2)}Gyr\n ' +
            f_burst_text +
            t_burst_text +
            Mstar_text,
            fontsize=14, transform=ax[0].transAxes,
               bbox=dict(boxstyle='round', facecolor='white'), zorder=20, va='top')
    
    ax[0].set_xlim(ax[0].get_xlim()[::-1])
    pipes.plotting.add_z_axis(ax[0])
    
    if plot_model_sfh:
        ax[1].plot(model_ages[::-1], c_model_sfh_val, zorder=9)
    ax[1].plot(post_ages[::-1], c_sfh_percentiles[1], color='k', zorder=8)
    if plot_mean:
        ax[1].plot(post_ages[::-1], c_sfh_mean, color='k', ls='--', zorder=6)
    ax[1].fill_between(post_ages[::-1], c_sfh_percentiles[0], c_sfh_percentiles[2], color='gray',
                       alpha=0.6, zorder=7)
    if ninty_region:
        c_ninty_sfh = np.percentile(c_sfh_samples, (5,95), axis=0)
        ax[1].fill_between(post_ages[::-1], c_ninty_sfh[0], c_ninty_sfh[1], color='gray',
                           alpha=0.3, zorder=5)
    ax[1].errorbar(tx_percentiles[:,1], np.linspace(0,1,5)[1:-1], xerr=[tx_percentiles[:,1]-tx_percentiles[:,0],
                                                                        tx_percentiles[:,2]-tx_percentiles[:,1]],
              color='red', label='calculated equivilent tx times (assuming 4 bins)', fmt='o', zorder=10)
    
    ax[1].set_xlim(ax[0].get_xlim())
    ax[1].set_ylim([0,1])
    ax[1].set_xlabel(ax[0].get_xlabel())
    ax[1].set_ylabel('fraction of cumulative mass formed')
    
    if plot_metallicity:
        # third plot
        ax[2].plot(post_ages, zmet_evo_percentiles[1], color='k', zorder=8)
        ax[2].fill_between(post_ages, zmet_evo_percentiles[0], zmet_evo_percentiles[2],
                       color='gray', alpha=0.6, zorder=7)
        if plot_mean:
            zmet_evo_mean = np.mean(zmet_evo, axis=0)
            ax[2].plot(post_ages, zmet_evo_mean, color='k', ls='--', zorder=6)
        if ninty_region:
            zmet_evo_ninty = np.percentile(zmet_evo, (5,95), axis=0)
            ax[2].fill_between(post_ages, zmet_evo_ninty[0], zmet_evo_ninty[1], color='gray',
                               alpha=0.3, zorder=5)
        zmet_ylims = ax[2].get_ylim()
        # vertical band of jump age
        if fit.fit_instructions[SFH_comp]['metallicity_type'] == 'psb_two_step':
            step_age_percentiles = age_of_universe - np.percentile(
                fit.posterior.samples[f'{SFH_comp}:burstage'], (16,50,84))
            ax[2].axvline(step_age_percentiles[1], color='steelblue', zorder=1)
            ax[2].fill_between([step_age_percentiles[0], step_age_percentiles[2]],
                               [zmet_ylims[0]]*2, [zmet_ylims[1]]*2, color='steelblue',
                               alpha=0.3, zorder=0)
        elif fit.fit_instructions[SFH_comp]['metallicity_type'] == 'two_step':
            step_age_percentiles = age_of_universe - np.percentile(
                fit.posterior.samples[f'{SFH_comp}:metallicity_step_age'], (16,50,84))
            ax[2].axvline(step_age_percentiles[1], color='steelblue', zorder=1)
            ax[2].fill_between([step_age_percentiles[0], step_age_percentiles[2]],
                               [zmet_ylims[0]]*2, [zmet_ylims[1]]*2, color='steelblue',
                               alpha=0.3, zorder=0)
        ax[2].set_xlim(ax[0].get_xlim())
        ax[2].set_ylim(zmet_ylims)
        ax[2].set_xlabel(ax[0].get_xlabel())
        ax[2].set_ylabel(metallicity_label)
        ax[2].text(0.03,0.90,
            f"model:{fit.fit_instructions[SFH_comp]['metallicity_type'].replace('_',' ')}",
            fontsize=14, transform=ax[2].transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='white'), zorder=20)
    
    if samples > 0:
        samp_ind = np.random.randint(fit.posterior.n_samples, size=samples)
        for samp_i in samp_ind:
            ax[0].plot(post_ages, fit.posterior.samples['sfh'][samp_i], color='black', alpha=0.3, ls='--',
                       zorder=5)
            ax[1].plot(post_ages[::-1], c_sfh_samples[samp_i], color='black', alpha=0.3, ls='--', zorder=5)
            if plot_metallicity:
                ax[2].plot(post_ages, zmet_evo[samp_i], color='black', alpha=0.3, ls='--', zorder=5)
    
    if save:
        fname_parts = fit.fname.split('/')
        fig.savefig('pipes/plots/'+fname_parts[2]+'/'+fname_parts[3]+'combined_sfh.pdf')
    plt.show()
    
    return fig,ax
