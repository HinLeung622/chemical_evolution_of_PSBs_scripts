import bagpipes as pipes
import numpy as np
import bagpipes.config as config
import bagpipes.utils as utils

# add VW07 dust and relevent functions needed to dust_attenuation_model.py
# and model_galaxy.py
def update(self, param):

    # Fixed-shape dust laws are pre-computed in __init__.
    if self.type in ["Calzetti", "Cardelli", "SMC"]:
        return

    # Variable shape dust laws have to be computed every time.
    if self.type in ["VW07"]:
        self.A_cont, self.A_line, self.A_cont_bc = getattr(self, self.type)(param)
        return

    self.A_cont, self.A_line = getattr(self, self.type)(param)
    
def VW07(self, param):
    """ Modified Charlot + Fall (2000) model of Carnall et al.
    (2018) and Carnall et al. (2019). """
    A_cont = (5500./self.wavelengths)**0.7
    A_cont_bc = (5500./self.wavelengths)**1.3
    A_line = (5500./config.line_wavs)**1.3

    return A_cont, A_line, A_cont_bc

def _calculate_full_spectrum(self, model_comp):
    """ This method combines the models for the various emission
    and absorption processes to generate the internal full galaxy
    spectrum held within the class. The _calculate_photometry and
    _calculate_spectrum methods generate observables using this
    internal full spectrum. """

    t_bc = 0.01
    if "t_bc" in list(model_comp):
        t_bc = model_comp["t_bc"]

    spectrum_bc, spectrum = self.stellar.spectrum(self.sfh.ceh.grid, t_bc)
    em_lines = np.zeros(config.line_wavs.shape)

    if self.nebular:
        em_lines += self.nebular.line_fluxes(self.sfh.ceh.grid, t_bc,
                                             model_comp["nebular"]["logU"])

        # All stellar emission below 912A goes into nebular emission
        spectrum_bc[self.wavelengths < 912.] = 0.
        spectrum_bc += self.nebular.spectrum(self.sfh.ceh.grid, t_bc,
                                             model_comp["nebular"]["logU"])

    # Add attenuation due to stellar birth clouds.
    if self.dust_atten:
        dust_flux = 0.  # Total attenuated flux for energy balance.

        # Add extra attenuation to birth clouds.
        eta = 1.
        if "eta" in list(model_comp["dust"]):
            eta = model_comp["dust"]["eta"]
            bc_Av_reduced = (eta - 1.)*model_comp["dust"]["Av"]
            if self.dust_atten.type == "VW07":
                bc_trans_red = 10**(-bc_Av_reduced*self.dust_atten.A_cont_bc/2.5)
            else:
                bc_trans_red = 10**(-bc_Av_reduced*self.dust_atten.A_cont/2.5)
            spectrum_bc_dust = spectrum_bc*bc_trans_red
            dust_flux += np.trapz(spectrum_bc - spectrum_bc_dust,
                                  x=self.wavelengths)

            spectrum_bc = spectrum_bc_dust

        # Attenuate emission line fluxes.
        bc_Av = eta*model_comp["dust"]["Av"]
        em_lines *= 10**(-bc_Av*self.dust_atten.A_line/2.5)

    spectrum += spectrum_bc  # Add birth cloud spectrum to spectrum.

    # Add attenuation due to the diffuse ISM.
    if self.dust_atten:
        trans = 10**(-model_comp["dust"]["Av"]*self.dust_atten.A_cont/2.5)
        dust_spectrum = spectrum*trans
        dust_flux += np.trapz(spectrum - dust_spectrum, x=self.wavelengths)

        spectrum = dust_spectrum

        # Add dust emission.
        qpah, umin, gamma = 2., 1., 0.01
        if "qpah" in list(model_comp["dust"]):
            qpah = model_comp["dust"]["qpah"]

        if "umin" in list(model_comp["dust"]):
            umin = model_comp["dust"]["umin"]

        if "gamma" in list(model_comp["dust"]):
            gamma = model_comp["dust"]["gamma"]

        spectrum += dust_flux*self.dust_emission.spectrum(qpah, umin,
                                                          gamma)

    spectrum *= self.igm.trans(model_comp["redshift"])

    # Convert from luminosity to observed flux at redshift z.
    self.lum_flux = 1.
    if model_comp["redshift"] > 0.:
        ldist_cm = 3.086*10**24*np.interp(model_comp["redshift"],
                                          utils.z_array, utils.ldist_at_z,
                                          left=0, right=0)

        self.lum_flux = 4*np.pi*ldist_cm**2

    spectrum /= self.lum_flux*(1. + model_comp["redshift"])
    em_lines /= self.lum_flux

    # convert to erg/s/A/cm^2, or erg/s/A if redshift = 0.
    spectrum *= 3.826*10**33
    em_lines *= 3.826*10**33

    self.line_fluxes = dict(zip(config.line_names, em_lines))
    self.spectrum_full = spectrum

pipes.models.dust_attenuation.update = update
pipes.models.dust_attenuation.VW07 = VW07
pipes.models.model_galaxy._calculate_full_spectrum = _calculate_full_spectrum
