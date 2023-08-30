import bagpipes as pipes
import celerite2
from celerite2 import terms
import numpy as np
from bagpipes.fitting.prior import prior, dirichlet
from bagpipes.fitting.calibration import calib_model
from bagpipes.fitting.noise import noise_model
from bagpipes.models.model_galaxy import model_galaxy

# noise.py adding a new kernel options
def GP_SHOTerm(self):
    """ celerite's SHOTerm (singular) """

    scaling = self.param["scaling"]

    norm = self.param["norm"]
    #length = self.param["length"]
    # the undamped period of the oscillator
    period = self.param["period"]
    # dampening quality factor Q
    Q = self.param["Q"]

    kernel = terms.SHOTerm(S0=norm**2, rho=period, Q=Q)
    self.gp = celerite2.GaussianProcess(kernel)
    self.gp.compute(self.x, yerr=self.y_err*scaling)

    self.corellated = True
    
pipes.fitting.noise.noise_model.GP_SHOTerm = GP_SHOTerm

def GP_SHOTerm2(self):
    """ celerite's SHOTerm (singular), swapped to use sigma as norm noise """

    scaling = self.param["scaling"]

    norm = self.param["norm"]
    #length = self.param["length"]
    # the undamped period of the oscillator
    period = self.param["period"]
    # dampening quality factor Q
    Q = self.param["Q"]

    kernel = terms.SHOTerm(sigma=norm, rho=period, Q=Q)
    self.gp = celerite2.GaussianProcess(kernel)
    self.gp.compute(self.x, yerr=self.y_err*scaling)

    self.corellated = True
    
pipes.fitting.noise.noise_model.GP_SHOTerm2 = GP_SHOTerm2

#fitted_model.py updates _lnlike_spec
def _lnlike_spec(self):
    """ Calculates the log-likelihood for spectroscopic data. This
    includes options for fitting flexible spectral calibration and
    covariant noise models. """

    # Optionally divide the model by a polynomial for calibration.
    if "calib" in list(self.fit_instructions):
        self.calib = calib_model(self.model_components["calib"],
                                 self.galaxy.spectrum,
                                 self.model_galaxy.spectrum)

        model = self.model_galaxy.spectrum[:, 1]/self.calib.model

    else:
        model = self.model_galaxy.spectrum[:, 1]

    # Calculate differences between model and observed spectrum
    diff = (self.galaxy.spectrum[:, 1] - model)

    if "noise" in list(self.fit_instructions):
        if self.galaxy.spec_cov is not None:
            raise ValueError("Noise modelling is not currently supported "
                             "with manually specified covariance matrix.")

        self.noise = noise_model(self.model_components["noise"],
                                 self.galaxy, model)
    else:
        self.noise = noise_model({}, self.galaxy, model)

    #
    if self.noise.corellated:
        lnlike_spec = self.noise.gp.log_likelihood(self.noise.diff)

        return lnlike_spec

    else:
        # Allow for calculation of chi-squared with direct input
        # covariance matrix - experimental!
        if self.galaxy.spec_cov is not None:
            diff_cov = np.dot(diff.T, self.galaxy.spec_cov_inv)
            self.chisq_spec = np.dot(diff_cov, diff)

            return -0.5*self.chisq_spec

        self.chisq_spec = np.sum(self.noise.inv_var*diff**2)

        if "noise" in list(self.fit_instructions):
            c_spec = -np.log(self.model_components["noise"]["scaling"])
            K_spec = self.galaxy.spectrum.shape[0]*c_spec

        else:
            K_spec = 0.

        return K_spec - 0.5*self.chisq_spec
    
pipes.fitting.fitted_model._lnlike_spec = _lnlike_spec
pipes.plotting.latex_names["period"] = "\\rho"
pipes.plotting.latex_names["Q"] = "Q"
