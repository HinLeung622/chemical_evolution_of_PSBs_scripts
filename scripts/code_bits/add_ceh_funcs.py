import bagpipes as pipes
import numpy as np

# adding to chemical_enrichment_history.py
"""
structure = within psb_wild2020
where dictionary metallicity has structure:
['metallicity_type'] = 'constant' or 'psb_two_step'
if 'constant':
    ['metallicity'] = xx
if 'psb_two_step':
    ['metallicity_old'] = xx
    ['metallicity_burst'] = xx
['metallicity_scatter'] = 'delta' or 'exp' or 'lognorm'
"""

pipes.config.zmet_sampling = np.arange(0., 10., 0.01) + 0.005
import bagpipes.config as config

def __init__(self, model_comp, sfh_weights):

    self.zmet_vals = config.metallicities
    self.zmet_lims = config.metallicity_bins

    self.grid_comp = {}
    self.grid = np.zeros((self.zmet_vals.shape[0],
                          config.age_sampling.shape[0]))

    for comp in list(sfh_weights):
        if comp is not "total":
            if 'metallicity_type' not in model_comp[comp].keys():
                self.grid_comp[comp] = self.delta(model_comp[comp],
                                                  sfh_weights[comp])
            else:
                self.grid_comp[comp] = getattr(self, model_comp[comp]['metallicity_type']
                                              )(model_comp[comp],sfh_weights[comp])

            self.grid += self.grid_comp[comp]

def delta(self, comp, sfh, zmet=None, nested=False):
    """ Delta function metallicity history. """

    if zmet is None:
        zmet = comp["metallicity"]

    weights = np.zeros(self.zmet_vals.shape[0])

    high_ind = self.zmet_vals[self.zmet_vals < zmet].shape[0]

    if high_ind == self.zmet_vals.shape[0]:
        weights[-1] = 1.

    elif high_ind == 0:
        weights[0] = 1.

    else:
        low_ind = high_ind - 1
        width = (self.zmet_vals[high_ind] - self.zmet_vals[low_ind])
        weights[high_ind] = (zmet - self.zmet_vals[low_ind])/width
        weights[high_ind-1] = 1 - weights[high_ind]

    if nested:
        return weights
    else:
        return np.expand_dims(weights, axis=1)*np.expand_dims(sfh, axis=0)
    
def exp(self, comp, sfh, zmet=None, nested=False):
    """ P(Z) = exp(-z/z_mean). Currently no age dependency! """

    if zmet is None:
        tau_zmet = comp["metallicity"]
    else:
        tau_zmet = zmet

    weights = np.zeros(self.zmet_vals.shape[0])

    vals_hr = np.arange(0., 10., 0.01) + 0.005

    factors_hr = (1./tau_zmet)*np.exp(-vals_hr/tau_zmet)

    for i in range(weights.shape[0]):
        lowmask = (vals_hr > self.zmet_lims[i])
        highmask = (vals_hr < self.zmet_lims[i+1])
        weights[i] = np.sum(0.01*factors_hr[lowmask & highmask])

    if nested:
        return weights
    else:
        return np.expand_dims(weights, axis=1)*np.expand_dims(sfh, axis=0)
    
def lognorm(self, comp, sfh, zmet=None, nested=False):
    """
    log normal metallicity distribution scatter. 
    Functional form: P(x) = 1/(x*sigma*np.sqrt(2*np.pi)) * np.exp(-(np.log(x)-mu)**2/(2*sigma**2))
    where mu = ln(metallicity mean), sigma = some concentration measurement
    """
    
    if zmet is None:
        log_mean_zmet = np.log(comp["metallicity"])
    else:
        log_mean_zmet = np.log(zmet)
    sigma = 0.45

    weights = np.zeros(self.zmet_vals.shape[0])

    vals_hr = np.arange(0., 10., 0.01) + 0.005

    factors_hr = 1/(vals_hr*sigma*np.sqrt(2*np.pi)) * np.exp(-(np.log(vals_hr)-log_mean_zmet)**2/(2*sigma**2))

    for i in range(weights.shape[0]):
        lowmask = (vals_hr > self.zmet_lims[i])
        highmask = (vals_hr < self.zmet_lims[i+1])
        weights[i] = np.sum(0.01*factors_hr[lowmask & highmask])

    if nested:
        return weights
    else:
        return np.expand_dims(weights, axis=1)*np.expand_dims(sfh, axis=0)
    
def constant(self, comp, sfh):
    """ constant metallicity without any variation in time """
    
    zmet = comp["metallicity"]
    if "metallicity_scatter" not in comp.keys():
        comp["metallicity_scatter"] = "delta"
    
    weights = getattr(self, comp['metallicity_scatter']
                     )(comp, sfh, zmet=zmet, nested=True)
    return np.expand_dims(weights, axis=1)*np.expand_dims(sfh, axis=0)

def two_step(self, comp, sfh):
    """ 2-step metallicities (time-varying!) time of shift as free parameter """
    
    zmet_old = comp["metallicity_old"]
    zmet_new = comp["metallicity_new"]
    step_age = comp["metallicity_step_age"]*10**9
    if "metallicity_scatter" not in comp.keys():
        comp["metallicity_scatter"] = 'delta'
    
    # get SSP ages
    SSP_ages = config.age_sampling
    SSP_age_bins = config.age_bins
    
    # loop through all SSP ages
    zmet_comp = np.zeros((self.zmet_vals.shape[0], sfh.shape[0]))
    for i,agei in enumerate(SSP_ages):
        # detect if the SSP age's higher boundary > tburst and lower boundary < tburst
        if SSP_age_bins[i+1]>step_age and SSP_age_bins[i]<step_age:
            # interp between to get metallicity at this SSP
            width = SSP_age_bins[i+1] - SSP_age_bins[i]
            old_weight = (SSP_age_bins[i+1] - step_age)/width
            burst_weight = (step_age - SSP_age_bins[i])/width
            SSP_zmet = old_weight*zmet_old + burst_weight*zmet_new
            # weights from metallicity scatter
            if comp['metallicity_scatter'] == 'concentrate':
                zmet_comp[:,i] = getattr(self, 'delta'
                                        )(comp, sfh, zmet=SSP_zmet, nested=True)
            else:
                zmet_comp[:,i] = getattr(self, comp['metallicity_scatter']
                                        )(comp, sfh, zmet=SSP_zmet, nested=True)
        
        # if before tburst
        elif SSP_age_bins[i]>step_age:
            # weights from metallicity scatter
            if comp['metallicity_scatter'] == 'concentrate':
                zmet_comp[:,i] = getattr(self, 'lognorm'
                                        )(comp, sfh, zmet=zmet_old, nested=True)
            else:
                zmet_comp[:,i] = getattr(self, comp['metallicity_scatter']
                                        )(comp, sfh, zmet=zmet_old, nested=True)
            
        # if after tburst
        elif SSP_age_bins[i+1]<step_age:
            # weights from metallicity scatter
            if comp['metallicity_scatter'] == 'concentrate':
                zmet_comp[:,i] = getattr(self, 'delta'
                                        )(comp, sfh, zmet=zmet_new, nested=True)
            else:
                zmet_comp[:,i] = getattr(self, comp['metallicity_scatter']
                                        )(comp, sfh, zmet=zmet_new, nested=True)
            
        #else:
        #    print('help')
        
    return zmet_comp*np.expand_dims(sfh, axis=0)

def psb_two_step(self, comp, sfh):
    """ 2-step metallicities (time-varying!) for psb SFH shape, shift at burstage """
    
    zmet_old = comp["metallicity_old"]
    zmet_burst = comp["metallicity_burst"]
    burstage = comp["burstage"]*10**9
    if "metallicity_scatter" not in comp.keys():
        comp["metallicity_scatter"] = 'delta'
    
    # get SSP ages
    SSP_ages = config.age_sampling
    SSP_age_bins = config.age_bins
    
    # loop through all SSP ages
    zmet_comp = np.zeros((self.zmet_vals.shape[0], sfh.shape[0]))
    for i,agei in enumerate(SSP_ages):
        # detect if the SSP age's higher boundary > tburst and lower boundary < tburst
        if SSP_age_bins[i+1]>burstage and SSP_age_bins[i]<burstage:
            # interp between to get metallicity at this SSP
            width = SSP_age_bins[i+1] - SSP_age_bins[i]
            old_weight = (SSP_age_bins[i+1] - burstage)/width
            burst_weight = (burstage - SSP_age_bins[i])/width
            SSP_zmet = old_weight*zmet_old + burst_weight*zmet_burst
            # weights from metallicity scatter
            if comp['metallicity_scatter'] == 'concentrate':
                zmet_comp[:,i] = getattr(self, 'delta'
                                        )(comp, sfh, zmet=SSP_zmet, nested=True)
            else:
                zmet_comp[:,i] = getattr(self, comp['metallicity_scatter']
                                        )(comp, sfh, zmet=SSP_zmet, nested=True)
        
        # if before tburst
        elif SSP_age_bins[i]>burstage:
            # weights from metallicity scatter
            if comp['metallicity_scatter'] == 'concentrate':
                zmet_comp[:,i] = getattr(self, 'lognorm'
                                        )(comp, sfh, zmet=zmet_old, nested=True)
            else:
                zmet_comp[:,i] = getattr(self, comp['metallicity_scatter']
                                        )(comp, sfh, zmet=zmet_old, nested=True)
            
        # if after tburst
        elif SSP_age_bins[i+1]<burstage:
            # weights from metallicity scatter
            if comp['metallicity_scatter'] == 'concentrate':
                zmet_comp[:,i] = getattr(self, 'delta'
                                        )(comp, sfh, zmet=zmet_burst, nested=True)
            else:
                zmet_comp[:,i] = getattr(self, comp['metallicity_scatter']
                                        )(comp, sfh, zmet=zmet_burst, nested=True)
            
        #else:
        #    print('help')
    
    return zmet_comp*np.expand_dims(sfh, axis=0)

def psb_linear_step(self, comp, sfh):
    """ 
    2-part metallicities (time-varying!) time of shift set as burstage.
    post-burst component is assumed constant, pre-burst component assumed linear
    with a free-varying slope.
    """
    
    zmet_zero = comp["metallicity_zero"]  # metallicity at 0 lookback time
    # slope of metallicity before burst, Zsun/Gyr
    zmet_slope = comp["metallicity_slope"]/10**9
    zmet_burst = comp["metallicity_burst"]
    burstage = comp["burstage"]*10**9
    if "metallicity_scatter" not in comp.keys():
        comp["metallicity_scatter"] = 'delta'
    
    # get SSP ages
    SSP_ages = config.age_sampling
    SSP_age_bins = config.age_bins
    
    # loop through all SSP ages
    zmet_comp = np.zeros((self.zmet_vals.shape[0], sfh.shape[0]))
    #SSP_zmets = []
    for i,agei in enumerate(SSP_ages):
        # detect if the SSP age's higher boundary > tburst and lower boundary < tburst
        if SSP_age_bins[i+1]>burstage and SSP_age_bins[i]<burstage:
            # interp between to get metallicity at this SSP
            width = SSP_age_bins[i+1] - SSP_age_bins[i]
            zmet_at_h = zmet_zero + zmet_slope*SSP_age_bins[i+1]
            zmet_at_burstage = zmet_zero + zmet_slope*burstage
            area = (zmet_at_h+zmet_at_burstage)*(SSP_age_bins[i+1]-burstage)/2 + \
                   zmet_burst*(burstage-SSP_age_bins[i])
            SSP_zmet = area/width
            # weights from metallicity scatter
            if comp['metallicity_scatter'] == 'concentrate':
                zmet_comp[:,i] = getattr(self, 'delta'
                                        )(comp, sfh, zmet=SSP_zmet, nested=True)
            else:
                zmet_comp[:,i] = getattr(self, comp['metallicity_scatter']
                                        )(comp, sfh, zmet=SSP_zmet, nested=True)
            #SSP_zmets.append(SSP_zmet)
        
        # if before tburst
        elif SSP_age_bins[i]>burstage:
            zmet_at_l = zmet_zero + zmet_slope*SSP_age_bins[i]
            zmet_at_h = zmet_zero + zmet_slope*SSP_age_bins[i+1]
            SSP_zmet = (zmet_at_l+zmet_at_h)/2
            # weights from metallicity scatter
            if comp['metallicity_scatter'] == 'concentrate':
                zmet_comp[:,i] = getattr(self, 'lognorm'
                                        )(comp, sfh, zmet=SSP_zmet, nested=True)
            else:
                zmet_comp[:,i] = getattr(self, comp['metallicity_scatter']
                                        )(comp, sfh, zmet=SSP_zmet, nested=True)
            #SSP_zmets.append(SSP_zmet)
            
        # if after tburst
        elif SSP_age_bins[i+1]<burstage:
            # weights from metallicity scatter
            if comp['metallicity_scatter'] == 'concentrate':
                zmet_comp[:,i] = getattr(self, 'delta'
                                        )(comp, sfh, zmet=zmet_burst, nested=True)
            else:
                zmet_comp[:,i] = getattr(self, comp['metallicity_scatter']
                                        )(comp, sfh, zmet=zmet_burst, nested=True)
            #SSP_zmets.append(zmet_burst)
    
    return zmet_comp*np.expand_dims(sfh, axis=0)

def linear_step(self, comp, sfh):
    """ 
    2-part metallicities (time-varying!) time of shift set as burstage.
    post-burst component is assumed constant, pre-burst component assumed linear
    with a free-varying slope.
    """
    
    zmet_zero = comp["metallicity_zero"]  # metallicity at 0 lookback time
    # slope of metallicity before burst, Zsun/Gyr
    zmet_slope = comp["metallicity_slope"]/10**9
    zmet_new = comp["metallicity_new"]
    step_age = comp["metallicity_step_age"]*10**9
    if "metallicity_scatter" not in comp.keys():
        comp["metallicity_scatter"] = 'delta'
    
    # get SSP ages
    SSP_ages = config.age_sampling
    SSP_age_bins = config.age_bins
    
    # loop through all SSP ages
    zmet_comp = np.zeros((self.zmet_vals.shape[0], sfh.shape[0]))
    #SSP_zmets = []
    for i,agei in enumerate(SSP_ages):
        # detect if the SSP age's higher boundary > tburst and lower boundary < tburst
        if SSP_age_bins[i+1]>step_age and SSP_age_bins[i]<step_age:
            # interp between to get metallicity at this SSP
            width = SSP_age_bins[i+1] - SSP_age_bins[i]
            zmet_at_h = zmet_zero + zmet_slope*SSP_age_bins[i+1]
            zmet_at_burstage = zmet_zero + zmet_slope*step_age
            area = (zmet_at_h+zmet_at_burstage)*(SSP_age_bins[i+1]-step_age)/2 + \
                   zmet_new*(step_age-SSP_age_bins[i])
            SSP_zmet = area/width
            # weights from metallicity scatter
            if comp['metallicity_scatter'] == 'concentrate':
                zmet_comp[:,i] = getattr(self, 'delta'
                                        )(comp, sfh, zmet=SSP_zmet, nested=True)
            else:
                zmet_comp[:,i] = getattr(self, comp['metallicity_scatter']
                                        )(comp, sfh, zmet=SSP_zmet, nested=True)
            #SSP_zmets.append(SSP_zmet)
        
        # if before tburst
        elif SSP_age_bins[i]>step_age:
            zmet_at_l = zmet_zero + zmet_slope*SSP_age_bins[i]
            zmet_at_h = zmet_zero + zmet_slope*SSP_age_bins[i+1]
            SSP_zmet = (zmet_at_l+zmet_at_h)/2
            # weights from metallicity scatter
            if comp['metallicity_scatter'] == 'concentrate':
                zmet_comp[:,i] = getattr(self, 'lognorm'
                                        )(comp, sfh, zmet=SSP_zmet, nested=True)
            else:
                zmet_comp[:,i] = getattr(self, comp['metallicity_scatter']
                                        )(comp, sfh, zmet=SSP_zmet, nested=True)
            #SSP_zmets.append(SSP_zmet)
            
        # if after tburst
        elif SSP_age_bins[i+1]<step_age:
            # weights from metallicity scatter
            if comp['metallicity_scatter'] == 'concentrate':
                zmet_comp[:,i] = getattr(self, 'delta'
                                        )(comp, sfh, zmet=zmet_new, nested=True)
            else:
                zmet_comp[:,i] = getattr(self, comp['metallicity_scatter']
                                        )(comp, sfh, zmet=zmet_new, nested=True)
            #SSP_zmets.append(zmet_new)
    
    return zmet_comp*np.expand_dims(sfh, axis=0)

pipes.models.chemical_enrichment_history.__init__ = __init__
pipes.models.chemical_enrichment_history.delta = delta
pipes.models.chemical_enrichment_history.exp = exp
pipes.models.chemical_enrichment_history.lognorm = lognorm
pipes.models.chemical_enrichment_history.constant = constant
pipes.models.chemical_enrichment_history.two_step = two_step
pipes.models.chemical_enrichment_history.psb_two_step = psb_two_step
pipes.models.chemical_enrichment_history.psb_linear_step = psb_linear_step
pipes.models.chemical_enrichment_history.linear_step = linear_step
