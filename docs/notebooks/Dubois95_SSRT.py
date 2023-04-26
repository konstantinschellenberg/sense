#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
# from sense.surface import Dubois95, Oh92
from sense.util import f2lam
from sense.model import RTModel
from sense.soil import Soil
from sense.canopy import OneLayer
import matplotlib.pyplot as plt

def run_models():

    # -----------------------------------
    # settings
    
    # soil model parameters
    s = 0.0015  # m
    eps = 15. - 4.0j
    # lam = f2lam(f)  # m
    
    # -----------------------------------
    # canopy model parameters (short alfalfa)
    
    d = 0.17
    tau = 2.5
    ke = tau / d
    omega = 0.27
    ks = omega * ke
    
    # -----------------------------------
    # Model initialization
    
    # Soil model
    S = Soil(surface=models["surface"], f=f, s=s, eps=eps)
    
    # Canopy model
    C = OneLayer(canopy=models['canopy'], ke_h=ke, ke_v=ke, d=d, ks_v=ks, ks_h=ks)
    
    # Combined Model initialization
    RT1 = RTModel(theta=theta, models=models, surface=S, canopy=C, freq=f)
    RT1.sigma0(pol=["VV"])
    RT1.pol
    RT1._sigma0()
    
    # todo: Throws RuntimeWarning for hv pol
    
    # -----------------------------------
    # canopy model parameters (long alfalfa)
    
    d = 0.55
    tau = 0.45
    ke = tau / d
    omega = 0.175
    ks = omega * ke
    
    # -----------------------------------
    # Model initialization
    
    S = Soil(surface=models["surface"], f=f, s=s, eps=eps)
    C = OneLayer(canopy=models['canopy'], ke_h=ke, ke_v=ke, d=d, ks_v=ks, ks_h=ks)
    RT2 = RTModel(theta=theta, models=models, surface=S, canopy=C, freq=f)
    RT2.sigma0(pol=["VV"])

    # -----------------------------------
    # viz
    # plot first part
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(theta_deg, 10. * np.log10(RT1.stot[pol]), label='short', color='b')
    ax.plot(theta_deg, 10. * np.log10(RT2.stot[pol]), label='tall', color='r')
    
    ax.legend()
    ax.set_title('Fig 11-15 Alfalfa')
    
    ax.grid()
    ax.set_xlabel('incidence angle [°]')
    ax.set_ylabel('sigma vv [dB]')
    ax.set_xlim(0., 70.)
    ax.set_ylim(-16., 6.)
    
    plt.show()
    pass


if __name__ == '__main__':
    
    # -----------------------------------
    # theta space
    theta_deg = np.arange(0., 70.)
    theta = np.deg2rad(theta_deg)
    
    # wavelength
    f = 5.405  # GHz

    # -----------------------------------
    # model selection
    
    models = {'surface': 'Dubois95', 'canopy': 'turbid_rayleigh'}
    pol = 'vv'
    
    run_models()
