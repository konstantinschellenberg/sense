#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
# from sense.surface import Dubois95, Oh92
from sense.util import f2lam
from sense.model import RTModel
from sense.soil import Soil
from sense.canopy import OneLayer
import matplotlib.pyplot as plt


def main():
    
    # -----------------------------------
    # theta
    theta_deg = np.arange(0., 70.)
    theta = np.deg2rad(theta_deg)
    
    # -----------------------------------
    # model selection
    
    models = {'surface': 'Dubois95', 'canopy': 'turbid_rayleigh'}
    pol = 'vv'
    
    # -----------------------------------
    # settings
    
    # soil model parameters
    f = 5.405  # GHz
    # lam = f2lam(f)  # m
    s = 0.0015  # m
    eps = 15. - 4.0j
    
    # -----------------------------------
    # canopy model parameters (short alfalfa)
    
    omega = 0.1
    d = 0.17
    tau = 2.5
    ke = tau / d
    omega = 0.27
    ks = omega * ke
    
    # -----------------------------------
    # Model initialization
    
    # Soil model
    S = Soil(f=f, s=s, eps=eps)
    
    # Canopy model
    C = OneLayer(ke_h=ke, ke_v=ke, d=d, ks_v=ks, ks_h=ks, canopy=models['canopy'])
    
    # Combined Model initialization
    RT = RTModel(theta=theta, models=models, surface=S, canopy=C, freq=f)
    
    # -----------------------------------
    # Run RT model
    RT.sigma0()
    back_short = RT.stot[pol]
    # tall alfalfa
    d = 0.55
    tau = 0.45
    ke = tau / d
    omega = 0.175
    ks = omega * ke
    S = Soil(f=f, s=s, eps=eps)
    C = OneLayer(ke_h=ke, ke_v=ke, d=d, ks_v=ks, ks_h=ks, canopy=models['canopy'])
    RT = RTModel(theta=theta, models=models, surface=S, canopy=C, freq=f)
    RT.sigma0()
    # plot first part
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(theta_deg, 10. * np.log10(back_short), label='short', color='b')
    ax.plot(theta_deg, 10. * np.log10(RT.stot[pol]), label='tall', color='r')
    
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
    main()
