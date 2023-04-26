"""
compare results of implemented models
against Fig 11.07 from Ulaby (2014) and
compare own results against references
from the Ulaby example codes provided
http://mrs.eecs.umich.edu/codes/Module11_1/Module11_1.html

Ulaby uses for online code and book figure the reflectivity without the roughness correction factor. Email response why he did that was: He don't know he actually would use reflectivity with correction factor. Used parameter within this code: reflectivity with roughness correction factor! Therefore slightly different results. Matlab/graphical interface and python code produce same results if the same term for reflectivity is used.

Difference between matlab code and graphical interface is the curve of the ground contribution for low incidence angles. Don't know why.

Implementation of SSRT-model should be fine!!!
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.sep + '..')

import numpy as np

# from sense.surface import Dubois95, Oh92
from sense.util import f2lam
from sense.model import RTModel

from sense.soil import Soil
from sense.canopy import OneLayer

import matplotlib.pyplot as plt

plt.close('all')

theta_deg = np.arange(0.,71.)
theta = np.deg2rad(theta_deg)

f  = 3.  # GHz
lam = f2lam(f)  # m

s = 0.01  # m
l = 0.1  # m

omega = 0.1



# canopy
ke=1.
ks=omega*ke
# ks=0.63
# results strongly depend on the surface scattering model chosen!
# Oh92 gives a much smoother response at low incidence angles compared
# to the Dubois95 model

# shape of ACL would certailny also play an important role!

models = {'surface' : 'Oh92', 'canopy' : 'turbid_rayleigh'}
S = Soil(f=f, s=s, l=l, mv=0.2, sand=0.3, clay=0.3)

pol='vv'

d = 0.22
C = OneLayer(ke_h=ke, ke_v=ke, d=d, ks_v=ks, ks_h=ks, canopy = models['canopy'])
RT = RTModel(theta=theta, models=models, surface=S, canopy=C, freq=f)
RT.sigma0()

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(theta_deg, 10.*np.log10(RT.stot[pol]), label='STOT', color='k')
ax.plot(theta_deg, 10.*np.log10(RT.s0g[pol]), label='SIGGROUND', color='r')
ax.plot(theta_deg, 10.*np.log10(RT.s0c[pol]), label='SIG can', color='b')
ax.plot(theta_deg, 10.*np.log10(RT.s0cgt[pol]), label='SIG can ground', color='g')
ax.plot(theta_deg, 10.*np.log10(RT.s0gcg[pol]), label='SIG ground can ground', color='k', linestyle='--')
#ax.plot(theta_deg, 10.*np.log10(RT.G.rt_s.vv), label='surface', linestyle='-.')

#ax.legend()
ax.set_title('d='+str(d))


ax.grid()
ax.set_xlabel('incidence angle')
ax.set_ylabel('sigma')
ax.set_xlim(0.,70.)
ax.set_ylim(-17.,-9.)

plt.show()
