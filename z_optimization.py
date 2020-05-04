
import numpy as np
from sense.canopy import OneLayer
from sense.soil import Soil
from sense import model
import pdb

### Optimization ###
#---------------------

def run_model(dic, models):

    ke = dic['coef'] * np.sqrt(dic['lai'])
    # ke = dic['coef'] * np.sqrt(dic['vwc'])
    # ke=1
    dic['ke'] = ke

    # surface
    soil = Soil(mv=dic['mv'], C_hh=dic['C_hh'], C_vv=dic['C_vv'], D_hh=dic['D_hh'], D_vv=dic['D_vv'], C_hv=dic['C_hv'], D_hv=dic['D_hv'], V2=dic['V2'], s=dic['s'], clay=dic['clay'], sand=dic['sand'], f=dic['f'], bulk=dic['bulk'], l=dic['l'])

    # canopy
    can = OneLayer(canopy=dic['canopy'], ke_h=dic['ke'], ke_v=dic['ke'], d=dic['d'], ks_h = dic['omega']*dic['ke'], ks_v = dic['omega']*dic['ke'], V1=dic['V1'], V2=dic['V2'], A_hh=dic['A_hh'], B_hh=dic['B_hh'], A_vv=dic['A_vv'], B_vv=dic['B_vv'], A_hv=dic['A_hv'], B_hv=dic['B_hv'])

    S = model.RTModel(surface=soil, canopy=can, models=models, theta=dic['theta'], freq=dic['f'])
    S.sigma0()
    return S.__dict__['stot']['vv'[::-1]], S.__dict__['stot']['vh'[::-1]]


def solve_fun(VALS, var_opt, dic, models):

    for i in range(len(var_opt)):
        dic[var_opt[i]] = VALS[i]

    vv, vh = run_model(dic, models)

    return vv, vh

def fun_opt(VALS, var_opt, dic, models, pol):

    if pol == 'vv':
        return(np.nansum(np.square(solve_fun(VALS, var_opt, dic, models)[0]-dic['vv'])))
    elif pol == 'vh':
        return(np.nansum(np.square(solve_fun(VALS, var_opt, dic, models)[1]-dic['vh'])))
    elif pol == 'vv_vh':
        return(np.nansum(np.square((solve_fun(VALS, var_opt, dic, models)[0]-dic['vv'])/2+(solve_fun(VALS, var_opt, dic, models)[1]-dic['vh'])/2)))

def data_optimized_run(n, field_data, theta_field, sm_field, height_field, lai_field, vwc_field, pol):
    n = np.int(np.floor(n/2))

    if n > 0:
        field_data = field_data.drop(field_data.index[-n:])
        field_data = field_data.drop(field_data.index[0:n])
        theta_field = theta_field.drop(theta_field.index[-n:])
        theta_field = theta_field.drop(theta_field.index[0:n])

    sm_field = field_data.filter(like='SM')
    height_field = field_data.filter(like='Height')/100
    lai_field = field_data.filter(like='LAI')
    vwc_field = field_data.filter(like='VWC')
    vwcpro_field = field_data.filter(like='watercontentpro')
    relativeorbit = field_data.filter(like='relativeorbit')

    vv_field = field_data.filter(like='sigma_sentinel_vv')
    vh_field = field_data.filter(like='sigma_sentinel_vh')

    pol_field = field_data.filter(like='sigma_sentinel_'+pol)
    return field_data, theta_field, sm_field, height_field, lai_field, vwc_field, vv_field, vh_field, pol_field, relativeorbit, vwcpro_field
