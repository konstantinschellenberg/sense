
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator
# import matplotlib.ticker
import numpy as np
from sense.canopy import OneLayer
from sense.soil import Soil
from sense import model
import scipy.stats
from scipy.optimize import minimize
import pdb
from z_helper import *
from z_optimization import *



### Data preparation ###
#-----------------------------------------------------------------
# storage information
path = '/media/tweiss/Work/z_final_mni_data_2017'
file_name = 'in_situ_s1' # theta needs to be changed to for norm multi
extension = '.csv'

path_agro = '/media/nas_data/2017_MNI_campaign/field_data/meteodata/agrarmeteorological_station'
file_name_agro = 'Eichenried_01012017_31122017_hourly'
extension_agro = '.csv'

field = '508_high'

field_plot = []
# field_plot = ['542_high','542_low','542_med']
# field_plot = ['301_high','301_low','301_med']
pol = 'vv'
# pol = 'vh'

# output path
plot_output_path = '/media/tweiss/Work/z_paper_rt_model_comparison/'

df, df_agro, field_data, field_data_orbit, theta_field, sm_field, height_field, lai_field, vwc_field, pol_field, vv_field, vh_field, relativeorbit, vwcpro_field = read_data(path, file_name, extension, field, path_agro, file_name_agro, extension_agro, pol)

### Run SenSe module
#-----------------------------------------------------------------
#### Choose models
#-----------------

surface_list = ['Oh92', 'Oh04', 'Dubois95', 'WaterCloud', 'I2EM']
canopy_list = ['turbid_isotropic', 'water_cloud']

# surface_list = ['Oh92']
# canopy_list = ['water_cloud']

### option for time invariant or variant calibration of parameter
#-------------------------------
opt_mod = 'time invariant'
opt_mod = 'time variant'
#---------------------------

### plot option: "single" or "all" modelcombination
#------------------------------
# plot = 'single'
plot = 'all'
#------------------------------

### plot option scatterplot or not
#-------------------------------
# style = 'scatterplot'
style = ''
# style = 'single_esu'

### plot option for scatterplot single ESU
#------------------------------------
style_2 = 'scatterplot_single_ESU'
style_2 = ''
#-----------------------------------

# Initialize plot settings
#---------------------------
if style == 'scatterplot':
    fig, ax = plt.subplots(figsize=(10, 10))
else:
    fig, ax = plt.subplots(figsize=(17, 10))
# plt.title('Winter Wheat')
plt.ylabel('Backscatter [dB]', fontsize=18)
plt.xlabel('Date', fontsize=18)
plt.tick_params(labelsize=17)


if pol == 'vv':
    ax.set_ylim([-25,-7.5])
elif pol == 'vh':
    ax.set_ylim([-30,-15])

colormaps = ['Greens', 'Purples', 'Blues', 'Oranges', 'Reds', 'Greys', 'pink', 'bone', 'Blues', 'Blues', 'Blues']
j = 0

colormap = plt.get_cmap(colormaps[j])
colors = [colormap(jj) for jj in np.linspace(0.35, 1., 3)]

for k in surface_list:

    for kk in canopy_list:
        df, df_agro, field_data, field_data_orbit, theta_field, sm_field, height_field, lai_field, vwc_field, pol_field, vv_field, vh_field, relativeorbit, vwcpro_field = read_data(path, file_name, extension, field, path_agro, file_name_agro, extension_agro, pol)
        freq = 5.405
        clay = 0.08
        sand = 0.12
        bulk = 1.5
        # s = 0.0105 # vv
        s = 0.0115
        # s = 0.009 # vh ?????

        C_hh = 0
        D_hh = 0
        C_hv = -22.5
        D_hv = 3.2
        C_vv = -14.609339
        D_vv = 12.884086

        ### Canopy
        # Water Cloud (A, B, V1, V2, theta)
        # SSRT (coef, omega, theta)
        #-----------------------------------
        A_hh = 0
        B_hh = 0
        A_hv = 0.029
        B_hv = 0.0013
        A_vv = 0.0029
        B_vv = 0.13
        V1 = lai_field.values.flatten()
        # V1 = vwc_field.values.flatten()
        # V1 = lai_field.values.flatten()*(vwcpro_field.values.flatten()/100)
        # V1 = vwc_field.values.flatten() / height_field.values.flatten()
        # V1 = (vwcpro_field.values.flatten()/100)**3
        # V1 = (vwcpro_field.values.flatten()/100)**3 * height_field.values.flatten()
        V2 = V1 # initialize in surface model
        coef = 1.
        omega = 0.027 # vv
        # omega = 0.015 # vh
        # omega = 0.015
        # omega = 1. # vh
        # IEM
        l = 0.01


        surface = k
        canopy = kk
        models = {'surface': surface, 'canopy': canopy}

        #### Optimization
        #-----------------

        if opt_mod == 'time invariant':

            dic = {"mv":sm_field.values.flatten(), "C_hh":C_hh, "C_vv":C_vv, "D_hh":D_hh, "D_vv":D_vv, "C_hv":C_hv, "D_hv":D_hv, "s":s, "clay":clay, "sand":sand, "f":freq, "bulk":bulk, "l":l, "canopy":canopy, "d":height_field.values.flatten(), "V1":V1, "V2":V2, "A_hh":A_hh, "B_hh":B_hh, "A_vv":A_vv, "B_vv":B_vv, "A_hv":A_hv, "B_hv":B_hv, "lai":lai_field.values.flatten(), "vwc":vwc_field.values.flatten(), "pol_value":pol_field.values.flatten(), "vv":vv_field.values.flatten(), "vh":vh_field.values.flatten(), "theta":theta_field.values.flatten(), "omega": omega, "coef": coef, "vwcpro": vwcpro_field.values.flatten()}

            if canopy == 'turbid_isotropic':
                var_opt = ['coef']
                guess = [2]
                bounds = [(0.00000000000001,5.5)]
                # var_opt = ['coef', 'omega']
                # guess = [0.5, 0.027]
                # bounds = [(0.00000000000001,5.5),(0.00001,0.09)]
            elif surface == 'WaterCloud' and canopy == 'water_cloud':
                var_opt = ['A_vv', 'B_vv', 'A_hv', 'B_hv', 'C_vv', 'D_vv', 'C_hv', 'D_hv']
                guess = [A_vv, B_vv, A_hv, B_hv, C_vv, D_vv, C_hv, D_hv]
                bounds = [(0.,1), (0.,1), (0.,1), (0.,1), (-20.,-1.), (1.,20.), (-20.,-1.), (1.,20.)]
            elif canopy == 'water_cloud':
                var_opt = ['A_vv', 'B_vv', 'A_hv', 'B_hv']
                guess = [A_vv, B_vv, A_hv, B_hv]
                bounds = [(0.,1), (0.,1), (0.,1), (0.,1)]

            method = 'L-BFGS-B'

            res = minimize(fun_opt,guess,args=(var_opt, dic, models, pol),bounds=bounds, method=method)

            fun_opt(res.x, var_opt, dic, models, pol)
            aaa = res.x

        if opt_mod == 'time variant':
            aaa = [[],[],[],[],[],[],[],[],[],[],[],[]]
            n=7

            for i in range(len(pol_field.values.flatten())-n+1):

                if type(coef) == float:
                    dic = {"mv":sm_field.values.flatten()[i:i+n], "C_hh":C_hh, "C_vv":C_vv, "D_hh":D_hh, "D_vv":D_vv, "C_hv":C_hv, "D_hv":D_hv, "V2":V2[i:i+n], "s":s, "clay":clay, "sand":sand, "f":freq, "bulk":bulk, "l":l, "canopy":canopy, "d":height_field.values.flatten()[i:i+n], "V1":V1[i:i+n], "A_hh":A_hh, "B_hh":B_hh, "A_vv":A_vv, "B_vv":B_vv, "A_hv":A_hv, "B_hv":B_hv, "lai":lai_field.values.flatten()[i:i+n], "vwc":vwc_field.values.flatten()[i:i+n], "pol_value":pol_field.values.flatten()[i:i+n], "vv":vv_field.values.flatten()[i:i+n], "vh":vh_field.values.flatten()[i:i+n], "theta":theta_field.values.flatten()[i:i+n], "omega": omega, "coef": coef, "vwcpro": vwcpro_field.values.flatten()[i:i+n]}
                else:
                    dic = {"mv":sm_field.values.flatten()[i:i+n], "C_hh":C_hh, "C_vv":C_vv, "D_hh":D_hh, "D_vv":D_vv, "C_hv":C_hv, "D_hv":D_hv, "V2":V2[i:i+n], "s":s, "clay":clay, "sand":sand, "f":freq, "bulk":bulk, "l":l, "canopy":canopy, "d":height_field.values.flatten()[i:i+n], "V1":V1[i:i+n], "A_hh":A_hh, "B_hh":B_hh, "A_vv":A_vv, "B_vv":B_vv, "A_hv":A_hv, "B_hv":B_hv, "lai":lai_field.values.flatten()[i:i+n], "vwc":vwc_field.values.flatten()[i:i+n], "pol_value":pol_field.values.flatten()[i:i+n], "vv":vv_field.values.flatten()[i:i+n], "vh":vh_field.values.flatten()[i:i+n], "theta":theta_field.values.flatten()[i:i+n], "omega": omega, "vwcpro": vwcpro_field.values.flatten()[i:i+n]}

                # if canopy == 'turbid_isotropic' and surface == 'WaterCloud':
                #     var_opt = ['coef', 'C_vv', 'D_vv', 'C_hv', 'D_hv']
                #     guess = [0.01, C_vv, D_vv, C_hv, D_hv]
                #     bounds = [(0.1,5.5), (-20.,-1.), (1.,20.), (-20.,-1.), (1.,20.)]
                if canopy == 'turbid_isotropic':
                    var_opt = ['coef']
                    guess = [0.1]
                    bounds = [(0.,5.5)]
                elif surface == 'WaterCloud' and canopy == 'water_cloud':
                    if i == 0:
                        var_opt = ['A_vv', 'B_vv', 'A_hv', 'B_hv']
                        guess = [A_vv, B_vv, A_hv, B_hv]
                        bounds = [(0.001,1), (0.01,1), (0.00001,1), (0.00001,1)]
                        # bounds = [(0.001,1), (guess[1]*0.3, guess[1]*1.7), (0.001,1), (guess[3]*0.75, guess[3]*1.25)]
                    else:
                        var_opt = ['A_vv', 'B_vv', 'A_hv', 'B_hv']
                        guess = [A_vv, B_vv, A_hv, B_hv]
                        # bounds = [(res.x[0]*0.7,res.x[0]*1.3), (res.x[1]*0.7,res.x[1]*1.3), (res.x[2]*0.7,res.x[2]*1.3), (res.x[3]*0.7,res.x[3]*1.3)]
                        # bounds = [(res.x[0]*0.5,res.x[0]*1.5), (res.x[1]*0.5,res.x[1]*1.5), (res.x[2]*0.5,res.x[2]*1.5), (res.x[3]*0.5,res.x[3]*1.5)]
                        bounds = [(res.x[0]*0.6,res.x[0]*1.4), (res.x[1]*0.8,res.x[1]*2.2), (res.x[2]*0.9,res.x[2]*1.1), (res.x[3]*0.9,res.x[3]*1.1)]
                        # bounds = [(res.x[0]*0.95,res.x[0]*1.05), (res.x[1]*0.95,res.x[1]*1.05), (res.x[2]*0.95,res.x[2]*1.05), (res.x[3]*0.95,res.x[3]*1.05)]

                    # var_opt = ['A_vv', 'B_vv', 'A_hv', 'B_hv', 'C_vv', 'D_vv', 'C_hv', 'D_hv']
                    # guess = [A_vv, B_vv, A_hv, B_hv, C_vv, D_vv, C_hv, D_hv]
                    # bounds = [(0.,1), (guess[1]*0.55, guess[1]*1.55), (0.,1), (guess[3]*0.75, guess[3]*1.25), (-20.,-1.), (1.,20.), (-20.,-1.), (1.,20.)]
                    # var_opt = ['C_vv', 'D_vv', 'C_hv', 'D_hv']
                    # guess = [C_vv, D_vv, C_hv, D_hv]
                    # bounds = [(-20.,-1.), (1.,20.), (-20.,-1.), (1.,20.)]
                elif canopy == 'water_cloud':
                    if i == 0:
                        var_opt = ['A_vv', 'B_vv', 'A_hv', 'B_hv']
                        guess = [A_vv, B_vv, A_hv, B_hv]
                        bounds = [(0.0001,0.001), (0.001,1), (0.00001,1), (0.00001,1)]
                    else:
                        var_opt = ['A_vv', 'B_vv', 'A_hv', 'B_hv']
                        guess = [A_vv, B_vv, A_hv, B_hv]
                        # bounds = [(res.x[0]*0.7,res.x[0]*1.3), (res.x[1]*0.7,res.x[1]*1.3), (res.x[2]*0.7,res.x[2]*1.3), (res.x[3]*0.7,res.x[3]*1.3)]
                        # bounds = [(res.x[0]*0.5,res.x[0]*1.5), (res.x[1]*0.5,res.x[1]*1.5), (res.x[2]*0.5,res.x[2]*1.5), (res.x[3]*0.5,res.x[3]*1.5)]
                        bounds = [(res.x[0]*0.9,res.x[0]*1.1), (res.x[1]*0.5,res.x[1]*2.2), (res.x[2]*0.9,res.x[2]*1.1), (res.x[3]*0.9,res.x[3]*1.1)]
                        # bounds = [(res.x[0]*0.95,res.x[0]*1.05), (res.x[1]*0.95,res.x[1]*5.05), (res.x[2]*0.95,res.x[2]*1.05), (res.x[3]*0.95,res.x[3]*1.05)]
                        # bounds = [(0.001,1), (0.001,1), (0.00001,1), (0.00001,1)]
                # var_opt = ['omega']
                # guess = [0.1]
                # bounds = [(0.,5.5)]

                # var_opt = ['s', 'coef', 'omega']
                # guess = [0.01, 0.1, 0.01]
                # bounds = [(0.001,0.03),(0.,2.5),(0.001,0.1)]

                # var_opt = ['C_hv', 'D_hv']
                # guess = [-13, 14]
                # bounds = [(-200.,100.),(-200.,400.)]

                # var_opt = ['A_vv', 'B_vv']

                # try:
                #     guess = [res.x[0], res.x[1]]
                # except:
                #     guess = [0.005, 0.09]
                # # bounds = [(0.000,5.),(0.001,5.)]
                # bounds = [(guess[0]*0.75, guess[0]*1.25), (guess[1]*0.75, guess[1]*1.25)]
                # bounds = [(guess[0]*0.9, guess[0]*1.1), (guess[1]*0.75, guess[1]*1.25)]
                # var_opt = ['coef', 'omega']
                # guess = [0.1, 0.22]
                # bounds = [(0.,5.5),(0.00001,0.2)]
                method = 'L-BFGS-B'
                # method = 'trust-exact'

                res = minimize(fun_opt,guess,args=(var_opt, dic, models, pol),bounds=bounds, method=method)

                fun_opt(res.x, var_opt, dic, models, pol)

                for j in range(len(res.x)):
                    aaa[j].append(res.x[j])

            field_data, theta_field, sm_field, height_field, lai_field, vwc_field, vv_field, vh_field, pol_field, relativeorbit, vwcpro_field = data_optimized_run(n, field_data, theta_field, sm_field, height_field, lai_field, vwc_field, pol)
            V1 = lai_field.values.flatten()
            # V1 = vwc_field.values.flatten()
            # V1 = lai_field.values.flatten() * vwcpro_field.values.flatten()/100
            # V1 = vwc_field.values.flatten() / height_field.values.flatten()
            # V1 = (vwcpro_field.values.flatten()/100)**3
            # V1 = (vwcpro_field.values.flatten()/100)**3 * height_field.values.flatten()
            V2 = V1 # initialize in surface model

        #-----------------------------------------------------------------

        for i in range(len(res.x)):
            exec('%s = %s' % (var_opt[i],aaa[i]))

        ke = coef * np.sqrt(lai_field.values.flatten())
        # ke = coef * np.sqrt(vwc_field.values.flatten())
        # ke = coef * np.sqrt(vwc_field.values.flatten()/height_field.values.flatten())
        # ke = coef * np.sqrt(lai_field.values.flatten()*vwcpro_field.values.flatten()/100)
        # ke = coef * np.sqrt(lai_field.values.flatten())*(vwcpro_field.values.flatten()/100)**4
        # ke = coef * (vwcpro_field.values.flatten()/100)**3
        # ke = coef * (vwcpro_field.values.flatten()/100)**3 * height_field.values.flatten()

        # print(str(coef)+models['surface'])
        # print(str(omega))
        # ke = smooth(ke, 11)

        soil = Soil(mv=sm_field.values.flatten(), C_hh=np.array(C_hh), C_vv=np.array(C_vv), D_hh=np.array(D_hh), D_vv=np.array(D_vv), C_hv=np.array(C_hv), D_hv=np.array(D_hv), s=s, clay=clay, sand=sand, f=freq, bulk=bulk, l=l)

        can = OneLayer(canopy=canopy, ke_h=ke, ke_v=ke, d=height_field.values.flatten(), ks_h = omega*ke, ks_v = omega*ke, V1=np.array(V1), V2=np.array(V2), A_hh=np.array(A_hh), B_hh=np.array(B_hh), A_vv=np.array(A_vv), B_vv=np.array(B_vv), A_hv=np.array(A_hv), B_hv=np.array(B_hv))

        S = model.RTModel(surface=soil, canopy=can, models=models, theta=theta_field.values.flatten(), freq=freq)
        S.sigma0()
#-----------------------------------------------------------------
        date = field_data.index

        colormap = plt.get_cmap(colormaps[j])
        colors = [colormap(jj) for jj in np.linspace(0.35, 1., 4)]

        # ax.plot(10*np.log10(pol_field), 'ks-', label='Sentinel-1 Pol: ' + pol, linewidth=3)
        # ax.plot(date, 10*np.log10(S.__dict__['s0g'][pol[::-1]]), color=colors[0], marker='s', linestyle='--', label=pol+' s0g')
        # ax.plot(date, 10*np.log10(S.__dict__['s0c'][pol[::-1]]), color=colors[1], marker='s', linestyle='--', label=pol+' s0c')
        # ax.plot(date, 10*np.log10(S.__dict__['s0cgt'][pol[::-1]]), 'ms-', label=pol+' s0cgt')
        # ax.plot(date, 10*np.log10(S.__dict__['s0gcg'][pol[::-1]]), 'ys-', label=pol+' s0gcg')

        mask = ~np.isnan(pol_field.values.flatten()) & ~np.isnan(S.__dict__['stot'][pol[::-1]])
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress((pol_field.values.flatten()[mask]), (S.__dict__['stot'][pol[::-1]][mask]))
        slope1, intercept1, r_value1, p_value1, std_err1 = scipy.stats.linregress(10*np.log10(pol_field.values.flatten())[mask], 10*np.log10(S.__dict__['stot'][pol[::-1]])[mask])
        rmse = rmse_prediction(10*np.log10(pol_field.values.flatten()), 10*np.log10(S.__dict__['stot'][pol[::-1]]))

        if k == 'Oh92':
            hm = 'Oh92'
            colors = 'b'
        elif k == 'Oh04':
            hm = 'Oh04'
            colors = 'r'
        elif k == 'Dubois95':
            hm='Dubois95'
            colors = 'y'
        elif k == 'WaterCloud':
            hm = 'WCM'
            colors = 'm'
        elif k == 'I2EM':
            hm = 'IEM'
            colors = 'g'

        if plot == 'all':
            if opt_mod == 'time invariant':
                if kk == 'turbid_isotropic':
                    ax.plot(date[3:-3], 10*np.log10(S.__dict__['stot'][pol[::-1]][3:-3]), color=colors, marker='s', linestyle='dashed', label = hm+ ' + ' +  'SSRT')
                else:
                    ax.plot(date[3:-3], 10*np.log10(S.__dict__['stot'][pol[::-1]][3:-3]), color=colors, marker='s', label = hm+ ' + ' +  'WCM')
            else:
                if kk == 'turbid_isotropic':
                    ax.plot(date, 10*np.log10(S.__dict__['stot'][pol[::-1]]), color=colors, marker='s', linestyle='dashed', label = hm+ ' + ' +  'SSRT')
                else:
                    ax.plot(date, 10*np.log10(S.__dict__['stot'][pol[::-1]]), color=colors, marker='s', label = hm+ ' + ' +  'WCM')

        if plot == 'single':
            if style == 'scatterplot':
                if pol == 'vv':
                    ax.set_xlim([-22.5,-7.5])
                elif pol == 'vh':
                    ax.set_xlim([-30,-15])

                if style_2 == 'scatterplot_single_ESU':
                    ax.plot(10*np.log10(pol_field.values.flatten()),10*np.log10(S.__dict__['stot'][pol[::-1]]), 'rs', label=field)

                    x = 10*np.log10(pol_field.values.flatten())
                    y = 10*np.log10(S.__dict__['stot'][pol[::-1]])

                    lower_position = np.nanargmin(x)
                    upper_position = np.nanargmax(x)

                    ax.plot(np.array((x[lower_position],x[upper_position])),np.array((y[lower_position],y[upper_position])), '--r')


                else:
                    aa = []
                    bb = []
                    # cc = []

                    # field_plot = ['508_high', '508_low', '508_med']
                    jj = 0
                    colors = ['ks', 'ys', 'ms', 'rs']

                    for field in field_plot:
                        df, df_agro, field_data, field_data_orbit, theta_field, sm_field, height_field, lai_field, vwc_field, pol_field, vv_field, vh_field, relativeorbit, vwcpro_field = read_data(path, file_name, extension, field, path_agro, file_name_agro, extension_agro, pol)
                        field_data, theta_field, sm_field, height_field, lai_field, vwc_field, vv_field, vh_field, pol_field, vwcpro_field = data_optimized_run(n, field_data, theta_field, sm_field, height_field, lai_field, vwc_field, pol)

                        soil = Soil(mv=sm_field.values.flatten(), C_hh=np.array(C_hh), C_vv=np.array(C_vv), D_hh=np.array(D_hh), D_vv=np.array(D_vv), C_hv=np.array(C_hv), D_hv=np.array(D_hv), s=s, clay=clay, sand=sand, f=freq, bulk=bulk, l=l)

                        can = OneLayer(canopy=canopy, ke_h=ke, ke_v=ke, d=height_field.values.flatten(), ks_h = omega*ke, ks_v = omega*ke, V1=np.array(V1), V2=np.array(V2), A_hh=np.array(A_hh), B_hh=np.array(B_hh), A_vv=np.array(A_vv), B_vv=np.array(B_vv), A_hv=np.array(A_hv), B_hv=np.array(B_hv))

                        S = model.RTModel(surface=soil, canopy=can, models=models, theta=theta_field.values.flatten(), freq=freq)
                        S.sigma0()

                        ax.plot(10*np.log10(pol_field.values.flatten()),10*np.log10(S.__dict__['stot'][pol[::-1]]), colors[jj], label=field)

                        slope, intercept, r_value, p_value, std_err = linregress(10*np.log10(pol_field.values.flatten())[~np.isnan(10*np.log10(S.__dict__['stot'][pol[::-1]]))], 10*np.log10(S.__dict__['stot'][pol[::-1]])[~np.isnan(10*np.log10(S.__dict__['stot'][pol[::-1]]))])
                        line = slope * 10*np.log10(S.__dict__['stot'][pol[::-1]]) + intercept

                        # ax.plot(10*np.log10(S.__dict__['stot'][pol[::-1]]), line)

                        lower_position = np.nanargmin(line)
                        upper_position = np.nanargmax(line)

                        ax.plot(np.array((10*np.log10(S.__dict__['stot'][pol[::-1]])[lower_position],10*np.log10(S.__dict__['stot'][pol[::-1]])[upper_position])),np.array((line[lower_position],line[upper_position])), '--'+colors[jj][0])

                        aa = np.append(aa, 10*np.log10(pol_field.values.flatten()))
                        bb = np.append(bb, 10*np.log10(S.__dict__['stot'][pol[::-1]]))
                        jj = jj+1
            elif style == 'single_esu':
                aa = []
                bb = []
                # cc = []

                # field_plot = ['508_high', '508_low', '508_med']
                jj = 0
                colors = ['orange', 'yellow', 'green', 'red']

                for field in field_plot:
                    df, df_agro, field_data, field_data_orbit, theta_field, sm_field, height_field, lai_field, vwc_field, pol_field, vv_field, vh_field, relativeorbit, vwcpro_field = read_data(path, file_name, extension, field, path_agro, file_name_agro, extension_agro, pol)
                    field_data, theta_field, sm_field, height_field, lai_field, vwc_field, vv_field, vh_field, pol_field,vwcpro_field = data_optimized_run(n, field_data, theta_field, sm_field, height_field, lai_field, vwc_field, pol)

                    soil = Soil(mv=sm_field.values.flatten(), C_hh=np.array(C_hh), C_vv=np.array(C_vv), D_hh=np.array(D_hh), D_vv=np.array(D_vv), C_hv=np.array(C_hv), D_hv=np.array(D_hv), s=s, clay=clay, sand=sand, f=freq, bulk=bulk, l=l)

                    can = OneLayer(canopy=canopy, ke_h=ke, ke_v=ke, d=height_field.values.flatten(), ks_h = omega*ke, ks_v = omega*ke, V1=np.array(V1), V2=np.array(V2), A_hh=np.array(A_hh), B_hh=np.array(B_hh), A_vv=np.array(A_vv), B_vv=np.array(B_vv), A_hv=np.array(A_hv), B_hv=np.array(B_hv))

                    S = model.RTModel(surface=soil, canopy=can, models=models, theta=theta_field.values.flatten(), freq=freq)
                    S.sigma0()

                    ax.plot(date, 10*np.log10(S.__dict__['stot'][pol[::-1]]), 'go-', color=colors[jj],  label=S.models['surface']+ ' + ' +  S.models['canopy'] + field+' Pol: ' + pol + '; RMSE???: ' + str(rmse)[0:4] + '; $R^2???$: ' + str(r_value)[0:4])
                    ax.plot(10*np.log10(pol_field), 'ks-',color=colors[jj], label='Sentinel-1 Pol: ' + pol, linewidth=3)
                    jj = jj+1











            else:
                ax.plot(date, 10*np.log10(S.__dict__['stot'][pol[::-1]]), color='orange', marker='s', label=S.models['surface']+ ' + ' +  S.models['canopy'] + ' Pol: ' + pol + '; RMSE: ' + str(rmse)[0:4] + '; $R^2$: ' + str(r_value)[0:4])
                ax.plot(date, 10*np.log10(S.__dict__['s0g'][pol[::-1]]), color='red', marker='s', label='Ground contribution')
                ax.plot(date, 10*np.log10(S.__dict__['s0c'][pol[::-1]]), color='green', marker='s', label='Canopy contribution')
                ax2 = ax.twinx()
                if canopy == 'turbid_isotropic':
                    ax2.plot(date,ke, label='ke')
                if canopy == 'water_cloud':
                    ax2.plot(date,np.array(A_vv)*100, label='A_vv')
                    ax2.plot(date,B_vv,  label='B_vv')
                ax2.set_ylim(0,0.8)


        j = j+1


if style == 'scatterplot':
    pass
else:
    if opt_mod == 'time invariant':
        ax.plot(10*np.log10(pol_field)[3:-3], 'ks-', label='Sentinel-1', linewidth=3)
    else:
        ax.plot(10*np.log10(pol_field), 'ks-', label='Sentinel-1', linewidth=3)
    plt.legend(prop={'size': 16})
    plt.grid(linestyle='dotted')
    # plt.title(field)

if plot == 'all':
    if opt_mod == 'time variant':
        opt_mod2 = 'time_variant'
    elif opt_mod == 'time invariant':
        opt_mod2 = 'time_invariant'

    plt.savefig(plot_output_path+pol+'_all_'+opt_mod2, bbox_inches = 'tight')

if plot == 'single':
    if style == 'scatterplot':
        plt.ylabel(surface + ' ' + canopy + ' [dB]')
        plt.xlabel('Sentinel-1 [dB]')
        plt.legend()
        x = np.linspace(np.min(10*np.log10(pol_field.values.flatten()))-2, np.max(10*np.log10(pol_field.values.flatten()))+2, 16)
        ax.plot(x,x)
        if style_2 == 'scatterplot_single_ESU':
            www = rmse_prediction(10*np.log10(pol_field).values.flatten(), 10*np.log10(S.__dict__['stot'][pol[::-1]]))
            plt.title(pol+' ' + field + ' ' + surface + ' ' + canopy + '$R^2$='+str(r_value)+' RMSE='+str(www))
            plt.savefig(plot_output_path+'scatterplot_fertig_single_'+field+'_'+pol+'_'+file_name+'_'+S.models['surface']+'_'+S.models['canopy'])
        else:
            www = rmse_prediction(aa, bb)
            # slope, intercept, r_value, p_value, std_err = linregress(aaa[~np.isnan(bbb)], bbb[~np.isnan(bbb)])
            plt.title(pol+' ' + field + ' ' + surface + ' ' + canopy + '$R^2$='+str(r_value)+' RMSE='+str(www))
            plt.savefig(plot_output_path+'scatterplot_fertig_'+field+'_'+pol+'_'+file_name+'_'+S.models['surface']+'_'+S.models['canopy'])
    else:
        plt.savefig(plot_output_path+pol+'50_single_'+opt_mod+'_'+S.models['surface']+'_'+S.models['canopy'])






