
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator
import matplotlib.dates as mdates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
# import matplotlib.ticker
import numpy as np
from sense.canopy import OneLayer
from sense.soil import Soil
from sense import model
import scipy.stats
from scipy.optimize import minimize
import pdb
import datetime
import seaborn as sns
import skill_metrics as sm

# Helper functions for statistical parameters
#--------------------------------------------
def rmse_prediction(predictions, targets):
    """ calculation of RMSE """
    return np.sqrt(np.nanmean((predictions - targets) ** 2))

def bias_prediction(predictions, targets):
    return np.nanmean(predictions - targets)

def ubrmse_prediction(rmse,bias):
    return np.sqrt(rmse ** 2 - bias ** 2)

def linregress(predictions, targets):
    """ Calculate a linear least-squares regression for two sets of measurements """
    ### get rid of nan data
    predictions2 = predictions[~np.isnan(predictions)]
    targets2 = targets[~np.isnan(predictions)]
    predictions3 = predictions2[~np.isnan(targets2)]
    targets3 = targets2[~np.isnan(targets2)]

    #linregress calculation
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(predictions3, targets3)
    return slope, intercept, r_value, p_value, std_err

def nan_values(predictions, targets):
    """ get rid of nan values"""
    predictions2 = predictions[~np.isnan(predictions)]
    targets2 = targets[~np.isnan(predictions)]
    predictions3 = predictions2[~np.isnan(targets2)]
    targets3 = targets2[~np.isnan(targets2)]
    return predictions3, targets3

def read_mni_data(path, file_name, extention, field, sep=','):
    """ read MNI campaign data """
    df = pd.io.parsers.read_csv(os.path.join(path, file_name + extension), header=[0, 1], sep=sep)
    df = df.set_index(pd.to_datetime(df[field]['date']))
    df = df.drop(df.filter(like='date'), axis=1)
    return df

def read_agrometeo(path, file_name, extentio, sep=';', decimal=','):
    """ read agro-meteorological station (hourly data) """
    df = pd.read_csv(os.path.join(path, file_name + extension), sep=sep, decimal=decimal)
    df['SUM_NN050'] = df['SUM_NN050'].str.replace(',','.')
    df['SUM_NN050'] = df['SUM_NN050'].str.replace('-','0').astype(float)

    df['date'] = df['Tag'] + ' ' + df['Stunde']

    df = df.set_index(pd.to_datetime(df['date'], format='%d.%m.%Y %H:%S'))
    return df

def filter_relativorbit(data, field, orbit1, orbit2=None, orbit3=None, orbit4=None):
    """ data filter for relativ orbits """
    output = data[[(check == orbit1 or check == orbit2 or check == orbit3 or check == orbit4) for check in data[(field,'relativeorbit')]]]
    return output

def smooth(x,window_len=11,window='hanning'):
        if x.ndim != 1:
                raise ValueError #, "smooth only accepts 1 dimension arrays."
        if x.size < window_len:
                raise ValueError #, "Input vector needs to be bigger than window size."
        if window_len<3:
                return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError #, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
                w=np.ones(window_len,'d')
        else:
                w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]

def read_data(path, file_name, extension, field, path_agro, file_name_agro, extension_agro):
    # Read MNI data
    df = read_mni_data(path, file_name, extension, field)

    # Read agro-meteorological station
    df_agro = read_agrometeo(path_agro, file_name_agro, extension_agro)

    # filter for field
    field_data = df.filter(like=field)

    # filter for relativorbit
    field_data_orbit = filter_relativorbit(field_data, field, 95)
    # field_data = field_data_orbit

    # get rid of NaN values
    parameter_nan = 'LAI'
    field_data = field_data[~np.isnan(field_data.filter(like=parameter_nan).values)]

    # available auxiliary data
    theta_field = np.deg2rad(field_data.filter(like='theta'))
    # theta_field[:] = 45
    sm_field = field_data.filter(like='SM')
    height_field = field_data.filter(like='Height')/100
    lai_field = field_data.filter(like='LAI')
    vwc_field = field_data.filter(like='VWC')
    pol_field = field_data.filter(like='sigma_sentinel_'+pol)
    vv_field = field_data.filter(like='sigma_sentinel_vv')
    vh_field = field_data.filter(like='sigma_sentinel_vh')
    relativeorbit = field_data.filter(like='relativeorbit')
    vwcpro_field = field_data.filter(like='watercontentpro')
    return df, df_agro, field_data, field_data_orbit, theta_field, sm_field, height_field, lai_field, vwc_field, pol_field, vv_field, vh_field, relativeorbit, vwcpro_field

### Optimization ###
#-----------------------------------------------------------------
def solve_fun(VALS):

    for i in range(len(var_opt)):
        dic[var_opt[i]] = VALS[i]

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

def run(VALS):

    for i in range(len(var_opt)):
        dic[var_opt[i]] = VALS[i]

    ke = dic['coef'] * np.sqrt(dic['lai'])
    dic['ke'] = ke

    # surface
    soil = Soil(mv=dic['mv'], C_hh=dic['C_hh'], C_vv=dic['C_vv'], D_hh=dic['D_hh'], D_vv=dic['D_vv'], C_hv=dic['C_hv'], D_hv=dic['D_hv'], V2=dic['V2'], s=dic['s'], clay=dic['clay'], sand=dic['sand'], f=dic['f'], bulk=dic['bulk'], l=dic['l'])

    # canopy
    can = OneLayer(canopy=dic['canopy'], ke_h=dic['ke'], ke_v=dic['ke'], d=dic['d'], ks_h = dic['omega']*dic['ke'], ks_v = dic['omega']*dic['ke'], V1=dic['V1'], V2=dic['V2'], A_hh=dic['A_hh'], B_hh=dic['B_hh'], A_vv=dic['A_vv'], B_vv=dic['B_vv'], A_hv=dic['A_hv'], B_hv=dic['B_hv'])

    S = model.RTModel(surface=soil, canopy=can, models=models, theta=dic['theta'], freq=dic['f'])
    S.sigma0()

    return S.__dict__['stot']['vv'[::-1]], S.__dict__['stot']['vh'[::-1]]

def fun_opt(VALS):


    # return(10.*np.log10(np.nansum(np.square(solve_fun(VALS)-dic['pol_value']))))
    return(np.nansum(np.square(solve_fun(VALS)[0]-dic['vv'])))
    # return(np.nansum(np.square((solve_fun(VALS)[0]-dic['vv'])/2+(solve_fun(VALS)[0]-dic['vh'])/2)))

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

### run model
#-----------------------------------------------------------------
def run_model():

    ke = dic['coef'] * np.sqrt(dic['lai'])
    dic['ke'] = ke

    # surface
    soil = Soil(mv=dic['mv'], C_hh=dic['C_hh'], C_vv=dic['C_vv'], D_hh=dic['D_hh'], D_vv=dic['D_vv'], C_hv=dic['C_hv'], D_hv=dic['D_hv'], V2=dic['V2'], s=dic['s'], clay=dic['clay'], sand=dic['sand'], f=dic['f'], bulk=dic['bulk'], l=dic['l'])

    # canopy
    can = OneLayer(canopy=dic['canopy'], ke_h=dic['ke'], ke_v=dic['ke'], d=dic['d'], ks_h = dic['omega']*dic['ke'], ks_v = dic['omega']*dic['ke'], V1=dic['V1'], V2=dic['V2'], A_hh=dic['A_hh'], B_hh=dic['B_hh'], A_vv=dic['A_vv'], B_vv=dic['B_vv'], A_hv=dic['A_hv'], B_hv=dic['B_hv'])

    S = model.RTModel(surface=soil, canopy=can, models=models, theta=dic['theta'], freq=dic['f'])
    S.sigma0()
    return S.__dict__['stot']['vv'[::-1]], S.__dict__['stot']['vh'[::-1]]
#-----------------------------------------------------------------

### Data preparation ###
#-----------------------------------------------------------------
# storage information
path = '/media/tweiss/Work/z_final_mni_data_2017'
file_name = 'in_situ_s1_buffer_50' # theta needs to be changed to for norm multi
extension = '.csv'

path_agro = '/media/nas_data/2017_MNI_campaign/field_data/meteodata/agrarmeteorological_station'
file_name_agro = 'Eichenried_01012017_31122017_hourly'
extension_agro = '.csv'

# field = '508_high'
field_list = ['508_high','508_low','508_med','301_high','301_low','301_med','542_high','542_low','542_med']

pol = 'vv'
# pol = 'vh'

# output path
plot_output_path = '/media/tweiss/Daten/plots/paper/new/'
plot_output_path = '/media/tweiss/Work/z_check_data/'

csv_output_path = plot_output_path+'csv/'

# df, df_agro, field_data, field_data_orbit, theta_field, sm_field, height_field, lai_field, vwc_field, pol_field, vv_field, vh_field = read_data(path, file_name, extension, field, path_agro, file_name_agro, extension_agro)

#-----------------------------------------------------------------
### Run SenSe module
#-----------------------------------------------------------------
#### Choose models
#-----------------
surface_list = ['Oh92', 'Oh04', 'Dubois95', 'WaterCloud', 'I2EM']
canopy_list = ['turbid_isotropic', 'water_cloud']

### option for time_invariant or variant calibration of parameter
#-------------------------------
opt_mod = ['time_invariant','time_variant']
# opt_mod = ['time_invariant']
#---------------------------

### plot option: "single" or "all" modelcombination
#------------------------------
plot = 'single'
# plot = 'all'
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
# if style == 'scatterplot':
#     fig, ax = plt.subplots(figsize=(10, 10))
# else:
#     fig, ax = plt.subplots(figsize=(17, 10))
# # plt.title('Winter Wheat')
# plt.ylabel('Backscatter [dB]', fontsize=15)
# plt.xlabel('Date', fontsize=15)
# plt.tick_params(labelsize=12)


# if pol == 'vv':
#     ax.set_ylim([-25,-7.5])
# elif pol == 'vh':
#     ax.set_ylim([-30,-15])

colormaps = ['Greens', 'Purples', 'Blues', 'Oranges', 'Reds', 'Greys', 'pink', 'bone', 'Blues', 'Blues', 'Blues']
j = 0

colormap = plt.get_cmap(colormaps[j])
colors = [colormap(jj) for jj in np.linspace(0.35, 1., 3)]

df_statistics = pd.DataFrame(columns=pd.MultiIndex(levels=[[],[],[],[]], codes=[[],[],[],[]]))

for field in field_list:
    break
    for k in surface_list:
        for kk in canopy_list:

            df_output = pd.DataFrame(columns=pd.MultiIndex(levels=[[],[],[],[],[]], codes=[[],[],[],[],[]]))

            df, df_agro, field_data, field_data_orbit, theta_field, sm_field, height_field, lai_field, vwc_field, pol_field, vv_field, vh_field, relativeorbit, vwcpro_field = read_data(path, file_name, extension, field, path_agro, file_name_agro, extension_agro)

            #### Optimization
            #-----------------

            for kkk in opt_mod:

                freq = 5.405
                clay = 0.0738
                sand = 0.2408
                bulk = 1.45
                s = 0.0105 # vv
                s = 0.0115
                # s = 0.009 # vh ?????

                C_hh = 0
                D_hh = 0
                C_hv = -22.5
                D_hv = 3.2
                C_vv = -14.609339
                C_vv = -11
                D_vv = 12.884086
                # D_vv = 0.1

                ### Canopy
                # Water Cloud (A, B, V1, V2, theta)
                # SSRT (coef, omega, theta)
                #-----------------------------------
                A_hh = 0
                B_hh = 0
                A_hv = 0.029
                B_hv = 0.0013
                A_vv = 0.0029
                # A_vv = 0.0019
                B_vv = 0.23
                V1 = lai_field.values.flatten()
                V2 = V1 # initialize in surface model
                coef = 1.
                omega = 0.027
                # omega = 0.017 # vv
                # omega = 0.015 # vh
                # IEM
                # l = (1.281 + 0.134*np.sin(theta_field.values.flatten()*0.19) ** (-1.59) * s) /100
                l = 0.01

                df_output = pd.DataFrame(columns=pd.MultiIndex(levels=[[],[],[],[],[]], codes=[[],[],[],[],[]]))

                surface = k
                canopy = kk
                models = {'surface': surface, 'canopy': canopy}

                if kkk == 'time_invariant':
                    n=0

                    dic = {"mv":sm_field.values.flatten(), "C_hh":C_hh, "C_vv":C_vv, "D_hh":D_hh, "D_vv":D_vv, "C_hv":C_hv, "D_hv":D_hv, "s":s, "clay":clay, "sand":sand, "f":freq, "bulk":bulk, "l":l, "canopy":canopy, "d":height_field.values.flatten(), "V1":V1, "V2":V2, "A_hh":A_hh, "B_hh":B_hh, "A_vv":A_vv, "B_vv":B_vv, "A_hv":A_hv, "B_hv":B_hv, "lai":lai_field.values.flatten(), "vwc":vwc_field.values.flatten(), "pol_value":pol_field.values.flatten(), "vv":vv_field.values.flatten(), "vh":vh_field.values.flatten(), "theta":theta_field.values.flatten(), "omega": omega, "coef": coef, "vwcpro": vwcpro_field.values.flatten()}

                    if canopy == 'turbid_isotropic':
                        var_opt = ['coef']
                        guess = [2.]
                        bounds = [(0.001,5.5)]
                        # guess = [0.5]
                        # bounds = [(0.00005,10)]
                    elif canopy == 'water_cloud':
                        var_opt = ['B_vv', 'B_hv']
                        guess = [B_vv, B_hv]
                        bounds = [ (0.00000001,1),  (0.00000001,1)]

                    method = 'L-BFGS-B'

                    res = minimize(fun_opt,guess,bounds=bounds, method=method)

                    fun_opt(res.x)
                    aaa = res.x

                if kkk == 'time_variant':
                    aaa = [[],[],[],[],[],[],[],[],[],[],[],[]]
                    n=7

                    for i in range(len(pol_field.values.flatten())-n+1):

                        if type(coef) == float:
                            dic = {"mv":sm_field.values.flatten()[i:i+n], "C_hh":C_hh, "C_vv":C_vv, "D_hh":D_hh, "D_vv":D_vv, "C_hv":C_hv, "D_hv":D_hv, "V2":V2[i:i+n], "s":s, "clay":clay, "sand":sand, "f":freq, "bulk":bulk, "l":l, "canopy":canopy, "d":height_field.values.flatten()[i:i+n], "V1":V1[i:i+n], "A_hh":A_hh, "B_hh":B_hh, "A_vv":A_vv, "B_vv":B_vv, "A_hv":A_hv, "B_hv":B_hv, "lai":lai_field.values.flatten()[i:i+n], "vwc":vwc_field.values.flatten()[i:i+n], "pol_value":pol_field.values.flatten()[i:i+n], "vv":vv_field.values.flatten()[i:i+n], "vh":vh_field.values.flatten()[i:i+n], "theta":theta_field.values.flatten()[i:i+n], "omega": omega, "coef": coef, "vwcpro": vwcpro_field.values.flatten()[i:i+n]}
                        else:
                            dic = {"mv":sm_field.values.flatten()[i:i+n], "C_hh":C_hh, "C_vv":C_vv, "D_hh":D_hh, "D_vv":D_vv, "C_hv":C_hv, "D_hv":D_hv, "V2":V2[i:i+n], "s":s, "clay":clay, "sand":sand, "f":freq, "bulk":bulk, "l":l, "canopy":canopy, "d":height_field.values.flatten()[i:i+n], "V1":V1[i:i+n], "A_hh":A_hh, "B_hh":B_hh, "A_vv":A_vv, "B_vv":B_vv, "A_hv":A_hv, "B_hv":B_hv, "lai":lai_field.values.flatten()[i:i+n], "vwc":vwc_field.values.flatten()[i:i+n], "pol_value":pol_field.values.flatten()[i:i+n], "vv":vv_field.values.flatten()[i:i+n], "vh":vh_field.values.flatten()[i:i+n], "theta":theta_field.values.flatten()[i:i+n], "omega": omega, "vwcpro": vwcpro_field.values.flatten()[i:i+n]}

                        if canopy == 'turbid_isotropic':
                            var_opt = ['coef']
                            guess = [0.5]
                            bounds = [(0.00005,10)]
                        elif canopy == 'water_cloud':
                            if i == 0:
                                var_opt = ['B_vv','B_hv']
                                guess = [B_vv, B_hv]
                                bounds = [ (0.001,1),  (0.001,1)]
                            else:
                                var_opt = ['B_vv', 'B_hv']
                                guess = [B_vv, B_hv]
                                bounds = [(res.x[0]*0.5,res.x[0]*2.2), (res.x[1]*0.9,res.x[1]*1.1)]

                        method = 'L-BFGS-B'

                        res = minimize(fun_opt,guess,bounds=bounds, method=method)

                        fun_opt(res.x)

                        for j in range(len(res.x)):
                            aaa[j].append(res.x[j])

                    field_data, theta_field, sm_field, height_field, lai_field, vwc_field, vv_field, vh_field, pol_field, relativeorbit, vwcpro_field = data_optimized_run(n, field_data, theta_field, sm_field, height_field, lai_field, vwc_field, pol)
                    V1 = lai_field.values.flatten()
                    V2 = V1 # initialize in surface model

                df_output[k,kk,kkk,field,'SM_insitu'] = sm_field.values.flatten()
                df_output = df_output.set_index(sm_field.index)

                df_output[k,kk,kkk,field,'LAI_insitu'] = lai_field.values.flatten()
                df_output[k,kk,kkk,field,'S1_vv'] = vv_field.values.flatten()
                df_output[k,kk,kkk,field,'S1_vh'] = vh_field.values.flatten()
                df_output[k,kk,kkk,field,'theta'] = theta_field.values.flatten()
                df_output[k,kk,kkk,field,'height'] = height_field.values.flatten()
                df_output[k,kk,kkk,field,'relativeorbit'] = relativeorbit.values.flatten()
                df_output[k,kk,kkk,field,'VWC'] = vwc_field.values.flatten()
                df_output[k,kk,kkk,field,'watercontentpro'] = vwcpro_field.values.flatten()

                if canopy == 'turbid_isotropic':
                    if kkk == 'time_variant':
                        mask = np.squeeze(np.isnan(sm_field.values))
                        bbb = np.array(aaa[0])
                        bbb[mask] = np.nan
                        df_output[k,kk,kkk,field,'coef'] = bbb
                        coef = bbb
                    else:
                        df_output[k,kk,kkk,field,'coef'] = aaa[0]
                        coef = aaa[0]
                elif canopy == 'water_cloud':
                    if kkk == 'time_variant':
                        mask = np.squeeze(np.isnan(sm_field.values))
                        bbb = np.array(aaa[0])
                        bbb[mask] = np.nan
                        ccc = np.array(aaa[1])
                        ccc[mask] = np.nan
                        df_output[k,kk,kkk,field,'B_vv'] = bbb
                        B_vv = bbb
                        df_output[k,kk,kkk,field,'B_vh'] = ccc
                        B_hv = ccc
                    else:
                        df_output[k,kk,kkk,field,'B_vv'] = aaa[0]
                        B_vv = aaa[0]
                        df_output[k,kk,kkk,field,'B_vh'] = aaa[1]
                        B_hv = aaa[1]

                dic = {"mv":sm_field.values.flatten(), "C_hh":C_hh, "C_vv":C_vv, "D_hh":D_hh, "D_vv":D_vv, "C_hv":C_hv, "D_hv":D_hv, "s":s, "clay":clay, "sand":sand, "f":freq, "bulk":bulk, "l":l, "canopy":canopy, "d":height_field.values.flatten(), "V1":lai_field.values.flatten(), "V2":lai_field.values.flatten(), "A_hh":A_hh, "B_hh":B_hh, "A_vv":A_vv, "B_vv":B_vv, "A_hv":A_hv, "B_hv":B_hv, "lai":lai_field.values.flatten(), "vwc":vwc_field.values.flatten(), "pol_value":pol_field.values.flatten(), "vv":vv_field.values.flatten(), "vh":vh_field.values.flatten(), "theta":theta_field.values.flatten(), "omega": omega, "coef": coef, "vwcpro": vwcpro_field.values.flatten()}

                print(k+'_'+kk+'_'+kkk+'_'+field)

                if canopy == 'turbid_isotropic':
                    vv_model, vh_model = run([np.array(df_output[k, kk, kkk, field, 'coef'])])

                elif canopy == 'water_cloud':
                    vv_model, vh_model = run([np.array(df_output[k, kk, kkk, field, 'B_vv']), np.array(df_output[k, kk, kkk, field, 'B_vh'])])

                df_output[k,kk,kkk,field,'vv_model'] = vv_model
                df_output[k,kk,kkk,field,'vh_model'] = vh_model

                df_output[k,kk,kkk,field,'diff'] = 10*np.log10(df_output[k,kk,kkk,field,'S1_vv']) - 10*np.log10(df_output[k,kk,kkk,field,'vv_model'])

                rmse_vv = rmse_prediction(10*np.log10(df_output[k,kk,kkk,field,'S1_vv']),10*np.log10(df_output[k,kk,kkk,field,'vv_model']))
                rmse_vv = rmse_prediction(10*np.log10(df_output[k,kk,kkk,field,'S1_vv'][0:38]),10*np.log10(df_output[k,kk,kkk,field,'vv_model'][0:38]))

                bias_vv = bias_prediction(10*np.log10(df_output[k,kk,kkk,field,'S1_vv']),10*np.log10(df_output[k,kk,kkk,field,'vv_model']))
                ubrmse_vv = ubrmse_prediction(rmse_vv,bias_vv)
                slope, intercept, r_value, p_value, std_err = linregress(10*np.log10(df_output[k,kk,kkk,field,'S1_vv']),10*np.log10(df_output[k,kk,kkk,field,'vv_model']))

                df_statistics[k,kk,kkk,field] = np.array([rmse_vv, ubrmse_vv, bias_vv, r_value])
                df_statistics.index = ['rmse_vv', 'ubrmse_vv', 'bias_vv', 'r_value_vv']

                fig, ax = plt.subplots(figsize=(17, 10))
                # plt.title('Winter Wheat')
                plt.ylabel('Backscatter [dB]', fontsize=15)
                plt.xlabel('Date', fontsize=15)
                plt.tick_params(labelsize=12)

                ax.set_ylim([-22,-5])

                # grid
                ax.grid(linestyle='dotted')

                # weekly grid for x-axis
                weeks = mdates.WeekdayLocator(byweekday=MO)
                daysFmt = mdates.DateFormatter('%d. %b')
                ax.xaxis.set_major_locator(weeks)
                ax.xaxis.set_major_formatter(daysFmt)

                # only ticks bottom and left
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()

                # tick size
                plt.tick_params(labelsize=12)

                ax.plot(10*np.log10(df_output[k,kk,kkk,field,'S1_vv']),label='S1_vv_'+str(rmse_vv))
                ax.plot(10*np.log10(df_output[k,kk,kkk,field,'vv_model']),label=k+' '+kk+' '+kkk+' '+field+' vv')
                ax.legend()
                plt.savefig(plot_output_path+kkk+'/'+kk+'/'+k+'_'+kk+'_'+kkk+'_'+field+'_50',dpi=300)
                plt.close()

                df_output.to_csv(csv_output_path+k+'_'+kk+'_'+kkk+'_'+field+'_50.csv')
                df_statistics.to_csv(csv_output_path+'all_statistics_50.csv')

### put all data into one csv
df_auswertung = pd.DataFrame()
for field in field_list:
    for k in surface_list:
        for kk in canopy_list:
            for kkk in opt_mod:
                df_daten = pd.read_csv(csv_output_path+k+'_'+kk+'_'+kkk+'_'+field+'_50.csv',header=[0,1,2,3,4,5])
                df_daten = df_daten.set_index(df_daten[df_daten.columns[0]])
                df_auswertung = df_auswertung.append(df_daten)
                df_insitu = df_auswertung.groupby(df_auswertung.index).mean()
                df_insitu.to_csv(csv_output_path+'all_50.csv')

### statistic one model combination
df_statistics = pd.DataFrame(columns=pd.MultiIndex(levels=[[],[],[]], codes=[[],[],[]]))
for k in surface_list:
    for kk in canopy_list:
        for kkk in opt_mod:
            s1_vv = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='S1_vv').values.flatten()
            model_vv = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='vv_model').values.flatten()

            rmse_vv = rmse_prediction(10*np.log10(s1_vv),10*np.log10(model_vv))
            bias_vv = bias_prediction(10*np.log10(s1_vv),10*np.log10(model_vv))
            ubrmse_vv = ubrmse_prediction(rmse_vv,bias_vv)
            slope, intercept, r_value, p_value, std_err = linregress(10*np.log10(s1_vv),10*np.log10(model_vv))

            df_statistics[k,kk,kkk] = np.array([rmse_vv, ubrmse_vv, bias_vv, r_value])
            df_statistics.index = ['rmse_vv', 'ubrmse_vv', 'bias_vv', 'r_value_vv']
            df_statistics.to_csv(csv_output_path+'all_combi_statistics_50.csv')

# ### statistic one model combination first vegetation half
# df_statistics = pd.DataFrame(columns=pd.MultiIndex(levels=[[],[],[]], codes=[[],[],[]]))
# for k in surface_list:
#     for kk in canopy_list:
#         for kkk in opt_mod:
#             s1_vv = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='S1_vv')
#             pdb.set_trace()
#             s1_vv = s1_vv[0:np.int(len(s1_vv)/2)].values.flatten()
#             model_vv = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='vv_model')
#             model_vv = model_vv[0:np.int(len(model_vv)/2)].values.flatten()

#             rmse_vv = rmse_prediction(10*np.log10(s1_vv),10*np.log10(model_vv))
#             bias_vv = bias_prediction(10*np.log10(s1_vv),10*np.log10(model_vv))
#             ubrmse_vv = ubrmse_prediction(rmse_vv,bias_vv)
#             slope, intercept, r_value, p_value, std_err = linregress(10*np.log10(s1_vv),10*np.log10(model_vv))

#             df_statistics[k,kk,kkk] = np.array([rmse_vv, ubrmse_vv, bias_vv, r_value])
#             df_statistics.index = ['rmse_vv', 'ubrmse_vv', 'bias_vv', 'r_value_vv']
#             df_statistics.to_csv(csv_output_path+'all_combi_statistics_50_first_veg_half.csv')

### Validation
validation_list = ['508_high','508_low','508_med','301_high','301_low','301_med','542_high','542_low','542_med']

sm_validation = pd.DataFrame()
height_validation = pd.DataFrame()
lai_validation = pd.DataFrame()
vv_validation = pd.DataFrame()
relativeorbit_validation = pd.DataFrame()
theta_validation = pd.DataFrame()
for field in validation_list:
    df, df_agro, field_data, field_data_orbit, theta_field, sm_field, height_field, lai_field, vwc_field, pol_field, vv_field, vh_field, relativeorbit, vwcpro_field = read_data(path, file_name, extension, field, path_agro, file_name_agro, extension_agro)
    sm_validation[field] = sm_field.values.flatten()
    height_validation[field] = height_field.values.flatten()
    lai_validation[field] = lai_field.values.flatten()
    vv_validation[field] = vv_field.values.flatten()
    relativeorbit_validation[field] = relativeorbit.values.flatten()
    theta_validation[field] = theta_field.values.flatten()

    sm_validation = sm_validation.set_index(sm_field.index)
    height_validation = height_validation.set_index(height_field.index)
    lai_validation = lai_validation.set_index(lai_field.index)
    vv_validation = vv_validation.set_index(vv_field.index)
    relativeorbit_validation = relativeorbit_validation.set_index(relativeorbit.index)
    theta_validation = theta_validation.set_index(theta_field.index)

### Model parameter
freq = 5.405
clay = 0.0738
sand = 0.2408
bulk = 1.45
s = 0.0105 # vv
s = 0.0115
# s = 0.009 # vh ?????

C_hh = 0
D_hh = 0
C_hv = -22.5
D_hv = 3.2
C_vv = -14.609339
C_vv = -11
D_vv = 12.884086
# D_vv = 0.1

### Canopy
# Water Cloud (A, B, V1, V2, theta)
# SSRT (coef, omega, theta)
#-----------------------------------
A_hh = 0
B_hh = 0
A_hv = 0.029
B_hv = 0.0013
A_vv = 0.0029
# A_vv = 0.0019
B_vv = 0.23
coef = 1.
omega = 0.027
# omega = 0.017 # vv
# omega = 0.015 # vh
# IEM
l = 0.01
df_output = pd.DataFrame(columns=pd.MultiIndex(levels=[[],[],[],[]], codes=[[],[],[],[]]))

df_statistics = pd.DataFrame(columns=pd.MultiIndex(levels=[[],[],[],[]], codes=[[],[],[],[]]))

vv_modeled = pd.DataFrame()
T_used = pd.DataFrame()

for field in validation_list:
    for k in surface_list:
        for kk in canopy_list:
            for kkk in opt_mod:
                if kk == 'turbid_isotropic':
                    data = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='coef')

                if kk == 'water_cloud':
                    data = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='B_vv')

                # data = data[data.columns.drop(list(data.filter(like='542')))]
                # data = data[data.columns.drop(list(data.filter(like='508')))]
                # data = data[data.columns.drop(list(data.filter(like='301_med')))]

                data = data[data.columns.drop(list(data.filter(like=field)))].mean(axis=1)

                if kk == 'turbid_isotropic':
                    T_used[field,k,kk,kkk,'coef'] = data

                if kk == 'water_cloud':
                    T_used[field,k,kk,kkk,'B_vv'] = data
                T_used.to_csv(csv_output_path+'all_vali_coef_B_vv_mean_50.csv')

                surface = k
                canopy = kk
                models = {'surface': surface, 'canopy': canopy}

                dic = {"mv":sm_validation[field].values, "C_hh":C_hh, "C_vv":C_vv, "D_hh":D_hh, "D_vv":D_vv, "C_hv":C_hv, "D_hv":D_hv, "V2":lai_validation[field].values, "s":s, "clay":clay, "sand":sand, "f":freq, "bulk":bulk, "l":l, "canopy":canopy, "d":height_validation[field].values, "V1":lai_validation[field].values, "A_hh":A_hh, "B_hh":B_hh, "A_vv":A_vv, "B_vv":data, "A_hv":A_hv, "B_hv":B_hv, "lai":lai_validation[field].values, "vwc":1, "pol_value":1, "vv":vv_validation[field].values, "vh":1, "theta":theta_validation[field].values, "omega": omega, "coef": data}

                vv, vh = run_model()

                rmse_vv = rmse_prediction(10*np.log10(vv_validation[field].values),10*np.log10(vv))
                bias_vv = bias_prediction(10*np.log10(vv_validation[field].values),10*np.log10(vv))
                ubrmse_vv = ubrmse_prediction(rmse_vv,bias_vv)
                slope, intercept, r_value, p_value, std_err = linregress(10*np.log10(vv_validation[field].values),10*np.log10(vv))


                vv_model_used, vv_s1_used = nan_values(10*np.log10(vv), 10*np.log10(vv_validation[field].values))

                crmsd_vv = sm.centered_rms_dev(vv_model_used,vv_s1_used)

                sdev_vv = np.std(vv_model_used)
                nse_vv = sm.nse(vv_model_used,vv_s1_used)

                df_statistics[k,kk,kkk,field] = np.array([rmse_vv, ubrmse_vv, bias_vv, r_value, sdev_vv,nse_vv, crmsd_vv])
                df_statistics.index = ['rmse_vv', 'ubrmse_vv', 'bias_vv', 'r_value_vv', 'sdev_vv', 'nse_vv','crmsd_vv']
                df_statistics.to_csv(csv_output_path+'all_vali_statistics_50.csv')

                vv_modeled[field,k,kk,kkk,'biasedmodel_'] = vv
                vv_modeled[field,k,kk,kkk,'S1_vv'] = vv_validation[field].values
                vv_modeled[field,k,kk,kkk,'unbiasedmodeldb'] = 10*np.log10(vv)+bias_vv
                vv_modeled.to_csv(csv_output_path+'all_vali_vv_50.csv')

                plt.plot(10*np.log10(vv),color='blue', label='RMSE:'+str(rmse_vv)[0:4]+' ubrmse:'+str(ubrmse_vv)[0:4]+' bias:'+str(bias_vv)[0:4])
                plt.plot(10*np.log10(vv_validation[field].values), color='black')
                plt.plot(10*np.log10(vv)+bias_vv,color='green')
                plt.legend()
                plt.savefig('/media/tweiss/Work/z_check_data/new/'+field+k+kk+'.png')
                plt.close()


# hm = df_statistics.filter(like='time_variant').filter(like=kk)

# sdev = hm.loc['sdev_vv'].values
# rmsd = hm.loc['rmse_vv'].values
# crmsd = hm.loc['crmsd_vv'].values
# ubrmsd = hm.loc['ubrmse_vv'].values
# coef = hm.loc['r_value_vv'].values

# label = {'ubrmsd': 'r', 'rmsd': 'b'}
# sm.taylor_diagram(sdev[0:2],ubrmsd[0:2],coef[0:2], markercolor = 'r')
# sm.taylor_diagram(sdev[0:2],rmsd[0:2],coef[0:2], markercolor = 'b', overlay = 'on', markerLabel= label)

# sm.taylor_diagram(sdev,ubrmsd,coef, markerLabel = hm.columns.values.tolist(), markerLabelColor = 'r', markerLegend = 'on', markerColor = 'r',styleOBS = '-', colOBS = 'r', markerobs = 'o',markerSize = 6, tickRMS = [0.0, 1.0, 2.0, 3.0],tickRMSangle = 115, showlabelsRMS = 'on',titleRMS = 'on', titleOBS = 'Ref', checkstats = 'on')

# sm.taylor_diagram(sdev,ubrmsd,coef, markerLabel = xxx.tolist(), locationColorBar = 'EastOutside', markerDisplayed = 'colorBar', titleColorBar = 'Bias', markerLabelColor='black', markerSize=10, markerLegend='off',  colRMS='g', styleRMS=':', widthRMS=2.0, titleRMS='on', colSTD='b', styleSTD='-.', widthSTD=1.0, titleSTD ='on', colCOR='k', styleCOR='--', widthCOR=1.0, titleCOR='on')




# label = []
# stats = {}
# stats['bias'] = []
# stats['rmsd'] = []
# stats['crmsd'] = []
# stats['sdev'] = []
# stats['ccoef'] = []
# stats['ss'] = []
# stats['nse'] = []


df_vali = pd.read_csv(csv_output_path+'all_vali_vv_50.csv',header=[0])

### statistic one model combination
df_statistics = pd.DataFrame(columns=pd.MultiIndex(levels=[[],[],[]], codes=[[],[],[]]))
for k in surface_list:
    for kk in canopy_list:
        for kkk in opt_mod:
            s1_vv = df_vali.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='S1_vv').values.flatten()
            model_vv = df_vali.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='biasedmodel_').values.flatten()
            model_vv_ub = df_vali.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='unbiasedmodeldb').values.flatten()
            rmse_vv = rmse_prediction(10*np.log10(s1_vv),10*np.log10(model_vv))
            ubrmse_vv = rmse_prediction(10*np.log10(s1_vv),model_vv_ub)
            slope, intercept, r_value, p_value, std_err = linregress(10*np.log10(s1_vv),model_vv_ub)

            sm.taylor_statistics(s1_vv,model_vv,'data')


            df_statistics[k,kk,kkk] = np.array([rmse_vv, ubrmse_vv, r_value])
            df_statistics.index = ['rmse_vv', 'ubrmse_vv', 'r_value_vv']
            df_statistics.to_csv(csv_output_path+'all_vali_combi_statistics_50.csv')




pdb.set_trace()

### get results
# calibration

df_cal = pd.read_csv(csv_output_path+'all_combi_statistics_50.csv',header=[0,1,2,3])
df_cal = df_cal.set_index(df_cal[df_cal.columns[0]])

# for k in surface_list:
#     for kk in canopy_list:
#         for kkk in opt_mod:
#             df_cal.filter(like=k).filter(like=kk).filter(like=kkk)








xxx = '95'
xxx = ''
df_insitu = pd.read_csv(csv_output_path+'all'+xxx+'.csv',header=[0,1,2,3,4,5],index_col=0)

field_short = ['508','301','542']
output_path_boxplot = '/media/tweiss/Work/z_check_data/boxplot/'

opt_mod = ['time_variant']

for k in surface_list:
    for kk in canopy_list:
        for kkk in opt_mod:
            if kk == 'turbid_isotropic':

                fig, ax = plt.subplots(figsize = (12,6))
                # coef = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='coef')
                # fig = sns.boxplot(data=coef.T,color='Black')
                coef = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='coef').filter(like='301')
                plt.fill_between(coef.index.values, coef.min(axis=1).values, coef.max(axis=1).values, facecolor='blue', alpha=0.5)
                plt.plot(coef,color='blue')
                coef = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='coef').filter(like='508')
                plt.fill_between(coef.index.values, coef.min(axis=1).values, coef.max(axis=1).values, facecolor='green', alpha=0.5)
                plt.plot(coef,color='green')
                coef = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='coef').filter(like='542')
                plt.fill_between(coef.index.values, coef.min(axis=1).values, coef.max(axis=1).values, facecolor='red', alpha=0.5)
                plt.plot(coef,color='red')

                # for patch in fig.artists:
                #     r, g, b, a = patch.get_facecolor()
                #     patch.set_facecolor((r, g, b, .3))

                ax.set(ylim=(0, 4))
                coef.index = pd.to_datetime(coef.index).strftime('%m-%d')
                ax.set_xticklabels(labels=coef.index, rotation=45, ha='right')
                plt.savefig(plot_output_path+kkk+'/'+kk+'/coef/'+k+'_'+kk+'_'+kkk+'_boxplot',dpi=300)
                plt.close()

                fig, ax = plt.subplots(figsize = (12,6))
                # coef = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='coef')
                # lai = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='LAI')
                # coef = coef.mul(np.sqrt(lai.values))*0.027
                # fig = sns.boxplot(data=coef.T,color='Black')
                coef = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='coef').filter(like='301')
                lai = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='LAI').filter(like='301')
                height = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='height').filter(like='301')
                theta = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='theta').filter(like='301')
                theta_sec = 1/np.cos(theta.values)
                coef = np.power(np.exp(-coef.mul(np.sqrt(lai.values),height.values,theta_sec)*0.027),2)
                plt.fill_between(coef.index.values, coef.min(axis=1).values, coef.max(axis=1).values, facecolor='blue', alpha=0.5)
                plt.plot(coef,color='blue')
                coef = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='coef').filter(like='508')
                lai = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='LAI').filter(like='508')
                height = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='height').filter(like='508')
                theta = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='theta').filter(like='508')
                theta_sec = 1/np.cos(theta.values)
                coef = np.power(np.exp(-coef.mul(np.sqrt(lai.values),height.values,theta_sec)*0.027),2)
                plt.fill_between(coef.index.values, coef.min(axis=1).values, coef.max(axis=1).values, facecolor='green', alpha=0.5)
                plt.plot(coef,color='green')
                coef = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='coef').filter(like='542')
                lai = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='LAI').filter(like='542')
                height = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='height').filter(like='542')
                theta = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='theta').filter(like='542')
                theta_sec = 1/np.cos(theta.values)
                coef = np.power(np.exp(-coef.mul(np.sqrt(lai.values),height.values,theta_sec)*0.027),2)
                plt.fill_between(coef.index.values, coef.min(axis=1).values, coef.max(axis=1).values, facecolor='red', alpha=0.5)
                plt.plot(coef,color='red')

                # for patch in fig.artists:
                #     r, g, b, a = patch.get_facecolor()
                #     patch.set_facecolor((r, g, b, .3))

                # ax.set(ylim=(0, 4))
                coef.index = pd.to_datetime(coef.index).strftime('%m-%d')
                ax.set_xticklabels(labels=coef.index, rotation=45, ha='right')
                plt.savefig(plot_output_path+kkk+'/'+kk+'/coef/T/'+k+'_'+kk+'_'+kkk+'_boxplot_T',dpi=300)
                plt.close()



            if kk == 'water_cloud':

                # fig, ax = plt.subplots(figsize = (12,6))
                # coef = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='A_vv')
                # fig = sns.boxplot(data=coef.T,color='Black')
                # coef = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='A_vv').filter(like='301')
                # fig = sns.boxplot(data=coef.T,color='Blue')
                # coef = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='A_vv').filter(like='508')
                # fig = sns.boxplot(data=coef.T,color='Green')
                # coef = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='A_vv').filter(like='542')
                # fig = sns.boxplot(data=coef.T,color='Red')

                # for patch in fig.artists:
                #     r, g, b, a = patch.get_facecolor()
                #     patch.set_facecolor((r, g, b, .3))

                # ax.set(ylim=(0, 0.08))
                # coef.index = pd.to_datetime(coef.index).strftime('%m-%d')
                # ax.set_xticklabels(labels=coef.index, rotation=45, ha='right')
                # plt.savefig(plot_output_path+kkk+'/'+kk+'/a_vv/'+k+'_'+kk+'_'+kkk+'_boxplot_a_vv',dpi=300)
                # plt.close()

                fig, ax = plt.subplots(figsize = (12,6))
                # coef = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='B_vv')
                # fig = sns.boxplot(data=coef.T,color='Black')
                coef = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='B_vv').filter(like='301')
                plt.fill_between(coef.index.values, coef.min(axis=1).values, coef.max(axis=1).values, facecolor='blue', alpha=0.5)
                plt.plot(coef,color='blue')
                coef = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='B_vv').filter(like='508')
                plt.fill_between(coef.index.values, coef.min(axis=1).values, coef.max(axis=1).values, facecolor='green', alpha=0.5)
                plt.plot(coef,color='green')
                coef = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='B_vv').filter(like='542')
                plt.fill_between(coef.index.values, coef.min(axis=1).values, coef.max(axis=1).values, facecolor='red', alpha=0.5)
                plt.plot(coef,color='red')

                # for patch in fig.artists:
                #     r, g, b, a = patch.get_facecolor()
                #     patch.set_facecolor((r, g, b, .3))

                ax.set(ylim=(0, 0.75))
                coef.index = pd.to_datetime(coef.index).strftime('%m-%d')
                ax.set_xticklabels(labels=coef.index, rotation=45, ha='right')
                plt.savefig(plot_output_path+kkk+'/'+kk+'/b_vv/'+k+'_'+kk+'_'+kkk+'_boxplot_b_vv',dpi=300)
                plt.close()

                fig, ax = plt.subplots(figsize = (12,6))
                # coef = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='B_vv')
                # lai = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='LAI')
                # coef = coef.mul(lai.values)*-2
                # fig = sns.boxplot(data=coef.T,color='Black')
                coef = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='B_vv').filter(like='301')
                lai = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='LAI').filter(like='301')
                theta = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='theta').filter(like='301')
                theta_sec = 1/np.cos(theta.values)
                coef = np.exp(-2 * coef.mul(lai.values,theta_sec))
                plt.fill_between(coef.index.values, coef.min(axis=1).values, coef.max(axis=1).values, facecolor='blue', alpha=0.5)
                plt.plot(coef,color='blue')
                coef = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='B_vv').filter(like='508')
                lai = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='LAI').filter(like='508')
                theta = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='theta').filter(like='508')
                theta_sec = 1/np.cos(theta.values)
                coef = np.exp(-2 * coef.mul(lai.values,theta_sec))
                plt.fill_between(coef.index.values, coef.min(axis=1).values, coef.max(axis=1).values, facecolor='green', alpha=0.5)
                plt.plot(coef,color='green')
                coef = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='B_vv').filter(like='542')
                lai = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='LAI').filter(like='542')
                theta = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='theta').filter(like='542')
                theta_sec = 1/np.cos(theta.values)
                coef = np.exp(-2 * coef.mul(lai.values,theta_sec))
                plt.fill_between(coef.index.values, coef.min(axis=1).values, coef.max(axis=1).values, facecolor='red', alpha=0.5)
                plt.plot(coef,color='red')

                # for patch in fig.artists:
                #     r, g, b, a = patch.get_facecolor()
                #     patch.set_facecolor((r, g, b, .3))

                # ax.set(ylim=(0, 0.75))
                coef.index = pd.to_datetime(coef.index).strftime('%m-%d')
                ax.set_xticklabels(labels=coef.index, rotation=45, ha='right')
                plt.savefig(plot_output_path+kkk+'/'+kk+'/b_vv/T/'+k+'_'+kk+'_'+kkk+'_boxplot_b_vv_T',dpi=300)
                plt.close()



pdb.set_trace()

# ### Plot 508 static
# fig, ax = plt.subplots(figsize=(17, 10))
# # plt.title('Winter Wheat')
# plt.ylabel('Backscatter [dB]', fontsize=15)
# plt.xlabel('Date', fontsize=15)
# plt.tick_params(labelsize=12)
# ax.set_ylim([-25,-7.5])
# for kkk in opt_mod:
#     fig, ax = plt.subplots(figsize=(17, 10))
#     # plt.title('Winter Wheat')
#     plt.ylabel('Backscatter [dB]', fontsize=15)
#     plt.xlabel('Date', fontsize=15)
#     plt.tick_params(labelsize=12)

#     ax.set_ylim([-25,-7.5])
#     for k in surface_list:
#         for kk in canopy_list:

#             vv_model = df_insitu.filter(like='508_high').filter(like=k).filter(like=kk).filter(like=kkk).filter(like='vv_model')

#             vv_s1 = df_insitu.filter(like='508_high').filter(like=k).filter(like=kk).filter(like=kkk).filter(like='S1_vv')

#             if kkk == 'time_invariant':
#                 vv_model = vv_model.values[3:-3]
#                 vv_s1 = vv_s1.values[3:-3]
#                 date = pd.to_datetime(df_insitu.index).strftime('%Y-%m-%d')
#                 date = pd.to_datetime(date[3:-3])
#             else:
#                 vv_model = vv_model.values
#                 vv_s1 = vv_s1.values
#                 date = pd.to_datetime(df_insitu.index).strftime('%Y-%m-%d')
#                 date = pd.to_datetime(date)

#             slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(10*np.log10(vv_s1[~np.isnan(vv_s1)]), 10*np.log10(vv_model[~np.isnan(vv_model)]))
#             rmse = rmse_prediction(10*np.log10(vv_s1), 10*np.log10(vv_model))

#             if k == 'Oh92':
#                 hm = 'Oh92'
#                 colors = 'blue'
#             elif k == 'Oh04':
#                 hm = 'Oh04'
#                 colors = 'red'
#             elif k == 'Dubois95':
#                 hm='Dubois95'
#                 colors = 'orange'
#             elif k == 'WaterCloud':
#                 hm = 'Water Cloud'
#                 colors = 'purple'
#             elif k == 'I2EM':
#                 hm = 'IEM'
#                 colors = 'green'

#             if kk == 'turbid_isotropic':
#                 ax.plot(date, 10*np.log10(vv_model), color=colors, marker='s', linestyle='dashed', label = hm+ ' + ' +  'SSRT' + '; Pol: ' + pol + '; RMSE: ' + str(rmse)[0:4] + '; $R^2$: ' + str(r_value)[0:4])
#             else:
#                 ax.plot(date, 10*np.log10(vv_model), color=colors, marker='s', label = hm+ ' + ' +  'Water Cloud' + '; Pol: ' + pol + '; RMSE: ' + str(rmse)[0:4] + '; $R^2$: ' + str(r_value)[0:4])

#     ax.plot(date, 10*np.log10(vv_s1), 'ks-', label='Sentinel-1 Pol: ' + pol, linewidth=3)
#     plt.legend()
#     plt.savefig(plot_output_path+'_all_'+kkk)
#     plt.close()


# ### Scatterplot
# opt_mod = ['time_invariant', 'time_variant']
# for field in field_list:
#     for k in surface_list:
#         for kk in canopy_list:
#             plt.close()
#             fig, ax = plt.subplots(figsize=(17, 10))
#             for kkk in opt_mod:
#                 plt.ylabel(k + ' ' + kk + ' [dB]')
#                 plt.xlabel('Sentinel-1 [dB]')
#                 vv_model = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='vv_model')
#                 vv_s1 = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='S1_vv')
#                 vv_model_flat = 10*np.log10(vv_model.values.flatten())
#                 vv_s1_flat = 10*np.log10(vv_s1.values.flatten())

#                 slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(vv_s1_flat[~np.isnan(vv_model_flat)],vv_model_flat[~np.isnan(vv_model_flat)])
#                 rmse = rmse_prediction(vv_s1_flat[~np.isnan(vv_model_flat)],vv_model_flat[~np.isnan(vv_model_flat)])


#                 ax.plot(10*np.log10(vv_model.values.flatten()), 10*np.log10(vv_s1.values.flatten()), 'rs', label='RMSE: '+ str(rmse)[0:4] + '; $R^2$: ' + str(r_value)[0:4])

#                 x = np.linspace(np.nanmin(10*np.log10(vv_s1.values))-2, np.nanmax(10*np.log10(vv_s1.values))+2, 16)
#                 xx = np.linspace(np.nanmin(10*np.log10(vv_s1.values)-2)-2, np.nanmax(10*np.log10(vv_s1.values)-2)+2, 16)
#                 ax.plot(x,x)
#                 ax.plot(x,xx)
#                 ax.plot(xx,x)
#                 ax.set_xlim([-22.5,-5])
#                 ax.set_ylim([-22.5,-5])
#                 plt.legend()
#                 plt.savefig(plot_output_path+kkk+'/scatterplot/'+k+'_'+kk)

#                 plt.close()


### Scatterplot
opt_mod = ['time_variant']

s1 = np.array([])
model = np.array([])

colormaps = ['Greens', 'Purples', 'Blues', 'Oranges', 'Reds', 'Greys', 'pink', 'bone', 'Blues', 'Blues', 'Blues', 'Blues', 'Blues']
# colors = ['ks', 'ys', 'ms', 'rs', 'bs', 'gs', 'cs', 'ks', 'ks']
# colors = ['orange', 'yellow', 'green', 'red']


for kk in canopy_list:
    for k in surface_list:
        fig, ax = plt.subplots(figsize=(17, 10))
        plt.ylabel(k + ' ' + kk + ' [dB]')
        plt.xlabel('Sentinel-1 [dB]')
        j=0
        jj=0
        jjj=0
        colormap = plt.get_cmap(colormaps[j])
        colors = [colormap(jj) for jj in np.linspace(0.35, 1., 3)]
        colors = ['lime', 'forestgreen', 'darkgreen', 'lightskyblue', 'blue', 'navy', 'lightcoral', 'red', 'darkred', 'Blues', 'Blues', 'Blues', 'Blues']

        for field in field_list:
            for kkk in opt_mod:
                vv_model = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='vv_model').filter(like=field)
                vv_s1 = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='S1_vv').filter(like=field)
                vv_model_flat = 10*np.log10(vv_model.values.flatten())
                vv_s1_flat = 10*np.log10(vv_s1.values.flatten())

                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(vv_s1_flat[~np.isnan(vv_model_flat)],vv_model_flat[~np.isnan(vv_model_flat)])
                rmse = rmse_prediction(vv_s1_flat[~np.isnan(vv_model_flat)],vv_model_flat[~np.isnan(vv_model_flat)])


                ax.plot(10*np.log10(vv_model.values.flatten()), 10*np.log10(vv_s1.values.flatten()), colors[jj] , linestyle='', marker='.', label='RMSE: '+ str(rmse)[0:4] + '; $R^2$: ' + str(r_value)[0:4] + field)

                jj = jj+1
                s1 = np.concatenate([s1,vv_s1_flat])
                model = np.concatenate([model,vv_model_flat])



        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(s1[~np.isnan(model)],model[~np.isnan(model)])
        rmse = rmse_prediction(s1[~np.isnan(model)],model[~np.isnan(model)])
        x = np.linspace(np.nanmin(s1)-2, np.nanmax(s1)+2, 16)
        xx = np.linspace(np.nanmin(s1-2)-2, np.nanmax(s1-2)+2, 16)
        ax.plot(x,x)
        ax.plot(x,xx)
        ax.plot(xx,x)
        ax.set_xlim([-22.5,-5])
        ax.set_ylim([-22.5,-5])
        plt.legend()
        plt.title('RMSE: '+ str(rmse)[0:4] + '; $R^2$: ' + str(r_value)[0:4])
        plt.savefig(plot_output_path+kkk+'/scatterplot/'+k+'_'+kk)

        plt.close()




pdb.set_trace()


fig, ax = plt.subplots(figsize=(17, 10))
opt_mod = ['time_variant']
for field in field_list:
    for k in surface_list:
        for kk in canopy_list:
            for kkk in opt_mod:
                if kk == 'turbid_isotropic' and k != 'WaterCloud':

                    df, df_agro, field_data, field_data_orbit, theta_field, sm_field, height_field, lai_field, vwc_field, pol_field, vv_field, vh_field, vwcpro_field = read_data(path, file_name, extension, field, path_agro, file_name_agro, extension_agro)

                    freq = 5.405
                    clay = 0.08
                    sand = 0.12
                    bulk = 1.5
                    s = 0.0105 # vv
                    s = 0.0115
                    # s = 0.009 # vh ?????

                    C_hh = 0
                    D_hh = 0
                    C_hv = -22.5
                    D_hv = 3.2
                    C_vv = -14.609339
                    C_vv = -11
                    D_vv = 12.884086
                    D_vv = 1

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
                    V2 = V1 # initialize in surface model
                    coef = 1.
                    omega = 0.027 # vv
                    # omega = 0.015 # vh
                    # IEM
                    l = 0.01

                    surface = k
                    canopy = kk
                    models = {'surface': surface, 'canopy': canopy}

                    dic = {"mv":sm_field.values.flatten(), "C_hh":C_hh, "C_vv":C_vv, "D_hh":D_hh, "D_vv":D_vv, "C_hv":C_hv, "D_hv":D_hv, "s":s, "clay":clay, "sand":sand, "f":freq, "bulk":bulk, "l":l, "canopy":canopy, "d":height_field.values.flatten(), "V1":V1, "V2":V2, "A_hh":A_hh, "B_hh":B_hh, "A_vv":A_vv, "B_vv":B_vv, "A_hv":A_hv, "B_hv":B_hv, "lai":lai_field.values.flatten(), "vwc":vwc_field.values.flatten(), "pol_value":pol_field.values.flatten(), "vv":vv_field.values.flatten(), "vh":vh_field.values.flatten(), "theta":theta_field.values.flatten(), "omega": omega, "coef": coef}

                    coef = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='coef')
                    vv_s1 = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='S1_vv').filter(like=field)
                    coef.T.mean().values
                    var_opt = ['coef']
                    vv,vh = run([coef.T.mean().values])

                    rmse_vv = rmse_prediction(10*np.log10(vv_s1[vv_s1.columns[0]].values),10*np.log10(vv))

                    fig, ax = plt.subplots(figsize=(17, 10))
                    # plt.title('Winter Wheat')
                    plt.ylabel('Backscatter [dB]', fontsize=15)
                    plt.xlabel('Date', fontsize=15)
                    plt.tick_params(labelsize=12)


                    ax.set_ylim([-22,-5])
                    # ax.set_ylim([-30,-15])


                    # grid
                    ax.grid(linestyle='dotted')

                    # weekly grid for x-axis
                    weeks = mdates.WeekdayLocator(byweekday=MO)
                    daysFmt = mdates.DateFormatter('%d. %b')
                    ax.xaxis.set_major_locator(weeks)
                    ax.xaxis.set_major_formatter(daysFmt)

                    # only ticks bottom and left
                    ax.get_xaxis().tick_bottom()
                    ax.get_yaxis().tick_left()

                    # tick size
                    plt.tick_params(labelsize=12)

                    ax.plot(10*np.log10(vv_s1[vv_s1.columns[0]].values),label='S1_vv_'+str(rmse_vv))
                    # plt.plot(10*np.log10(df_output[k,kk,kkk,field,'S1_vh']),label='S1_vh'+str(rmse_vh))
                    # plt.plot(10*np.log10(df_output[k,kk,kkk,field,'vh_model']),label='vh_model')
                    ax.plot(10*np.log10(vv),label=k+' '+kk+' '+kkk+' '+field+' vv')
                    ax.legend()
                    plt.savefig(output_path_boxplot+k+'_'+kk+'_'+kkk+'_'+field+'_middle',dpi=300)

                    plt.close()






pdb.set_trace()
#### Boxplot alt

for field in field_short:
    for k in surface_list:
        for kk in canopy_list:
            for kkk in opt_mod:

                # if kk == 'turbid_isotropic' and k == 'WaterCloud':
                #     coef = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='coef')
                #     coef.T.boxplot()
                #     plt.savefig(output_path_boxplot+k+'_'+kk+'_'+kkk+'_coef',dpi=300)
                #     plt.close()
                #     (coef.T-coef.T.mean()).boxplot()
                #     plt.savefig(output_path_boxplot+k+'_'+kk+'_'+kkk+'_coef-mean',dpi=300)
                #     plt.close()

                #     lai = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='LAI')

                #     ke = pd.DataFrame(coef.values*np.sqrt(lai.values),columns=coef.columns, index=coef.index)
                #     ke.T.boxplot()
                #     plt.savefig(output_path_boxplot+k+'_'+kk+'_'+kkk+'_ke',dpi=300)
                #     plt.close()
                #     (ke.T-ke.T.mean()).boxplot()
                #     plt.savefig(output_path_boxplot+k+'_'+kk+'_'+kkk+'_ke-mean',dpi=300)
                #     plt.close()

                #     coef_2 = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='coef').filter(like=field)
                #     coef_2.T.boxplot()
                #     plt.savefig(output_path_boxplot+k+'_'+kk+'_'+kkk+'_'+field+'_coef_2',dpi=300)
                #     plt.close()
                #     (coef_2.T-coef_2.T.mean()).boxplot()
                #     plt.savefig(output_path_boxplot+k+'_'+kk+'_'+kkk+'_'+field+'_coef_2-mean',dpi=300)
                #     plt.close()
                #     lai_2 = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='LAI').filter(like=field)
                #     ke_2 = pd.DataFrame(coef_2.values*np.sqrt(lai_2.values),columns=coef_2.columns, index=coef_2.index)
                #     ke_2.T.boxplot()
                #     plt.savefig(output_path_boxplot+k+'_'+kk+'_'+kkk+'_'+field+'_ke_2',dpi=300)
                #     plt.close()
                #     (ke_2.T-ke_2.T.mean()).boxplot()
                #     plt.savefig(output_path_boxplot+k+'_'+kk+'_'+kkk+'_'+field+'_ke_2-mean',dpi=300)
                #     plt.close()

                #     C_vv = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='C_vv')
                #     C_vv.T.boxplot()
                #     plt.savefig(output_path_boxplot+k+'_'+kk+'_'+kkk+'_C_vv',dpi=300)
                #     plt.close()
                #     (C_vv.T-C_vv.T.mean()).boxplot()
                #     plt.savefig(output_path_boxplot+k+'_'+kk+'_'+kkk+'_C_vv-mean',dpi=300)
                #     plt.close()

                #     D_vv = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='D_vv')
                #     D_vv.T.boxplot()
                #     plt.savefig(output_path_boxplot+k+'_'+kk+'_'+kkk+'_D_vv',dpi=300)
                #     plt.close()
                #     (D_vv.T-D_vv.T.mean()).boxplot()
                #     plt.savefig(output_path_boxplot+k+'_'+kk+'_'+kkk+'_D_vv_mean',dpi=300)
                #     plt.close()

                if kk == 'turbid_isotropic':
                    coef = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='coef')
                    coef.T.boxplot()
                    plt.savefig(output_path_boxplot+k+'_'+kk+'_'+kkk+'_coef',dpi=300)
                    plt.close()
                    (coef.T-coef.T.mean()).boxplot()
                    plt.savefig(output_path_boxplot+k+'_'+kk+'_'+kkk+'_coef-mean',dpi=300)
                    plt.close()

                    lai = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='LAI')

                    ke = pd.DataFrame(coef.values*np.sqrt(lai.values),columns=coef.columns, index=coef.index)
                    ke.T.boxplot()
                    plt.savefig(output_path_boxplot+k+'_'+kk+'_'+kkk+'_ke',dpi=300)
                    plt.close()
                    (ke.T-ke.T.mean()).boxplot()
                    plt.savefig(output_path_boxplot+k+'_'+kk+'_'+kkk+'_ke-mean',dpi=300)
                    plt.close()

                    coef_2 = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='coef').filter(like=field)
                    coef_2.T.boxplot()
                    plt.savefig(output_path_boxplot+k+'_'+kk+'_'+kkk+'_'+field+'_coef_2',dpi=300)
                    plt.close()
                    (coef_2.T-coef_2.T.mean()).boxplot()
                    plt.savefig(output_path_boxplot+k+'_'+kk+'_'+kkk+'_'+field+'_coef_2-mean',dpi=300)
                    plt.close()
                    lai_2 = df_insitu.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='LAI').filter(like=field)
                    ke_2 = pd.DataFrame(coef_2.values*np.sqrt(lai_2.values),columns=coef_2.columns, index=coef_2.index)
                    ke_2.T.boxplot()
                    plt.savefig(output_path_boxplot+k+'_'+kk+'_'+kkk+'_'+field+'_ke_2',dpi=300)
                    plt.close()
                    (ke_2.T-ke_2.T.mean()).boxplot()
                    plt.savefig(output_path_boxplot+k+'_'+kk+'_'+kkk+'_'+field+'_ke_2-mean',dpi=300)
                    plt.close()

                elif k == 'WaterCloud' and kk == 'water_cloud':
                    pass
                    # df_output[k,kk,kkk,field,'A_vv'] = aaa[0]
                    # A_vv = aaa[0]
                    # df_output[k,kk,kkk,field,'B_vv'] = aaa[1]
                    # B_vv = aaa[1]
                    # df_output[k,kk,kkk,field,'A_vh'] = aaa[2]
                    # A_hv = aaa[2]
                    # df_output[k,kk,kkk,field,'B_vh'] = aaa[3]
                    # B_vh = aaa[3]
                    # df_output[k,kk,kkk,field,'C_vv'] = aaa[4]
                    # B_vh = aaa[4]
                    # df_output[k,kk,kkk,field,'D_vv'] = aaa[5]
                    # B_vh = aaa[5]
                    # df_output[k,kk,kkk,field,'C_vh'] = aaa[6]
                    # B_vh = aaa[6]
                    # df_output[k,kk,kkk,field,'D_vh'] = aaa[7]
                    # B_vh = aaa[7]
                elif kk == 'water_cloud' and k != 'WaterCloud':
                    pass
                    # df_output[k,kk,kkk,field,'A_vv'] = aaa[0]
                    # A_vv = aaa[0]
                    # df_output[k,kk,kkk,field,'B_vv'] = aaa[1]
                    # B_vv = aaa[1]
                    # df_output[k,kk,kkk,field,'A_vh'] = aaa[2]
                    # A_hv = aaa[2]
                    # df_output[k,kk,kkk,field,'B_vh'] = aaa[3]
                    # B_hv = aaa[3]
















pdb.set_trace()
### Nur time_invariant ersten paar Tage zur Berechnung von C_vv und D_vv


df_new = pd.DataFrame()

for field in field_list:

    for k in surface_list:

        for kk in canopy_list:

            df_output = pd.DataFrame(columns=pd.MultiIndex(levels=[[],[],[],[],[]], codes=[[],[],[],[],[]]))

            df, df_agro, field_data, field_data_orbit, theta_field, sm_field, height_field, lai_field, vwc_field, pol_field, vv_field, vh_field, vwcpro_field = read_data(path, file_name, extension, field, path_agro, file_name_agro, extension_agro)
            freq = 5.405
            clay = 0.08
            sand = 0.12
            bulk = 1.5
            s = 0.0105 # vv
            s = 0.0115
            # s = 0.009 # vh ?????

            C_hh = 0
            D_hh = 0
            C_hv = -22.5
            D_hv = 3.2
            C_vv = -14.609339
            C_vv = -11
            D_vv = 12.884086
            D_vv = 10.0

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
            V2 = V1 # initialize in surface model
            coef = 1.
            omega = 0.027 # vv
            # omega = 0.015 # vh
            # IEM
            l = 0.01


            surface = k
            canopy = kk
            models = {'surface': surface, 'canopy': canopy}

            #### Optimization
            #-----------------

            for kkk in opt_mod:
                if kkk == 'time_invariant':
                    df_output = pd.DataFrame(columns=pd.MultiIndex(levels=[[],[],[],[],[]], codes=[[],[],[],[],[]]))
                    n=0
                    theta_field[theta_field.index < '2017-04-04 01:00:00'].values.flatten()
                    dic = {"mv":sm_field[sm_field.index < '2017-04-04 01:00:00'].values.flatten(), "C_hh":C_hh, "C_vv":C_vv, "D_hh":D_hh, "D_vv":D_vv, "C_hv":C_hv, "D_hv":D_hv, "s":s, "clay":clay, "sand":sand, "f":freq, "bulk":bulk, "l":l, "canopy":canopy, "d":height_field[height_field.index < '2017-04-04 01:00:00'].values.flatten(), "V1":lai_field[lai_field.index < '2017-04-04 01:00:00'].values.flatten(), "V2":lai_field[lai_field.index < '2017-04-04 01:00:00'].values.flatten(), "A_hh":A_hh, "B_hh":B_hh, "A_vv":A_vv, "B_vv":B_vv, "A_hv":A_hv, "B_hv":B_hv, "lai":lai_field[lai_field.index < '2017-04-04 01:00:00'].values.flatten(), "vwc":vwc_field[vwc_field.index < '2017-04-04 01:00:00'].values.flatten(), "pol_value":pol_field[pol_field.index < '2017-04-04 01:00:00'].values.flatten(), "vv":vv_field[vv_field.index < '2017-04-04 01:00:00'].values.flatten(), "vh":vh_field[vh_field.index < '2017-04-04 01:00:00'].values.flatten(), "theta":theta_field[theta_field.index < '2017-04-04 01:00:00'].values.flatten(), "omega": omega, "coef": coef}

                    if canopy == 'turbid_isotropic' and surface == 'WaterCloud':
                        var_opt = ['coef', 'C_vv', 'D_vv', 'C_hv', 'D_hv']
                        guess = [0.01, C_vv, D_vv, C_hv, D_hv]
                        bounds = [(0.1,5.5), (-20.,-1.), (-10,20.), (-20.,-1.), (-10,20.)]
                    elif canopy == 'turbid_isotropic':
                        # var_opt = ['coef']
                        # guess = [2.]
                        # bounds = [(0.001,5.5)]
                        var_opt = ['coef']
                        guess = [0.1]
                        bounds = [(0.005,2)]
                    elif surface == 'WaterCloud' and canopy == 'water_cloud':
                        var_opt = ['A_vv', 'B_vv', 'A_hv', 'B_hv', 'C_vv', 'D_vv', 'C_hv', 'D_hv']
                        guess = [A_vv, B_vv, A_hv, B_hv, C_vv, D_vv, C_hv, D_hv]
                        bounds = [(0.00000001,1), (00000001.,1), (000000001.,1), (0.00000001,1), (-20.,-1.), (1.,20.), (-20.,-1.), (1.,20.)]
                    elif canopy == 'water_cloud':
                        var_opt = ['A_vv', 'B_vv', 'A_hv', 'B_hv']
                        guess = [A_vv, B_vv, A_hv, B_hv]
                        bounds = [(0.00000001,1), (0.00000001,1), (0.00000001,1), (0.00000001,1)]

                    method = 'L-BFGS-B'

                    res = minimize(fun_opt,guess,bounds=bounds, method=method)

                    fun_opt(res.x)
                    aaa = res.x

                else:
                    pass


                df_output[k,kk,kkk,field,'SM_insitu'] = sm_field[sm_field.index < '2017-04-04 01:00:00'].values.flatten()
                df_output = df_output.set_index(sm_field[sm_field.index < '2017-04-04 01:00:00'].index)

                df_output[k,kk,kkk,field,'LAI_insitu'] = lai_field[lai_field.index < '2017-04-04 01:00:00'].values.flatten()
                df_output[k,kk,kkk,field,'S1_vv'] = vv_field[vv_field.index < '2017-04-04 01:00:00'].values.flatten()
                df_output[k,kk,kkk,field,'S1_vh'] = vh_field[vh_field.index < '2017-04-04 01:00:00'].values.flatten()
                df_output[k,kk,kkk,field,'theta'] = theta_field[theta_field.index < '2017-04-04 01:00:00'].values.flatten()
                df_output[k,kk,kkk,field,'height'] = height_field[height_field.index < '2017-04-04 01:00:00'].values.flatten()

                if canopy == 'turbid_isotropic' and surface == 'WaterCloud':
                    df_output[k,kk,kkk,field,'coef'] = aaa[0]
                    coef = aaa[0]
                    df_output[k,kk,kkk,field,'C_vv'] = aaa[1]
                    C_vv = aaa[1]
                    df_output[k,kk,kkk,field,'D_vv'] = aaa[2]
                    D_vv = aaa[2]
                    df_output[k,kk,kkk,field,'C_vh'] = aaa[3]
                    C_hv = aaa[3]
                    df_output[k,kk,kkk,field,'D_vh'] = aaa[4]
                    D_hv = aaa[4]
                if canopy == 'turbid_isotropic':
                    df_output[k,kk,kkk,field,'coef'] = aaa[0]
                    coef = aaa[0]
                elif surface == 'WaterCloud' and canopy == 'water_cloud':
                    df_output[k,kk,kkk,field,'A_vv'] = aaa[0]
                    A_vv = aaa[0]
                    df_output[k,kk,kkk,field,'B_vv'] = aaa[1]
                    B_vv = aaa[1]
                    df_output[k,kk,kkk,field,'A_vh'] = aaa[2]
                    A_hv = aaa[2]
                    df_output[k,kk,kkk,field,'B_vh'] = aaa[3]
                    B_vh = aaa[3]
                    df_output[k,kk,kkk,field,'C_vv'] = aaa[4]
                    B_vh = aaa[4]
                    df_output[k,kk,kkk,field,'D_vv'] = aaa[5]
                    B_vh = aaa[5]
                    df_output[k,kk,kkk,field,'C_vh'] = aaa[6]
                    B_vh = aaa[6]
                    df_output[k,kk,kkk,field,'D_vh'] = aaa[7]
                    B_vh = aaa[7]
                elif canopy == 'water_cloud' and surface != 'WaterCloud':
                    df_output[k,kk,kkk,field,'A_vv'] = aaa[0]
                    A_vv = aaa[0]
                    df_output[k,kk,kkk,field,'B_vv'] = aaa[1]
                    B_vv = aaa[1]
                    df_output[k,kk,kkk,field,'A_vh'] = aaa[2]
                    A_hv = aaa[2]
                    df_output[k,kk,kkk,field,'B_vh'] = aaa[3]
                    B_hv = aaa[3]

                dic = {"mv":sm_field[sm_field.index < '2017-04-04 01:00:00'].values.flatten(), "C_hh":C_hh, "C_vv":C_vv, "D_hh":D_hh, "D_vv":D_vv, "C_hv":C_hv, "D_hv":D_hv, "s":s, "clay":clay, "sand":sand, "f":freq, "bulk":bulk, "l":l, "canopy":canopy, "d":height_field[height_field.index < '2017-04-04 01:00:00'].values.flatten(), "V1":lai_field[lai_field.index < '2017-04-04 01:00:00'].values.flatten(), "V2":lai_field[lai_field.index < '2017-04-04 01:00:00'].values.flatten(), "A_hh":A_hh, "B_hh":B_hh, "A_vv":A_vv, "B_vv":B_vv, "A_hv":A_hv, "B_hv":B_hv, "lai":lai_field[lai_field.index < '2017-04-04 01:00:00'].values.flatten(), "vwc":vwc_field[vwc_field.index < '2017-04-04 01:00:00'].values.flatten(), "pol_value":pol_field[pol_field.index < '2017-04-04 01:00:00'].values.flatten(), "vv":vv_field[vv_field.index < '2017-04-04 01:00:00'].values.flatten(), "vh":vh_field[vh_field.index < '2017-04-04 01:00:00'].values.flatten(), "theta":theta_field[theta_field.index < '2017-04-04 01:00:00'].values.flatten(), "omega": omega, "coef": coef}

                print(k+'_'+kk+'_'+kkk+'_'+field)

                if canopy == 'turbid_isotropic' and surface == 'WaterCloud':
                    vv_model, vh_model = run([np.array(df_output[k, kk, kkk, field, 'coef']), np.array(df_output[k, kk, kkk, field, 'C_vv']), np.array(df_output[k, kk, kkk, field, 'D_vv']), np.array(df_output[k, kk, kkk, field, 'C_vh']), np.array(df_output[k, kk, kkk, field, 'D_vh'])])
                elif canopy == 'turbid_isotropic':
                    vv_model, vh_model = run([np.array(df_output[k, kk, kkk, field, 'coef'])])
                # elif surface == 'WaterCloud' and canopy == 'water_cloud' and kkk == 'time_invariant':
                #     vv_model, vh_model = run([np.array(df_output[k, kk, kkk, field, 'A_vv']), np.array(df_output[k, kk, kkk, field, 'B_vv']), np.array(df_output[k, kk, kkk, field, 'A_vh']), np.array(df_output[k, kk, kkk, field, 'B_vh']),aaa[4],aaa[5],aaa[6],aaa[7]])
                elif surface == 'WaterCloud' and canopy == 'water_cloud':
                    vv_model, vh_model = run([np.array(df_output[k, kk, kkk, field, 'A_vv']), np.array(df_output[k, kk, kkk, field, 'B_vv']), np.array(df_output[k, kk, kkk, field, 'A_vh']), np.array(df_output[k, kk, kkk, field, 'B_vh']), np.array(df_output[k, kk, kkk, field, 'C_vv']), np.array(df_output[k, kk, kkk, field, 'D_vv']), np.array(df_output[k, kk, kkk, field, 'C_vh']), np.array(df_output[k, kk, kkk, field, 'D_vh'])])
                elif canopy == 'water_cloud' and surface != 'WaterCloud':
                    vv_model, vh_model = run([np.array(df_output[k, kk, kkk, field, 'A_vv']), np.array(df_output[k, kk, kkk, field, 'B_vv']), np.array(df_output[k, kk, kkk, field, 'A_vh']), np.array(df_output[k, kk, kkk, field, 'B_vh'])])

                df_output[k,kk,kkk,field,'vv_model'] = vv_model
                df_output[k,kk,kkk,field,'vh_model'] = vh_model

                df_output[k,kk,kkk,field,'diff'] = 10*np.log10(df_output[k,kk,kkk,field,'S1_vv']) - 10*np.log10(df_output[k,kk,kkk,field,'vv_model'])

                rmse_vv = rmse_prediction(10*np.log10(df_output[k,kk,kkk,field,'S1_vv']),10*np.log10(df_output[k,kk,kkk,field,'vv_model']))
                # rmse_vh = rmse_prediction(10*np.log10(df_output[k,kk,kkk,field,'S1_vh']),10*np.log10(df_output[k,kk,kkk,field,'vh_model']))



                fig, ax = plt.subplots(figsize=(17, 10))
                # plt.title('Winter Wheat')
                plt.ylabel('Backscatter [dB]', fontsize=15)
                plt.xlabel('Date', fontsize=15)
                plt.tick_params(labelsize=12)


                ax.set_ylim([-22,-5])
                # ax.set_ylim([-30,-15])


                # grid
                ax.grid(linestyle='dotted')

                # weekly grid for x-axis
                weeks = mdates.WeekdayLocator(byweekday=MO)
                daysFmt = mdates.DateFormatter('%d. %b')
                ax.xaxis.set_major_locator(weeks)
                ax.xaxis.set_major_formatter(daysFmt)

                # only ticks bottom and left
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()

                # tick size
                plt.tick_params(labelsize=12)

                ax.plot(10*np.log10(df_output[k,kk,kkk,field,'S1_vv']),label='S1_vv_'+str(rmse_vv))
                # plt.plot(10*np.log10(df_output[k,kk,kkk,field,'S1_vh']),label='S1_vh'+str(rmse_vh))
                # plt.plot(10*np.log10(df_output[k,kk,kkk,field,'vh_model']),label='vh_model')
                ax.plot(10*np.log10(df_output[k,kk,kkk,field,'vv_model']),label=k+' '+kk+' '+kkk+' '+field+' vv')
                ax.legend()
                plt.savefig(plot_output_path+'hm/'+k+'_'+kk+'_'+kkk+'_'+field+'_50',dpi=300)
                plt.close()

                df_output.to_csv(csv_output_path+k+'_'+kk+'_'+kkk+'_'+field+'_50_xx.csv')


pdb.set_trace()

