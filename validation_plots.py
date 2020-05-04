

import pandas as pd
import pdb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter
from matplotlib.ticker import NullFormatter
import skill_metrics as sm
import numpy as np
from matplotlib import rcParams

#### Choose models
#-----------------
surface_list = ['Oh92', 'Oh04', 'Dubois95', 'WaterCloud', 'I2EM']
canopy_list = ['turbid_isotropic', 'water_cloud']
# canopy_list = ['water_cloud']
### option for time_invariant or variant calibration of parameter
#-------------------------------
opt_mod = ['time_invariant','time_variant']
opt_mod = ['time_variant']

csv_file = '/media/tweiss/Work/z_check_data/csv/all_vali_coef_B_vv_mean_50.csv'


df = pd.read_csv(csv_file,header=[0])
colors = ['blue', 'red', 'y', 'purple', 'green', 'yellow']
colors = ['b', 'r', 'y', 'm', 'g', 'y']




for ii in canopy_list:

    fig, ax = plt.subplots(figsize=(20, 10))
    plt.xlabel('Date', fontsize=18)
    plt.tick_params(labelsize=14)
    j=0
    for i in surface_list:
        for iii in opt_mod:
            coef = df.filter(like=i).filter(like=ii).filter(like=iii)
            coef = coef.set_index(pd.to_datetime(df[df.columns[0]], format='%Y-%m-%d %H:%M:%S'))
            fig = sns.boxplot(data=coef[3:-3].T,color=colors[j])
            j=j+1


    patch1 = mpatches.Patch(color=colors[0], label=surface_list[0])
    patch2 = mpatches.Patch(color=colors[1], label=surface_list[1])
    patch3 = mpatches.Patch(color=colors[2], label=surface_list[2])
    patch4 = mpatches.Patch(color=colors[3], label='WCM')
    patch5 = mpatches.Patch(color=colors[4], label='IEM')




    plt.legend(handles=[patch1,patch2,patch3,patch4,patch5],prop={'size': 16})
    plt.grid(linestyle='dotted')

    coef.index = pd.to_datetime(coef.index).strftime('%m-%d')
    ax.set_xticklabels(labels=coef.index, rotation=45, ha='right')
    plt.xlabel('Vegetation period 2017', fontsize=18)
    if ii == 'turbid_isotropic':
        plt.ylabel('empirical parameter coef', fontsize=18)
    else:
        plt.ylabel('empirical parameter B', fontsize=18)

    plt.savefig('/media/tweiss/Work/z_check_data/boxplot_'+ii, bbox_inches = 'tight')
    plt.close()


df_vali = pd.read_csv('/media/tweiss/Work/z_check_data/csv/all_vali_vv_50.csv',header=[0])

### statistic one model combination
df_statistics = pd.DataFrame(columns=pd.MultiIndex(levels=[[],[],[]], codes=[[],[],[]]))
field_short = ['508','301','542']
field_short = ['508_high','508_low','508_med','301_high','301_low','301_med','542_high','542_low','542_med']
colors = ['lime', 'forestgreen', 'darkgreen', 'lightskyblue', 'blue', 'navy', 'lightcoral', 'red', 'darkred', 'blue', 'blue', 'blue', 'blue']
markercolors = ['b', 'r', 'y', 'm', 'g', 'y']

y=0
puh = []
# fig, ax = plt.subplots(figsize=(17, 10))
fig, ax = plt.subplots(figsize=(8, 6))
# Set the figure properties (optional)
# rcParams["figure.figsize"] = [8.0, 6.4]
# rcParams['lines.linewidth'] = 1 # line width for plots
# rcParams.update({'font.size': 12}) # font size of axes text
# rcParams.update({'lines.markersize' : 12})
# rcParams.update({'xtick.labelsize' : 18})
# rcParams.update({'ytick.labelsize' : 12})

for kk in canopy_list:
    fig, ax = plt.subplots(figsize=(8, 6))
    yy=0
    # fig, ax = plt.subplots(figsize=(17, 10))
    for k in surface_list:
        y=0
        for kkk in opt_mod:
            for kkkk in field_short:
                s1_vv = df_vali.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='S1_vv').filter(like=kkkk).values.flatten()
                model_vv = df_vali.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='biasedmodel_').filter(like=kkkk).values.flatten()
                model_vv_ub = df_vali.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='unbiasedmodeldb').filter(like=kkkk).values.flatten()
                # rmse_vv = rmse_prediction(10*np.log10(s1_vv),10*np.log10(model_vv))
                # ubrmse_vv = rmse_prediction(10*np.log10(s1_vv),model_vv_ub)
                # slope, intercept, r_value, p_value, std_err = linregress(10*np.log10(s1_vv),model_vv_ub)

                s1_vv = 10*np.log10(s1_vv)
                model_vv_ub = model_vv_ub
                # model_vv_ub = 10*np.log10(model_vv)

                predictions = model_vv_ub[~np.isnan(model_vv_ub)]
                targets = s1_vv[~np.isnan(model_vv_ub)]
                predictions = predictions[~np.isnan(targets)]
                targets = targets[~np.isnan(targets)]

                stats = sm.taylor_statistics(predictions,targets,'data')

                if y == 0:
                    ccoef = stats['ccoef']
                    crmsd = stats['crmsd']
                    sdev = stats['sdev']
                    label = ['hm',kkkk]
                else:
                    ccoef = np.append(ccoef,stats['ccoef'][1])
                    crmsd = np.append(crmsd,stats['crmsd'][1])
                    sdev = np.append(sdev,stats['sdev'][1])
                    if kkkk == 'I2EM':
                        label.append('IEM')
                    elif kkkk == 'WaterCloud':
                        label.append('WCM')
                    else:
                        label.append(kkkk)

                y=y+1






    # sm.taylor_diagram(sdev,crmsd,ccoef, markerLabel = label, markerLabelColor = 'r',
    #                   markerLegend = 'on', markerColor = 'r',
    #                   styleOBS = '-', colOBS = 'r', markerobs = 'o',
    #                   markerSize = 6, tickRMS = [0.0, 1.0, 2.0, 3.0],
    #                   tickRMSangle = 115, showlabelsRMS = 'on',
    #                   titleRMS = 'on', titleOBS = 'Ref', checkstats = 'on')



            if yy == 0:
                sm.taylor_diagram(np.array(sdev), np.array(crmsd), np.array(ccoef), alpha = 1.0, markercolor=markercolors[yy], markerSize=4, markerLabel = label, markerLabelColor = colors[y], markerLegend = 'on', colCOR = 'k', colRMS='k', styleOBS = '-', colOBS = 'r', markerobs = 'o', titleOBS = 'Ref')
            else:
                sm.taylor_diagram(np.array(sdev), np.array(crmsd), np.array(ccoef), alpha = 1.0, markercolor=markercolors[yy], overlay='on',markerSize=8, markerLabel = label, markerLabelColor = 'b', markerLegend = 'on', colCOR = 'k', colRMS='k')
            yy=yy+1

            # sm.taylor_diagram(sdev,crmsd,ccoef, markerLabel = label,
            #       markerLabelColor = 'r',
            #       tickRMS= np.arange(0,60,10),
            #       tickRMSangle = 110.0,
            #       colRMS = 'm', styleRMS = ':', widthRMS = 2.0,
            #       tickSTD = np.arange(0,80,20), axismax = 60.0,
            #       colSTD = 'b', styleSTD = '-.', widthSTD = 1.0,
            #       colCOR = 'k', styleCOR = '--', widthCOR = 1.0)
    # patch1 = mpatches.Patch(color=markercolors[0], label=surface_list[0], hatch='+')
    # patch2 = mpatches.Patch(color=markercolors[1], label=surface_list[1])
    # patch3 = mpatches.Patch(color=markercolors[2], label=surface_list[2])
    # patch4 = mpatches.Patch(color=markercolors[3], label=surface_list[3])
    # patch5 = mpatches.Patch(color=markercolors[4], label=surface_list[4])

    legend_elements = [
    Line2D([0], [0], color='w', lw=4, label='508-1', marker='P',markerfacecolor='k', markerSize=12), Line2D([0], [0], color='w', lw=4, label='508-2', marker='o',markerfacecolor='k', markerSize=12), Line2D([0], [0], color='w', lw=4, label='508-3', marker='X',markerfacecolor='k', markerSize=12), Line2D([0], [0], color='w', lw=4, label='301-1', marker='s',markerfacecolor='k', markerSize=12), Line2D([0], [0], color='w', lw=4, label='301-2', marker='d',markerfacecolor='k', markerSize=12), Line2D([0], [0], color='w', lw=4, label='301-3', marker='^',markerfacecolor='k', markerSize=12), Line2D([0], [0], color='w', lw=4, label='542-1', marker='v',markerfacecolor='k', markerSize=12), Line2D([0], [0], color='w', lw=4, label='542-2', marker='p',markerfacecolor='k', markerSize=12), Line2D([0], [0], color='w', lw=4, label='542-3', marker='h',markerfacecolor='k', markerSize=12)]

    legend_elements2 = [mpatches.Patch(color=markercolors[0], label=surface_list[0]),mpatches.Patch(color=markercolors[1], label=surface_list[1]),mpatches.Patch(color=markercolors[2], label=surface_list[2]),mpatches.Patch(color=markercolors[3], label=surface_list[3]),mpatches.Patch(color=markercolors[4], label=surface_list[4])]

    leg = ax.legend(handles=legend_elements, prop={'size': 10},loc='center left', bbox_to_anchor=(1.1, 0.3))
    leg1 = ax.legend(handles=legend_elements2, prop={'size': 10},loc='center left', bbox_to_anchor=(1.1, 0.8))
    ax.add_artist(leg)
    plt.savefig('/media/tweiss/Work/z_check_data/taylor_'+kk, bbox_inches = 'tight')
    plt.close()
    # plt.legend(handles=[patch1,patch2,patch3,patch4,patch5],prop={'size': 16})
pdb.set_trace()




csv_file = '/media/tweiss/Work/z_check_data/csv/all_vali_statistics_50.csv'
field_short = ['508','301','542']
df = pd.read_csv(csv_file,header=[0,1,2,3])
df = df.set_index(df.columns[0])
y=0
fig, ax = plt.subplots(figsize=(17, 10))
for ii in canopy_list:
    for i in surface_list:

        for iii in opt_mod:
            for iiii in field_short:
                # stats = df.filter(like=i).filter(like=ii).filter(like=iii).filter(like=iiii)
                stats = df.filter(like=ii).filter(like=iii).filter(like=iiii)

                sdev = stats.loc['sdev_vv'].values
                rmsd = stats.loc['rmse_vv'].values
                crmsd = stats.loc['crmsd_vv'].values
                ubrmsd = stats.loc['ubrmse_vv'].values
                coef = stats.loc['r_value_vv'].values
                bias = stats.loc['bias_vv'].values

                if y == 0:
                    sm.target_diagram(bias,crmsd,rmsd, markercolor=colors[y], ticks=np.arange(-3,4,1.),circles = [1.0, 2.0, 3.0])
                else:
                    sm.target_diagram(bias,crmsd,rmsd, markercolor=colors[y], overlay = 'on', ticks=np.arange(-4,4,1.),circles = [1.0, 2.0, 3.0])
                y=y+1


            pdb.set_trace()






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






pdb.set_trace()
