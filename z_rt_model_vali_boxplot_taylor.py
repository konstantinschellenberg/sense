
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
import skill_metrics as sm
import numpy as np
import os

#### Choose models
#-----------------
surface_list = ['Oh92', 'Oh04', 'Dubois95', 'WaterCloud', 'I2EM']
canopy_list = ['turbid_isotropic', 'water_cloud']
# canopy_list = ['water_cloud']

### option for time_invariant or variant calibration of parameter
#-------------------------------
# opt_mod = ['time_invariant','time_variant']
opt_mod = ['time_variant']

input_path = '/media/tweiss/Work/z_check_data/csv'
output_path = '/media/tweiss/Work/z_check_data'

boxplot_data =  'all_vali_coef_B_vv_mean_50.csv'
boxplot_file = os.path.join(input_path,boxplot_data)
df_boxplot = pd.read_csv(boxplot_file,header=[0])

taylor_data = 'all_vali_vv_50.csv'
taylor_file = os.path.join(input_path,taylor_data)
df_taylor = pd.read_csv(taylor_file,header=[0])

colors = ['b', 'r', 'y', 'm', 'g', 'y']

### Boxplot ###
#-----------------
for ii in canopy_list:

    fig, ax = plt.subplots(figsize=(20, 10))
    plt.xlabel('Date', fontsize=18)
    plt.tick_params(labelsize=14)
    j=0
    for i in surface_list:
        for iii in opt_mod:
            coef = df_boxplot.filter(like=i).filter(like=ii).filter(like=iii)
            coef = coef.set_index(pd.to_datetime(df_boxplot[df_boxplot.columns[0]], format='%Y-%m-%d %H:%M:%S'))
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

    plt.savefig(os.path.join(output_path,'boxplot_'+ii), bbox_inches = 'tight')
    plt.close()


### Taylor plot
#------------------
# Info: Made some changes within skill_metrics package (rename of RMSD to ubRMSE!)

field_short = ['508_high','508_low','508_med','301_high','301_low','301_med','542_high','542_low','542_med']

for kk in canopy_list:
    fig, ax = plt.subplots(figsize=(8, 6))
    yy=0
    for k in surface_list:
        y=0
        for kkk in opt_mod:
            for kkkk in field_short:
                s1_vv = df_taylor.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='S1_vv').filter(like=kkkk).values.flatten()
                model_vv = df_taylor.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='biasedmodel_').filter(like=kkkk).values.flatten()
                model_vv_ub = df_taylor.filter(like=k).filter(like=kk).filter(like=kkk).filter(like='unbiasedmodeldb').filter(like=kkkk).values.flatten()

                s1_vv = 10*np.log10(s1_vv)
                model_vv_ub = model_vv_ub

                predictions = model_vv_ub[~np.isnan(model_vv_ub)]
                targets = s1_vv[~np.isnan(model_vv_ub)]
                predictions = predictions[~np.isnan(targets)]
                targets = targets[~np.isnan(targets)]

                stats = sm.taylor_statistics(predictions,targets,'data')

                if y == 0:
                    ccoef = stats['ccoef']
                    crmsd = stats['crmsd']
                    sdev = stats['sdev']
                    label = ['',kkkk]
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

            if yy == 0:
                sm.taylor_diagram(np.array(sdev), np.array(crmsd), np.array(ccoef), alpha = 1.0, markercolor=colors[yy], markerSize=4, markerLabel = label, markerLabelColor = 'b', markerLegend = 'on', colCOR = 'k', colRMS='k', styleOBS = '-', colOBS = 'r', markerobs = 'o', titleOBS = 'Ref')
            else:
                sm.taylor_diagram(np.array(sdev), np.array(crmsd), np.array(ccoef), alpha = 1.0, markercolor=colors[yy], overlay='on',markerSize=8, markerLabel = label, markerLabelColor = 'b', markerLegend = 'on', colCOR = 'k', colRMS='k')
            yy=yy+1

    legend_elements = [
    Line2D([0], [0], color='w', lw=4, label='508-1', marker='P',markerfacecolor='k', markerSize=12), Line2D([0], [0], color='w', lw=4, label='508-2', marker='o',markerfacecolor='k', markerSize=12), Line2D([0], [0], color='w', lw=4, label='508-3', marker='X',markerfacecolor='k', markerSize=12), Line2D([0], [0], color='w', lw=4, label='301-1', marker='s',markerfacecolor='k', markerSize=12), Line2D([0], [0], color='w', lw=4, label='301-2', marker='d',markerfacecolor='k', markerSize=12), Line2D([0], [0], color='w', lw=4, label='301-3', marker='^',markerfacecolor='k', markerSize=12), Line2D([0], [0], color='w', lw=4, label='542-1', marker='v',markerfacecolor='k', markerSize=12), Line2D([0], [0], color='w', lw=4, label='542-2', marker='p',markerfacecolor='k', markerSize=12), Line2D([0], [0], color='w', lw=4, label='542-3', marker='h',markerfacecolor='k', markerSize=12)]

    legend_elements2 = [mpatches.Patch(color=colors[0], label=surface_list[0]),mpatches.Patch(color=colors[1], label=surface_list[1]),mpatches.Patch(color=colors[2], label=surface_list[2]),mpatches.Patch(color=colors[3], label=surface_list[3]),mpatches.Patch(color=colors[4], label=surface_list[4])]

    leg = ax.legend(handles=legend_elements, prop={'size': 10},loc='center left', bbox_to_anchor=(1.1, 0.3))
    leg1 = ax.legend(handles=legend_elements2, prop={'size': 10},loc='center left', bbox_to_anchor=(1.1, 0.8))
    ax.add_artist(leg)
    plt.savefig(os.path.join(output_path,'taylor_'+kk), bbox_inches = 'tight')
    plt.close()


