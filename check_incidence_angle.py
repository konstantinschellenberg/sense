
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


path = '/media/tweiss/Work/z_final_mni_data_2017'
file_name = 'in_situ_s1_buffer_50'


def filter_relativorbit(data, field, orbit1, orbit2=None, orbit3=None, orbit4=None):
    """ data filter for relativ orbits """
    output = data[[(check == orbit1 or check == orbit2 or check == orbit3 or check == orbit4) for check in data[(field,'relativeorbit')]]]
    return output

def rmse_prediction(predictions, targets):
    """ calculation of RMSE """
    return np.sqrt(np.nanmean((predictions - targets) ** 2))

def bias_prediction(predictions, targets):
    return np.nanmean(predictions - targets)




df = pd.read_csv(os.path.join(path,file_name+'.csv'),header=[0,1])
df.index = pd.to_datetime(df['301_high']['date'])

f508 = df.filter(like='508_high')

vv = f508.filter(like='sigma_sentinel_vv')

fa44 = filter_relativorbit(f508,'508_high',44)
fa117 = filter_relativorbit(f508,'508_high',117)
fd95 = filter_relativorbit(f508,'508_high',95)
fd168 = filter_relativorbit(f508,'508_high',168)

vv44 = 10*np.log10(fa44.filter(like='sigma_sentinel_vv'))
vv117 = 10*np.log10(fa117.filter(like='sigma_sentinel_vv'))
vv95 = 10*np.log10(fd95.filter(like='sigma_sentinel_vv'))
vv168 = 10*np.log10(fd168.filter(like='sigma_sentinel_vv'))


xxx = 12.8 - 14.7
xxx2 = 11.1 - 14.7
xxx3 = 13.01 - 14.7
fig, ax1 = plt.subplots()

ax1.plot(10*np.log10(vv))
ax1.plot(10*np.log10(fa44.filter(like='sigma_sentinel_vv'))+xxx2,'bo')
ax1.plot(10*np.log10(fa117.filter(like='sigma_sentinel_vv'))+xxx,'ro')
ax1.plot(10*np.log10(fd95.filter(like='sigma_sentinel_vv')),'yo')
ax1.plot(10*np.log10(fd168.filter(like='sigma_sentinel_vv'))+xxx3,'go')

ax2 = ax1.twinx()
ax2.plot(f508.filter(like='SM'),'r-')
ax2.plot(fa44.filter(like='SM'),'bo')
ax2.plot(fa117.filter(like='SM'),'ro')
ax2.plot(fd95.filter(like='SM'),'yo')
ax2.plot(fd168.filter(like='SM'),'go')
ax2.set_ylim(0,0.5)
ax1.set_ylim(-20,0)
plt.grid()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()



