min#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:17:30 2020

@author: sorensnielsen
"""



import os 
import time
import numpy as np
from PIL import Image
from multiprocessing import Pool 
from pims import ImageSequence
import numpy as np
import pandas as pd
import scipy
import matplotlib  as mpl
import matplotlib.pyplot as plt
from skimage import feature
import scipy.ndimage as ndimage
from skimage.feature import blob_log
import trackpy as tp
import os
from scipy.ndimage.filters import gaussian_filter
from timeit import default_timer as timer 
from glob import glob
from iminuit import Minuit
from probfit import BinnedLH, Chi2Regression, Extended, BinnedChi2, UnbinnedLH, gaussian
from pims import TiffStack
import pandas as pd
from skimage.feature import blob_log
import seaborn as sns
from tqdm import tqdm
from astropy.stats import RipleysKEstimator
from scipy import stats
import numpy as np

from pims import ImageSequence
import numpy as np
import pandas as pd
import trackpy as tp
import matplotlib  as mpl
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy import ndimage
from skimage.feature import blob_log
from skimage import feature
from scipy.stats.stats import pearsonr 
import os
import scipy
import scipy.ndimage as ndimage
from skimage import measure
from skimage.color import rgb2gray 
from skimage import io

   
from pims import TiffStack

import scipy

from skimage import feature

    

from skimage.feature import peak_local_max
from scipy import ndimage
from skimage.feature import blob_log
from skimage import feature
from scipy.stats.stats import pearsonr 
import os
import scipy
import scipy.ndimage as ndimage
from skimage import measure
from skimage.color import rgb2gray
import matplotlib.patches as mpatches   
import glob
from skimage import measure

from pomegranate import *
import time

import random
import itertools

import cython
import probfit

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn import neighbors
from tqdm import tqdm
import iminuit as Minuit

from probfit import BinnedLH, Chi2Regression, Extended, BinnedChi2, UnbinnedLH, gaussian
from matplotlib import patches

save_path = '/Users/sorensnielsen/Documents/Other people/philip/tracking articles/plots/'

blanchard_cols = ["#FDF9E9", "#FDF9E9", "#FFFFFF", "#BACAE9", "#88AAD8", "#527FC0", "#2E5992", "#00AAA4", "#3AB64A", "#CADB2A", "#FFF203", "#FFD26F", "#FAA841", "#F5834F", "#ED1D25"]
ccmap_lines =  ["lightgrey", "#FFFFFF", "#BACAE9", "#88AAD8", "#527FC0", "#2E5992", "#00AAA4", "#3AB64A", "#CADB2A", "#FFF203", "#FFD26F", "#FAA841", "#F5834F", "#ED1D25"]
   



def return_coordinates(df):
    x_tmp = np.asarray(df['x'].tolist())
    y_tmp = np.asarray(df['y'].tolist())
    df['normed_step'] = df['steplength']/np.max(df['steplength'])
    true_step = np.asarray(df['steplength'].tolist())
    step = np.asarray(df['normed_step'].tolist())
    
    return x_tmp,y_tmp,step,true_step
#for Figure 1, use the thing pml made, but Create a nice overlay of traces 
def fix_ax_probs(ax,x_label,y_label):
    ax.set_ylabel(y_label, size = 12)
    ax.set_xlabel(x_label, size = 12)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 10)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 10)
    ax.grid(False)
    return ax

def fit_func_up(x,rate,p,base_line):             # function to fit - p
    if rate <= 0 or p <0 or rate > 100000:
        return x*0
    else:
        return (p*(np.exp((rate)*x)))+base_line

def fit_func2(x,rate,p,base_line):             # function to fit - p
    if rate <= 0 or p <0 or rate > 100000:
        return x*0
    else:
        return (p*(np.exp(-(1/rate)*x)))+base_line

def fit_func_lin_up(x,a,b):
    if b<0:
        return x*0
    else:
        return a*x+b
    
    
def fit_double_exp(x,rate1,rate2,p1,p2,base_line):    
    if rate1 <= 0 or p1 <0 or rate1 > 100000:
        return x*0
    if rate2 <= 0 or p2 <0 or rate2 > 100000:
        return x*0
    else:
        return (p2*(np.exp(-(rate2)*x)))+ (p1*(np.exp(-(rate1)*x)))+base_line

def fit_rate_up_lin(data):
    import probfit
    import iminuit
    from probfit import BinnedLH, Chi2Regression, Extended, BinnedChi2, UnbinnedLH, gaussian
    
    x_vals = np.arange(1,len(data)+1)
    
    chi2 = Chi2Regression(fit_func_lin_up,x_vals,data)
    m = iminuit.Minuit(chi2,a = 1000.,b =np.mean(data[:10]),print_level=0)                           
    m.migrad()

    return m.values

def fit_rate_double(data):
    import probfit
    import iminuit
    from probfit import BinnedLH, Chi2Regression, Extended, BinnedChi2, UnbinnedLH, gaussian
    
    x_vals = np.arange(1,len(data)+1)
    
    chi2 = Chi2Regression(fit_double_exp,x_vals,data)
    m = iminuit.Minuit(chi2,rate1 = 0.001,p1 =10000,rate2 = 0.01,p2 =10000,print_level=0)                           
    m.migrad()

    return m.values,m

def fit_rate(data):
    import probfit
    import iminuit
    from probfit import BinnedLH, Chi2Regression, Extended, BinnedChi2, UnbinnedLH, gaussian
    
    x_vals = np.arange(1,len(data)+1)
    
    chi2 = Chi2Regression(fit_func2,x_vals,data)
    m = iminuit.Minuit(chi2,rate = 0.001,p =200,print_level=0)                           
    m.migrad()

    return m.values


def fit_rate_up(data):
    import probfit
    import iminuit
    from probfit import BinnedLH, Chi2Regression, Extended, BinnedChi2, UnbinnedLH, gaussian
    
    x_vals = np.arange(1,len(data)+1)
    
    chi2 = Chi2Regression(fit_func_up,x_vals,data)
    m = iminuit.Minuit(chi2,rate = 45.,p =200,print_level=0)                           
    m.migrad()

    return m.values
"""
# fitting potassium
rate = []
lip_size = []
green_size_init = []
green_size_end = []
p = []
base_line = []
file= []
names = []





dfs = ['/Volumes/Nebuchadnez/Soren/dopamin/new_encap/potassium 0.5um dopamin/tifs/_0_signal_df.csv',
       '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/potassium 0.5um dopamin/tifs/_1_signal_df.csv',
       '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/potassium 0.5um dopamin/tifs/_2_signal_df.csv', 
       '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/potassium 0.5um dopamin/tifs/_3_signal_df.csv', 
       '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/potassium 0.5um dopamin/tifs/_4_signal_df.csv',
       '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/potassium 0.5um dopamin/tifs/_5_signal_df.csv']





data_number     = 5
particle_name   = 75

data = pd.read_csv(dfs[data_number], low_memory=False, sep = ',')

particle_list = [5_28, 5_34, 5_85, 5_91, 5_93, 5_97, 5_114, 5_117, 5_122, 5_124, 5_134, 5_136, 5_144, 5_147, 5_159, 5_163, 5_164, 5_169, 5_170, 5_172, 5_173, 5_176, 5_179, 5_183, 5_191, 5_196, 5_199, 5_206, 5_216, 5_220, 5_225, 5_229, 5_233, 5_234, 5_237, 5_239, 5_246, 5_260, 5_264, 5_267, 5_268, 5_275, 5_278, 5_279, 5_294, 5_295, 5_300, 5_301, 5_303, 5_305, 5_333, 5_341]


for particle_name in tqdm(particle_list):
    particle_name = str(particle_name)
    particle_name = int(particle_name[1:])
    par  =data[data.particle == particle_name]
    
    
    
    start = 20
    
    end =   697
    
    
    
    
    fig,ax = plt.subplots(figsize = (4,2.5))
    ax.plot(par.frame.values,par.green_int_corrected.values,color = "seagreen",alpha =0.8,label = "K+ signal")
    
    ax.set_xlim(0,100)
    
    
    
    
    fit = fit_rate(par.green_int_corrected.values[start:end])
    
    fig,ax = plt.subplots(figsize = (4,2.5))
    ax.plot(par.frame.values,par.green_int_corrected.values,color = "seagreen",alpha =0.8,label = "K+ signal")
    ax.plot(par.frame.values[start:],fit_func2(par.frame.values[:-start],fit['rate'],fit['p'],fit['base_line']),color = "firebrick",linestyle = "--",alpha =0.8,label = "Single exp fit")
    ax.legend()
    ax = fix_ax_probs(ax,'Time [frame]','Intensity [A.U.]')
    fig.tight_layout()
    fig.savefig(str('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/treatment/potassium_0.5um_20200519/_'+str(data_number)+'_'+str(particle_name)+'_fit.pdf'))
    
    
    
    
    
    
    
    rate.append(fit['rate'])
    lip_size.append(np.mean(par.red_int_corrected.values[:10])**0.5)
    green_size_init.append(np.mean(par.green_int_corrected.values[:10]))
    green_size_end.append(np.mean(par.green_int_corrected.values[650:]))
    p.append(fit['p'])
    base_line.append(fit['base_line'])
    file.append(data_number)
    names.append(particle_name)
    
    
    
    fig,ax = plt.subplots(figsize = (4,2.5))
    ax.plot(par.frame.values,par.green_int_corrected.values,color = "seagreen",alpha =0.8,label = "K+ signal")
    ax.legend()
    ax = fix_ax_probs(ax,'Time [frame]','Intensity [A.U.]')
    fig.tight_layout()
    fig.savefig(str('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/treatment/potassium_0.5um_20200519/_'+str(data_number)+'_'+str(particle_name)+'_raw.pdf'))
    plt.close('all')

fits = pd.DataFrame(
                {'lip_size_relative'          : lip_size,
                 'sensor_init'         : green_size_init,
                 'sensor_end'     : green_size_end,
                 'particle'         : names,
                 'df_number'        :file,
                 'rate'             : rate, 
                 'base_line'        :base_line,
                 'p'                :p})    
fits['condition'] = 'potassium_05um_dopamin_20200519' 

fits.to_csv(str('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/treatment/'+'fitting_stats_potassium_05um_dopamin_20200519.csv'), header=True, index=None, sep=',', mode='w')     
  
rate =np.asarray(rate)
lip_size =np.asarray(lip_size)
rate_sort = rate[rate<1000]
lip_size_sort = lip_size[rate<1000]
green_size_init=np.asarray(green_size_init)
green_size_end=np.asarray(green_size_end)

drop =green_size_init-green_size_end
    
fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(rate,30,range = (0,1000),density = True,color = "gray",alpha = 0.8,label = str("K+ sensor, N: "+str(len(rate))))
ax.legend()
ax =fix_ax_probs(ax,'Rate [I/f]','Density')
fig.tight_layout()
fig.savefig(str('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/treatment/unconverted_rates_potassium_05um_dopamin_20200519.pdf'))

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.scatter(lip_size_sort,rate_sort, marker = ".",color = "gray",alpha = 0.8,label = str("K+ sensor, N: "+str(len(rate))))
ax.legend()
ax =fix_ax_probs(ax,'Relative size','Rate')
fig.tight_layout()
fig.savefig(str('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/treatment/unconverted_rates_vs_size_potassium_05um_dopamin_20200519.pdf'))







ints = []
encaps = []
encaps_end = []
particle = []
df_number = []
n = 0
for tmp in dfs:
    data = pd.read_csv(tmp, low_memory=False, sep = ',')
    data = data[data.y > 25 ]
    data = data[data.y < 512+25 ]
    data = data[data.x > 25 ]
    data = data[data.x < 512+25 ]
    grp = data.groupby('particle')
    for name, par in tqdm(grp):       
        ints.append(np.mean(par['red_int_corrected'].values[:10]))
        encaps.append(np.mean(par['green_int_corrected'].values[:10]))
        encaps_end.append(np.mean(par['green_int_corrected'].values[-10:]))
        df_number.append(n)
        particle.append(name)
    n+=1
    
    
    
ints = np.asarray(ints)**0.5  
encaps = np.asarray(encaps)**0.5  
encaps_end = np.asarray(encaps_end)**0.5    
features = pd.DataFrame(
                {'lip_size_relative'          : ints,
                 'sensor_init_relative'         : encaps,
                 'sensor_end_relative'     : encaps_end,
                 'particle'         : particle,
                 'df_number'        :df_number})    
features['condition'] = 'potassium_05um_dopamin_20200519' 
features.to_csv(str('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/treatment/'+'sizes_potassium_05um_dopamin_20200519.csv'), header=True, index=None, sep=',', mode='w')     
  
    
    
fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(ints,30,range = (0,800),density = True,color = "firebrick",alpha = 0.8,label = str("liposomes, N: "+str(len(ints))))
ax.legend()
ax =fix_ax_probs(ax,'Relative lip size','Density')
fig.tight_layout()
fig.savefig(str('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/treatment/size_liposomes_potassium_05um_dopamin_20200519.pdf'))

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(encaps,30,range = (0,800),density = True,color = "seagreen",alpha = 0.8,label = str("K+ init, N: "+str(len(ints))))
ax.legend()
ax =fix_ax_probs(ax,'Relative lip size','Density')
fig.tight_layout()
fig.savefig(str('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/treatment/size_encaps_init_potassium_05um_dopamin_20200519.pdf'))

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(encaps_end,30,range = (0,800),density = True,color = "seagreen",alpha = 0.8,label = str("K+ end, N: "+str(len(ints))))
ax.legend()
ax =fix_ax_probs(ax,'Relative lip size','Density')
fig.tight_layout()
fig.savefig(str('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/treatment/size_encaps_end_potassium_05um_dopamin_20200519.pdf'))




# fitting sodium


lip_size = []
green_size_init = []
green_size_end = []
start_decrease = []
start_increase = []
end_increase = []
rate_decrease = []
p_decrease = []
base_line_decrease = []

rate_increase = []
p_increase = []
base_line_increase = []

steady_state_length = []
steady_state_int = []
file= []
names = []


dfs = ['/Volumes/Nebuchadnez/Soren/dopamin/new_encap/Ver2 sodium indicator 0.5uM dopamin 400mM NaCl/tifs/_0_signal_df.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/Ver2 sodium indicator 0.5uM dopamin 400mM NaCl/tifs/_1_signal_df.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/Ver2 sodium indicator 0.5uM dopamin 400mM NaCl/tifs/_2_signal_df.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/Ver2 sodium indicator 0.5uM dopamin 400mM NaCl/tifs/_3_signal_df.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/Ver2 sodium indicator 0.5uM dopamin 400mM NaCl/tifs/_4_signal_df.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/Ver2 sodium indicator 0.5uM dopamin 400mM NaCl/tifs/_5_signal_df.csv']





data_number     = 0
particle_name   = 66

data = pd.read_csv(dfs[data_number], low_memory=False, sep = ',')



par  =data[data.particle == particle_name]

fig,ax = plt.subplots(figsize = (4,2.5))
ax.plot(par.frame.values,par.green_int_corrected.values,color = "seagreen",alpha =0.8,label = "K+ signal")

ax.set_xlim(0,120)


start_dec = 98
end_dec =   397


start_inc = 20
end_inc = 50




fit = fit_rate(par.green_int_corrected.values[start_dec:end_dec])

fit_up = fit_rate_up(par.green_int_corrected.values[start_inc:end_inc])


fig,ax = plt.subplots(figsize = (4,2.5))
ax.plot(par.frame.values,par.green_int_corrected.values,color = "royalblue",alpha =0.8,label = "Na+ signal")
ax.plot(par.frame.values[start_dec:],fit_func2(par.frame.values[:-start_dec],fit['rate'],fit['p'],fit['base_line']),color = "firebrick",linestyle = "--",alpha =0.8,label = "Single exp fit")
ax.plot(par.frame.values[start_inc:end_inc],fit_func_up(par.frame.values[:int(end_inc-start_inc)],fit_up['rate'],fit_up['p'],fit_up['base_line']),color = "black",linestyle = "--",alpha =0.8, label= "Single exp fit")


ax.legend()
ax = fix_ax_probs(ax,'Time [frame]','Intensity [A.U.]')
fig.tight_layout()
fig.savefig(str('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/treatment/_'+str(data_number)+'_'+str(particle_name)+'_fit.pdf'))







lip_size.append(np.mean(par.red_int_corrected.values[:10])**0.5)
green_size_init.append(np.mean(par.green_int_corrected.values[:10]))
green_size_end.append(np.mean(par.green_int_corrected.values[380:]))
start_decrease.append(start_dec)
start_increase.append(start_inc)
end_increase.append(end_inc)
rate_decrease.append(fit['rate'])
p_decrease.append(fit['p'])
base_line_decrease.append(fit['base_line'])

rate_increase.append(fit_up['rate'])
p_increase.append(fit_up['p'])
base_line_increase.append(fit_up['base_line'])

steady_state_length.append(start_dec-end_inc)
steady_state_int.append(np.mean(par.green_int_corrected.values[end_inc:start_dec]))
file.append(data_number)
names.append(particle_name)








dfs = ['/Volumes/Nebuchadnez/Soren/dopamin/new_encap/Ver2 sodium indicator 0.5uM dopamin 400mM NaCl/tifs/_0_signal_df.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/Ver2 sodium indicator 0.5uM dopamin 400mM NaCl/tifs/_1_signal_df.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/Ver2 sodium indicator 0.5uM dopamin 400mM NaCl/tifs/_2_signal_df.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/Ver2 sodium indicator 0.5uM dopamin 400mM NaCl/tifs/_3_signal_df.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/Ver2 sodium indicator 0.5uM dopamin 400mM NaCl/tifs/_4_signal_df.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/Ver2 sodium indicator 0.5uM dopamin 400mM NaCl/tifs/_5_signal_df.csv']





data_number     = 0
particle_name   = 185

data = pd.read_csv('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/potassium 0.5um dopamin/tifs/_0_signal_df.csv', low_memory=False, sep = ',')



par  =data[data.particle == particle_name]



fig,ax = plt.subplots(figsize = (4,2.5))
ax.plot(par.frame.values,par.green_int_corrected.values,color = "seagreen",alpha =0.8,label = "K+ signal")
ax.set_xlim(0,200)

ax.legend()
ax = fix_ax_probs(ax,'Time [frame]','Intensity [A.U.]')
fig.tight_layout()
fig.savefig(str('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/treatment/_'+str(data_number)+'_'+str(particle_name)+'K_repres5.pdf'))



"""


lip_size = []
encaps_init_raw = []
encaps_end_raw = []
particle_name_id = []
df_number_id = []
rate_dec = []
p_dec = []
baseline_dec = []
start_value_dec = []



list_of_dfs = ['/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/k indicator 100mM nacl 100kcl no dopamin/tifs/_0_signal_df_corrected_for_lips.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/k indicator 100mM nacl 100kcl no dopamin/tifs/_1_signal_df_corrected_for_lips.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/k indicator 100mM nacl 100kcl no dopamin/tifs/_2_signal_df_corrected_for_lips.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/k indicator 100mM nacl 100kcl no dopamin/tifs/_3_signal_df_corrected_for_lips.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/k indicator 100mM nacl 100kcl no dopamin/tifs/_4_signal_df_corrected_for_lips.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/k indicator 100mM nacl 100kcl no dopamin/tifs/_5_signal_df_corrected_for_lips.csv']

accepted_particles = '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/k indicator 100mM nacl 100kcl no dopamin/accepted_particles.csv'

data = pd.read_csv(accepted_particles, low_memory=False, sep = ',')
grp = data.groupby('df')

list_of_traces_in_dfs = []

for name, df in grp:
    list_of_traces_in_dfs.append(df['particle'].tolist())

#list_of_traces_in_dfs = [   [15,18,21,24,27,32,37,56,58,64,65,67,70,75,79,87,88,89,96,103,109,116,121,133,148,172],
#                            [17,2,151,156,157,163,1640,21,23,27,29,32,39,49,56,60,62,73,76,82,85,96,110,118,123,144],
#                            [19,21,26,28,40,44,46,48,50,58,59,63,67,68,69,71,78,93,94,95,97,107,110,123,127,132],
#                            [26,30,34,50,54,63,65,74,76,77,80,86,96,98,110,117,118,124,125,127,140,143,156,158],
#                            [],
#                            []]

df_numbers = [0,1,2,3,4,5]
couterasd = 0
for df_number in df_numbers:
    data = pd.read_csv(list_of_dfs[df_number], low_memory=False, sep = ',')
    list_of_traces_in_df = list_of_traces_in_dfs[df_number]
    for particle_name in list_of_traces_in_df:
        #particle_name = str(particle_name)
        #particle_name = int(particle_name[1:])
    
        par  =data[data.particle == particle_name]
        
        
        cont = 0
        x_fit_line = 30
        x_end = 300
        tmp = par.green_int_corrected.values/np.max(par.green_int_corrected.values)
  
#            
#        while cont <1:
#            fit = fit_rate(par.green_int_corrected.values[x_fit_line:x_end])  
#            x_vals = np.linspace(0,len(par.green_int_corrected.values[x_fit_line:x_end]),1000)
#
#            
#            
#            fig,ax = plt.subplots(1,2)    
#            ax[0].plot(par.frame.values,par.green_int_corrected.values,color = "seagreen",alpha =0.8,label = str(format((fit['rate']/3.358),'.5f')))
#            ax[0].axvline(x_fit_line)
#            ax[0].legend()
#            ax[0].axvline(x_end)
#            ax[0].plot(x_vals+x_fit_line,fit_func2(x_vals,fit['rate'],fit['p'],fit['base_line']))
#            ax[1].plot(par.frame.values,par.green_int_corrected.values,color = "seagreen",alpha =0.8,label = "K+ signal")
#            ax[1].set_xlim(x_fit_line-10,x_fit_line+100)
#            ax[1].axvline(x_fit_line)
#            ax[1].axvline(x_end)
#            ax[0].set_xlabel(str(str(df_number)+'_'+str(couterasd)))
#            plt.figure()
#            plt.show()
#            plt.ion()
#            raw = input("Change x: Enter -y- if good\n")
#            if raw == "y":
#                x_end = int(len(par.frame.values)-5)
#                cont += 5
#            else:
#                x_fit_line = input("Enter x:\n")
#
#                x_fit_line =int(x_fit_line)
#                
#                x_end = input("Enter x end:\n")
#
#                x_end =int(x_end)
#        
        fit = fit_rate(par.green_int_corrected.values[x_fit_line:x_end])    
        
        lip_size.append(np.mean(par.red_int_corrected.values[:10])**0.5)
        encaps_init_raw.append(np.mean(par.green_int_corrected.values[:10]))
        encaps_end_raw.append(np.mean(par.green_int_corrected.values[len(par.frame.values)-20:]))
        particle_name_id.append(particle_name)
        df_number_id.append(df_number)
        rate_dec.append(fit['rate'])
        p_dec.append(fit['p'])
        baseline_dec.append(fit['base_line'])
        start_value_dec.append(x_fit_line)
        couterasd +=1


features = pd.DataFrame(
                {'lip_size_relative'          : lip_size,
                 'sensor_init_relative'         : encaps_init_raw,
                 'sensor_end_relative'     : encaps_end_raw,
                 'particle'         : particle_name_id,
                 'df_number'        :df_number_id,
                 'rate' :rate_dec,
                 'p_dec': p_dec,
                 'baseline_dec': baseline_dec}) 


features['condition'] = 'K+ indicator 100KCl 100NaCl dopamin' 
features.to_csv(str('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/k indicator 100mM nacl 100kcl no dopamin/'+str('all')+'rate_fit_data_tester.csv'), header=True, index=None, sep=',', mode='w')     
 



"""



lip_size = []
encaps_init_raw = []
encaps_end_raw = []
particle_name_id = []
df_number_id = []
rate_dec = []
p_dec = []
baseline_dec = []

rate_inc = []
p_inc = []
baseline_inc = []

start_value_dec = []
start_value_inc = []
length_of_platau = []

encaps_bf_inc = []
encaps_peak = []
encaps_dec_end = []




list_of_dfs = ['/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/na indikator no edta, vit c or dopamin/tifs/_0_signal_df_corrected_for_lips.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/na indikator no edta, vit c or dopamin/tifs/_1_signal_df_corrected_for_lips.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/na indikator no edta, vit c or dopamin/tifs/_2_signal_df_corrected_for_lips.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/na indikator no edta, vit c or dopamin/tifs/_3_signal_df_corrected_for_lips.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/na indikator no edta, vit c or dopamin/tifs/_4_signal_df.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/na indikator no edta, vit c or dopamin/tifs/_5_signal_df.csv']

accepted_particles = '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/na indikator no edta, vit c or dopamin/accepted_particles.csv'

data = pd.read_csv(accepted_particles, low_memory=False, sep = ',')
grp = data.groupby('df')

list_of_traces_in_dfs = []

for name, df in grp:
    list_of_traces_in_dfs.append(df['particle'].tolist())

counteredssf = 0
df_numbers = [0,1,2,3,4,5]
for df_number in df_numbers:
    data = pd.read_csv(list_of_dfs[df_number], low_memory=False, sep = ',')
    list_of_traces_in_df = list_of_traces_in_dfs[df_number]
    for particle_name in list_of_traces_in_df:
        par  =data[data.particle == particle_name]
        cont = 0
        y_fit_line = 10
        y_fit_line_end = 20
        y_height = 50
        y_end = 200
        counteredssf +=1


        while cont <1:
            y_fit_line = int(y_fit_line)
            y_fit_line_end = int(y_fit_line_end)
            y_height = int(y_height)
            y_end = int(y_end)
            
            fig,ax = plt.subplots()    
            ax.plot(par.frame.values,par.green_int_corrected.values,color = "royalblue",alpha =0.8,label = "Na+ signal")
            ax.scatter(y_fit_line,par.green_int_corrected.values[y_fit_line],color = "red",label = "Init")
            ax.scatter(y_fit_line_end,par.green_int_corrected.values[y_fit_line_end],color = "blue",label = "fit end")
            ax.scatter(y_height,par.green_int_corrected.values[y_height],color = "green",label = "height end")
            ax.scatter(y_end,par.green_int_corrected.values[y_end],color = "m",label = "end")
            ax.set_xlabel(str(counteredssf))
            ax.legend()
            
            plt.figure()
            plt.show()
            plt.ion()
            print (str("Current are: Init: ")+str(y_fit_line))
            print (str("Current are: fit end: ")+str(y_fit_line_end))
            print (str("Current are: height end: ")+str(y_height))
            print (str("Current are: end: ")+str(y_end))
            
            raw = input("Change x: Enter -no- if good\n")
            if raw == "no":
                cont += 5
            else:
                y_fit_line = input("Enter init:\n")
                y_fit_line =int(y_fit_line)
                
                y_fit_line_end = input("Enter fit end:\n")
                y_fit_line_end =int(y_fit_line_end)
                
                
                y_height = input("Enter height end:\n")
                y_height =int(y_height)
                
                y_end = input("Enter end:\n")
                y_end =int(y_end)


        
        
        fit = fit_rate(par.green_int_corrected.values[y_height:y_end])
        fit_up = fit_rate_up(par.green_int_corrected.values[y_fit_line:y_fit_line_end])
    
        lip_size.append(np.mean(par.red_int_corrected.values[:10])**0.5)
        encaps_init_raw.append(np.mean(par.green_int_corrected.values[:10]))
        encaps_end_raw.append(np.mean(par.green_int_corrected.values[len(par.frame.values)-20:]))
        particle_name_id.append(particle_name)
        df_number_id.append(df_number)
        
        
        
        
        
        
        rate_dec.append(fit['rate'])
        p_dec.append(fit['p'])
        baseline_dec.append(fit['base_line'])
        rate_inc.append(fit_up['rate'])
        p_inc.append(fit_up['p'])
        baseline_inc.append(fit_up['base_line'])
    
        start_value_dec.append(y_height)
        start_value_inc.append(y_fit_line)
        length_of_platau.append(y_height-y_fit_line_end)
    
        encaps_bf_inc.append(np.mean(par.green_int_corrected.values[5-y_fit_line:y_fit_line]))
        encaps_peak.append(np.mean(par.green_int_corrected.values[5-y_height:y_height]))
        encaps_dec_end.append(np.mean(par.green_int_corrected.values[5-y_end:y_end]))
        

features = pd.DataFrame(
                {'lip_size_relative'          : lip_size,
                 'sensor_init_relative'         : encaps_init_raw,
                 'sensor_end_relative'     : encaps_end_raw,
                 'particle'         : particle_name_id,
                 'df_number'        :df_number_id,
                 'rate_dec' :rate_dec,
                 'p_dec': p_dec,
                 'baseline_dec': baseline_dec,
                 'rate_inc' :rate_inc,
                 'p_inc': p_inc,
                 'baseline_inc': baseline_inc,
                 'start_value_dec' :start_value_dec,
                 'start_value_inc': start_value_inc,
                 'length_of_platau': length_of_platau,
                 'encaps_bf_inc_raw' :start_value_dec,
                 'encaps_peak_raw': start_value_inc,
                 'encaps_dec_end_raw': length_of_platau}) 


features['condition'] = 'na indikator no edta, vit c or dopamin' 
features.to_csv(str('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/na indikator no edta, vit c or dopamin/'+'rate_fit_data.csv'), header=True, index=None, sep=',', mode='a')     
 




"""

"""
cond = 'no_dopamine'
list_of_dfs = ['/Volumes/Nebuchadnez/Soren/dopamin/new_encap/Potassium/Controls/20200525control without dopamin/tifs/_0_signal_df_corrected_for_lips.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/Potassium/Controls/20200525control without dopamin/tifs/_1_signal_df_corrected_for_lips.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/Potassium/Controls/20200525control without dopamin/tifs/_2_signal_df_corrected_for_lips.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/Potassium/Controls/20200525control without dopamin/tifs/_3_signal_df_corrected_for_lips.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/Potassium/Controls/20200525control without dopamin/tifs/_4_signal_df_corrected_for_lips.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/Potassium/Controls/20200525control without dopamin/tifs/_5_signal_df_corrected_for_lips.csv']

df_rate_list = ['/Volumes/Nebuchadnez/Soren/dopamin/new_encap/Potassium/Controls/20200525control without dopamin/tifs/__allrate_fit_data.csv']


lips = []
lips_act = []
rates = []
drop = []

for df in list_of_dfs:
    data = pd.read_csv(df, low_memory=False, sep = ',')
    grp = data.groupby('particle')
    for name,par in tqdm(grp):
        lips.append(np.mean(par.red_int_corrected[:10])**0.5)

for df in df_rate_list:
    data = pd.read_csv(df, low_memory=False, sep = ',')
    lips_act.extend(data.lip_size_relative.values)
    rates.extend(data.rate.values)
    tmp_init = data.sensor_init_relative.values
    tmp_end = data.sensor_end_relative.values
    drop.extend(tmp_init-tmp_end)

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(lips,30,range = (0,1000),density = True,color = "gray",alpha =0.8,label = str("All lips, N: "+str(len(lips))))
ax=fix_ax_probs(ax,'Relative size','Density')
ax.legend()
fig.tight_layout()
fig.savefig(str('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/Potassium/'+str(cond)+'lip_size.pdf'))

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(lips_act,30,range = (0,1000),density = True,color = "firebrick",alpha =0.8,label = str("Active traces, N: "+str(len(lips_act))))
ax=fix_ax_probs(ax,'Relative size','Density')
ax.legend()
fig.tight_layout()
fig.savefig(str('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/Potassium/'+str(cond)+'act_lip_size.pdf'))

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(lips,30,range = (0,1000),density = True,color = "gray",alpha =0.8,label = str("all lips, N: "+str(len(lips))))
ax.hist(lips_act,30,range = (0,1000),density = True,color = "firebrick",alpha =0.8,label = str("Active traces, N: "+str(len(lips_act))))
ax=fix_ax_probs(ax,'Relative size','Density')
ax.legend()
fig.tight_layout()
fig.savefig(str('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/Potassium/'+str(cond)+'act_and_all_lip_size.pdf'))




fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(rates,30,range = (0,1000),density = True,color = "royalblue",alpha =0.8,label = str("Rates, N: "+str(len(lips_act))))
ax=fix_ax_probs(ax,'Rate I/Frame','Density')
ax.legend()
fig.tight_layout()
fig.savefig(str('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/Potassium/'+str(cond)+'rate.pdf'))


fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.scatter(lips_act,rates,marker = ".",color ="royalblue",alpha = 0.8)
ax.set_xlim(0,1000)
ax.set_ylim(0,1000)
ax=fix_ax_probs(ax,'relative size','Rate')
ax.legend()
fig.tight_layout()
fig.savefig(str('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/Potassium/'+str(cond)+'size_v_rate_scatter.pdf'))


save_path = '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/plots_20200713/'

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.bar(1,15.52821813,width = 0.5,yerr = 1.063244489,color = "seagreen",edgecolor = "black",alpha =.8,label = "100:100, -Dop")
ax.bar(2,0.119047619,width = 0.5,yerr = 0.291605922,color = "seagreen",edgecolor = "black",alpha =.8,label = "100:100, +Dop + inhib")
ax.bar(3,18.09429951,width = 0.5,yerr = 1.824918132,color = "seagreen",edgecolor = "black",alpha =.8,label = "100:100, +Dop")
ax.set_ylim(0,30)
ax = fix_ax_probs(ax,'','Percentage')
plt.xticks([1,2,3], ['-Dop','+Dop, +Inhib','+Dop'], rotation=-45)
fig.tight_layout()
fig.savefig(str(save_path+'Potassium_100_100_buffer.pdf'))

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.bar(2,0.422754767,width = 0.5,yerr = 0.381185182,color = "seagreen",edgecolor = "black",alpha =.8,label = "200NaCL, +Dop + inhib")
ax.bar(3,15.3521518,width = 0.5,yerr = 4.66044984,color = "seagreen",edgecolor = "black",alpha =.8,label = "200NaCl, +Dop")
ax.bar(1,15.61974998,width = 0.5,yerr = 4.303629158,color = "seagreen",edgecolor = "black",alpha =.8,label = "200NaCl, -Dop")
ax.set_ylim(0,30)
ax = fix_ax_probs(ax,'','Percentage')
plt.xticks([1,2,3], ['-Dop','+Dop, +Inhib','+Dop'], rotation=-45)
fig.tight_layout()
fig.savefig(str(save_path+'Potassium_200NaCL_buffer.pdf'))

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.bar(1,8.364813471,width = 0.5,yerr = 0.864211948,color = "royalblue",edgecolor = "black",alpha =.8,label = "200NaCL, -Dop")
ax.bar(2,0,width = 0.5,yerr = 0,color = "royalblue",edgecolor = "black",alpha =.8,label = "200NaCl, +Dop + Inhib")
ax.bar(3,11.46559624,width = 0.5,yerr =2.162598692,color = "royalblue",edgecolor = "black",alpha =.8,label = "200NaCl, +Dop")
ax.set_ylim(0,30)
ax = fix_ax_probs(ax,'','Percentage')
plt.xticks([1,2,3], ['-Dop','+Dop, +Inhib','+Dop'], rotation=-45)
fig.tight_layout()
fig.savefig(str(save_path+'Sodium_200NaCL_buffer_NMDG.pdf'))

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.bar(1,3.536266942,width = 0.5,yerr = 0.60155381,color = "royalblue",edgecolor = "black",alpha =.8,label = "100:100, -Dop")
#ax.bar(2,0,width = 0.5,yerr = 0,color = "royalblue",edgecolor = "black",alpha =.8,label = "100:100, +Dop + Inhib")
ax.bar(2,3.814372291,width = 0.5,yerr =0.804289308,color = "royalblue",edgecolor = "black",alpha =.8,label = "100:100, +Dop")
ax.set_ylim(0,30)
ax = fix_ax_probs(ax,'','Percentage')
plt.xticks([1,2,3], ['-Dop','+Dop','+Dop, +Inhib'], rotation=-45)
fig.tight_layout()
fig.savefig(str(save_path+'Sodium_100-100aCL_buffer.pdf'))

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.bar(1,4.52874466,width = 0.5,yerr = 3.142744996,color = "royalblue",edgecolor = "black",alpha =.8,label = "200Nacl, -Dop")
ax.bar(2,9.151360753,width = 0.5,yerr =2.715915117,color = "royalblue",edgecolor = "black",alpha =.8,label = "200NaCl, +Dop")
ax.set_ylim(0,30)
ax = fix_ax_probs(ax,'','Percentage')
plt.xticks([1,2,3], ['-Dop','+Dop','+Dop, +Inhib'], rotation=-45)
fig.tight_layout()
fig.savefig(str(save_path+'Sodium_200NaCL_buffer.pdf'))
"""
def plot_condition_k(df,all_dfs,path):
    all_lips = []
    for i in all_dfs:
        tmp  = pd.read_csv(i, low_memory=False, sep = ',')
        grp = tmp.groupby('particle')
        for name, da in grp:
            all_lips.extend(da.red_int_corrected.values[:10]**0.5)
    data =  pd.read_csv(df, low_memory=False, sep = ',')        
    all_lips = np.asarray(all_lips)
    
    data = data[data.lip_size_relative >0]
    
    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.hist(all_lips,20,range = (0,1000),density = True,edgecolor = "black",color = "gray",alpha =0.8,label = "All lips")
    ax = fix_ax_probs(ax,'Relative size','Density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(path+'_all_liposome_size.pdf'))
    
    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.hist(data.lip_size_relative.values,20,range = (0,1000),density = True,edgecolor = "black",color = "gray",alpha =0.8,label = "Active")
    ax = fix_ax_probs(ax,'Relative size','Density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(path+'_active_liposome_size.pdf'))
    
    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.hist(data.sensor_init_relative.values-data.sensor_end_relative.values,20,range = (0,200000),density = True,edgecolor = "black",color = "gray",alpha =0.8,label = "Dec raw")
    ax = fix_ax_probs(ax,'Raw Int','Density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(path+'raw_decrease.pdf'))
    
    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.hist(data.rate.values,20,range = (0,400),density = True,edgecolor = "black",color = "gray",alpha =0.8,label = "Rate")
    ax = fix_ax_probs(ax,'Rate','Density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(path+'rate.pdf'))
    
    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.scatter(data.lip_size_relative.values,data.rate.values,marker = '.',color = "black",alpha =0.8)
    ax.set_xlim(0,1000)
    ax.set_ylim(0,400)
    ax = fix_ax_probs(ax,'Lip size','Rate Dec')
    fig.tight_layout()
    fig.savefig(str(path+'_scatter___lip_vs_ratedec.pdf'))
    
    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.scatter(data.sensor_init_relative.values,data.rate.values,marker = '.',color = "black",alpha =0.8)
    ax.set_xlim(0,200000)
    ax.set_ylim(0,400)
    ax = fix_ax_probs(ax,'Encaps Int','Rate Dec')
    fig.tight_layout()
    fig.savefig(str(path+'_scatter___encaps_int_vs_ratedec.pdf'))
    
    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.scatter(data.sensor_init_relative.values,data.sensor_end_relative.values,marker = '.',color = "black",alpha =0.8)
    ax.set_xlim(0,200000)
    ax.set_ylim(0,200000)
    ax = fix_ax_probs(ax,'Encaps Int','encaos end')
    fig.tight_layout()
    fig.savefig(str(path+'_scatter___encaps_int_vs_encaps_int_end.pdf'))
    plt.close('all')
    

    with open(str(path+"correlations.txt"), "w") as text_file:
        text_file.write(str('size to rate dec')+  "\n")
        text_file.write(str(stats.pearsonr(data.lip_size_relative.values,data.rate.values))+  "\n\n")
        

    
  

def plot_condition_na(df,all_dfs,path):
    all_lips = []
    for i in all_dfs:
        tmp  = pd.read_csv(i, low_memory=False, sep = ',')
        grp = tmp.groupby('particle')
        for name, da in grp:
            all_lips.extend(da.red_int_corrected.values[:10]**0.5)
    data =  pd.read_csv(df, low_memory=False, sep = ',')        
    all_lips = np.asarray(all_lips)
    
    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.hist(all_lips,20,range = (0,1000),density = True,edgecolor = "black",color = "gray",alpha =0.8,label = "All lips")
    ax = fix_ax_probs(ax,'Relative size','Density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(path+'_all_liposome_size.pdf'))
    
    
    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.hist(data.lip_size_relative.values,20,range = (0,1000),density = True,edgecolor = "black",color = "gray",alpha =0.8,label = "Active")
    ax = fix_ax_probs(ax,'Relative size','Density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(path+'_active_liposome_size.pdf'))
    
    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.hist(data.rate_inc.values,20,range = (0,300),density = True,edgecolor = "black",color = "gray",alpha =0.8,label = "Inc. Rate")
    ax = fix_ax_probs(ax,'Rate','Density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(path+'_rate_inc.pdf'))
    
    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.hist(data.rate_dec.values,20,range = (0,400),density = True,edgecolor = "black",color = "gray",alpha =0.8,label = "Dec. Rate")
    ax = fix_ax_probs(ax,'Rate','Density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(path+'_rate_dec.pdf'))
    
    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.hist((data.p_dec.values-data.baseline_inc.values),20,range = (0,300000),density = True,edgecolor = "black",color = "gray",alpha =0.8,label = "peak increase")
    ax = fix_ax_probs(ax,'Int inc','Density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(path+'_peak_increase.pdf'))
    
    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.hist((data.baseline_inc.values-data.baseline_dec.values),20,range = (0,100000),density = True,edgecolor = "black",color = "gray",alpha =0.8,label = "init-end")
    ax = fix_ax_probs(ax,'Int start - end','Density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(path+'_decrease level.pdf'))
    
    
    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.scatter(data.lip_size_relative.values,data.rate_inc.values,marker = '.',color = "black",alpha =0.8)
    ax.set_xlim(0,1000)
    ax.set_ylim(0,300)
    fig.tight_layout()
    ax = fix_ax_probs(ax,'Lip size','Rate Inc')
    fig.savefig(str(path+'_scatter___lip_vs_rateinc.pdf'))
    
    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.scatter(data.lip_size_relative.values,data.rate_dec.values,marker = '.',color = "black",alpha =0.8)
    ax.set_xlim(0,1000)
    ax.set_ylim(0,400)
    fig.tight_layout()
    ax = fix_ax_probs(ax,'Lip size','Rate Dec')
    fig.savefig(str(path+'_scatter___lip_vs_ratedec.pdf'))
    
    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.scatter(data.rate_inc.values,data.rate_dec.values,marker = '.',color = "black",alpha =0.8)
    ax.set_xlim(0,300)
    ax.set_ylim(0,400)
    fig.tight_layout()
    ax = fix_ax_probs(ax,'Rate Inc','Rate Dec')
    fig.savefig(str(path+'_scatter___rateinc_vs_ratedec.pdf'))
    
    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.scatter(data.p_dec.values,data.rate_inc.values,marker = '.',color = "black",alpha =0.8)
    ax.set_xlim(0,200000)
    ax.set_ylim(0,300)
    fig.tight_layout()
    ax = fix_ax_probs(ax,'Int Inc','Rate Inc')
    fig.savefig(str(path+'_scatter___peak_height_vs_rateinc.pdf'))
    
    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.scatter(data.lip_size_relative.values,data.p_dec.values-data.baseline_inc.values,marker = '.',color = "black",alpha =0.8)
    ax.set_xlim(0,1000)
    ax.set_ylim(0,200000)
    fig.tight_layout()
    ax = fix_ax_probs(ax,'Int Inc','Rate Inc')
    fig.savefig(str(path+'_scatter___lip_size_vs_peak_inc.pdf'))
    
    
    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.scatter(data.p_dec.values,data.rate_dec.values,marker = '.',color = "black",alpha =0.8)
    ax.set_xlim(0,200000)
    ax.set_ylim(0,400)
    fig.tight_layout()
    ax = fix_ax_probs(ax,'Int Inc','Rate Dec')
    fig.savefig(str(path+'_scatter___peak_height_vs_ratedec.pdf'))
    plt.close('all')
    
    
    
    with open(str(path+"correlations.txt"), "w") as text_file:
        text_file.write(str('rate inc to rate dec')+  "\n")
        text_file.write(str(stats.pearsonr(data.rate_inc.values,data.rate_dec.values))+  "\n\n")
        
        text_file.write(str('lip size to rate inc')+  "\n")
        text_file.write(str(stats.pearsonr(data.lip_size_relative.values,data.rate_inc.values))+  "\n\n")
        
        text_file.write(str('lip size to rate dec')+  "\n")
        text_file.write(str(stats.pearsonr(data.lip_size_relative.values,data.rate_dec.values))+  "\n\n")

        text_file.write(str('lip size to peak_increase')+  "\n")
        text_file.write(str(stats.pearsonr(data.lip_size_relative.values,data.p_dec.values-data.baseline_inc.values))+  "\n\n")

mains = ['/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200630_new_k_exp/20200630/100mM NaCl +100 mM KCl +0.5uM Da w.kcl/', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200630_new_k_exp/20200630/100mM NaCl +100 mM KCl no Da w.kcl/', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200630_new_k_exp/20200630/200mM NaCl +0.5uM Da w.kcl/', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200630_new_k_exp/20200630/200mM NaCl no Da w. KCl/',
         '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200707_new_NA_exp/Na indicator 200mM NMDG 0.5uM dopamin_bad/', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200707_new_NA_exp/Na indicator 200mM NMDG no dopamin/', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200707_new_NA_exp/ver2 Na indicator 200mM NMDG 0.5uM dopamin/']


for path in mains:
    df = str(path+'/rate_fit_data.csv')
    all_dfs = [str(path+'tifs/_0_signal_df_corrected_for_lips.csv'),
               str(path+'tifs/_1_signal_df_corrected_for_lips.csv'),
               str(path+'tifs/_2_signal_df_corrected_for_lips.csv'),
               str(path+'tifs/_3_signal_df_corrected_for_lips.csv'),
               str(path+'tifs/_4_signal_df_corrected_for_lips.csv'),
               str(path+'tifs/_5_signal_df_corrected_for_lips.csv')]
    #plot_condition_na(df,all_dfs,path)



mains = ['/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200707_new_NA_exp/K+ indicator 100KCl 100NaCl no dopamin/',
         '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200626_new_k_exp/k+ ind and 100mM NaCl 100 KCl 0.5uM Da uptake/', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200626_new_k_exp/k+ ind and 200mM NaCl 0.5uM Da 500uM inhibitor uptake/', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200626_new_k_exp/k+ ind and 200mM NaCl 0.5uM Da uptake/', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200626_new_k_exp/k+ ind and 200mM NaCl no Da uptake/', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200626_new_k_exp/ver2 k+ ind and 200mM NaCl 0.5uM Da uptake/']




for path in mains:
    df = str(path+'allrate_fit_data.csv')
    all_dfs = [str(path+'tifs/_0_signal_df_corrected_for_lips.csv'),
               str(path+'tifs/_1_signal_df_corrected_for_lips.csv'),
               str(path+'tifs/_2_signal_df_corrected_for_lips.csv'),
               str(path+'tifs/_3_signal_df_corrected_for_lips.csv'),
               str(path+'tifs/_4_signal_df_corrected_for_lips.csv'),
               str(path+'tifs/_5_signal_df_corrected_for_lips.csv')]
    #plot_condition_k(df,all_dfs,path)



def base_fit(x,lag,baseline):
    if x <lag:
        return baseline+x*0
    else:
        return 0
def base_drop(x,N,k,baseline2):
    return (N*(np.exp(-(1/k)*x)))+baseline2

def awesome_fit(x,lag,k,N,baseline,baseline2):

    return base_fit(x,baseline)
 
    
    
    #if lag< 5 or lag >30:
    #    return 0
    #if k<0 or k>10000:
    #    return 0
    #if baseline<0 or baseline2<0:
    #    return 0
    #if x<0:
    #    return 0
    #if N<0:
    #    return 0
    #if x<lag:
    #    return base_fit(baseline)
    #else:
    #    return 0

"""

data = [10,11,12,13,12,12,11,12,14,11,10,8,12,13,10,14,15,14,13,15,10,8,7,5,3,2,2,3,2,1,2,3]
data = np.asarray(data)
import probfit
import iminuit
from probfit import BinnedLH, Chi2Regression, Extended, BinnedChi2, UnbinnedLH, gaussian

x_vals = np.arange(1,len(data)+1)

chi2 = Chi2Regression(awesome_fit,x_vals,data)
m = iminuit.Minuit(chi2)                           
m.migrad()

fig,ax = plt.subplots()
ax.plot(x_vals,data)
ax.plot(x_vals,awesome_fit(x_vals,20,0,10,11,2))

m.args
awesome_fit(x_vals,20,10,10,11,2)

for val in x_vals:
    print (awesome_fit(val,20,10,50,11,2))
"""

"""






list_of_dfs = ['/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/na indikator 200mM nacl outside 200NMDG inside no dopamin/tifs/_0_signal_df_corrected_for_lips.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/na indikator 200mM nacl outside 200NMDG inside no dopamin/tifs/_1_signal_df_corrected_for_lips.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/na indikator 200mM nacl outside 200NMDG inside no dopamin/tifs/_2_signal_df_corrected_for_lips.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/na indikator 200mM nacl outside 200NMDG inside no dopamin/tifs/_3_signal_df_corrected_for_lips.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/na indikator 200mM nacl outside 200NMDG inside no dopamin/tifs/_4_signal_df_corrected_for_lips.csv', '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/na indikator 200mM nacl outside 200NMDG inside no dopamin/tifs/_5_signal_df_corrected_for_lips.csv']

accepted_particles ='/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/na indikator 200mM nacl outside 200NMDG inside no dopamin/accepted_particles.csv'

path = '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/na indikator 200mM nacl outside 200NMDG inside no dopamin/'

data = pd.read_csv(accepted_particles, low_memory=False, sep = ',')
grp = data.groupby('df')

list_of_traces_in_dfs = []

for name, df in grp:
    list_of_traces_in_dfs.append(df['particle'].tolist())

sum_of_active = []
sum_of_lips = []
percentage = []


df_numbers = [0,1,2,3,4,5]
for df_number in df_numbers:
    data = pd.read_csv(list_of_dfs[df_number], low_memory=False, sep = ',')
    
    
    list_of_traces_in_df = list_of_traces_in_dfs[df_number]
    
    
    
    sum_of_active.append(len(list_of_traces_in_df))
    sum_of_lips.append(len(set(data['particle'])))
    
    percentage.append(100*len(list_of_traces_in_df)/len(set(data['particle'])))

with open(str(path+"stats.txt"), "w") as text_file:
    text_file.write(str("Lips: "+ str(sum(sum_of_lips))+'\n'))
    text_file.write(str("Active: "+ str(sum(sum_of_active))+'\n'))
    text_file.write(str("Percentage: "+ str(np.mean(percentage))+'\n'))
    text_file.write(str("Std dev: "+ str(np.std(percentage))+'\n'))
    
    text_file.write('\n\n')
    text_file.write(str("Std dev: "+ str((percentage))+'\n'))

print ("Lips: "+ str(sum(sum_of_lips)))
print ("Active: "+ str(sum(sum_of_active)))

print ("Percentage: "+ str(np.mean(percentage)))
print ("Std dev: "+ str(np.std(percentage)))








fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.bar(3,39.902561,width = 0.5,yerr = 3.9405444,color = "seagreen",edgecolor = "black",alpha =.8)
ax.bar(2,15.3521518,width = 0.5,yerr = 4.66044984,color = "seagreen",edgecolor = "black",alpha =.8)
ax.bar(1,15.61974998,width = 0.5,yerr = 4.303629158,color = "seagreen",edgecolor = "black",alpha =.8)
ax.set_ylim(0,30)
ax = fix_ax_probs(ax,'','Percentage')
plt.xticks([1,2,3], ['-Dop','+Dop, +Inhib','+Dop'], rotation=-45)
fig.tight_layout()
fig.savefig(str(path+'Potassium_200NaCL_buffer.pdf'))




list_of_dfs = ['/Volumes/Nebuchadnez/Soren/dopamin/publication figures and data/20200626 potasium first purification/100 nacl 100 kcl dopamine/all_liposomes.csv', '/Volumes/Nebuchadnez/Soren/dopamin/publication figures and data/20200626 potasium first purification/200 nack dopamine/all_liposomes.csv', '/Volumes/Nebuchadnez/Soren/dopamin/publication figures and data/20200626 potasium first purification/200 nacl dopamine inhib/all_liposomes.csv', '/Volumes/Nebuchadnez/Soren/dopamin/publication figures and data/20200626 potasium first purification/200 nacl dopamine vers 2/all_liposomes.csv', '/Volumes/Nebuchadnez/Soren/dopamin/publication figures and data/20200630 sodium experiments first purification/sodium 100 nacl 100 kcl dopamine/all_liposomes.csv', '/Volumes/Nebuchadnez/Soren/dopamin/publication figures and data/20200630 sodium experiments first purification/sodium 100 nacl 100 kcl no dopamine/all_liposomes.csv', '/Volumes/Nebuchadnez/Soren/dopamin/publication figures and data/20200630 sodium experiments first purification/sodium 200 NaCl dopamine inhib/all_liposomes.csv', '/Volumes/Nebuchadnez/Soren/dopamin/publication figures and data/20200630 sodium experiments first purification/sodium 200 nacl no dopamine/all_liposomes.csv', '/Volumes/Nebuchadnez/Soren/dopamin/publication figures and data/20200630 sodium experiments first purification/sodium 200 ncl dopamine/all_liposomes.csv', '/Volumes/Nebuchadnez/Soren/dopamin/publication figures and data/20200707 first purification k indicator 100 nacl 100 kcl no dopamine/all_liposomes.csv', '/Volumes/Nebuchadnez/Soren/dopamin/publication figures and data/20200707 first purification na indicator 200mM nMDG dopamine/all_liposomes.csv', '/Volumes/Nebuchadnez/Soren/dopamin/publication figures and data/20200707 first purification na indicator 200mM nMDG no dopamine/all_liposomes.csv', '/Volumes/Nebuchadnez/Soren/dopamin/publication figures and data/20200819 second purification potassium data 100 nacl 100 klc/dopamine/all_liposomes.csv', '/Volumes/Nebuchadnez/Soren/dopamin/publication figures and data/20200819 second purification potassium data 100 nacl 100 klc/no dopamine/all_liposomes.csv']


path ='/Volumes/Nebuchadnez/Soren/dopamin/publication figures and data/plots_for_publication/'


df_numbers = [0,1,2,3,4,5]
all_lips = []
for df_number in range(len(list_of_dfs)):
    tmp = pd.read_csv(list_of_dfs[df_number], low_memory=False, sep = ',')
  
    all_lips.extend(tmp.liposomes.values)


features = pd.DataFrame(
                {'liposomes'          : all_lips}) 


features.to_csv(str(path+'all_liposomes_across_conditions.csv'), header=True, index=None, sep=',', mode='w')     
 











lips_act = data.lip_size_relative.values

rates = data.rate.values


fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(lips,30,range = (0,1000),density = True,color = "gray",alpha =0.8,label = str("All lips, N: "+str(len(lips))))
ax=fix_ax_probs(ax,'Relative size','Density')
ax.legend()
fig.tight_layout()
fig.savefig(str(path+'lip_size.pdf'))

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(lips_act,30,range = (0,1000),density = True,color = "firebrick",alpha =0.8,label = str("Active traces, N: "+str(len(lips_act))))
ax=fix_ax_probs(ax,'Relative size','Density')
ax.legend()
fig.tight_layout()
fig.savefig(str(path+'act_lip_size.pdf'))

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(lips,30,range = (0,1000),density = True,color = "gray",alpha =0.8,label = str("all lips, N: "+str(len(lips))))
ax.hist(lips_act,30,range = (0,1000),density = True,color = "firebrick",alpha =0.8,label = str("Active traces, N: "+str(len(lips_act))))
ax=fix_ax_probs(ax,'Relative size','Density')
ax.legend()
fig.tight_layout()
fig.savefig(str(path+'act_and_all_lip_size.pdf'))



fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(rates,25,range = (0,700),density = True,color = "royalblue",alpha =0.8,label = str("Rates, N: "+str(len(lips_act))))
ax=fix_ax_probs(ax,'Rate I/Frame','Density')
ax.legend()
fig.tight_layout()
fig.savefig(str(path+'rate.pdf'))


fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.scatter(lips_act,rates,marker = ".",color ="royalblue",alpha = 0.8)
ax.set_xlim(0,1000)
ax.set_ylim(0,1000)
ax=fix_ax_probs(ax,'relative size','Rate')
ax.legend()
fig.tight_layout()
fig.savefig(str(path+'size_v_rate_scatter.pdf'))


rates_no_dopa = rates


data = pd.read_csv('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200819/20200819/k+ indicator 0.5uM dopamin 100mM nacl 100mM Kcl/allrate_fit_data.csv', low_memory=False, sep = ',')


data_no = pd.read_csv('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200819/20200819/k+ indicator no dopamin 100mM nacl 100mM Kcl/allrate_fit_data.csv', low_memory=False, sep = ',')




fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(rates,25,range = (0,700),density = True,color = "gray",alpha =0.8,label = str("Dopa, Rates, N: "+str(len(lips_act))))
ax.hist(rates_no_dopa,25,range = (0,700),density = True,color = "firebrick",alpha =0.8,label = str("Rates, N: "+str(len(lips_act))))

ax=fix_ax_probs(ax,'Rate I/Frame','Density')
ax.legend()
fig.tight_layout()
fig.savefig(str('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200819/20200819/comp_rate.pdf'))


fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(data.sensor_end_relative.values,25,range = (0,200000),density = True,color = "gray",alpha =0.8,label = str("Dopa, Rates, N: "+str(len(lips_act))))
ax.hist(data_no.sensor_end_relative.values,25,range = (0,200000),density = True,color = "firebrick",alpha =0.8,label = str("Rates, N: "+str(len(lips_act))))

ax=fix_ax_probs(ax,'Rate I/Frame','Density')
ax.legend()
fig.tight_layout()
fig.savefig(str('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200819/20200819/dec.pdf'))



fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.scatter(data_no.p_dec.values,data_no.rate.values,marker = ".",color ="royalblue",alpha = 0.8)

ax.set_xlim(0,200000)
ax.set_ylim(0,1000)
ax=fix_ax_probs(ax,'relative size','Rate')
ax.legend()
fig.tight_layout()
fig.savefig(str('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200819/20200819/no_dopa_scatter_amp_vs_rate.pdf'))





def log_norm( x, mu, sigma,scale):
    return  (1. / (sigma * np.sqrt( 2. * np.pi ) ) * np.exp( -( np.log( x ) - mu )**2 / ( 2. * sigma**2 ) ))*scale
 
def single_log_norm(x,s,loc, scale):
    from scipy import signal, stats
    return stats.lognorm.pdf(x, s, loc, scale)



def Convolved_dist(x,rate1,rate2, sigma1, sigma2, mean_transporter,N_max=10,withdopamin=False):
    out = np.zeros(x.shape)
    for i in range(1,N_max+1):
        if not withdopamin:
            out += stats.poisson.pmf(i,mean_transporter)*stats.norm.pdf(x,rate1*i,np.sqrt(i*sigma1))
        else:
            out += stats.poisson.pmf(i,mean_transporter)*stats.norm.pdf(x,rate1*i+rate2*i,np.sqrt(i*sigma1+i*sigma2))
    return out 

fig,ax=plt.subplots(1,1,figsize=(12,8))
xmin,xmax = 0,200
bins=400
x_axis = np.linspace(xmin,xmax,bins)
y_axis = Convolved_dist(x_axis,25,20,100,100,1)
ax.plot(x_axis,y_axis)


#making figures for the paper k indicator

Dopamine_100_100_init = pd.read_csv('/Volumes/Nebuchadnez/Soren/dopamin/publication figures and data/20200626 potasium first purification/100 nacl 100 kcl dopamine/allrate_fit_data.csv', low_memory=False, sep = ',')
No_dopamine_100_100_init = pd.read_csv('/Volumes/Nebuchadnez/Soren/dopamin/publication figures and data/20200707 first purification k indicator 100 nacl 100 kcl no dopamine/allrate_fit_data.csv', low_memory=False, sep = ',')

Dopamine_100_100_second = pd.read_csv('/Volumes/Nebuchadnez/Soren/dopamin/publication figures and data/20200819 second purification potassium data 100 nacl 100 klc/dopamine/allrate_fit_data.csv', low_memory=False, sep = ',')
No_dopamine_100_100_second = pd.read_csv('/Volumes/Nebuchadnez/Soren/dopamin/publication figures and data/20200819 second purification potassium data 100 nacl 100 klc/no dopamine/allrate_fit_data.csv', low_memory=False, sep = ',')



Dopamine_200 = pd.read_csv( '/Volumes/Nebuchadnez/Soren/dopamin/publication figures and data/20200626 potasium first purification/200 nacl dopamine vers 2/allrate_fit_data.csv', low_memory=False, sep = ',')
Dopamine_200_2 = pd.read_csv('/Volumes/Nebuchadnez/Soren/dopamin/publication figures and data/20200626 potasium first purification/200 nack dopamine/allrate_fit_data.csv', low_memory=False, sep = ',')
Dopamine_200 = Dopamine_200.append(Dopamine_200_2)
del(Dopamine_200_2)
No_dopamine_200 = pd.read_csv('/Volumes/Nebuchadnez/Soren/dopamin/publication figures and data/20200626 potasium first purification/200 nacl no dopamine/allrate_fit_data.csv', low_memory=False, sep = ',')

mean_dist_dls =    5.1979 # from mettes data np.log(180.9)
mean_dist_tif = 5.128 # sigma 0.514 from fit

conversion_factor = 1.0136 # multiplied to microscope sizes
path = '/Volumes/Nebuchadnez/Soren/dopamin/publication figures and data/plots_for_publication/plotting all/'

df_frames = [Dopamine_100_100_init,
             No_dopamine_100_100_init,
             Dopamine_100_100_second,
             No_dopamine_100_100_second,
             Dopamine_200,
             No_dopamine_200]

conditions = ['Dopamine_100_100_init',
             'No_dopamine_100_100_init',
             'Dopamine_100_100_second',
             'No_dopamine_100_100_second',
             'Dopamine_200',
             'No_dopamine_200']

df_frames_corr = []
for df in df_frames:
    df = df[df.rate<1000]
    df = df[df.rate>0]
    
    df = df[df.lip_size_relative<1000]
    df = df[df.lip_size_relative>0]
    
    df['rate'] = df['rate']/3.358
    df['lip_size_relative'] = df['lip_size_relative']*1.0136
    df_frames_corr.append(df)

with open(str(path+"ks_tests_rate.txt"), "w") as text_file:
        text_file.write(str('100_100 1st rate Dopamine vs 100_100 1st no dopamine')+  "\n")
        text_file.write(str(stats.ks_2samp(df_frames_corr[0].rate.values,df_frames_corr[1].rate.values))+  "\n\n")
        
        text_file.write(str('100_100 1st rate Dopamine vs 100_100 2nd dopamine')+  "\n")
        text_file.write(str(stats.ks_2samp(df_frames_corr[0].rate.values,df_frames_corr[2].rate.values))+  "\n\n")
        
        text_file.write(str('100_100 1st rate Dopamine vs 200 1st dopamine')+  "\n")
        text_file.write(str(stats.ks_2samp(df_frames_corr[0].rate.values,df_frames_corr[4].rate.values))+  "\n\n")
        
        text_file.write(str('100_100 1st rate No Dopamine vs 100_100 2nd no dopamine')+  "\n")
        text_file.write(str(stats.ks_2samp(df_frames_corr[1].rate.values,df_frames_corr[3].rate.values))+  "\n\n")
        
        text_file.write(str('100_100 1st rate No Dopamine vs 200 1st no dopamine')+  "\n")
        text_file.write(str(stats.ks_2samp(df_frames_corr[1].rate.values,df_frames_corr[5].rate.values))+  "\n\n")
        
        text_file.write(str('100_100 2nd rate No Dopamine vs 100 2nd dopamine')+  "\n")
        text_file.write(str(stats.ks_2samp(df_frames_corr[2].rate.values,df_frames_corr[3].rate.values))+  "\n\n\n\n\n\n")
    
         
        
for df, cond in zip(df_frames_corr,conditions):
    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.hist(df.rate.values,20,range = (0,250),density = True,ec = "gray",color = "gray",alpha =0.7,label = str(cond))
    ax=fix_ax_probs(ax,'Rate [I/s]','Density')
    ax.legend()
    ax.set_ylim(0,0.03)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig(str(path+'__'+cond+'rate.pdf'))
    
    
    features = pd.DataFrame( {'rates'          : df.rate.values}) 
    features.to_csv(str(path+cond+'rate.csv'), header=True, index=None, sep=',', mode='w')
    
    
    shape, loc, scale = stats.lognorm.fit(df.rate.values)
    x_vals = np.linspace(1,250,1000)
    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.hist(df.rate.values,20,range = (0,250),density = True,ec = "gray",color = "gray",alpha =0.7,label = str(cond))
    ax.plot(x_vals,single_log_norm(x_vals,shape, loc, scale),color = "firebrick",linestyle = "--",alpha =0.8,label = "Fit")
    ax.legend()
    ax.set_ylim(0,0.04)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig(str(path+'__'+cond+'rate_fit_lognorm.pdf'))
    
    with open(str(path+cond+"rate_log_norm_fit.txt"), "w") as text_file:
        text_file.write(str(str('shape, loc, scale'))+  "\n")
        text_file.write(str(str(shape)+'  '+str(loc)+'  '+str(scale))+  "\n\n")

    
        
    
    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.hist(df.lip_size_relative.values,20,range = (0,600),density = True,ec = "black",color = "black",alpha =0.7,label = str(cond))
    
    #ax.hist(rates_to_plot2,20,range = (0,600),density = True,ec = "gray",color = "gray",alpha =0.7,label = str('No Dopamine'))
    #ax.hist(rates_to_plot,20,range = (0,600),density = True,ec = "firebrick",color = "firebrick",alpha =0.7,label = str('Dopamine'))
    ax=fix_ax_probs(ax,'Size [nm]','Density')
    ax.legend()
    #ax.set_ylim(0,0.03)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig(str(path+'__'+cond+'size.pdf'))
    features = pd.DataFrame( {'size'          : df.lip_size_relative.values}) 
    features.to_csv(str(path+cond+'size.csv'), header=True, index=None, sep=',', mode='w')


    fig,ax = plt.subplots(figsize = (2.5,2.5))
    ax.scatter(df.lip_size_relative.values,df.rate.values,marker = ".",color ="gray",alpha = 0.8)
    ax.set_xlim(0,600)
    ax.set_ylim(0,250)
    ax=fix_ax_probs(ax,'Size [nm]','Rate [I/s]')
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(path+'__'+cond+'scatter.pdf'))
    
    features = pd.DataFrame( {'size'          : df.lip_size_relative.values,
                              'rate':df.rate.values}) 
    features.to_csv(str(path+cond+'size_to_rate.csv'), header=True, index=None, sep=',', mode='w')
    plt.close('all')


    print (stats.pearsonr(df.lip_size_relative.values,df.rate.values))
    with open(str(path+cond+"size_to_rate_corr.txt"), "w") as text_file:
        text_file.write(str(str(cond))+  "\n")
        text_file.write(str(str(stats.pearsonr(df.lip_size_relative.values,df.rate.values))+  "\n\n"))

#sorting 



######## SIZES #########
######################33


name = str(condition+'decrease_rate_no_dopamine')


fig,ax = plt.subplots(figsize = (2.5,2.5))

ax.hist(rates_to_plot2,20,range = (0,250),density = True,ec = "gray",color = "gray",alpha =0.7,label = str('No Dopamine'))
#ax.hist(rates_to_plot,20,range = (0,250),density = True,ec = "firebrick",color = "firebrick",alpha =0.7,label = str('Dopamine'))
ax=fix_ax_probs(ax,'I/s','Density')
ax.legend()
ax.set_ylim(0,0.03)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.tight_layout()
fig.savefig(str(path+'__'+name+'.pdf'))

features = pd.DataFrame( {'rates'          : rates_to_plot2}) 
features.to_csv(str(path+name+'.csv'), header=True, index=None, sep=',', mode='w')     
 

# fit rates with log norm dist

fit_sizes = np.asarray(rates_to_plot)

compdf1 = probfit.functor.AddPdfNorm(log_norm)


shape, loc, scale = stats.lognorm.fit(df_frames_corr[0].rate.values)
shape2, loc2, scale2 = stats.lognorm.fit(df_frames_corr[1].rate.values)
    
ulh1 = UnbinnedLH(single_log_norm, fit_sizes, extended=False)
import iminuit
m1 = iminuit.Minuit(ulh1,mu=1.4,sigma=14.9,scale = 0.4,pedantic= False,print_level = 0)
m1.migrad(ncall=30000)


dop =np.log(df_frames_corr[0].rate.values)
nodop =np.log(df_frames_corr[1].rate.values)

x_vals = np.linspace(1,250,1000)
fig,ax = plt.subplots(2,1,figsize = (5,2.5))
ax[0].hist(df_frames_corr[0].rate.values,20,range = (0,250),density = True,ec = "gray",color = "gray",alpha =0.7,label = str('Dopamine'))
ax[1].hist(df_frames_corr[1].rate.values,20,range = (0,250),density = True,ec = "gray",color = "gray",alpha =0.7,label = str('No Dopamine'))



ax[0].plot(x_vals,single_log_norm(x_vals,shape, loc, scale))

ax[1].plot(x_vals,single_log_norm(x_vals,shape2, loc2, scale2))

######## SIZES #########
######################33

mean_dist_dls =    5.1979 # from mettes data np.log(180.9)
mean_dist_tif = 5.128 # sigma 0.514 from fit

conversion_factor = 1.0136 # multiplied to microscope sizes


name = str(condition+'lip_size_all')





fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(np.asarray(all_lips)*1.0136,20,range = (0,600),density = True,ec = "black",color = "black",alpha =0.7,label = str('All liposomes'))

#ax.hist(rates_to_plot2,20,range = (0,600),density = True,ec = "gray",color = "gray",alpha =0.7,label = str('No Dopamine'))
#ax.hist(rates_to_plot,20,range = (0,600),density = True,ec = "firebrick",color = "firebrick",alpha =0.7,label = str('Dopamine'))
ax=fix_ax_probs(ax,'Size [nm]','Density')
ax.legend()
#ax.set_ylim(0,0.03)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.tight_layout()
fig.savefig(str(path+'__'+name+'.pdf'))

features = pd.DataFrame( {'lip_size'          : rates_to_plot}) 
features.to_csv(str(path+name+'.csv'), header=True, index=None, sep=',', mode='w')     






# plot some nice traes
"""

path ='/Users/sorensnielsen/Documents/Projects/Dopamine/article plots/'
name = "representative_traces_potassium_w_fit"
tmp = pd.read_csv('/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/k indicator 100mM nacl 100kcl 0.5uM dopamin/tifs/_0_signal_df_corrected_for_lips.csv', low_memory=False, sep = ',')
list(tmp)


tmp =tmp[tmp.particle ==146]



fit = fit_rate(tmp.green_int_corrected.values[40:200])    
x_vals = np.linspace(0,len(tmp.green_int_corrected.values[40:]),1000)


fig,ax = plt.subplots(figsize = (5,2))
ax.plot((1/60)*(tmp.frame.values*3.358),tmp.green_int_corrected.values, color = "royalblue", alpha = 0.8,label = "K+ signal")
ax.plot((1/60)*(x_vals+40)*3.358,fit_func2(x_vals,fit[0],fit[1],fit[2]),color = "black", linestyle = '--',linewidth = 3,alpha =0.8, label = "Fit")

ax=fix_ax_probs(ax,'Time [min]','Intensity [A.U.]')
ax.legend()
ax.set_ylim(5e4,20e4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(str(path+'__'+name+'.pdf'))


features = pd.DataFrame( {'time'          : (1/60)*(tmp.frame.values*3.358),
                          'signal':tmp.green_int_corrected.values}) 
features.to_csv(str(path+'/chart_data/'+name+'.csv'), header=True, index=None, sep=',', mode='w')     


"""







"""



Dopamine = pd.read_csv( '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/k indicator 100mM nacl 100kcl 0.5uM dopamin/allrate_fit_data_tester.csv', low_memory=False, sep = ',')
no_Dopamine = pd.read_csv( '/Volumes/Nebuchadnez/Soren/dopamin/new_encap/20200910 mette/k indicator 100mM nacl 100kcl no dopamin/allrate_fit_data_tester.csv', low_memory=False, sep = ',')

fig,ax = plt.subplots(figsize = (2.5,2.5))
ax.hist(Dopamine.rate.values,20,density = True,ec = "gray",color = "gray",alpha =0.7,label = str('Dopamine'))
ax.hist(no_Dopamine.rate.values,20,density = True,ec = "firebrick",color = "firebrick",alpha =0.7,label = str('No Dopamine'))
ax=fix_ax_probs(ax,'Rate [I/s]','Density')


ax.legend()
ax.set_ylim(0,0.03)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.tight_layout()




















