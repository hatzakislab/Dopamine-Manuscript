

import os
import time

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
from tqdm import tqdm

lip_int_size = 30   
lip_BG_size = 60







sep = 15     # afstand mellem centrum af to liposomer   skal være 15
mean_multiplier = 1.5  #  correleret med antal liposomer, hvor int skal liposomet være for at blive accepteret  skal være 1.5
sigmas = 0.9   # 
memory = 10   #frame   # var 10 da jeg treatede
search_range = 10 # pixels   
duration_min = 20 # min duration of track
appear_min = 50   # maske liposomer som opstår efter frame bliver ikke talt med   # skulle være 5  var 50



#first reverse green videos

def image_loader_video(video):
    from skimage import io
    images_1 = io.imread(video)
    return np.asarray(images_1)  # fjern frame 1 


def green_video_reverser(vid_save_path):
    print ('Fixing vid: ',str(vid_save_path))
    vid = image_loader_video(vid_save_path)
    vid = np.asarray(vid)
    vid = vid[::-1]
    from tifffile import imsave
    imsave(str(vid_save_path), vid)

def ext_pir(x, y, frame):
    x, y, frame = map(np.asarray, [x, y, frame])

    mn, mx = frame.min(), frame.max() + 1
    d = np.diff(np.append(frame, mx))
    r = np.arange(len(frame))
    i = r.repeat(d)

    return x[i], y[i], np.arange(mn, mx)


def extend_arrs(x, y, frame):
    # Convert to arrays
    frame = np.asarray(frame)
    x = np.asarray(x)
    y = np.asarray(y)

    l = frame[-1] - frame[0] + 1
    id_ar = np.zeros(l, dtype=int)
    id_ar[frame - frame[0]] = 1
    idx = id_ar.cumsum() - 1
    return np.r_[frame[0]:frame[-1] + 1], x[idx], y[idx]

def position_extractor(tracked, max_length):
    x_pos = []
    y_pos = []
    frames = []
    names = []

    group_all = tracked.groupby('particle')
    for name, group in group_all:
        frame = group.frame.tolist()

        frame = frame[0:(len(frame) - 3)]
        tmp = max_length - 1
        frame.append(tmp)

        # frame = [0,1,2,(max_length-1)]

        x_tmp = group.x.tolist()
        y_tmp = group.y.tolist()
        frames_full = np.arange(min(frame), max(frame) + 1, 1)

        frame, x, y = extend_arrs(x_tmp, y_tmp, frame)
        # x,y,frame = ext_pir(x_tmp, y_tmp, frame)
        x_pos.extend(x)
        y_pos.extend(y)
        frames.extend(frame)
        names.extend([name] * len(x))

    final_df = pd.DataFrame(
        {'particle': names,
         'frame': frames,
         'x': x_pos,
         'y': y_pos})
    return final_df

def get_video_files(main_folder_path):
    files = glob(str(main_folder_path+'*.tif'))        
    for file in files:
        if file.find('green')      != -1:
            green = file
        elif file.find('blue')      != -1:
            blue = file
        elif file.find('red')      != -1:
            red = file    
    return [red,green,blue]  


def tracker(video, mean_multiplier, sep):
    mean = np.mean(video[0])
    print ('tracking')
    full = tp.batch(video, 11, minmass=mean * mean_multiplier, separation=sep,noise_size = 3);
    print ('1000')
    # check for subpixel accuracy
    tp.subpx_bias(full)

    full_tracked = tp.link_df(full, search_range, memory=memory)

    full_tracked['particle'] = full_tracked['particle'].transform(int)
    full_tracked['duration'] = full_tracked.groupby('particle')['particle'].transform(len)
    full_tracked['t_appear'] = full_tracked.groupby('particle')['frame'].transform(min)

    full_tracked = full_tracked[full_tracked.duration > duration_min]
    full_tracked = full_tracked[full_tracked.t_appear < appear_min]

    return full_tracked



def fix_green(green_vid):
    
    new= []
    for i in range(len(green_vid)):
        for j in range(10):
            new.append(green_vid[i])
            
    return np.asarray(new, dtype=np.float32)
 
def signal_extractor(video, final_df, red_blue,roi_size,bg_size):  # change so taht red initial is after appearance timing
    lip_int_size= roi_size
    lip_BG_size = bg_size
    def cmask(index, array, BG_size, int_size):
        a, b = index
        nx, ny = array.shape
        y, x = np.ogrid[-a:nx - a, -b:ny - b]
        mask = x * x + y * y <= lip_int_size  # radius squared - but making sure we dont do the calculation in the function - slow
        mask2 = x * x + y * y <= lip_int_size   # to make a "gab" between BG and roi
    
        BG_mask = (x * x + y * y <= lip_BG_size)
        BG_mask = np.bitwise_xor(BG_mask, mask2)
        return (sum((array[mask]))), np.median(((array[BG_mask])))

    final_df = final_df.sort_values(['particle', 'frame'], ascending=True)

    def df_extractor2(row):
        b, a = row['x'], row['y'] #b,a
        frame = int(row['frame'])
        array = video[frame]
        nx, ny = array.shape
        y, x = np.ogrid[-a:nx - a, -b:ny - b]
        mask = x * x + y * y <= lip_int_size  # radius squared - but making sure we dont do the calculation in the function - slow
        mask2 = x * x + y * y <= lip_int_size  # to make a "gab" between BG and roi
        BG_mask = (x * x + y * y <= lip_BG_size)
        BG_mask = np.bitwise_xor(BG_mask, mask2)

        return np.sum((array[mask])), np.median(((array[BG_mask]))) # added np in sum

    size_maker = np.ones(video[0].shape)
    ind = 25, 25  # dont ask - leave it here, it just makes sure the below runs
    mask_size, BG_size = cmask(ind, size_maker, lip_BG_size, lip_int_size)
    mask_size = np.sum(mask_size)

    a = final_df.apply(df_extractor2, axis=1)
    # a = df_extractor2(final_df, video)

    intensity = []
    bg = []
    for line in a:
        i, b = line
        bg.append(b)
        intensity.append(i)
    if red_blue == 'blue' or red_blue == 'Blue':
        final_df['np_int'] = intensity
        final_df['np_bg'] = bg
        final_df['np_int_corrected'] = (final_df['np_int']/mask_size) - (final_df['np_bg'])
    elif red_blue == 'red' or red_blue == 'Red':
        final_df['lip_int'] = intensity
        final_df['lip_bg'] = bg
        final_df['lip_int_corrected'] = (final_df['lip_int']/mask_size) - (final_df['lip_bg'])
    else:
        final_df['green_int'] = intensity
        final_df['green_bg'] = bg
        final_df['green_int_corrected'] = (final_df['green_int']/mask_size) - (final_df['green_bg'])

    return final_df
def signal_extractor_no_pos(video, final_df, red_blue,roi_size,bg_size):  # change so taht red initial is after appearance timing
    lip_int_size= roi_size
    lip_BG_size = bg_size
    def cmask(index, array, BG_size, int_size):
        a, b = index
        nx, ny = array.shape
        y, x = np.ogrid[-a:nx - a, -b:ny - b]
        mask = x * x + y * y <= lip_int_size  # radius squared - but making sure we dont do the calculation in the function - slow
        mask2 = x * x + y * y <= lip_int_size   # to make a "gab" between BG and roi
    
        BG_mask = (x * x + y * y <= lip_BG_size)
        BG_mask = np.bitwise_xor(BG_mask, mask2)
        return (sum((array[mask]))), np.median(((array[BG_mask])))

    final_df = final_df.sort_values(['frame'], ascending=True)

    def df_extractor2(row):
        b, a = row['x'], row['y'] #b,a
        frame = int(row['frame'])
        array = video[frame]
        nx, ny = array.shape
        y, x = np.ogrid[-a:nx - a, -b:ny - b]
        mask = x * x + y * y <= lip_int_size  # radius squared - but making sure we dont do the calculation in the function - slow
        mask2 = x * x + y * y <= lip_int_size  # to make a "gab" between BG and roi
        BG_mask = (x * x + y * y <= lip_BG_size)
        BG_mask = np.bitwise_xor(BG_mask, mask2)

        return np.sum((array[mask])), np.median(((array[BG_mask]))) # added np in sum

    size_maker = np.ones(video[0].shape)
    ind = 25, 25  # dont ask - leave it here, it just makes sure the below runs
    mask_size, BG_size = cmask(ind, size_maker, lip_BG_size, lip_int_size)
    mask_size = np.sum(mask_size)

    a = final_df.apply(df_extractor2, axis=1)
    # a = df_extractor2(final_df, video)

    intensity = []
    bg = []
    for line in a:
        i, b = line
        bg.append(b)
        intensity.append(i)
    if red_blue == 'blue' or red_blue == 'Blue':
        final_df['blue_int'] = intensity
        final_df['blue_bg'] = bg
        final_df['blue_int_corrected'] = (final_df['blue_int']) - (final_df['blue_bg']*mask_size)
    elif red_blue == 'red' or red_blue == 'Red':
        final_df['red_int'] = intensity
        final_df['red_bg'] = bg
        final_df['red_int_corrected'] = (final_df['red_int']) - (final_df['red_bg']*mask_size)
    else:
        final_df['green_int'] = intensity
        final_df['green_bg'] = bg
        final_df['green_int_corrected'] = (final_df['green_int']) - (final_df['green_bg']*mask_size)

    return final_df
   
   
def big_red_fix(red):
    new = []
    for i in range(len(red)):
        if i %9 ==0:
            new.append(red[i])
            new.append(red[i])
        else:
            new.append(red[i])
    return np.asarray(new)    
        
def retreater(df,video,main_folder):  
    df = df.sort_values(['particle', 'frame'], ascending=True)
    x_pos_final = np.asarray(df['x'].tolist())
    y_pos_final = np.asarray(df['y'].tolist())    

  
    video_g = video
    video_g,pos = cut_video(x_pos_final[0],y_pos_final[0],video_g)    


    from tifffile import imsave    
    
    video_g=np.asarray(video_g, dtype=np.float32)

    video_g = fix_green(video_g)
    
    imsave(str(main_folder+'green_vbig.tif'), video_g)


def cmask_plotter(index, array, BG_size, int_size):
    a, b = index
    nx, ny = array.shape
    y, x = np.ogrid[-a:nx - a, -b:ny - b]
    mask = x * x + y * y <= lip_int_size  # radius squared - but making sure we dont do the calculation in the function - slow
    mask2 = x * x + y * y <= lip_int_size   # to make a "gab" between BG and roi

    BG_mask = (x * x + y * y <= lip_BG_size)
    BG_mask = np.bitwise_xor(BG_mask, mask2)
    return mask,BG_mask
    


def step_tracker(df):
    microns_per_pixel   = 1
    steps = []
    msd = []
    lag = []
    df['x'] = df['x']* microns_per_pixel 
    df['y'] = df['y']* microns_per_pixel 
    group_all = df.groupby('particle')
    x_step = []
    y_step = []
    
    # easiest: compute step in x, step in y and then steps
    for name, group in group_all:
        x_list = group.x.tolist()
        x_tmp = [y - x for x,y in zip(x_list,x_list[1:])] 
        x_tmp.insert(0, 0.)
        
        y_list = group.y.tolist()
        y_tmp = [y - x for x,y in zip(y_list,y_list[1:])]
        y_tmp.insert(0, 0.)
        y_step.extend(y_tmp)
        x_step.extend(x_tmp)
        step_tmp = [np.sqrt(y**2+x**2) for y,x in zip(y_tmp,x_tmp)]
        
        #msd_tmp,lag_tmp = msd_straight_forward(x_tmp,y_tmp)
        
        #msd.extend(msd_tmp)
        #lag.extend(lag_tmp)
        steps.extend(step_tmp)  
        
    df['x_step'] = x_step  
    df['y_step'] = y_step 
    df['steplength'] = steps 
    #df['lag'] = lag
    #df['msd'] = msd
    return   df   

def get_meanx_y_df(df):
    df = step_tracker(df)
    df = df.sort_values(['frame'], ascending=True)
    grp = df.groupby('frame')
    x_list = []
    y_list = []
    for name,df_t in grp:
        x_list.append(np.mean(df_t['x_step']))
        y_list.append(np.mean(df_t['y_step']))
    return x_list,y_list




def fix_drift(video,df):
    x,y = get_meanx_y_df(df)
    x = np.cumsum(x)
    y = np.cumsum(y)
    tmp = np.zeros((len(video),512+50,512+50))   #512+100
    
    
    for i in range(len(video)):
        x_offset = 25-int(y[i])  # 0 would be what you wanted
        y_offset =25-int(x[i])  # 0 in your case
        
        tmp[i][x_offset:512+x_offset,y_offset:512+y_offset] = video[i]
        tmp[i][x_offset:512+x_offset,y_offset:512+y_offset] = video[i]   # skal nok slettes
        
        mask = tmp[i]==0
        mask2 = tmp[i]!=0
        
        mean = np.mean(tmp[i][mask2],dtype = float)
        std = np.std(tmp[i][mask2],dtype = float)
        
        c = np.count_nonzero(mask)
        nums = np.random.normal(mean,std, int(c))
        
        tmp[i][mask] = nums
                   
    return np.asarray(tmp,dtype=np.float32)
    


reds =[]
greens =[]
blues = []

def combined_fixer(reds,greens,save_path):
    from tifffile import imsave
    mean_multiplier = 0.5
    counter = 0
    for red,green in tqdm(zip(reds,greens)):
        print ('retrack_single_combined')
   
        video_r = image_loader_video(red)
        print ('1')
        video_g = image_loader_video(green)
        print ('2')
        video_blend =  video_r + video_g
        print ('3')
        
        final_df=tracker(video_blend, mean_multiplier, sep)
        print ('4')
        max_length = len(video_blend)
        final_df = position_extractor(final_df, max_length)
        print ('5')
        
        video_load = image_loader_video(red)
        print ('6')
        print (red)
        video_load=fix_drift(video_load,final_df)
        
        
        if not os.path.exists(str(save_path+str(counter)+'/')):                      # creates a folder for saving the stuff in if it does not exist
            os.makedirs(str(save_path+str(counter)+'/'))
        
        imsave(str(save_path+str(counter)+'/fixed_red.tif'), video_load)
        
        
        video_load = image_loader_video(green)
        video_load = fix_drift(video_load,final_df)
        imsave(str(save_path+str(counter)+'/fixed_green.tif'), video_load)
        
        del(video_r)
        counter +=1        
     
        
def pure_signal_extraction(reds,greens,save_path,color_plot):
    mean_multiplier = 0.5 # was 0.5
    counter = 0
    for red,green in tqdm(zip(reds,greens)):
        print ('pure signal')
   
        video_r = image_loader_video(red)
        video_g = image_loader_video(green)
        
        mean=np.mean(video_r[10])
        f = tp.locate(video_r[10], 11, minmass=mean * mean_multiplier, separation=sep,noise_size =3)
       
        f['particle'] = range(len(f))
        
        
        new =  pd.DataFrame() 
        
        
        for i in range(len(video_r)):
            f['frame'] = i
            new= new.append(f, ignore_index = True)
            

            
        new = signal_extractor_no_pos(video_g, new, 'green',lip_int_size,lip_BG_size)
        

        new = signal_extractor_no_pos(video_r, new, 'red',lip_int_size,lip_BG_size)
        
            
        new.to_csv(str(save_path+'_'+str(counter)+'_signal_df.csv'), header=True, index=None, sep=',', mode='w')
         
        grp = new.groupby('particle')
        import matplotlib
        my_cmap = matplotlib.cm.Reds
        my_cmap.set_under('k', alpha=0)
        my_cmap.set_bad('k', alpha=0)
        
        my_cmap2 = matplotlib.cm.Greens
        my_cmap2.set_under('k', alpha=0)
        my_cmap2.set_bad('k', alpha=0)
        
        size_maker = np.ones(video_r[0].shape)
        for par,dat in tqdm(grp):
            
            x_aq = dat.x.values[10]
    
            y_aq = dat.y.values[10]
            ind = (y_aq,x_aq)  # dont ask - leave it here, it just makes sure the below runs
            mask, BG_mask = cmask_plotter(ind, size_maker, lip_BG_size, lip_int_size)
            
            #fig,ax =plt.subplots(3,2,figsize = (10,8))
            #ax[0,0].plot(dat.frame.values,dat.green_int_corrected.values,color = color_plot,alpha =0.8,label ='Corrected sensor')
            #ax[0,1].plot(dat.frame.values,dat.red_int_corrected.values,color = "firebrick",alpha =0.8,label ='Corrected Lip')           
            #ax[1,0].plot(dat.frame.values,dat.green_int.values,color = color_plot,alpha =0.8,label ='Raw sensor')
            #ax[1,1].plot(dat.frame.values,dat.red_int.values,color = "firebrick",alpha =0.8,label ='raw Lipo') 
            #ax[0,0].legend()
            #ax[0,1].legend()
            #ax[1,0].legend()
            #ax[0,0].legend()
            
            #ax[2,1].hist(new.red_int_corrected.values**0.5,30,range =(0,800),color = "firebrick",alpha =0.8,label ='All sizes')
            #ax[2,1].axvline(np.mean(dat.red_int_corrected.values[:10])**0.5,color = "black", label = "Current liposome")
            #ax[2,1].legend()
            
            #ax[2,0].imshow(video_r[10])
            #ax[2,0].imshow(BG_mask , cmap=my_cmap2,interpolation='none',vmin = 0.1,alpha = 0.4)
            #ax[2,0].set_xlim(x_aq-20,x_aq+20)
            #ax[2,0].set_ylim(y_aq-20,y_aq+20)
            
            
            
            #fig.tight_layout()
            #fig.savefig(str(save_path+str(counter)+'_'+str(par)+'_.pdf'))
            #plt.close('all')
        counter +=1
def combined_fixer_single(reds,save_path):
    from tifffile import imsave
    mean_multiplier = 0.5
    counter = 0
    for red in reds:
        print ('retrack_single_combined')
   
        video_r = image_loader_video(red)

        video_blend =  video_r + video_r

        final_df=tracker(video_blend, mean_multiplier, sep)

        max_length = len(video_blend)
        final_df = position_extractor(final_df, max_length)

        
        video_load = image_loader_video(red)

        video_load=fix_drift(video_load,final_df)
        
        
        if not os.path.exists(str(save_path+str(counter)+'/')):                      # creates a folder for saving the stuff in if it does not exist
            os.makedirs(str(save_path+str(counter)+'/'))
        
        imsave(str(save_path+str(counter)+'/fixed_red.tif'), video_load)
        
        
        del(video_r)
        counter +=1        
     


def plotter_of_data(paths,save_path,reds_fil,color_plot):
    counter =0
    for df in paths:
        data = pd.read_csv(df, low_memory=False, sep = ',')
        new = data
        
        data = data[data.red_int_corrected >0 ]
        data = data[data.y > 25 ]
        data = data[data.y < 512+25 ]
        data = data[data.x > 25 ]
        data = data[data.x < 512+35 ]
         
        
        df2 = df[:-4]
        df2 = str(df2+'_corrected_for_lips.csv')
        data.to_csv(str(df2), header=True, index=None, sep=',', mode='w')
    
        grp = new.groupby('particle')
        import matplotlib
        my_cmap = matplotlib.cm.Reds
        my_cmap.set_under('k', alpha=0)
        my_cmap.set_bad('k', alpha=0)
        
        my_cmap2 = matplotlib.cm.Greens
        my_cmap2.set_under('k', alpha=0)
        my_cmap2.set_bad('k', alpha=0)
        video_r = image_loader_video(reds_fil[counter])
        size_maker = np.ones(video_r[0].shape)
        for par,dat in tqdm(grp):
            a = np.mean(dat.red_int_corrected.values[:10])
            if a<0:
                continue
            x_aq = dat.x.values[10]
    
            y_aq = dat.y.values[10]
            ind = (y_aq,x_aq)  # dont ask - leave it here, it just makes sure the below runs
            mask, BG_mask = cmask_plotter(ind, size_maker, lip_BG_size, lip_int_size)
            
            fig,ax =plt.subplots(3,2,figsize = (10,8))
            ax[0,0].plot(dat.frame.values,dat.green_int_corrected.values,color = color_plot,alpha =0.8,label ='Corrected sensor')
            ax[0,1].plot(dat.frame.values,dat.red_int_corrected.values,color = "firebrick",alpha =0.8,label ='Corrected Lip')           
            ax[1,0].plot(dat.frame.values,dat.green_int.values,color = color_plot,alpha =0.8,label ='Raw sensor')
            ax[1,1].plot(dat.frame.values,dat.red_int.values,color = "firebrick",alpha =0.8,label ='raw Lipo') 
            ax[0,0].legend()
            ax[0,1].legend()
            ax[1,0].legend()
            ax[0,0].legend()
            
            #ax[2,1].hist(new.red_int_corrected.values**0.5,30,range =(0,800),color = "firebrick",alpha =0.8,label ='All sizes')
            ax[2,1].axvline(np.mean(dat.red_int_corrected.values[:10])**0.5,color = "black", label = "Current liposome")
            ax[2,1].legend()
            
            ax[2,0].imshow(video_r[10])
            ax[2,0].imshow(BG_mask , cmap=my_cmap2,interpolation='none',vmin = 0.1,alpha = 0.4)
            ax[2,0].set_xlim(x_aq-20,x_aq+20)
            ax[2,0].set_ylim(y_aq-20,y_aq+20)
            
            
            
            fig.tight_layout()
            fig.savefig(str(save_path+str(counter)+'_'+str(par)+'_.pdf'))
            plt.close('all')
        counter +=1
def pure_signal_extraction_single(reds,save_path):
    mean_multiplier = 0.5 # was 0.5
    counter = 0
    for red in reds:
        print ('pure signal')
   
        video_r = image_loader_video(red)
        
        mean=np.mean(video_r[10])
        f = tp.locate(video_r[10], 11, minmass=mean * mean_multiplier, separation=sep,noise_size =3)
       
        f['particle'] = range(len(f))
        
        
        new =  pd.DataFrame() 
        
        
        for i in range(len(video_r)):
            f['frame'] = i
            new= new.append(f, ignore_index = True)
            

        new = signal_extractor_no_pos(video_r, new, 'red',lip_int_size,lip_BG_size)
        
            
        new.to_csv(str(save_path+'_'+str(counter)+'_signal_df.csv'), header=True, index=None, sep=',', mode='w')
         
        counter +=1
        
def plotter_of_data_single(paths,save_path,reds_fil):
    counter =0
    for df in paths:
        data = pd.read_csv(df, low_memory=False, sep = ',')
        new = data
        
        data = data[data.red_int_corrected >0 ]
        data = data[data.y > 25 ]
        data = data[data.y < 512+25 ]
        data = data[data.x > 25 ]
        data = data[data.x < 512+35 ]
         
        
        df2 = df[:-4]
        df2 = str(df2+'_corrected_for_lips.csv')
        data.to_csv(str(df2), header=True, index=None, sep=',', mode='w')
    
        grp = new.groupby('particle')
        import matplotlib
        my_cmap = matplotlib.cm.Reds
        my_cmap.set_under('k', alpha=0)
        my_cmap.set_bad('k', alpha=0)
        
        my_cmap2 = matplotlib.cm.Greens
        my_cmap2.set_under('k', alpha=0)
        my_cmap2.set_bad('k', alpha=0)
        video_r = image_loader_video(reds_fil[counter])
        size_maker = np.ones(video_r[0].shape)
        for par,dat in tqdm(grp):
            a = np.mean(dat.red_int_corrected.values[:10])
            if a<0:
                continue
            x_aq = dat.x.values[10]
    
            y_aq = dat.y.values[10]
            ind = (y_aq,x_aq)  # dont ask - leave it here, it just makes sure the below runs
            mask, BG_mask = cmask_plotter(ind, size_maker, lip_BG_size, lip_int_size)
            
            fig,ax =plt.subplots(2,2,figsize = (10,8))
            
            ax[0,0].plot(dat.frame.values,dat.red_int_corrected.values,color = "firebrick",alpha =0.8,label ='Corrected Lip')           
            ax[1,0].plot(dat.frame.values,dat.red_int.values,color = "firebrick",alpha =0.8,label ='raw Lipo') 
            ax[0,0].legend()
            ax[1,0].legend()
            
            ax[0,1].imshow(video_r[10])
            ax[0,1].imshow(BG_mask , cmap=my_cmap2,interpolation='none',vmin = 0.1,alpha = 0.4)
            ax[0,1].set_xlim(x_aq-20,x_aq+20)
            ax[0,1].set_ylim(y_aq-20,y_aq+20)
            
            
            
            fig.tight_layout()
            fig.savefig(str(save_path+str(counter)+'_'+str(par)+'_.pdf'))
            plt.close('all')
        counter +=1        
############3

#SRB fusion 1120ms temp




tifs = ['/Volumes/Nebuchadnez/Mette/Mette NP2/20200928/0.05% detergent 0.76pmol NP with label/tifs/']

for tif_path in tqdm(tifs):
    
    
    reds = [ '/Volumes/Nebuchadnez/Mette/Mette NP2/20200928/0.05% detergent 0.76pmol NP with label/tifs/Experiment_Process_001_20200928.tif', '/Volumes/Nebuchadnez/Mette/Mette NP2/20200928/0.05% detergent 0.76pmol NP with label/tifs/Experiment_Process_003_20200928.tif', '/Volumes/Nebuchadnez/Mette/Mette NP2/20200928/0.05% detergent 0.76pmol NP with label/tifs/Experiment_Process_005_20200928.tif', '/Volumes/Nebuchadnez/Mette/Mette NP2/20200928/0.05% detergent 0.76pmol NP with label/tifs/Experiment_Process_007_20200928.tif', '/Volumes/Nebuchadnez/Mette/Mette NP2/20200928/0.05% detergent 0.76pmol NP with label/tifs/Experiment_Process_009_20200928.tif', '/Volumes/Nebuchadnez/Mette/Mette NP2/20200928/0.05% detergent 0.76pmol NP with label/tifs/Experiment_Process_011_20200928.tif']    
    greens = ['/Volumes/Nebuchadnez/Mette/Mette NP2/20200928/0.05% detergent 0.76pmol NP with label/tifs/Experiment_Process_002_20200928.tif', '/Volumes/Nebuchadnez/Mette/Mette NP2/20200928/0.05% detergent 0.76pmol NP with label/tifs/Experiment_Process_004_20200928.tif', '/Volumes/Nebuchadnez/Mette/Mette NP2/20200928/0.05% detergent 0.76pmol NP with label/tifs/Experiment_Process_006_20200928.tif', '/Volumes/Nebuchadnez/Mette/Mette NP2/20200928/0.05% detergent 0.76pmol NP with label/tifs/Experiment_Process_008_20200928.tif', '/Volumes/Nebuchadnez/Mette/Mette NP2/20200928/0.05% detergent 0.76pmol NP with label/tifs/Experiment_Process_010_20200928.tif', '/Volumes/Nebuchadnez/Mette/Mette NP2/20200928/0.05% detergent 0.76pmol NP with label/tifs/Experiment_Process_012_20200928.tif']
    combined_fixer(reds,greens,tif_path) 

## signal extractoion

for tif_path in tqdm(tifs):
    
    reds = [ str(tif_path+'0/fixed_red.tif'), 
             str(tif_path+'1/fixed_red.tif'), 
             str(tif_path+'2/fixed_red.tif'), 
             str(tif_path+'3/fixed_red.tif'),
             str(tif_path+'4/fixed_red.tif'), 
             str(tif_path+'5/fixed_red.tif')]
    
    greens = [ str(tif_path+'0/fixed_green.tif'), 
               str(tif_path+'1/fixed_green.tif'), 
               str(tif_path+'2/fixed_green.tif'), 
               str(tif_path+'3/fixed_green.tif'),
               str(tif_path+'4/fixed_green.tif'), 
               str(tif_path+'5/fixed_green.tif')]
    
    pure_signal_extraction(reds,greens,  tif_path, "royalblue")



tif_path = '/Volumes/Nebuchadnez/Mette/Mette NP2/20200928/control with 0.1% detergent/tifs/'
tifs =['/Volumes/Nebuchadnez/Mette/Mette NP2/20200928/control with 0.1% detergent/tifs/Experiment_Process_001_20200928.tif', '/Volumes/Nebuchadnez/Mette/Mette NP2/20200928/control with 0.1% detergent/tifs/Experiment_Process_002_20200928.tif', '/Volumes/Nebuchadnez/Mette/Mette NP2/20200928/control with 0.1% detergent/tifs/Experiment_Process_003_20200928.tif', '/Volumes/Nebuchadnez/Mette/Mette NP2/20200928/control with 0.1% detergent/tifs/Experiment_Process_004_20200928.tif', '/Volumes/Nebuchadnez/Mette/Mette NP2/20200928/control with 0.1% detergent/tifs/Experiment_Process_005_20200928.tif', '/Volumes/Nebuchadnez/Mette/Mette NP2/20200928/control with 0.1% detergent/tifs/Experiment_Process_006_20200928.tif']

combined_fixer_single(tifs,tif_path) 

reds = [ str(tif_path+'0/fixed_red.tif'), 
             str(tif_path+'1/fixed_red.tif'), 
             str(tif_path+'2/fixed_red.tif'), 
             str(tif_path+'3/fixed_red.tif'),
             str(tif_path+'4/fixed_red.tif'), 
             str(tif_path+'5/fixed_red.tif')]

pure_signal_extraction_single(reds,  tif_path)


tifs = ['/Volumes/Nebuchadnez/Mette/Mette NP2/20200928/0.05% detergent 0.76pmol NP with label/tifs/']

for save_path in tqdm(tifs):
    print ("Plotting")
    color_plot = "royalblue"
    
    paths = glob(str(save_path+'*.csv'))   
    reds_fil = [ str(save_path+'0/fixed_red.tif'), 
                 str(save_path+'1/fixed_red.tif'), 
                 str(save_path+'2/fixed_red.tif'), 
                 str(save_path+'3/fixed_red.tif'),
                 str(save_path+'4/fixed_red.tif'), 
                 str(save_path+'5/fixed_red.tif')]
    
    plotter_of_data(paths,save_path,reds_fil,color_plot)

tifs = ['/Volumes/Nebuchadnez/Mette/Mette NP2/20200928/control with 0.1% detergent/tifs/']

for save_path in tqdm(tifs):
    print ("Plotting")
    paths = glob(str(save_path+'*.csv'))   
    reds_fil = [ str(save_path+'0/fixed_red.tif'), 
                 str(save_path+'1/fixed_red.tif'), 
                 str(save_path+'2/fixed_red.tif'), 
                 str(save_path+'3/fixed_red.tif'),
                 str(save_path+'4/fixed_red.tif'), 
                 str(save_path+'5/fixed_red.tif')]
    
    plotter_of_data_single(paths,save_path,reds_fil)


