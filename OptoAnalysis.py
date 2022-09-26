"""
MIT License

Copyright (c) 2021 Derek Britain

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import collections
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu 
from skimage.measure import  regionprops
from skimage.segmentation import  watershed
from skimage.morphology import binary_erosion, binary_dilation
from skimage.feature import peak_local_max
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from tifffile import TiffFile
from tifffile import TiffWriter
import seaborn as sns
from scipy.optimize import curve_fit
from skimage.morphology import remove_small_objects

#fitting parameters for washout data for function converting LED intensity to half-lifes
pLED = [4.50067076e-06, 2.64353635e-01, 5.96607443e-02] 

#fitting parameters for converting LED percantage into LED intensity from imaging puddle of FITC with LED
conv_led = np.poly1d([ -1.74070543, 515.89030873, -34.28032294])


def analyze_data(data_dir, pp2_dir=None, data_flat_dir=None, lov2flat_dir=None, dark_dir=None, good_RICM=False, normalize_type='final',\
    cond_endpoints=None, max_endpoints=None, conditions=None, data_channel=1, lov_channel=2, RICM_channel=0, current_pos=0, num_bins=10.0,\
    filter_stat='mean',respond_filt=None, supress_filt=None, p0=[1,1000,1], plot_stat='mean_intensity_Normalized', subtract_lov_bg = True, lov_limit=None, lov_stat='mean_intensity_sub',\
    min_filt=None, max_filt=None, fatcells=False):

##Search directory for tif files to analyze
    data_list = glob.glob(data_dir + '/*.tif')
    print('The data dir found is ' + data_list[0])
    

    ##Make output directory if it does not exist
    output_dir = data_dir + '/output_stat_' + filter_stat + '_norm_' + normalize_type + '_datachan' + str(data_channel) + '_plotstat' + plot_stat
    try:
        os.mkdir(output_dir)
        print('made output dir')
    except OSError as error: 
        print('output dir exists')

    #check if an output file for this data already exists and load that instead
    if os.path.isfile(data_list[0] + '_channel_' + str(data_channel) + '_data_df.pkl') and os.path.isfile(data_list[0] + '_lov_df.pkl'):
        print('loading existing data and lov pickles')
        data_df = pd.read_pickle(data_list[0] + '_channel_' + str(data_channel) + '_data_df.pkl')
        new_lov_df = pd.read_pickle(data_list[0] + '_lov_df.pkl')

    #Otherwise load the images
    else:
        #load image stack
        data_stack = get_pos_stacks(data_list[0])

   
        #Check if segmentation label stacks already exist and use them. Otherwise create segmentation images
        if os.path.isfile(data_list[0] + '_label_stack.ome.tif'):

            print('loading existing label stacks')
            lab_stack = np.moveaxis(get_pos_stacks(data_list[0] + '_label_stack.ome.tif')[0],0,2)
            bg_lab_stack = np.moveaxis(get_pos_stacks(data_list[0] + '_bg_stack.ome.tif')[0],0,2)

        #Generate labeled images for the foreground and background
        else:
            print('Using Watershed Segmentation')
            lab_stack, bg_lab_stack = make_label_stack_watershed(data_stack[current_pos], good_RICM=good_RICM, fatcells=fatcells)

            # Save foreground and background label stacks
            with TiffWriter(data_list[0] + '_label_stack.ome.tif') as tif:
                tif.save(np.moveaxis(lab_stack,2,0), compress = 6)

            with TiffWriter(data_list[0] + '_bg_stack.ome.tif') as tif:
                tif.save(np.moveaxis(bg_lab_stack,2,0), compress = 6)


    
        
        #Check if reporter channel data alread exists and load instead of processing new data
        if os.path.isfile(data_list[0] + '_channel_' + str(data_channel) + '_data_df.pkl'):
            print('loading exisitng data pickle')
            data_df = pd.read_pickle(data_list[0] + '_channel_' + str(data_channel) + '_data_df.pkl')

        #Using labeled image to segment and generate dataset
        else:
            print('computing new data frames')

            # Use a FLat field correction if directory supplied to flatfield image stack
            if data_flat_dir is not None:
                dark_val = None

                #Check if Flattened data exists and us it instead
                if os.path.isfile(data_list[0] + '_ff_corrected_data_chanel' + str(data_channel) + '.ome.tif'):
                    print('using existing flattened data')
                    data_ff = get_pos_stacks(data_list[0] + '_ff_corrected_data_chanel' + str(data_channel) + '.ome.tif')

                #compute flat field image and save
                else:
                    if dark_val == None:
                        dark_list = glob.glob(dark_dir + '/*.tif')
                        dark_data_list = get_pos_stacks(dark_list[0])
                        dark_val = np.mean(dark_data_list[0])

                    ff561_list = glob.glob(data_flat_dir + '/*.tif')
                    print('The 561flat dir found is ' + ff561_list[0])
                    ff561_data_list = get_pos_stacks(ff561_list[0])
                    data_ff = get_flatfield_correct(data_stack,ff561_data_list,data_channel,data_dir=data_list[0], filename='data_ff_norm_channel_' + str(data_channel), dark_val=dark_val)

                   
                    with TiffWriter(data_list[0] + '_ff_corrected_data_chanel' + str(data_channel) + '.ome.tif') as tif:
                        tif.save(data_ff[current_pos], compress = 6)

                #compute reporter channel data
                data_df = get_regionprop_dataframe(lab_stack, data_ff[current_pos])

            
            #If not flat field image directory is supplies, comput reporter channel data on raw images
            else:
                data_df = get_regionprop_dataframe(lab_stack, data_stack[current_pos][:,data_channel,:,:])


            #save reporter channel data
            print('saving data pickle')
            data_df.to_pickle(data_list[0] + '_channel_' + str(data_channel) + '_data_df.pkl')
            


        #Load LOV2 channel data if it exists
        if os.path.isfile(data_list[0] + '_lov_df.pkl'):
            print('loading existing lov pickle')
            new_lov_df = pd.read_pickle(data_list[0] + '_lov_df.pkl')
        
        #Compute new LOV2 channel data
        else:
            print('computing new lov dataframes')

            #compute flat field image for LOV2 channel if directory supplied
            if lov2flat_dir is not None:

                #Check if Flattened data exists and us it instead
                if os.path.isfile(data_list[0] + '_ff_corr_lov.ome.tif'):
                    print('using existing flattened LOV')
                    LOV_ff = get_pos_stacks(data_list[0] + '_ff_corr_lov.ome.tif')

                # compute flattened data
                else:
                    if dark_val == None:
                        dark_list = glob.glob(dark_dir + '/*.tif')
                        dark_data_list = get_pos_stacks(dark_list[0])
                        dark_val = np.mean(dark_data_list[0])

                    ff488_list = glob.glob(lov2flat_dir + '/*.tif')
                    print('The 488flat dir found is ' + ff488_list[0])
                    ff488_data_list = get_pos_stacks(ff488_list[0])
                    LOV_ff = get_flatfield_correct(data_stack,ff488_data_list,lov_channel,data_dir=data_list[0], filename='lov_ff_norm', dark_val=dark_val)

                #Saving flat field corrected lov2 channel
                    with TiffWriter(data_list[0] + '_ff_corr_lov.ome.tif') as tif:
                        tif.save(LOV_ff[current_pos], compress = 6)


                #compute LOV2 data frame using flattened data
                LOV_df = get_regionprop_dataframe(lab_stack, LOV_ff[current_pos])
                LOV_bg_df = get_regionprop_dataframe(bg_lab_stack, LOV_ff[current_pos])

            #compute LOV2 data frame using raw data
            else:
                LOV_df = get_regionprop_dataframe(lab_stack, data_stack[current_pos][:,lov_channel,:,:])
                LOV_bg_df = get_regionprop_dataframe(bg_lab_stack, data_stack[current_pos][:,lov_channel,:,:])
            

            
            #Subtract off the immediate background intensity for each cell from LOV2 data using background label image stack
            if subtract_lov_bg == True:
                print('subtracting background lov')

                new_lov_df = pd.merge(LOV_df, LOV_bg_df, suffixes=('','_bg'), how='outer', left_on=['TimeFrame','label'], right_on = ['TimeFrame','label'])

                new_lov_df['total_intensity_sub'] = (new_lov_df.intensity_image - new_lov_df.mean_intensity_bg).apply(np.clip, args=(0,None)).apply(np.sum)
                new_lov_df['mean_intensity_sub'] =  new_lov_df.total_intensity_sub / new_lov_df.image.apply(np.count_nonzero) 

            else:
                new_lov_df = LOV_df
                new_lov_df['total_intensity'] = new_lov_df.intensity_image.sum()
            
            
            print('saving lov pickle')
            new_lov_df.to_pickle(data_list[0]+ '_lov_df.pkl')



    #Normalized reporter data in multiple methods
    print('normalizing data')

    #filter out cells not present for the entire timecourse
    update_Max_Count(data_df)
    data_df = data_df[data_df.Max_Count == max(data_df.Max_Count)]


    #Normalize each cell to it's average intensity in full suppresive light for mulitple parameters
    normalize_mean_endpoint_data(data_df, 'mean_intensity', cond_endpoints)
    normalize_mean_endpoint_data(data_df, 'max_intensity', cond_endpoints)
    normalize_mean_endpoint_data(data_df, 'top_mean_intensity', cond_endpoints)

    #Filter cell data
    print('filtering data')
    filt_data_df, filt_lov_df = filter_cells2(data_df, new_lov_df, normalize=normalize_type, stat=filter_stat, responded=respond_filt, suppressed=supress_filt, min_filt=min_filt, max_filt=max_filt)


    #compute fit to kinetic proofreading model and make heat maps if timepoints (cond_endpoints) and LED percent values (conditions) are given
    if cond_endpoints is not None and conditions is not None:
        print('making heatmap')

        #compute the endpoint values for each LED condition in a time series for the reporter channel 
        data_response_df = compute_endpoint_data(filt_data_df, plot_stat, cond_endpoints, avg_range=3 ,conditions=conditions, subtract_supressed=True)

        #compute the endpoint values for each LED condition in a time series for the LOV2 channel 
        lov_response_df = compute_endpoint_data(filt_lov_df, lov_stat ,cond_endpoints, avg_range=0 ,conditions=conditions, subtract_supressed=False)

        #Combine the Reporter and LOV2 data into one dataframe
        df = pd.merge(data_response_df, lov_response_df, left_on=['label','cond_order','conditions'], right_on=['label','cond_order','conditions'], how='outer', suffixes=['_data', '_lov'])
        df.fillna(value=0, inplace=True)

        #Filter off set ceilling of LOV2 data to have equal data range across all halflife conditions, also filter off background label (0)
        if lov_limit is not None:
            if type(lov_limit) is int: 
                dfplot = df[(df.response_lov<lov_limit) & (df.label != 0)]
            else:
                dfplot = df[(df.response_lov<lov_limit[1]) & (df.response_lov>lov_limit[0]) & (df.label != 0)]
              
        else:
            dfplot = df[df.label != 0]

        #plot heatmaps and compute fit to proofreading model 
        #compute xbin edges
        conditions2 = conditions + [max(conditions) + 1] #add extra dumby condition to the list of conditions for far right bind edge
        xbins = np.unique(np.array(sorted(conditions2)))

        #compute ybin edges
        bin_size = max(dfplot.response_lov)/num_bins #make numb_bins number of rows in heatmap
        ybins = np.arange(min(dfplot.response_lov), max(dfplot.response_lov)+bin_size, bin_size)

        #plot heatmap, compute fit, and save image
        try:
            fig = make_heatmap(dfplot, xbins, ybins,p0,lov_limit)
            plt.savefig(output_dir + '/Heatmap.png')
            plt.close(fig=fig)

        #if no fit is found, still return data
        except RuntimeError:
            dfplot['Occupancy'] = pd.cut(dfplot.response_lov, ybins, labels=False, include_lowest=True)
            dfplot['Halflife'] = pd.cut(dfplot.conditions, xbins, labels=False, right=False, include_lowest=True)

            print('No Fit Found')

        return filt_data_df, filt_lov_df, dfplot


        

    #If no conditions or condition endpoints are given return the reporter data frames and the lov data frames on their own
    else:
        return filt_data_df, filt_lov_df 




def get_pos_stacks(file_path):
    data_list=[]
    with TiffFile(file_path) as tif:
        pos_num = len(tif.series)
        print('found ' + str(pos_num) + ' positions')
        for i in range(pos_num):
            print('Reading pos ' + str(i+1))
            data = tif.asarray(series=i)
            print(np.shape(data))
            data_list.append(data)
    return data_list



#Take an image stack with an RICM channel specified by 'segmentation_channel' and compute foreground and background labeled image stack. 
def make_label_stack_watershed(img_stack, segmentation_channel=0):

    data=[]
    data_bg=[]
    
    #segmentation and tracking for no time dimension image stack (channel,x,y)
    if len(np.shape(img_stack)) == 3:
        for j in range(0,np.shape(img_stack)[1]):
            if j == segmentation_channel:
                img = img_stack[j,:,:]

                #compute background RICM by heavily blurring image
                bg = gaussian_filter(img,(30,30))

                #smooth RICM image and subtrack background value
                smoothed = gaussian_filter(img,(3,3)) - bg* 0.9

                #threshold image with otsu method and compute binary image
                thresh = threshold_otsu(smoothed) * 1
                binary_img = smoothed < thresh

                #close any hole in binary image
                binary_img = binary_dilation(binary_img)
                binary_img = binary_erosion(binary_img)

                #remove small binary regions
                remove_small_objects(binary_img, min_size=10, in_place=True)

                #compute local maximums and create distance image
                distance = ndi.distance_transform_edt(binary_img)
                local_maxi = peak_local_max(distance, footprint=np.ones((20, 20)), min_distance=20, indices=False , labels=binary_img)
                
                #find markers at center of regions in distance image
                markers = ndi.label(local_maxi)[0]

                #Use watershed to label cellular regions and apply
                labels = watershed(-distance, markers = markers, mask=binary_img)


                data.append(labels)

                #compute the background labeled image
                data_bg.append(get_bg_labelstack(labels))


    #Same seg as above, but for img stack including time dimension (time,channel,x,y)          
    if len(np.shape(img_stack)) == 4:
        for i in range(0,np.shape(img_stack)[0]): 
            for j in range(0,np.shape(img_stack)[1]):
                if j == segmentation_channel:
                    img = img_stack[i,j,:,:]

                   #image processing same as above for non-time data
                    bg = gaussian_filter(img,(30,30))
                    smoothed = gaussian_filter(img,(3,3)) - bg* 0.9
                    thresh = threshold_otsu(smoothed) * 1
                    binary_img = smoothed < thresh

                    binary_img = binary_dilation(binary_img)
                    binary_img = binary_erosion(binary_img)
                    remove_small_objects(binary_img, min_size=10, in_place=True)

                    distance = ndi.distance_transform_edt(binary_img)
                    local_maxi = peak_local_max(distance, footprint=np.ones((20, 20)), min_distance=20, indices=False , labels=binary_img)


                    #if first timeframe in the stack, find first marker locations
                    if i==0:

                        #compute marker from peak max image
                        markers = ndi.label(local_maxi)[0]

                        #apply watershed algorithm with distanc image and markers
                        labels = watershed(-distance, markers = markers, mask=binary_img)
                        
                        
                    #Use previouse timepoints markers for watched of next timepoint to track cells regions
                    else:

                        #generate image of all markers
                        new_markers = ndi.label(local_maxi)[0]

                        #use previouse label image the eliminate markers that already exists inside a label region (a cell)
                        new_markers[labels>0] = 0 
                        
                        #get list of all current cell label values exluding background label
                        old_labels = list(np.unique(labels))[1:] #[1:] ignores 0 labeled background pixels

                        #iterate through all new marker labels found in this image (new cells)
                        for x in list(np.unique(new_markers))[1:]:

                            #generate a new label value (1 greater than last image)
                            val = max(old_labels) + 1

                            #set pixel in marker image that equal current x to new val
                            new_markers[new_markers==x] = val 

                            #add val to old_label list to not use again
                            old_labels.append(val)
                        
                        #update marker image to include previouse label image and new found markers
                        markers = labels + new_markers
                        
                        #perform watershed on marker image and generate new label image for next iteration
                        labels = watershed(-distance, markers = markers, mask=binary_img)

                    
                    data.append(labels)
                    data_bg.append(get_bg_labelstack(labels))


    labeled_stack = np.dstack(data)
    bg_labeled_stack = np.dstack(data_bg)
    
    
    return labeled_stack, bg_labeled_stack




def get_bg_labelstack(stack):

    a = stack

    b = maximum_filter(a, 10)
    b[a != 0] = a[a != 0]
    
    output = b - a

    output[binary_erosion(binary_erosion(output>0)) == False] = 0

    return output


def get_flatfield_correct(data_list, ff_imgs, channel_num, dark_val = None, data_dir=None, filename=None):  

    avg_field = np.median(ff_imgs[0], axis=0)

    if dark_val is not None:
        avg_field = avg_field - dark_val 

    avg_field_norm = avg_field / np.max(avg_field)

    if data_dir is not None and filename is not None:
        with TiffWriter(data_dir + '_' + filename + '.ome.tif') as tif:
            tif.save(avg_field_norm, compress = 6)
            
    ff_cor_list=[]
    if dark_val is not None:
        for data_stack in data_list:
            ff_cor_list.append(np.trunc(np.divide((data_stack[:,channel_num,:,:] - dark_val), avg_field_norm)))
    else:
        for data_stack in data_list:
            ff_cor_list.append(np.trunc(np.divide(data_stack[:,channel_num,:,:], avg_field_norm)))

    
    return ff_cor_list


def get_regionprop_dataframe(labeled_stack, intensity_img_stack):
    data =[] 
    for i in range(0,np.shape(labeled_stack)[2]):
        regions = regionprops(labeled_stack[:,:,i], intensity_image=intensity_img_stack[i,:,:])

        for region in regions:

            
            #find mean of top 10% of pics
            cell = region.intensity_image.flatten()
            cell_pix = cell[cell>0]
            num_pix = np.size(cell_pix)
            thresh = 0.05
            x = int(num_pix * thresh)
            top = np.mean(cell[np.argpartition(cell,-x)[-x:]])
    

            prop_labels = ['TimeFrame','top_mean_intensity']
            properties = [i, top]

            for prop in region:
                prop_labels.append(prop)
                properties.append(region[prop])

            prop_dict = dict(zip(prop_labels,properties))
            data.append(prop_dict)

    df = pd.DataFrame(data)

    return df


def update_Max_Count(df, var = 'label'):
    df['Max_Count'] = df.groupby(var)[var].transform('count')


def normalize_mean_endpoint_data(df, var, endpoints, group_on = 'label'):  # used for "suppressmean" normalization

    supress_ranges = []

    #assuming 3 min period, 10s int with full supression just before for 2 min
    for i in endpoints:
        supress_ranges.append(np.arange(i-19, i-17, 1))

    timepoints = np.concatenate( supress_ranges, axis=0 ) #combine all times into 1D array since taking average of all values

    frame_list = []
    for key, grp in df.groupby(group_on):
    
        mean_response = grp['mean_intensity'][grp.TimeFrame.isin(timepoints)].mean()
        normalized = grp[var].divide(mean_response)
        frame_list.append(normalized)

    data = pd.concat(frame_list)
    new_name = var + '_Normalized'
    df[new_name] = data  
        
    return


def filter_cells2(df, other_df = None, stat=None, responded = None, suppressed= None, min_filt=None, max_filt=None):

    #accept cells only present durring full timecourse
    update_Max_Count(df)
    df = df[df.Max_Count == max(df.Max_Count)]

    if stat == 'mean':

        if suppressed is not None:
            for filt in suppressed:  
                df = mean_group_response(df,'mean_intensity_Normalized', filt[0])
            
                df = df[df['mean_intensity_Normalized_' + str(filt[0]) + '_Mean'] < filt[1]]

        if responded is not None:
            for filt in responded:
                df = mean_group_response(df,'mean_intensity_Normalized', filt[0])
                
                df = df[df['mean_intensity_Normalized_' + str(filt[0]) + '_Mean'] > filt[1]]

        df = min_max_limit_filter(df, 'mean_intensity_Normalized', min_filt=min_filt, max_filt=max_filt)  
           
  
    elif stat == 'max':
        if responded is not None:
            for filt in responded:
                df = mean_group_response(df,'max_intensity_Normalized', filt[0])
                df = df[df['max_intensity_Normalized_' + str(filt[0]) + '_Mean'] > filt[1]]
        
        if suppressed is not None:
            for filt in suppressed:
                df = mean_group_response(df,'max_intensity_Normalized', filt[0])
                df = df[df['max_intensity_Normalized_' + str(filt[0]) + '_Mean'] < filt[1]]

        df = min_max_limit_filter(df, 'max_intensity_Normalized', min_filt=min_filt, max_filt=max_filt)

    elif stat == 'top':

        if responded is not None:
            for filt in responded:
                df = mean_group_response(df,'top_mean_intensity_Normalized', filt[0])
                df = df[df['top_mean_intensity_Normalized_' + str(filt[0]) + '_Mean'] > filt[1]]
        
        if suppressed is not None:
            for filt in suppressed:
                df = mean_group_response(df,'top_mean_intensity_Normalized', filt[0])
                df = df[df['top_mean_intensity_Normalized_' + str(filt[0]) + '_Mean'] < filt[1]]

        df = min_max_limit_filter(df, 'top_mean_intensity_Normalized', min_filt=min_filt, max_filt=max_filt)

    else:

        if responded is not None:
            for filt in responded:
                df = mean_group_response(df, stat, filt[0])
                df = df[df[stat + '_' + str(filt[0]) + '_Mean'] > filt[1]]
        
        if suppressed is not None:
            for filt in suppressed:
                df = mean_group_response(df, stat, filt[0])
                df = df[df[stat + '_' + str(filt[0]) + '_Mean'] < filt[1]]

        df = min_max_limit_filter(df, stat, min_filt=min_filt, max_filt=max_filt)


    update_Max_Count(df)
    df = df[df.Max_Count == max(df.Max_Count)]

    if other_df is not None:
        other_df = other_df[other_df['label'].isin(df.label.unique())]
        return df, other_df
    
    return df 



def mean_group_response(df, var, endpoint, group_on = 'label',  avg_range = 3 ):

    new_name = var + '_' + str(endpoint) + '_Mean'
    
    df = df.groupby(group_on).apply(get_group_mean, endpoint, new_name, var, avg_range)

    return df 




def get_group_mean(grp, endpoint, new_name, var, avg_range):
    
    final_value = grp[var][(grp.TimeFrame < endpoint) & (grp.TimeFrame>(endpoint-avg_range))].mean()
    grp[new_name]= final_value
    
    return grp




def min_max_limit_filter(df, stat, min_filt=None, max_filt=None):

    if max_filt is not None:
        bad_cells = df.label[(df[stat] > max_filt) & (df.TimeFrame>42)]
        df = df[~df.label.isin(bad_cells)]

    if min_filt is not None:
        bad_cells = df.label[(df[stat] < min_filt) & (df.TimeFrame>42)]
        df = df[~df.label.isin(bad_cells)]

    update_Max_Count(df)
    df = df[df.Max_Count == max(df.Max_Count)]

    return df



def compute_endpoint_data(df, var, endpoints, avg_range = 3, conditions = None, group_on = 'label', subtract_supressed=False):
    
    data_ranges = []
    supress_ranges = []

    for i in endpoints:
        data_ranges.append(np.arange(i-avg_range, i+1, 1))

    #assuming 3 min period, 10s interval with full supression just before each condition for 2 minute duration
    if subtract_supressed == True:
        for i in endpoints:
            supress_ranges.append(np.arange(i-20, i-17, 1))

    if conditions == None:
        conditions = [str(i) for i in endpoints] 
    
    frame_list = collections.defaultdict(list)

    for key, grp in df.groupby(group_on):

        cell_data=[]
        for k, j in enumerate(data_ranges):
      
            mean_response = grp[var][grp.TimeFrame.isin(j)].mean()
            if subtract_supressed == True:
                supress_response = grp[var][grp.TimeFrame.isin(supress_ranges[k])].mean()

                #round negative output values to zero
                if mean_response-supress_response > 0:
                    cell_data.append(mean_response-supress_response)
                else:
                    cell_data.append(0.0)
            else:
                cell_data.append(mean_response)
        
        frame_list[key] = pd.DataFrame(list(zip(cell_data,conditions)), columns=['response','conditions'])


    data = pd.concat(frame_list, axis=0).reset_index().rename(columns={'level_0': group_on, 'level_1' : 'cond_order'})

    return data


def make_heatmap(df,xbins,ybins,p0,lov_limit, pLED, conv_led, values = 'response_data', vmax = None, return_fit = False, ax = None):


    popt, pcov = compute_KP_model(df,p0, lov_limit,pLED,conv_led)
    


    df['Occupancy'] = pd.cut(df.response_lov, ybins, labels=False, include_lowest=True)
    df['Halflife'] = pd.cut(df.conditions, xbins, labels=False, right=False, include_lowest=True)

    
    data = df.groupby(['Occupancy','Halflife']).mean().reset_index()
    result = data.pivot_table(index='Occupancy', columns='Halflife', values=values)
    

    if ax == None:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
    

    xlabel = np.around(hl_func(conv_led(xbins[:-1]), *pLED), 2)
    ylabel = np.trunc(ybins[:-1])



    if vmax is not None:
        sns.heatmap(result, vmin=0, vmax=vmax, cmap='magma',annot=False, xticklabels = xlabel, yticklabels = ylabel, ax=ax)  #with Vminmax limits
    else:
        sns.heatmap(result, cmap='magma',annot=False, xticklabels = xlabel, yticklabels = ylabel, ax=ax) # No Vminmax limits


    ax.invert_yaxis()
    ax.invert_xaxis()

    
    if return_fit == True:
        return fig, [popt, pcov]
    else:
        return fig


def compute_KP_model(df,p0,pLED,conv_led):
    
    y = np.array([conv_led(x) for x in df.conditions])
    
    ydata = df.response_data
    
    occ = df.response_lov 

    hl = hl_func(y, *pLED)  #hl func takes in LED img intensity and outputs halflife (ln(2)/k in function)

    X = occ, hl

    popt, pcov = curve_fit(KP_func, X, ydata, p0=p0, method='lm', maxfev=10000)

    return popt, pcov

def hl_func(v, k1, k2, k3):
    return np.log(2) / ((k1 * v * k2) / ((k1 * v) + k2) + k3)


def KP_func(X, n, K, b):
    occ, hl = X
    return (occ * (hl**n))/(K + occ * (hl**n) ) + b
