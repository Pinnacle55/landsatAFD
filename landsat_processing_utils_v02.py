#!/usr/bin/env python
# coding: utf-8

# In[1]:


from glob import glob
import os
import numpy as np
import rioxarray as rxr
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
import earthpy.spatial as es
from tqdm import tqdm
import argparse


# In[ ]:


### Additional utils - may have to update to use rxr workflow

def histogram_stretch(img, min_vals = None, max_vals = 99):
    """
    Performs a histogram_stretch on an image. DO NOT use this for analytical workflows - 
    this should only be used to improve image visualization
    
    img: an unmasked 3D raster 
    
    min_vals: percentile that you wish to crop to
        will be np.zeros by default
    max_vals: percentile that you wish to crop to
        will be np.nanpercentile(img, 99) by default # crops to 99th percentile
    """
    if img.ndim != 3:
        print("ValueError: Your raster must have three dimensions.")
        return
    
    # This returns the max_valth percentile
    max_vals = np.nanpercentile(img, max_vals, axis = (1,2)).reshape(img.shape[0],1,1) 
    # min_vals = np.nanmin(tcc_toa, axis = (1,2)).reshape(3,1,1) # Use this to stretch to minimum values
    if min_vals is not None:
        min_vals = np.nanpercentile(img, min_vals, axis = (1,2)).reshape(img.shape[0],1,1)
    else:
        min_vals = np.zeros(img.shape[0]).reshape(img.shape[0],1,1)
    
    # Perform normalization
    img_stretched = (img - min_vals) / (max_vals - min_vals)
    
    # Clip values above 1
    img_stretched[img_stretched > 1] = 1
    
    return img_stretched

def landsat_l2_scaling(stacked_mb_raster):    
    '''
    stacked_mb_raster: filepath to stacked l2 raster to be scaled (must be ordered bands 1-X then 10)
        OR 3D stacked numpy array (must be ordered bands 1-X then 10)
    
    returns 3D numpy array of the scaled, stacked bands as well as the profile information
    '''
    if isinstance(stacked_mb_raster, str):
        with rio.open(stacked_mb_raster) as src:
            profile = src.profile
            stack = src.read()
            src.close()
    else:
        stack = stacked_mb_raster

    # change the stack from DNs to reflectances
    # remember temperature uses a different scaling factor (temp is in Kelvin)
    scaled_stack = stack[:-1,...] * 0.0000275 - 0.2
    scaled_temp = stack[-1,...] * 0.00341802 + 149.0 

    # need to change scaled_temp from 2D to 3D
    scaled_stack = np.vstack((scaled_stack, np.expand_dims(scaled_temp, axis = 0))) 
    
    return scaled_stack, profile

def prep_training_data(raster, gdf, scale = False):
    '''
    Prepares X_train and y_train data from a raster image given a GeoDataFrame of classified polygons
    
    raster: Path to stacked 3D raster. If conducting scaling, 3D raster should be in the form (Bands 1-X, 10).
    
    gdf: GeoDataFrame containing labelled polygons. Must have "geometry" col and the labels col must be named "labels"
        May want to adjust this in the future.
        
    scale: Flag to indicate if the raster should be scaled from DNs to reflectances. Current implementation only works
        with Landsat Level 2 scaling. Please run process_folder() to do TOA scaling on Landsat level 1 products prior to using
        this function.
    '''

    # note that although we have scaled_stack, rio.mask.mask requires a dataset in read mode
    with rio.open(raster) as src:
        profile = src.profile
        
        # To prep the data for ML analysis, we need two numpy arrays:
        # X: a numpy array that contains all of the band data for the pixel
        # y: the labels for training

        # Creates an empty array with X columns, where X is the number of bands in the multiband raster
        X_train = np.array([], dtype = np.float32).reshape(0, profile["count"])
        y_train = np.array([], dtype = np.string_) # labels for training
        
        # Iterate over each polygon in our landuse ground truth dataset
        for index, row in gdf.iterrows():
            feature = [row["geometry"]]

            # crop the image - mask function returns a tuple
            out_image, out_transform = rio.mask.mask(src, feature, crop = True)

            # out_image has a shape (8, height, width)
            # Since this returns a rectangular array, and our shape is not rectangular, there will be
            # a bunch of nodata - get rid of them.

            # ~np.any(np.isnan(out_image), axis = 0) if any of the bands are np.nan

            # note that this gets rid of any column that has ANY nans
            # The following code returns a column for each band, with each row representing a pixel
            out_image_trimmed = out_image[:, ~np.any(out_image == profile['nodata'], axis = 0)]

            # We actually want this the other way around - we want a row for pixel, and a column for each
            # band - so we transpose the image
            out_image_trimmed = out_image_trimmed.T
            
            if scale:
                out_bands = out_image_trimmed[:,:-1] * 0.0000275 - 0.2
                out_temp = out_image_trimmed[:,-1] * 0.00341802 + 149.0 

                out_raster = np.hstack((out_bands, np.expand_dims(out_temp, axis = 1)))
            else:
                out_raster = out_image_trimmed
            
            # We append the labels to the answer array equal to the number of pixels:
            # Remember to put brackets around the row["LULC"], or else you'll get "forestforestforest"
            # out_image_trimmed.shape[0] is the number of pixels in the training data
            y_train = np.append(y_train, [row["labels"]] * out_raster.shape[0])

            # vstack is like concat for rows. Note that to vstack correctly, the array tuple that you feed
            # to vstack must have the same column dimension
            X_train = np.vstack((X_train, out_raster))

    src.close()
    
    return X_train, y_train

# Run a conservative cloud detection
def qa_pixel_interp_aggressive(number):
    '''
    Helps interpret the 16bit data in the landsat qa pixels
    
    returns True if there is mid confidence cirrus, snow/ice, cloud shadow, OR clouds
    '''
    binary = bin(number)[2:].zfill(16)
    
    # if medium to high confidence cirrus, snow/ice, cloud shadow, and clouds
    if int(binary[:2]) > 1:
        return True
    elif int(binary[2:4]) > 1:
        return True
    elif int(binary[4:6]) > 1:
        return True
    elif int(binary[6:8]) > 1:
        return True
    else:
        return False
    
def qa_pixel_interp_conservative(number):
    '''
    Helps interpret the 16bit data in the landsat qa pixels
    
    returns True if there is mid confidence cirrus, snow/ice, cloud shadow, OR clouds
    '''
    binary = bin(number)[2:].zfill(16)
    
    # if high confidence cirrus, snow/ice, cloud shadow, and clouds
    # 01 - low, 10 - medium, 11 - high
    if int(binary[:2]) > 10:
        return True
    elif int(binary[2:4]) > 10:
        return True
    elif int(binary[4:6]) > 10:
        return True
    elif int(binary[6:8]) > 10:
        return True
    else:
        return False
    
def qa_pixel_interp_conserv_water(number):
    '''
    Helps interpret the 16bit data in the landsat qa pixels
    
    returns True if there is mid confidence cirrus, snow/ice, cloud shadow, clouds OR WATER
    '''
    binary = bin(number)[2:].zfill(16)
    
    # if high confidence cirrus, snow/ice, cloud shadow, and clouds
    # 01 - low, 10 - medium, 11 - high
    if int(binary[:2]) > 10:
        return True
    elif int(binary[2:4]) > 10:
        return True
    elif int(binary[4:6]) > 10:
        return True
    elif int(binary[6:8]) > 10:
        return True
    # if water return true
    elif int(binary[8]) == 1:
        return True
    else:
        return False
    
def apply_array_func(func, x):
    '''
    Applies a function element-wise across a 1D array
    '''
    return np.array([func(xi) for xi in x])

def run_qa_parser(qa_raster, func):
    '''
    Accepts a QA_PIXEL array bundled with Landsat l2 product consisting of 16 bit unsigned integers.
    Generates a binary cloud mask using the function provided.
    
    qa_raster: numpy array of qa_raster. Works both stacked and unstacked.
    func: interpretation scheme to be used to generate the cloud mask. Current schemes include:
        qa_pixel_interp_conserv_water: conservative cloud detection, also masks water bodies
        qa_pixel_interp_conservative: conservative cloud detection
        qa_pixel_interp_aggressive: aggressive cloud detection
    
    Returns a squeezed binary cloud mask
    '''
    unique_vals = np.unique(qa_raster)
    masked_vals = apply_array_func(func, unique_vals)
    masked_vals = unique_vals[masked_vals]
    cl_mask = np.isin(qa_raster, masked_vals)
    
    return cl_mask.squeeze()


# In[2]:


def parse_mtl(mtl_file):
    """
    Parses the landsat metadata file into a dictionary of dictionaries.
    
    Dictionary is split into several sub-dicts including PRODUCT_CONTENTS and IMAGE_ATTRIBUTES
    """
    
    with open(mtl_file) as f:
        lines = f.readlines()
        f.close()

    clean_lines = [element.strip("\n").strip() for element in lines]

    ### PARSE THE MTL FILE INTO A DICTIONARY ###
    # Find all major groups in the metadata
    groups = [element for element in clean_lines if element.startswith("GROUP")]

    group_dict = dict()

    # We don't need the overarching metadata group
    for group in groups[1:]:
        # Return the part of list that the group contains
        contents = clean_lines[clean_lines.index(group)+1:clean_lines.index(f"END_{group}")]

        data_dict = {}
        # Iterate through the elements in the list
        for element in contents:
            # Split the element by "="
            parts = element.split("=")
            if len(parts) == 2:
                # Assign A as key and B as value to the dictionary
                key = parts[0].strip()  # Remove leading/trailing whitespace
                value = parts[1].strip()  # Remove leading/trailing whitespace
                data_dict[key] = value.strip("\"") # Remove quotation marks

        group_dict[group.replace("GROUP = ", "", 1)] = data_dict
    
    return group_dict

def toa_reflectance(raster, band_num, metadata, sun_corr = True):
    """
    raster: requires a 2D xarray as read by rxr.open_rasterio
    NB - array should be masked since landsat uses 0 for np.nan
    
    band_num: the landsat band number associated with that raster
    sun_corr: indicate if you want sun elevation correction (default true)
    
    returns the landsat level 1 product raster corrected for TOA
    Note that these are center image sun corrected - you can do pixel level sun correction but it
    takes a lot more work
    """
    # Get TOA reflectance
    toa_ref = raster * float(metadata["LEVEL1_RADIOMETRIC_RESCALING"][f"REFLECTANCE_MULT_BAND_{band_num}"]) + float(metadata["LEVEL1_RADIOMETRIC_RESCALING"][f"REFLECTANCE_ADD_BAND_{band_num}"])
    
    # Correct for sun elevation
    if sun_corr:
        toa_ref = toa_ref / np.sin(np.deg2rad(float(metadata["IMAGE_ATTRIBUTES"]["SUN_ELEVATION"])))
    
    # Clip any values that are larger than 1 to 1
    # must use .where() method
    toa_ref.where(toa_ref <= 1, 1)
    
    # toa_ref[toa_ref > 1] = 1 
    
    return toa_ref


# In[61]:


#### variables required: 
## filepath - folder containing the MTL as well as the landsat images
## mask = None - shapefile containing the masking polygons. 
## bounds = True - Bool, whether the data should be masked by the polygon exactly or if the raster should be masked by the total 
##          bounds of the polygon. Default uses total_bounds of polygon (results in rectangular image)
## sun_corr = True - Bool, whether TOA should take into account sun correctoin
## stack = True - Bool, whether a stacked raster containing the cropped, TOA corrected rasters should be created
## outdir = None - output directory where the output files should be saved. If not specified, save in the folder where this program
##          is located

def process_folder(filepath, mask = None, bounds = True, sun_corr = True, stack = True, outdir = None):
    # identify the mtl_file
    mtl_file = glob(os.path.join(filepath, '*MTL.txt'))

    # If no mtl_file found or more than one mtl_file found
    if len(mtl_file) != 1:
        print('No MTL file found or more than one MTL file found. Please check the folder.')
        return

    # Error handling if the metadata file can't be read
    try:
        metadata = parse_mtl(mtl_file[0])
    except:
        print("Metadata could not be read.")
        return

    # Find all bands from the MTL file
    filenames = [value for key, value in metadata["PRODUCT_CONTENTS"].items() if key.startswith("FILE_NAME_BAND")]

    # convert to the correct filenames
    filenames = [os.path.join(filepath, filename) for filename in filenames]

    band_list = []
    target_crs = es.crs_check(filenames[0])

    for file in tqdm(filenames):
        band_num = int(file.split("B")[1].split(".")[0])

        # print(f'Processing band {band_num}...')

        # Skip band 8 completely (panchromatic band)
        if band_num == 8:
            continue

        with rxr.open_rasterio(file, masked = True) as ds:

            # If mask has been specified, crop before loading
            if mask is not None:
                # reproject the polygon to the same CRS as the landsat image
                polygon = gpd.read_file(mask)
                polygon = polygon.to_crs(target_crs)

                # if 
                if bounds:
                    xmin, ymin, xmax, ymax = polygon.total_bounds
                    target_shapefile = box(np.floor(xmin), np.floor(ymin), np.ceil(xmax), np.ceil(ymax))
                    ds = ds.rio.clip([target_shapefile], from_disk = True).squeeze()
                else:
                    ds = ds.rio.clip(polygon.geometry, from_disk = True).squeeze()

            if mask:
                savename = f'{os.path.basename(file).split(".")[0]}_TOA_crop.tif'
            else:
                savename = f'{os.path.basename(file).split(".")[0]}_TOA.tif'

            if outdir is not None:
                out_path = os.path.join(outdir, savename)
            else:
                out_path = savename

            # catch temperature bands and save raster
            if band_num > 9:
                band_list.append(ds)
                ds.rio.to_raster(out_path)
                continue

            # Do TOA reflectance correction
            ds = toa_reflectance(ds, band_num, metadata, sun_corr = sun_corr)

            # adjust nodata value to np.nan (since 0 is used for DNs)
            # NB: This is extremely important - if you do not explicitly state a nodata value, you won't be able to 
            # save xr.Dataset() - raises ufunc isnan error
            ds = ds.rio.write_nodata(np.nan, inplace = True)
            band_list.append(ds)
            
            # Save raster
            ds.rio.to_raster(out_path)

    if stack:
        print('Stacking...')
        
        # Use xarray dataset to insert band names
        
        bands=['Coastal','Blue','Green','Red','NIR','SWIR-1','SWIR-2','Cirrus', 'TIRS-1', 'TIRS-2']
        
        stacked_array = xr.Dataset()
        
        for idx, band in enumerate(bands):
            stacked_array[band] = band_list[idx]
            
        # get landsat code
        stack_name = metadata["PRODUCT_CONTENTS"]["LANDSAT_PRODUCT_ID"]

        if mask:
            stack_name = f'{stack_name}_TOA_crop_stacked.tif'
        else:
            stack_name = f'{stack_name}_TOA_stacked.tif'

        if outdir is not None: 
            stacked_array.rio.to_raster(os.path.join(outdir, stack_name))
        else:
            stacked_array.rio.to_raster(stack_name)
            
    print('TOA processing complete.')


# In[ ]:


if __name__ == "__main__":
    ### useful tutorial https://docs.python.org/3/howto/argparse.html#argparse-tutorial
    parser = argparse.ArgumentParser(
        description='Calculates TOA reflectance for Level 1 Landsat products. Additional flags can be set to crop, \
        stack, and ignore sun elevation correction.')

    # add positional argument (i.e., required argument)
    parser.add_argument('filepath',
                       help = 'Path to folder containing all Landsat bands and MTL file.')

    # optional flags - the name of the variable is the -- option
    parser.add_argument('-m', '--mask', help = 'Provide a shapefile that the images will be cropped to') 

    # on/off flags - action indicates what the program should do
    # if flag is called (default will be the opposite for on/off)
    parser.add_argument('-b', '--bounds', action='store_false', help = "Add this flag if you want to crop the raster to exact polygon geometries. Otherwise, the rasters will be cropped to the total bounds of the shapefile (default; results in rectangular raster)")
    parser.add_argument('-s', '--stack', action='store_false', help = 'Add this flag if you DONT want to create a stacked raster of the images') 
    parser.add_argument('-c', '--sun_corr', action='store_false', help = "Add this flag if you DON'T want to do a sun elevation correction")
    parser.add_argument('-o', '--outdir', help = "Specify an output folder to save TOA corrected images")


    ### Preview arguments
    # parser.parse_args('LC08 -c -m test.geojson'.split(' '))
    
    # grab arguments from command line
    args = parser.parse_args()
    
    # calculate TOA
    process_folder(args.filepath, 
                   mask = args.mask, 
                   stack = args.stack, 
                   sun_corr = args.sun_corr, 
                   outdir = args.outdir,
                   bounds = args.bounds)


# In[62]:


# Speed of raster
# %%time
# process_folder('LC08_L1TP_180035_20230710_20230718_02_T1', outdir = 'Test Outputs', mask = 'rhodes.geojson')

