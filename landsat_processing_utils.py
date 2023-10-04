#!/usr/bin/env python
# coding: utf-8

# In[30]:


### Attempt to create a TOA_reflectance_stacker with the appropriate arguments and flags
import argparse
from glob import glob
import os, sys
import rasterio as rio
import numpy as np
import earthpy.spatial as es
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm


# In[ ]:


### Additional utils

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


# In[39]:


### TOA Processing Utils

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
    raster: requires a 2D numpy array as read from rasterio
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
    toa_ref[toa_ref > 1] = 1
    
    return toa_ref

def resample_tif(raster_file, target_height, target_width):
    """
    given a raster file and a height/width with the same aspect ratio, 
    
    output a masked 2D array of resampled data
    """
    # we need to resample the land_use geotiff because it has a 10m scale
    with rio.open(raster_file) as dataset:

        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                target_height, # height and width of the up/downsampled tiff - this will force the 
                target_width  # opened landuse dataset into this shape
            ),
            resampling=Resampling.nearest, # nearest is good for land use, use cubicspline for DEM
            masked = True
        )

        # scale image transform object
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )

        resampled_profile = dataset.profile
        resampled_profile.update(transform = transform, 
                       width = data.shape[-1], 
                       height = data.shape[-2])

    dataset.close()
    
    # data = masked numpy array, squeeze removes any dimensions of length 1 (i.e., 3D array with only
    # one stack will be converted into a 2D array)
    return data.squeeze(), resampled_profile

def earthpy_cropper(filenames, target_shapefile, profile, cleanup = False):
    '''
    Crops files to the bounds of the specified shapefile
    
    filenames: list of landsat images
    target_shapefile: this should be a shapely polygon. the image will be cropped according to the .total_bounds_ attribute
        of the given polygon.
    profile: rasterio metadata object - used to convert the shapefile to the raster CRS
    cleanup: indicates whether the original raster files should be deleted. This will save a lot of disk space.
    
    returns a list of band_paths 
    '''
    
    target_shapefile = gpd.read_file(target_shapefile)
    
    # convert to the correct CRS
    target_shapefile = target_shapefile.to_crs(profile['crs'].to_epsg())
    xmin, ymin, xmax, ymax = target_shapefile.total_bounds
    target_shapefile = box(np.floor(xmin), np.floor(ymin), np.ceil(xmax), np.ceil(ymax))
    
    band_paths = es.crop_all(
        filenames, os.path.dirname(filenames[0]), [target_shapefile], overwrite=True
    )

    # deletes uncropped files - will only delete files in which TOA reflectance has been conducted
    # i.e., will not delete any of the original landsat data files
    if cleanup:
        for file in filenames:
            if "TOA" in os.path.basename(file):
                os.remove(file)
            
    return band_paths

def process_folder(filepath, mask = None, stack = False, sun_corr = True, outdir = None, cleanup = False):
    '''
    Path to the landsat folder containing the landsat images as well as the MTL.txt file.    
    '''
    
    # identify the mtl_file
    mtl_file = glob(os.path.join(filepath, '*MTL.txt'))
    
    # If no mtl_file found or more than one mtl_file found
    if len(mtl_file) != 1:
        print('No MTL file found or more than one MTL file found. Please check the folder.')
        return
        
    try:
        metadata = parse_mtl(mtl_file[0])
    except:
        print("Metadata could not be read.")
    
    # Find all bands
    filenames = [value for key, value in metadata["PRODUCT_CONTENTS"].items() if key.startswith("FILE_NAME_BAND")]
    
    # convert to the correct filenames
    filenames = [os.path.join(filepath, filename) for filename in filenames]
    
    # Create an initial profile from band 1. This will be used for transformations and defaults.
    with rio.open(filenames[0]) as src:
        profile = src.profile
        src.close()
        
    profile.update(dtype = np.float32, nodata = np.nan)
    
    processed_filenames = list()
    
    for band in tqdm(filenames):
        
        # get band number
        band_num = int(band.split("B")[1].split(".")[0])
        
        # Skip band 8 completely (panchromatic band)
        if band_num == 8:
            continue
        
        # if no output directory specified, place the TOA corrected rasters in the original folder
        if outdir is not None:
            out_path = os.path.join(outdir, f'{os.path.basename(band).split(".")[0]}_TOA.tif')
        else:
            out_path = os.path.join(filepath, f'{os.path.basename(band).split(".")[0]}_TOA.tif')
        
        # catch temperature bands
        if band_num > 9:
            processed_filenames.append(band)
            continue
        
        # read original raster
        with rio.open(band) as src:
            raster = src.read(1, masked = True)
            src.close()
        
        # perform correction
        toa_ref = toa_reflectance(raster, band_num, metadata, sun_corr = sun_corr)
        
        # save corrected image (note change to dtype float32 to accommodate np.nan)
        with rio.open(out_path, 'w', **profile) as dst:
            dst.write(toa_ref.astype(np.float32), 1)
            dst.close()
        
        # add filepaths to processed_filenames
        processed_filenames.append(out_path)
        
    # Crop the data if a shapefile has been provided
    # NB this always produces a rectangular raster based on the total_bounds_ of the given polygon
    
    if mask is not None:
        processed_filenames = earthpy_cropper(processed_filenames, mask, profile, cleanup)
    
    # If stack, create stacked data
    # NB: THIS WILL BUG OUT IF TEMP DATA IS NOT SAME RESOLUTION AS OTHER BAND DATA
    if stack:
        print("Stacking...")
        if outdir is not None:
            out_path = os.path.join(outdir, f"{os.path.basename(filepath)}_TOA_STACKED.tif")
        else:
            out_path = os.path.join(filepath, f"{os.path.basename(filepath)}_TOA_STACKED.tif")
        
        try:
            stack, metadata = es.stack(processed_filenames, out_path = out_path)
        except Exception as e:
            print('Issue with stacking - attempting to stack bands 1-7 only.')
            print(e)
            try:
                stack, metadata = es.stack(processed_filenames[:7], out_path = out_path)
            except Exception as e:
                print('Issue with stacking - please see error below for more info')
                print(e)
                
    print('TOA calculation complete.')


# In[38]:


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
    parser.add_argument('-s', '--stack', action='store_true', help = 'Create a stacked raster of the images - default False') 
    parser.add_argument('-c', '--sun_corr', action='store_false', help = "add this flag if you DON'T want to do a sun elevation correction - default True")
    parser.add_argument('-o', '--outdir', help = "Specify an output folder to save TOA corrected images")
    parser.add_argument('-d', '--cleanup', action='store_true', help = "If cropping, choose whether to delete the uncropped TOA images - default False")


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
                   cleanup = args.cleanup)


# In[44]:


# parser.parse_args('-m "study_area.geojson" -s -c -o "20230710_TOA" "LC08_L1TP_180035_20230710_20230718_02_T1"'.split(' '))


# In[46]:





# In[ ]:




