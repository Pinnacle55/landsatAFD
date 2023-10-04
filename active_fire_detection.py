#!/usr/bin/env python
# coding: utf-8

# In[3]:


import rasterio as rio
import numpy as np
from landsat_processing_utils import histogram_stretch
from rasterio.plot import show
from scipy.ndimage import generic_filter, uniform_filter
import earthpy.spatial as es
import os
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import rasterio.features as rf
import shapely
import geopandas as gpd
import argparse


# In[1]:


def load_path(filepath):
    '''
    Prepares stacked raster for fire detection algorithm
    
    Inputs
    ======
    filepath: Path to stacked raster. The first 7 bands of the raster must be reflectance corrected Bands 1-7.
    
    
    Returns
    =======
    band: dictionary of bands with keys 1-7
    profile: src.profile of the original raster
    '''
    with rio.open(filepath) as src:
        profile = src.profile
        stack = src.read()
        src.close()
        
    # Extract stacks for clarity
    band = dict()

    for i in range(7):
        band[i+1] = stack[i, ...]
        
    return band, profile
    
def dn_folding(band):
    '''
    Identify DN folding, which indicates an unambiguous fire pixel.
    
    Input
    =====
    band: dictionary containing bands with keys 1-7
    
    Returns 
    =======
    condition: A boolean array containing true values where a pixel has undergone folding
    
    '''
    condition = ((band[6] > 0.8) & (band[1] < 0.2) & ((band[5] > 0.4) | (band[7] < 0.1))) 
    
    return condition

def unambiguous_fire(band, dn_folded):
    '''
    Identify unambiguous fire pixels.
    
    Input 
    =====
    band:      dictionary containing bands with keys 1-7
    dn_folded: boolean array indicating DN folded pixels (from dn_folding())
    
    
    Returns 
    =======
    uafp: A boolean array containing true values where unambiguous fire pixels were detected.
    '''
    uafp = ((band[7] / band[5]) > 2.5) & ((band[7] - band[5]) > 0.3) & (band[7] > 0.5)
    
    uafp = (uafp | dn_folded)
    
    return uafp

def water(band):
    '''
    Used to detect water pixels (both ocean and inland water bodies)
    
    Input
    =====
    band: Dictionary containing bands with keys 1-7
    
    Returns
    =======
    Returns a NumPy boolean array containing true values where water pixels were detected.
    '''
    condition_1 = band[4] > band[5]
    condition_2 = band[5] > band[6]
    condition_3 = band[6] > band[7]
    condition_4 = (band[1] - band[7]) < 0.2
    
    test_1 = (condition_1 & condition_2 & condition_3 & condition_4)
    
    condition_5 = band[3] > band[2]
    condition_6 = band[1] > band[2]
    condition_7 = band[2] > band[3]
    condition_8 = band[3] > band[4]
    
    test_2 = ((condition_5) | (condition_6 & condition_7 & condition_8))
    
    return (test_1 & test_2)

def background(band, unambiguous_array, water_array):
    '''
    Used to identify valid background pixels for potential fire detection. 
    
    These pixels are used in the calculation of the mean and stddev values required
    for threshold and contextual tests for potential fire pixel detection
    
    Inputs:
    ======
    band: Dict with keys 1-7 containing reflectance bands 1-7 
    unambiguous_array: Boolean array indicating the presence of an unambiguous fire pixel
    water_array: Boolean array indicating the presence of a water pixel
    
    Returns a boolean array indicating whether a pixel is a valid background pixel.
    '''
    # background array is anything that isn't a unambgiuous fire OR water pixel
    background = np.invert(unambiguous_array | water_array)
    
    # and band[7] > 0
    background = (background & (band[7] > 0))
    
    return background

def potential_fire(band, background_mask, size = 61):
    '''
    This helps classify potential fire pixels. First, it conducts threshold and contextual tests to determine if the 
    pixel is valid, then checks if it meets the other requirements.
    
    Inputs:
    ======
    band: Dict with keys 1-7 containing reflectance bands 1-7 
    background_mask: boolean array of valid background pixels (i.e., not water or fire) 
    
    Returns
    =======
    Boolean array where true values indicate the presence of potential fire pixels
    '''
    # background_mask = True/False
    # background_array = 1./0.
    background_array = background_mask.astype(np.float32)
    
    # this is used to get back the actual means from uniform_filter
    # weights = uniform_filter(background_array, size = size, mode = "constant")
    weights = uniform_filter(background_array, size = size)
    
    ratio75 = band[7] / band[5]
    
    # masked version of band7 and ratio75 where invalid pixels are set to 0.
    band7_bgmasked = background_mask * band[7]
    ratio75_bgmasked = background_mask * ratio75
    
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        ### vectorized mean of band 7
        # mean of the 61x61 kernel, but includes 0s
        # band7_uf = uniform_filter(band7_bgmasked, size = size, mode = "constant")
        band7_uf = uniform_filter(band7_bgmasked, size = size)
        # true mean of the 61x61 kernel, not including 0s
        # use weights > 1/size to avoid floating point errors
        band7_mean = np.where(weights > (1/size), band7_uf/weights, np.nan)
        
        
        # convert nans to 0 for later calcs.
        band7_mean = np.nan_to_num(band7_mean, nan=0)

        ### vectorized mean of ratio75
        # ratio75_uf = uniform_filter(ratio75_bgmasked, size = size, mode = "constant")
        ratio75_uf = uniform_filter(ratio75_bgmasked, size = size)
        ratio75_mean = np.where(weights > (1/size), ratio75_uf/weights, np.nan)
        ratio75_mean = np.nan_to_num(ratio75_mean, nan=0)

        ### vectorized calc of band7
        # std = sqrt(mean(X^2) - (mean(X))^2)
        # calc mean of band7 squared
        band7_sq = band7_bgmasked ** 2

        # band7_sq_uf = uniform_filter(band7_sq, size = size, mode = "constant")
        band7_sq_uf = uniform_filter(band7_sq, size = size)
        band7_sq_mean = np.where(weights > (1/size), band7_sq_uf/weights, np.nan)
        band7_sq_mean = np.nan_to_num(band7_sq_mean, nan=0)

        band7_std = np.sqrt(band7_sq_mean - (band7_mean)**2)

        ### vectorized stddev of ratio75
        ratio75_sq = ratio75_bgmasked ** 2

        # ratio75_sq_uf = uniform_filter(ratio75_sq, size = size, mode = "constant")
        ratio75_sq_uf = uniform_filter(ratio75_sq, size = size)
        ratio75_sq_mean = np.where(weights > (1/size), ratio75_sq_uf/weights, np.nan)
        ratio75_sq_mean = np.nan_to_num(ratio75_sq_mean, nan=0)

        ratio75_std = np.sqrt(ratio75_sq_mean - (ratio75_mean)**2)

        ### threshold tests
        band7_add = band7_std * 3
        band7_add[band7_add < 0.08] = 0.08
        
        ratio75_add = ratio75_std * 3
        ratio75_add[ratio75_add < 0.8] = 0.8
        
        threshold_1 = band[7] > (band7_mean + band7_add)
        threshold_2 = ratio75 > (ratio75_mean + ratio75_add)
        threshold_3 = (band[7] / band[6]) > 1.6

        condition_1 = (threshold_1 & threshold_2 & threshold_3)

        ### candidates
        condition_2 = (ratio75 > 1.8) & ((band[7] - band[5]) > 0.17)
    
    # fire pixel must meet ALL conditions and must be a valid background pixel.
    potential_array = (condition_1 & condition_2 & background_mask)
    
    return potential_array

def create_fire_detection_array(band, size = 61):
    '''
    Combines all of the above functions and classifies your raster
    
    Inputs
    ======
    band: dict with keys 1-7 containing the reflectances in bands 1-7s
    size: size of kernel to use for background checks; default 61 as according to Schroeder et al. 2016
    
    Returns
    =======
    An unsigned 8-bit integer array where 0 is not a fire pixel, 
                                        1 is a potential fire pixel, and 
                                        2 is an unambiguous fire pixel
    '''
    
    # create dn_folding and background_masks
    dn_folded = dn_folding(band)
    water_mask = water(band)
    fire_mask = unambiguous_fire(band, dn_folded)
    
    background_mask = background(band, fire_mask, water_mask)
    
    pfire_mask = potential_fire(band, background_mask, size)
    
    fire_detection = np.zeros((band[1].shape[0], band[1].shape[1]))
    
    fire_detection = fire_detection + pfire_mask
    fire_detection = fire_detection + (fire_mask * 2)
    
    return fire_detection.astype(np.uint8)

def conduct_afd(filepath, size = 61, write_tif = False, polygonize = False):
    '''
    Connect to stacked raster and conduct active fire detection.
    
    Inputs
    ======
    filepath: Path to stacked raster - the first seven bands of the raster must be reflectance corrected bands 1-7
    size: size of kernel to use for analysis of pixels; default 61 as per Schroeder et al. 2016
    write_tif: writes the resulting fire_array as a TIF file with the following:
        0 is not a fire pixel, 
        1 is a potential fire pixel
        2 is an unambiguous fire pixel
        255 is a nodata value
    polygonize: converts fire_array into polygons for use in other software
    
    Returns
    =======
    Classified fire_array where:
        0 is not a fire pixel, 
        1 is a potential fire pixel
        2 is an unambiguous fire pixel
        255 is a nodata value
        
    If write_tif and polygonize were set to True, will also produce a raster and a shapefile in the same directory as
        the stacked raster, respectively.
    
    '''
    
    band, profile = load_path(filepath)
    fire_array = create_fire_detection_array(band, size)
    
    if write_tif:
        # update profile data
        profile.update(count = 1, dtype = np.uint8, nodata = 255)
        
        # prep filename
        basename, extension = os.path.splitext(filepath)
        basename = basename + "_AFD"
        afd_filepath = basename + extension
        
        with rio.open(afd_filepath, "w", **profile) as dst:
            dst.write(fire_array, 1)
            dst.close()
    
    if polygonize:
        # shapes is a generator object
        shapes = rf.shapes(
            fire_array,
            mask = fire_array > 0, # This must evaluate to a boolean array
            transform = profile['transform']
        )
        
        polygons = list(shapes)
        
        geom = [shapely.geometry.shape(x[0]) for x in polygons]
        values = [x[1] for x in polygons]
        fire_polygons = gpd.GeoDataFrame({"fire_id": values, "geometry": geom}, crs = profile["crs"])
        
        basename, extension = os.path.splitext(filepath)
        basename = basename + "_AFDshapefile"
        output_file_path = basename + ".GeoJSON"
        
        fire_polygons.to_file(output_file_path, driver='GeoJSON')
    
    return fire_array


# In[10]:


if __name__ == "__main__":
    ### useful tutorial https://docs.python.org/3/howto/argparse.html#argparse-tutorial
    parser = argparse.ArgumentParser(
        description='Identifies unambiguous and potential fire pixels in a stacked Landsat image.'
    )

    # add positional argument (i.e., required argument)
    parser.add_argument('filepath', 
                        help = 'Path to the stacked Landsat 8/9 images. Note that the stacked raster must be \
                        in TOA reflectance values and must have bands 1-7 in the first seven bands.')
    
     # optional flags - the name of the variable is the -- option
    parser.add_argument('-s', '--size', type = int, default = 61,
                        help = 'Indicate the size of kernel to use (strongly recommend not using this flag)') 

    # on/off flags - action indicates what the program should do
    # if flag is called (default will be the opposite for on/off)
    parser.add_argument('-w', '--write_tif', 
                        action='store_true', 
                        help = 'Call if you want ADF to write a uint8 raster of the fire_array to disk - default False') 
    parser.add_argument('-p', '--polygonize', 
                        action='store_true', 
                        help = "Call if you want ADF to write a GeoJSON shapefile of the fire_array to disk - default False")
    
        # grab arguments from command line
    args = parser.parse_args()
    
    
    # calculate TOA
    fire_array = conduct_afd(args.filepath, 
                size = args.size, 
                write_tif = args.write_tif, 
                polygonize = args.polygonize)


# In[ ]:




