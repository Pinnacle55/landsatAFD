# Active Fire Detection

![alt_text](https://github.com/Pinnacle55/landsatAFD/blob/f8ed3a062aa5ee0b7d636f99080d0eade96732d3/Images/AFD_GIS_rhodes.png)

There are many natural and anthropogenic causes of fire; these include forest fires caused by lightning strikes or volcanic activity or anthropogenic fires that can be caused by slash-and-burn agriculture or biomass burning. Fires can either be geohazards by themselves, And can be the cause of significant infrastructural damage as well as the loss of human life. In addition to these safety issues, there are several negative ecological effects of fires such as a significant reduction in air quality and this changes in soil chemistry.

This is repository contains an algorithm that can be used for active fire detection using remote sensing data from Landsat satellites. It uses Level 1 products from Landsat 8/9 that have been corrected for top-of-atmosphere reflectances. Final product is a Python file that can be run on the command line and can be used to rapidly prepare both raster and vector data about any fire-affected pixels in the dataset being analysed. The algorithms used here were applied to two case studies: Landsat data from the 2023 Rhodes fires and 2015 data from Kalimantan on the island of Borneo.

The active fire detection algorithm implemented in this repository is based on the methodology described in Schroeder et al. 2016. It is currently implemented for daytime landsat data.

This particular algorithm requires you to have preprocessed your Landsat images. Specifically, it requires you to have corrected your data for tablet atmosphere advances as well as stack your data, such that the first seven bands raster image represent Bands 1 to 7.

While the algorithm itself cannot do this for you, included in this repository is a separate algorithm called `landsat_processing_utils.py` which can be used for a variety of image preprocessing. This includes the TOA reflectance calculation, cropping, and stacking of Landsat data. An example of how you would use this utility from the command line to create a stacked raster file is shown below:

```
python landsat_processing_utils.py [filepath to folder containing landsat image data] -s -c -o [output_directory]
```

Note the use of the -c flag - the algorithm used by Schroeder et al. specifically requires the TOA reflectances to NOT be corrected for solar elevation.

## Implementation

`active_fire_detection.py` uses a variety of algebraic inequalities between Landsat bands to identify whether a pixel is an unambiguous fire pixel or a potential fire pixel. While most of these algebraic inequalities are computationally rapid to calculate, one of the requirements is the calculation of the standard deviation of a 61 x 61 kernel. Furthermore, this standard deviation must only take into account valid background pixels which must be identified via a different set of threshold and context requirements. This is extremely tricky to implement in a computationally efficient manner. 

The traditional way of implementing a specific filter onto a kernel of pixels requires the use of `scipy.ndimage.generic_filter`. However, although this function is extremely powerful and can be used to run any function, it is extremely slow especially when applied to raster images. In most cases, the application of `generic_filter` requires the use of multi-threading in order for the algorithm to run at acceptable speeds. It is therefore preferable to identify an alternative, faster means of implementing the function you want if at all possible. 

Thankfully, this is possible for standard deviations. The code snippet below compares the performance of two ways of calculating the standard deviation of a 61x61 kernel on a 1000x1000 pixel numpy array.

```
%%time
np.sqrt(uniform_filter(test**2, size = 61, mode = "constant") - uniform_filter(test, size = 61, mode = "constant")**2)

>>>Wall time: 27.4 ms

%%time
generic_filter(test, np.std, size = 61, mode = "constant")

>>>Wall time: 19.9 s
```

Based on the work described by [Horizon Maps](https://github.com/horizonmaps/landsatfire/tree/main), it is possible to create an alternative algorithm that applies the necessary standard deviation filter to the data by using `uniform_filter()`, which is a linear function find the mean of a kernel and is much faster than the `generic_filter` implementation. You can see the differences in speed in the example below - the proposed solution is almost three orders of magnitude faster than using `generic_filter()`.

The end result after processing is a numpy array that contains 0 in all pixels where no fires were detected, 1 in all pixels that are potential fire pixels, and 2 in all pixels that are unambiguously fire pixels. 255 is set as the no-data value. active_fire_detection.py allows you to store this data as both raster (uint8) and vector data, which are much more space efficient that other storage options (~50 MB and ~200kb for the raster/vector data taken from an uncropped Landsat image, respectively).

## Visualization

The fire data is best presented using GIS software, is it allows for more fine tuning of the map compared to the available tools in matplotlib. In particular, using a shapefile to present the fire data is much more effective than using raster data. An example visualisation of the extent of the fire that occurred in Rhodes, Greece in 2023, which ravaged about 135,000 hectares of forest and vegetation, burned more than 50,000 olive trees and many domestic animals, destroyed about 50 homes and led to the mass evacuation of tourists from the area. 

![alt_text](https://github.com/Pinnacle55/landsatAFD/blob/f8ed3a062aa5ee0b7d636f99080d0eade96732d3/Images/TCC_GIS_rhodes.png) ![alt_text](https://github.com/Pinnacle55/landsatAFD/blob/f8ed3a062aa5ee0b7d636f99080d0eade96732d3/Images/AFD_GIS_rhodes.png)

The fire detection algorithm was also applied to 2015 images from Kalimantan, Indonesia. These fires are the result of biomass burning on local farms due to the slash-and-burn agriculture practiced in this region. The smoke produced from these fires results in haze, which travels westward and severely affects the air quality in many parts of Southeast Asia. This has been an ongoing international issue for many years.

