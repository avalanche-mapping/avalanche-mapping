# Import packages and set working directory if needed here
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.image as mpimg

import rasterio as rio
from rasterio.mask import mask
from rasterio.plot import plotting_extent
from rasterio.merge import merge

import geopandas as gpd
from shapely.geometry import mapping, box
from shapely.geometry import Polygon

import earthpy as et
import earthpy.spatial as es
import earthpy.clip as cl

# set working directory
data_dir = os.path.join(et.io.HOME, 'earth-analytics')

def mergeNAIP(src_dir, out_fn):
    """
    Merges the NAIP .tif files into one .tif file, if it doesn't already exist
    
    Parameters
    ----------
    src_dir : string
        The directory that contains the NAIP .tif files
    out_fn : string
        The output file name, including the directory path
    
    Returns
    -------
    Nothing : Creates and saves a merged .tif file to directory and file name specified in out_fn
    """

    # If the output file doesn't already exist, create it
    if os.path.exists(out_fn) == False:
    
        # Create list of files to merge
        search_criteria = "m*.tif"
        file_list = glob(os.path.join(src_dir, search_criteria))

        # Open each file in rasterio
        # Create an empty list that will hold all the opened files
        src_files_to_mosaic = []

        # Loop through and open each file and add it to the mosaic list
        for file in file_list:
            src = rio.open(file)
            src_files_to_mosaic.append(src)

        # Using rasterio.merge, merge all files into one array
        mosaic, out_trans = merge(src_files_to_mosaic)
    
        # Write the mosaic to an output file
        # Set up the metadata for the output file
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": mosaic.shape[1],
                         "width": mosaic.shape[2],
                         "transform": out_trans,
                         "crs": src.crs})

        with rio.open(out_fn, "w", **out_meta) as dest:
            dest.write(mosaic)
        
    return

def cropNAIP(src_fn, boundary_fn, out_fn):
    """
    Crops a NAIP .tif file given a site boundary shapefile and saves it to a .tif, if it doesn't already exist
    
    Parameters
    ----------
    src_fn : string
        The file name and directory of the .tif you want to crop
    boundary_fn : string
        The file name and direcotory of the shape file you want to crop with
    out_fn : string
        The output file name, including the directory path
    
    Returns
    -------
    Nothing : Creates and saves a cropped .tif file to directory and file name specified in out_fn
    """
    
    if os.path.exists(out_fn) == False:
        # Open site boundary shapefile
        #clip_out_path = "data/final-project/cottonwood-heights-utah/vector-clip/utah-avalanche-clip.shp"
        site_boundary = gpd.read_file(boundary_fn)
    
        with rio.open(src_fn) as src:
            mosaic_crop, mosaic_meta = es.crop_image(
                src, site_boundary)
            mosaic_extent = plotting_extent(
                mosaic_crop[0], mosaic_meta['transform'])

        # Set up the metadata for the output file
        out_meta_crop = src.meta.copy()
        out_meta_crop.update({"driver": "GTiff",
                         "height": mosaic_crop.shape[1],
                         "width": mosaic_crop.shape[2],
                         "transform": mosaic_meta['transform'],
                         "crs": src.crs})

        with rio.open(out_fn, "w", **out_meta_crop) as dest:
            dest.write(mosaic_crop)
        
    return

def plotHogumFork(src_fn, avi_paths, png_fn):
    """
    Plots an NDVI subset of the avalanche paths called Hogum Fork along with a png image of a Google Earth version
    
    Parameters
    ----------
    src_fn : string
        The file name and directory of the .tif you want to crop and compute NDVI from
    avi_path : geopandas dataframe
        The avalanche paths that fall inside the Little Cottonwood Pass site boundary
    png_fn : string
        The filename of the png image to plot
    
    Returns
    -------
    Nothing : Creates and saves a cropped .tif file to directory and file name specified in out_fn
    """
    
    # Set the extent upside down so that the plot matches the google earth orientation
    hogum_fork_extent = (441000.0, 437000.0, 4491500.0, 4486400.0)

    # Grab only the avalanche paths within the hogum fork area
    avi_paths.Shape_Area = avi_paths.Shape_Area.astype(int)
    hogum_fork_df = avi_paths[avi_paths['Shape_Area'].isin(
        [383328, 180726, 2031608, 907672, 979786, 212712, 113984, 50346, \
         63824, 27014, 26540, 51545, 15734, 150553, 67556, 38878, 33543, 47367])]
    
    # Create a clipping bounding box for hogum fork area
    SHAPE_CRS = '+init=epsg:26912'
    hogum_fork_area = Polygon([[437000.0, 4487000.0],
                               [437000.0, 4491500.0],
                               [441000.0, 4491500.0],
                               [441000.0, 4487000.0]])
    hogum_fork_area_gdf = gpd.GeoDataFrame(geometry=[hogum_fork_area], crs=SHAPE_CRS)
    
    # Clip the data to the hogum fork site
    #mosaic_2011_fp = "data/final-project/cottonwood-heights-utah/naip/outputs/naip_2011_mosaic_crop.tif"
    with rio.open(src_fn) as hogum_fork_src:
        hogum_fork_2011, hogum_fork_2011_affine = mask(hogum_fork_src, hogum_fork_area_gdf.geometry, crop=True)
        
    # create ndvi of hogum fork
    hogum_fork_2011 = hogum_fork_2011.astype(int)
    ndvi_hogum_fork = es.normalized_diff(b2=hogum_fork_2011[3], b1=hogum_fork_2011[2])
    
    # Open google earth image
    img=mpimg.imread(png_fn)
    
    # Plot the data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 17))

    # Plot the google earth image
    imgplot = ax1.imshow(img)
    ax1.set_axis_off()

    # Plot the hogum fork NDVI mosaic
    hogum_fork = ax2.imshow(ndvi_hogum_fork, cmap="PiYG",
                            extent=hogum_fork_extent)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    fig.colorbar(hogum_fork, cax=cax)

    hogum_fork_df.boundary.plot(ax=ax2, color="black")
    ax2.text(0.5,
            0.01,
            'NAIP 2011 Data',
            clip_on=True,
            color='white',
            backgroundcolor='black',
            ha='center',
            transform=ax1.transAxes)
    ax2.set_axis_off()
    
    return fig, ax1, ax2