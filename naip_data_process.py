# Import packages and set working directory if needed here
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
os.chdir(os.path.join(et.io.HOME, 'earth-analytics'))

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