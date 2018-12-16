# Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

import earthpy as et

os.chdir(os.path.join(et.io.HOME, 'earth-analytics'))

def contextMap():
    """
    Creates the site context map for Little Cottonwood Canyon, Utah
    
    Parameters
    ----------
    None
    
    Returns
    -------
    fig, ax : figure and axes objects
        The resulting fig and ax objects
    """

    # Create context location map
    # Import world boundary shapefile and state boundary shape file
    worldBound = gpd.read_file(os.path.join(
        'data', 'spatial-vector-lidar', 'global', 'ne_110m_land', 'ne_110m_land.shp'))
    state_boundary_us = gpd.read_file(os.path.join(
        'data', 'spatial-vector-lidar', 'usa', 'usa-states-census-2014.shp'))
    cottonwood_heights_utah = np.array([[-111.6578, 40.5725]])
    geometry = [Point(xy) for xy in cottonwood_heights_utah]
    utah_loc = gpd.GeoDataFrame(geometry,
                                columns=['geometry'],
                                crs={'init': 'epsg:4326'})

    # plot state boundary
    fig, ax = plt.subplots(figsize=(12, 8))
    state_boundary_us.plot(ax=ax, facecolor='white', edgecolor='black')
    utah_boundary = state_boundary_us[state_boundary_us.STUSPS == 'UT']
    utah_boundary.plot(ax=ax,
                  color='silver')
    utah_loc.plot(ax=ax,
                  markersize=52,
                  marker='*',
                  color='darkgreen')
    plt.title('Site Map: Little Cottonwood Canyon, Utah', fontsize=18)
    ax.set_axis_off()
    
    return fig, ax

def createBoundary():
    """
    Creates the site boundary and saves it as a shapefile
    
    Parameters
    ----------
    None
    
    Returns
    -------
    Nothing : Creates and saves the shapefile, if it doesn't already exist
    """
        
    # Create dataframe that holds the site bouding box
    poly_inters = Polygon([(436438, 4484000), (452700, 4492000),
                           (452700, 4496000), (436438, 4493600), (436438, 4484000)])
    poly_in_gdf = gpd.GeoDataFrame([1],
                                   geometry=[poly_inters],
                                   crs={'init': 'epsg:26912'})

    # Rename the columns to work with to_file()
    poly_in_gdf.rename(columns={0: 'poly_no'}, inplace=True)

    # Save to shapefile
    clip_out_path = os.path.join('final-project', 'avalanche-mapping', 'data', 'vector', 'site-boundary', 'site-boundary.shp')
    if os.path.exists(clip_out_path) == False:
        poly_in_gdf.to_file(clip_out_path, driver='ESRI Shapefile')
        
    return