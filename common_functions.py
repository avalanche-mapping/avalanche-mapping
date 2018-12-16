import os

from shapely.geometry import (box, Polygon)
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import earthpy.spatial as es
import earthpy as et
import pandas as pd
import rasterio as rio
import rasterstats as rs
from mpl_toolkits.axes_grid1 import make_axes_locatable

# The CRS of all hard-coded coordinates relating to bounding boxes and polygons
SHAPE_CRS = '+init=epsg:4326'

# Defining our bounding box coordinates
MIN_Y = 40.5042792
MAX_Y = 40.6134677
MIN_X = -111.7501813
MAX_X = -111.5591603

# Defining our study area polygon and associated dataframe
study_area = Polygon([[-111.7501813, 40.5042792], 
                      [-111.5588601, 40.5774339], 
                      [-111.5591603, 40.6134677], 
                      [-111.7511872, 40.5943609]])
study_area_gdf = gpd.GeoDataFrame(geometry=[study_area], crs=SHAPE_CRS)

# Defining our study area bounding box and associated dataframe
study_area_box = box(MIN_X, MIN_Y, MAX_X, MAX_Y)
study_area_box_gdf = gpd.GeoDataFrame(geometry=[study_area_box], crs=SHAPE_CRS)

home_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(home_dir, 'data')

vector_dir = os.path.join(data_dir, 'vector')
original_vector_data = os.path.join(vector_dir, 'original')
modified_vector_data = os.path.join(vector_dir, 'modified')

raster_dir = os.path.join(data_dir, 'raster')
original_raster_data = os.path.join(raster_dir, 'original')
modified_raster_data = os.path.join(raster_dir, 'modified')

image_dir = os.path.join(data_dir, 'images')

avalanche_shapes_path = os.path.join(modified_vector_data, "Cottonwood_UT_paths_intersection", "avalanche_intersection.shp")
# Taken from the metadata of the shapefile
avalanche_crs = "+init=epsg:2152"
avalanche_shapes_object = gpd.read_file(avalanche_shapes_path, crs=avalanche_crs)

snowfall_data_path = os.path.join(original_vector_data, "snowfall.csv")
snowfall_data_df = pd.read_csv(snowfall_data_path)

elevation_dem_path = os.path.join(original_raster_data, "ASTGTM2_N40W112", "ASTGTM2_N40W112_dem.tif")


def rasterstats_grouped_by_height(shape, data, data_transform, statistic):
    """
    Given an input shape which consists of the union between avalanche 
    shapes and elevation buckets, determine the statistic of the input 
    data using rasterstats.  Then, group this data into separate height_bucket
    groups and determine the mean difference in the input data between shapes
    that are part of an avalanche path and shapes that are not part of an avalanche 
    path.  Finally, return a dataframe with these values and the resulting difference
    between these values.  See the Returns note for additional details.

    Parameters
    ----------
    shape: geopandas object
        The input shape which is the resulting union of avalanche shapes and elevation buckets.
        The shape input must have the following columns:    
            avalanche_id: a unique identifier/integer for each avalanche
            height_bucket: The height bucket that this avalanche falls under
            geometry_sq_meters: The size of the geometry of each row
    data: ndarray
        The values which you would like to use in this calculation
    data_affine: rasterio.transform
        The transform for the data array
    statistic: string
        The statistic you would like to perform with rasterstats.
    
    Returns
    ----------
    merged_results: geopandas dataframe
        A geopandas dataframe containing the following columns:
            height_bucket: The bucketed height in intervals as specified within the input shape
            [statistic]_avalanche: The mean of the input data at this height where there was no avalanche shapefile
            [statistic]_no_avalanche: The mean of the input data at this height where there was an avalanche shapefile
            difference: The avalanche column subtracted by the no_avalanche column

    """
    results = get_zonal_stats_dataframe(shape, data, data_transform, statistic)
    
    results_avalanche = (results
                        .where(~(pd.isna(results['avalanche_id'])))
                        .groupby(["height_bucket"])[statistic]
                        .mean())
    results_no_avalanche = (results
                        .where(pd.isna(results['avalanche_id']))
                        .groupby(["height_bucket"])[statistic]
                        .mean())

    merged_results = pd.merge(results_avalanche.to_frame(), 
                            results_no_avalanche.to_frame(),
                            how='inner', 
                            on='height_bucket',
                            suffixes=('_avalanche', '_no_avalanche'))

    merged_results["difference"] = merged_results[statistic + "_avalanche"] - merged_results[statistic + "_no_avalanche"]
    merged_results = merged_results.reset_index()
    return merged_results


def get_zonal_stats_dataframe(shape, data, data_transform, statistic):
    """
    A wrapper around zonal stats, this packages the output into a geodataframe containing the following columns:
        avalanche_id
        height_bucket
        [statistic]
        size

    Parameters
    ----------
    shape: geopandas object
        The input shape which is the resulting union of avalanche shapes and elevation buckets.
        The shape input must have the following columns:    
            avalanche_id: a unique identifier/integer for each avalanche
            height_bucket: The height bucket that this avalanche falls under
            geometry_sq_meters: The size of the geometry of each row
    data: ndarray
        The values which you would like to use in this calculation
    data_transform: rasterio.transform
        The transform for the data array
    statistic: string
        The statistic you would like to perform with rasterstats.

    Returns
    ----------
    results: geopandas dataframe
        A geopandas dataframe containing the following columns:
            height_bucket: The bucketed height in intervals as specified within the input shape
            avalanche_id: A unique identifier to indicate what avalanche path this shape is part of
            [statistic]: The statistic over this shape
            size: The size of this shape
    """
    shape = shape[shape['height_bucket'] != 0]
    results_geojson = rs.zonal_stats(shape,
                                     data,
                                     affine=data_transform,
                                     stats=statistic,
                                     all_touched=True,
                                     nodata=False)
    shape[statistic] = [i[statistic] for i in results_geojson]
    return shape


def get_height_bins(dem_path, data, data_transform, data_crs, statistic, input_mask, height_interval=100):
    """
    Given a digital elevation model path, input data and a given statistic (eg max, min, mean), 
    calculate the statistic of the input data for each height_interval of that data.
    
    Parameters
    ----------
    dem_path: string 
        Path to the DEM data
    data: ndarray
        An array of data that you would like to calculate the input 
        statistic on in each generated height bin
    data_transform: affine
        Affine for the input ndarray
    statistic: string
        The statistic for rasterstats
    input_mask: geopandas object
        A shape to mask the incoming DEM
    height_interval: int
        An interval at which you would like to bin the height
    
    Returns
    ----------
    height_list: list of ints
        A list of integers representing the height for each bin
    statistic_list: list of floats
        A list of calculated statistics for each height_list bin on the data
    """
    gpd_height_buckets = dem_to_height_polygon_gdf(dem_path, input_mask, height_interval=height_interval)
    
    data_in_height_buckets = rs.zonal_stats(gpd_height_buckets.to_crs(data_crs),
                                            data,
                                            affine=data_transform,
                                            geojson_out=True,
                                            stats=statistic,
                                            nodata=False)
    height_list = []
    statistic_list = []
    for height_bucket in data_in_height_buckets:
        height_list.append(height_bucket['id'])
        statistic_list.append(height_bucket['properties'][statistic])
    return height_list, statistic_list


def dem_to_height_polygon_gdf(dem_path, input_mask, height_interval=100):
    """
    Given a digital elevation model path, generate a shapefile with shapes according to a specific height interval.
    
    Parameters
    ----------
    dem_path: string
        A path to a specified dem file
    input_mask: geopandas object
        A shape to mask the incoming DEM
    height_interval: int
        The height interval at which to create the DEM polygons (in units of the DEM data)

    Returns
    ----------
    gpd_height_buckets: geopandas dataframe
        A dataframe in the DEM projection that contains shapes that are broken in a contour at every 
        height_interval units in the DEM.
    """
    with rio.open(dem_path) as src:
        dem_projection = src.crs
        mask_reprojected = input_mask.to_crs(dem_projection)
        masked_band, out_transform = rio.mask.mask(src, mask_reprojected.geometry, crop=True)            
    masked_band = np.squeeze(masked_band)
    
    # Make polygon that is defined by heights that are in the specified intervals
    height_buckets = (np.round(masked_band / height_interval) * height_interval).astype(int16)
    attribute_name = 'height_bucket'
    height_buckets_geometry = (
        {'properties': {attribute_name: v}, 'geometry': s}
        for i, (s, v) 
        in enumerate(
            rio.features.shapes(height_buckets, transform=out_transform)))
    # Create a dataframe from our new attributes and dissolve similar height buckets together
    gpd_polygonized_raster = gpd.GeoDataFrame.from_features(list(height_buckets_geometry), crs=dem_projection)
    gpd_height_buckets = gpd_polygonized_raster.dissolve(by=attribute_name)
    
    return gpd_height_buckets
                

def plot_bar(df, 
             x_name, 
             x_label, 
             y_name_list, 
             y_label, 
             title, 
             source, 
             series_names=None, 
             ax=None, 
             fname=None, 
             display_plot=True):
    """
    Create a bar chart from a dataframe.
    
    Parameters
    ----------
    df: dataframe
        Input dataframe (geo or pandas)
    x_name: string
        The column name for the x axis
    x_label: string
        The x-axis label
    y_list: list or ndarray
        A list of column names for the y-axis
    y_label: string
        The y-axis label
    title: string
        Title for the chart
    source: string
        Source text
    series_names: list
        Alternative names to use to represent the series in the legend
    ax: pyplot axes
        axes object to use when plotting.  If none is specified, then one is created in this function
    fname: string
        If specified and a valid writable filename, write this plot to disk at this

    Returns
    ----------
    ax: axes object
        The ax parameter or an ax object that was created in the function
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(16, 9))

    if series_names is not None:        
        df = df.rename(columns={i:j for i, j in zip(y_name_list, series_names)})
        df.plot(ax=ax, x=x_name, y=series_names, kind="bar")
    else:
        df.plot(ax=ax, x=x_name, y=y_name_list, kind="bar")

    ax.set_xlabel(x_label, fontsize=22)
    ax.set_ylabel(y_label, fontsize=22)
    ax.set_title(title, fontsize=25)
    ax.tick_params(axis='x', rotation=45)
    add_data_source_text(ax, source)
    
    # Save and plot fig
    if fname is not None:
        plt.savefig(os.path.join(image_dir, fname))
    if display_plot is True:
        plt.show()
    return ax


def make_shapefile_inverse_within_box(within, shapes):
    """
    From a given shape/collection of shapes, create a shape that
    fills the negative space within a given bounding_box.

    Parameters
    ----------
    within: pandas geodataframe
        The bounding polygon defining the maximum extent of the output polygon
    shapes: pandas geodataframe
        The polygons which will act as cookie-cutters within the incoming polygon

    Returns
    ----------
    inverse: pandas geodataframe
        The result of cookie-cutting within with shapes
    """
    inverse = gpd.overlay(within, shapes, how='symmetric_difference')
    
    # For some reason the result is a crs-naive dataframe.  
    # Update it so it has the same crs as the input.
    inverse.crs = within.crs
    return inverse


def calculate_NDVI(data, red_idx=0, nir_idx=3):
    """
    Calculate the NDVI for a given raster (numpy) input with specified red/infrared indices.
    
    Parameters
    ----------
    data: ndarray
        A 3-d numpy array whose first dimensions represent radiation bands.
    red_idx: int
        The index that corresponds to the red band.
    nir_idx: int
        The index that corresponds to the near-infrared band.
    
    Returns
    ----------
    [NDVI index]: float
    """
    nir = data[nir_idx]
    red = data[red_idx]
    nir[(nir + red) == 0] = np.nan
    red[(nir + red) == 0] = np.nan
    return (nir - red) / (nir + red)


def add_data_source_text(ax, text):
    """
    Adds a black text box with white text centered in the bottom of a given axes object.  
    Alters in-place with no return.
    
    Parameters
    ----------
    ax: matplotlib axes object
        The input axes object
    text: str
        The label for the text.
    """
    ax.text(0.5, 
         0.01, 
         text, 
         clip_on=True, 
         color='white', 
         backgroundcolor='black', 
         ha='center', 
         transform=ax.transAxes)


def plot_rgb_and_vector(raster, 
                        raster_crs,
                        shapes,
                        title, 
                        data_source_text, 
                        plot_paths=True,
                        plot_study_area=True,
                        fname=None,
                        rgb_order=[0, 1, 2],
                        color=None,
                        vmax=1,
                        vmin=-1,
                        cmap=None,
                        display_plot=True):
    """
    Plot incoming 3xnxm numpy array.  If plot_paths or plot_study_area are true, 
    plot the incoming shapes and the study area overlayed on the raster data, with the colormap
    plotted for the "color" column if specified.
    
    Parameters
    ----------
    raster: ndarray
        The raster array which contains at least 3 stacked 2-d arrays.
    raster_crs: dict
        The crs of the incoming raster.
    shapes: pandas geodataframe
        A geodataframe to plot over the raster
    title: str
        A title for the plot.
    data_source_text: str
        The data source text.
    plot_paths: bool
        A bool indicating whether you would like to plot avalanche paths.
    plot_study_area: bool
        A bool indicating whether you would like to plot the study area polygon.
    fname: str
        The filename output.  If None, no file is saved.        
    rgb_order: list
        The equivalent red, green and blue bands to be plotted if raster has at least 3 stacked 2-d arrays.
    color: string
        A column in the shapes dataframe to plot the color of.
    cmap: str
        The color map to use if color is specified
        
    Returns
    ----------
    fig, ax: figure and axes objects
        The resulting fig and ax objects
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    if color is not None and cmap is None:
        cmap = 'RdYlGn'
    # Get the extent of the plotted area
    extent = get_extent(raster_crs)
    
    # Plot the raster
    if raster.ndim == 3:
        es.plot_rgb(raster, rgb=[0, 1, 2], ax=ax, extent=extent)
    else:
        raise ValueError("Cannot plot raster with %d dimension(s)." % raster.ndim)
        
    if plot_paths:
        # Plot the avalanche shapes
        shapes = shapes.to_crs(raster_crs)
        if color is not None:
            plot_boundary = False
        else:
            plot_boundary = True
        plot_dataframe(ax, shapes, color=color, cmap=cmap, vmax=vmax, vmin=vmin, plot_boundary=plot_boundary)
    if plot_study_area:
        # Plot the study area outline
        polygon_geodataframe = study_area_gdf.to_crs(raster_crs)
        plot_dataframe(ax, polygon_geodataframe, plot_boundary=True)
    
    # Add title and plot configuration
    ax.set_title(title, fontsize=30)            
    add_data_source_text(ax, data_source_text)
    ax.get_legend()
    ax.axis('off')
    
    # Save and plot fig
    if fname is not None:
        plt.savefig(os.path.join(image_dir, fname))
    if display_plot:
        plt.show()
    plt.close()
    return fig, ax


def plot_array_and_vector(raster, 
                          raster_crs,
                          shapes,
                          title, 
                          data_source_text, 
                          plot_paths=True,
                          plot_study_area=True,
                          fname=None,
                          color=None,
                          cmap_shapes=None,
                          vmax=1,
                          vmin=-1,
                          rgb_order=[0, 1, 2],
                          cmap_array='PiYG'):
    """
    Exact same as plot_rgb_and_vector(), except plotting a single band instead.
    
    Parameters
    ----------
    raster: ndarray
        The raster array which contains at least 3 stacked 2-d arrays.
    raster_crs: dict
        The crs of the incoming raster.
    shapes: pandas geodataframe
        A geodataframe to plot over the raster
    title: str
        A title for the plot.
    data_source_text: str
        The data source text.
    plot_paths: bool
        A bool indicating whether you would like to plot avalanche paths.
    plot_study_area: bool
        A bool indicating whether you would like to plot the study area polygon.
    fname: str
        The filename output.  If None, no file is saved.        
    rgb_order: list
        The equivalent red, green and blue bands to be plotted if raster has at least 3 stacked 2-d arrays.
    cmap: str
        The color map to use if plotting a single band
        
    Returns
    ----------
    fig, ax: figure and axes objects
        The resulting fig and ax objects
    """
    fig, ax = plt.subplots(figsize=(16, 9))

    # Get the extent of the plotted area
    extent = get_extent(raster_crs)
    
    # Plot the raster
    if raster.ndim == 2:
        colorbar_handle = ax.imshow(raster, cmap=cmap_array, extent=extent, vmax=vmax, vmin=vmin)
        
        # Add colorbar on the right-hand-side
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(colorbar_handle, fraction=.05, cax=cax)
    else:
        raise ValueError("Cannot plot raster with %d dimension(s)." % raster.ndim)

    if plot_paths:        
        # Plot the avalanche shapes
        shapes = shapes.to_crs(raster_crs)
        if color is not None:
            plot_boundary = False
        else:
            plot_boundary = True
        plot_dataframe(ax, shapes, color=color, cmap=cmap_shapes, plot_boundary=plot_boundary)
    if plot_study_area:
        # Plot the study area outline
        polygon_geodataframe = study_area_gdf.to_crs(raster_crs)
        plot_dataframe(ax, polygon_geodataframe, plot_boundary=True)
    
    # Add title and plot configuration
    ax.set_title(title, fontsize=30)            
    add_data_source_text(ax, data_source_text)
    ax.get_legend()
    ax.axis('off')
    
    # Save and plot fig
    if fname is not None:
        plt.savefig(os.path.join(image_dir, fname))
    plt.show()

    return fig, ax


def get_extent(crs):
    """
    Get bounds for the study area given a specific CRS.
    
    Parameters
    ----------
    crs: src.crs object
        The CRS of the shape.
    
    Returns
    ----------
    extent: list
        A list indicating the extent 
    """
    inProj = pyproj.Proj(SHAPE_CRS)
    outProj = pyproj.Proj(crs)
        
    min_x_trans, min_y_trans = pyproj.transform(inProj, outProj, MIN_X, MIN_Y)
    max_x_trans, max_y_trans = pyproj.transform(inProj, outProj, MAX_X, MAX_Y)
    extent = [min_x_trans, max_x_trans, min_y_trans, max_y_trans]
    return extent


def plot_dataframe(ax, polygon, opacity=0.75, plot_boundary=False, color=None, cmap=None, vmax=None, vmin=None):
    """
    Given a file name and axes object, plot a shapefile.
    
    Parameters
    ----------
    ax: axes object
        Incoming axes object.
    polygon_path: str
        Path to the shapefile.
    opacity: float
        The alpha (opacity) of the plotted shape.
    plot_boundary: bool
        Indicates whether you would like to plot only the boundary or the entire polygon.
    """
    if plot_boundary:
        polygon.boundary.plot(ax=ax, alpha=opacity, color='black')
    if color is not None:
        ax = polygon.plot(ax=ax, alpha=opacity, column=color, cmap=cmap, legend=False, vmax=vmax, vmin=vmin)
        fig = plt.gcf()
        scatter = ax.collections[0]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(scatter, fraction=.05, cax=cax)

def generate_unioned_avalanche_overlay(crs, load_from_file_if_available=True):
    """
    Either load the given avalanche union geojson file or create a new overlay from between the height_polygon_gdf and 
    avalanche_shapes.  This is VERY slow if the overlay is calculated (hence why caching this in a file is a good idea)

    Parameters
    ----------
    crs: crs object
        The CRS of the desired output.  Ignored if loading from file (uses the file's CRS)
    load_from_file_if_available: bool
        If this is False, re-create the overlay whether or not there is a geojson representation already on disk.

    Returns
    ----------
    avalanche_overlay_shape: pandas geodataframe
        The union of the avalanche shapefile and the binned-elevation shapefile.
    """
    target_file = os.path.join(modified_vector_data, "union_dataset.geojson")
    if load_from_file_if_available and os.path.isfile(target_file):
        avalanche_overlay_shape = gpd.read_file(target_file)
    else:
        avalanche_overlay_shape = gpd.overlay(height_polygon_gdf
                                            .reset_index()
                                            .to_crs(crs), 
                                        avalanche_shapes_object
                                            .reset_index()
                                            .rename({'index':'avalanche_id'}, axis='columns')
                                            .to_crs(crs), 
                                        how='union')
        avalanche_overlay_shape.crs = crs
        # Calculate the size of each polygon so every polygon has appropriate representation
        avalanche_overlay_shape['geometry_sq_meters'] = avalanche_overlay_shape['geometry'] \
                                                        .to_crs({'init': 'epsg:3395'}) \
                                                        .map(lambda p: p.area / 10**6)
        avalanche_overlay_shape.to_file(target_file, driver="GeoJSON")
    return avalanche_overlay_shape

height_polygon_gdf = dem_to_height_polygon_gdf(elevation_dem_path, study_area_box_gdf)
