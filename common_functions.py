import os

from shapely.geometry import (box, Polygon)
import geopandas as gpds
import matplotlib.pyplot as plt
import pyproj
import earthpy.spatial as es
import earthpy as et
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
study_area_gdf = gpds.GeoDataFrame(geometry=[study_area], crs=SHAPE_CRS)

# Defining our study area bounding box and associated dataframe
study_area_box = box(MIN_X, MIN_Y, MAX_X, MAX_Y)
study_area_box_gdf = gpds.GeoDataFrame(geometry=[study_area_box], crs=SHAPE_CRS)

HOME_DIR = os.path.join(et.io.HOME, 'data')
original_data = os.path.join(HOME_DIR, 'original')
avalanche_shapes_path = os.path.join(original_data, "Cottonwood_UT_paths_intersection", "avalanche_intersection.shp")
 
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
    
    return (data[nir_idx] - data[red_idx]) / (data[nir_idx] + data[red_idx])


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


def plot_array_and_vector(raster, 
                          raster_crs,
                          title, 
                          data_source_text, 
                          plot_paths=True,
                          plot_study_area=True,
                          fname=None,
                          rgb_order=[0, 1, 2],
                          cmap='PiYG'):
    """
    Plot incoming numpy array.  If the array is a single band, plot using given cmap.  
    If array is 3 or more bands, plot as RGB with the requested rgb order.  If plot_paths 
    or plot_study_area are true, plot the avalanche paths and the study area overlayed 
    on the raster data.
    
    Parameters
    ----------
    raster: ndarray
        The raster array which contains at least 3 stacked 2-d arrays.
    raster_crs: dict
        The crs of the incoming raster.
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
    fig, ax = plt.subplots(figsize=(10, 10))

    # Get the extent of the plotted area
    extent = get_extent(raster_crs)
    
    # Plot the raster
    if raster.ndim == 3:
        es.plot_rgb(raster, rgb=[0, 1, 2], ax=ax, extent=extent)
    elif raster.ndim == 2:
        colorbar_handle = ax.imshow(raster, cmap=cmap, extent=extent)
        
        # Add colorbar on the right-hand-side
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(colorbar_handle, fraction=.05, cax=cax)
    else:
        raise ValueError("Cannot plot raster with %d dimension." % raster.ndim)
        
    # Plot the avalanche shapes
    avalanche_shapes_object = gpds.read_file(avalanche_shapes_path)
    plot_dataframe(ax, avalanche_shapes_object.geometry, plot_boundary=True)
    
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
        plt.savefig(fname)
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


def plot_dataframe(ax, polygon, opacity=1.0, plot_boundary=False):
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
        polygon = polygon.boundary
    polygon.plot(ax=ax, alpha=opacity)