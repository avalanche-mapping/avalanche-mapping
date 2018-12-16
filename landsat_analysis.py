# Import packages and set working directory if needed here
import datetime
import glob
import os

import tempfile

import earthpy.spatial as es
import geopandas as gpds
from matplotlib import pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
from pyproj import Proj, transform
from rasterio import mask
from rasterio.transform import from_origin
import rasterio as rio
import rasterio.plot
import rasterstats as rs
import tarfile
import warnings; 
warnings.simplefilter('ignore')

import common_functions as common

landsat_file_root = os.path.join(common.original_raster_data, 'landsat_summer')
landsat_raw_data = os.path.join(common.original_raster_data, 'landsat_raw')
data_year_list = ['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']

avalanche_overlap_shape = None

# This should match the qa layer
qa_match = 'pixel_qa'

# Set an elevation threshold, in meters, to limit our elevation for our dNDVI analysis
minimum_elevation_threshold = 1000
maximum_elevation_threshold = 3000

running_max_dndvi = None

# The order which we will be concatenating our tifs
color_order = ['red', 'green', 'blue', 'nir']

ndvi_by_year = {}

mean_below_thresh = pd.DataFrame(columns=["year", 
                                          "mean_ndvi_in_slide", 
                                          "mean_ndvi_out_of_slide", 
                                          "mean_dndvi_in_slide", 
                                          "mean_dndvi_out_of_slide", 
                                          "snow_depth"])
    
file_list = glob.glob(os.path.join(landsat_file_root, "L*"))
total_array_count = 0
running_ndvi_sum_array = None
only_analyze_version_number = None
avalanches_only = None

band_colors_landsat_versions = {
                                   '7':{  
                                      'blue':'band1',
                                      'green':'band2',
                                      'red':'band3',
                                      'nir':'band4'
                                   },
                                   '8':{  
                                      'blue':'band2',
                                      'green':'band3',
                                      'red':'band4',
                                      'nir':'band5'
                                   }
                                }

def open_and_crop_geotiff(geotiff_path, out_path, crop_by):
    """
    Open geotiff and crop to the extent of the crop_by geodataframe
    
    Parameters
    ----------
    geotiff_path: string
        Path of the input geotiff to be cropped
    out_path: string
        Target path to write the output to
    crop_by: pandas geodataframe
        Geodataframe or shape to crop by 
    """
    with rio.open(geotiff_path) as src:
        # Reproject our shape to whatever projection the landsat data is in
        src_meta = src.meta.copy()
        crop_by_reprojected = crop_by.to_crs(src.crs)
        band_masked, transform_cropped = mask.mask(src, crop_by_reprojected.geometry, crop=True)
        src_meta['transform'] = transform_cropped
        print(transform_cropped)
        src_meta['width'] = band_masked.shape[2]
        src_meta['height'] = band_masked.shape[1]
    with rasterio.open(out_path, 'w', **src_meta) as dst:
        dst.write(band_masked)   
    

def create_archive_from_tgz(zipped_data_dir, target_data_dir):
    """
    Unzip the tgz files that are in the zipped_data_folder and move them to the
    target_dir only if they don't already exist in the target_dir.
    
    Parameters
    ----------
    zipped_data_dir: string
        Source directory
    target_dir: string
        Target directory
    """
    file_list = glob.glob(os.path.join(zipped_data_dir, "*.tar.gz"))
    print("Number of files found: %d" % len(file_list))
    files_in_data_dir = [os.path.basename(base_name) for base_name in glob.glob(os.path.join(target_data_dir, "*"))]
    print("Number of files already present: %d" % len(files_in_data_dir))
    files_to_upload = [file_to_unzip 
                       for file_to_unzip 
                       in file_list 
                       if os.path.basename(file_to_unzip.replace(".tar.gz", ""))
                       not in files_in_data_dir]    
    print("Number of files to upload: %d" % len(files_to_upload))
    for file_name in files_to_upload:
        with tempfile.TemporaryDirectory() as temporary_directory:
            scene_name = unzip_file(file_name, target_data_dir=temporary_directory)    
            band_list = glob.glob(os.path.join(temporary_directory, scene_name, "*.tif"))
            landsat_version = landsat_path_to_version(scene_name)
            bands_to_save = [i for i in band_colors_landsat_versions[landsat_version].values()] + ["qa"]
            band_list = [fname 
                            for fname in band_list 
                            for bands_to_save_match in bands_to_save 
                            if bands_to_save_match in fname]
            scene_directory = os.path.join(target_data_dir, scene_name)
            os.mkdir(scene_directory)
            for band in band_list:
                band_basename = os.path.basename(band)
                open_and_crop_geotiff(band, os.path.join(scene_directory, band_basename), common.study_area_box_gdf)
        
def unzip_file(file_name, target_data_dir="./"):    
    """
    Unzip all tgz files in file_list and save to target_data_dir.  Each file will
    be saved into its own directory, named the same as the .tgz name, without
    the extension.
    
    Parameters
    ----------
    file_name: string
        Path of the file to unzip
    target_data_dir: str
        Directory to put the files into
    """
    tar = tarfile.open(file_name, "r:gz")
    file_base_no_ext = os.path.basename(file_name.replace(".tar.gz", ""))
    directory_name = os.path.join(target_data_dir, file_base_no_ext)
    tar.extractall(path=directory_name)
    tar.close()
    return file_base_no_ext


def mask_clouds(qa_arr, landsat_ver="8"):
    """
    Creates a cloud mask given a qa_raster.
    
    Parameters
    ----------
    qa_arr: ndarray
        A qa raster containing information about cloud cover.
    landsat_ver: str
        A string representation of the landsat version.
        
    Returns
    ----------
    cloud_mask: ndarray
        A boolean array the same shape as qa_raster containing 
        True values where clouds are present and False values 
        where there are no clouds.
    """    
    if landsat_ver == "8":
        # Much of the terrain was being marked as cloud with the cloud_shadow and cloud 
        # mask values, so had to only mask high confidence clouds.
        cloud_shadow = []#[328, 392, 840, 904, 1350]
        cloud = []#[352, 368, 416, 432, 480, 864, 880, 928, 944, 992]
        high_confidence_cloud = [480, 992]
        high_confidence_cirrus = [834, 836, 840, 848, 864, 880, 898, 900, 904, 912, 928, 944, 992]
        snow_ice = [336, 368, 400, 432, 848, 880, 912, 944, 1352]
        water = [324, 388, 836, 900, 1348]
        combined_list = list(set(cloud_shadow + 
                                 cloud + 
                                 high_confidence_cloud + 
                                 high_confidence_cirrus + 
                                 snow_ice + 
                                 water))
    elif landsat_ver == "7":
        # Much of the terrain was being marked as cloud with the cloud_shadow and cloud 
        # mask values, so had to only mask high confidence clouds.
        cloud_shadow = []#[72, 136]
        cloud = [] #[96, 112, 160, 176, 224]
        low_confidence_cloud = [] #[66, 68, 72, 80, 96, 112]
        medium_confidence_cloud = [] #[130, 132, 136, 144, 160, 176]
        high_confidence_cloud = [224]
        snow_ice = [80, 112, 144, 176]
        water = [68, 132]
        combined_list = list(set(cloud_shadow + 
                                 cloud + 
                                 low_confidence_cloud + 
                                 medium_confidence_cloud + 
                                 high_confidence_cloud + 
                                 snow_ice + 
                                 water))
    else:
        print("Landsat version %s not recognized.  No cloud removal performed." % landsat_ver)
        combined_list = []
    # Create a mask with True values indicating non-cloud pixels
    all_masked_values = np.array(combined_list)
    cloud_mask = np.isin(qa_arr, all_masked_values)
    
    return cloud_mask


def files_from_pattern(pattern, expect_single_file=False):
    """
    From a given pattern, retrieve the filenames.  If expect_single_file is True,
    raise an error if multiple files are returned.  If no files are returned, print
    a message.
    
    TODO: expand pattern to regex instead of only wildcards.
    
    Parameters
    ----------
    pattern: str
        A pattern to match filename.  At this time, only wildcards are accepted (no regular
        expressions).
    expect_single_file: bool
        When True, a valueError is raised if more than one file is returned.
    
    Returns
    ----------
    [file names]: list
        A list of returned file names.
    """
    returned_files = glob.glob(pattern)
    if len(returned_files) == 0:
        print("No files found for pattern %s." % pattern)
    if expect_single_file and len(returned_files) > 1:
        raise ValueError("Expecting a single value to be returned "
                         "and found %d values for pattern %s." 
                         % (len(returned_files), pattern))
    return returned_files

def scene_path_to_year(path):
    """
    Determine the year given a scene ID
    
    Parameters
    ----------
    path: string
        Landsat scene ID (see https://landsat.usgs.gov/landsat-collections#Prod%20IDs)
    
    Returns
    ----------
    [year]: string
        Landsat year
    """
    return os.path.basename(path)[10:14]

def landsat_path_to_version(path):
    """
    Determine the landsat version given a scene ID
    
    Parameters
    ----------
    path: string
        Landsat scene ID (see https://landsat.usgs.gov/landsat-collections#Prod%20IDs)
    
    Returns
    ----------
    [year]: string
        Landsat version (single digit string)
    """
    return os.path.basename(path)[3:4]


def generate_ndvi():
    ndvi_df = pd.DataFrame()

    # Loop through each year in the outer loop
    for year in data_year_list:                
        print("Analyzing year %s" % year)
        year_list_subset = [file for file in file_list if scene_path_to_year(file) == year]
        if not year_list_subset:
            print("No files found for year %s" % (year))
            continue
        accumulated_ndvi_arrays = []
        
        # Loop through each file for the specified year in the inner loop
        for file in year_list_subset:
            file_basename = os.path.basename(file)
            landsat_version_number = landsat_path_to_version(file_basename)
            
            # If this landsat version number is not to be analyzed, continue to the next iteration
            if only_analyze_version_number is not None and only_analyze_version_number == landsat_version_number:
                continue
            band_colors = band_colors_landsat_versions[landsat_version_number]        
            accumulated_bands_list = []
            accumulated_bands_list_unmasked = []       

            # Get QA layer to create our cloud mask
            qa_file_name = files_from_pattern(os.path.join(file, "*%s*" % qa_match), 
                                                expect_single_file=True)[0]

            with rio.open(qa_file_name) as src:
                
                # Reproject our shape to whatever projection the landsat data is in
                landsat_crs = src.crs
                landsat_affine = src.transform
                qa_arr = src.read()
                qa_arr = np.squeeze(qa_arr)
                cloud_mask = mask_clouds(qa_arr, landsat_ver=landsat_version_number)

            # Loop through the colors necessary to create NDVI
            for color in color_order:
                band_file_name = files_from_pattern(os.path.join(file, "*%s*" % band_colors[color]), 
                                                    expect_single_file=True)[0]
                with rio.open(band_file_name) as src:
                    
                    # Reproject our shape to whatever projection the landsat data is in
                    band = src.read()

                    # Cast to float so we can assign nan values
                    band = np.squeeze(band).astype("float")
                    
                    # Mask invalid values
                    band[band == src.nodatavals] = False
                    
                    # Remove the banding effect due to sattelite malfunction with landsat 7 after 2003
                    if landsat_version_number == 7 and int(year) > 2003:
                        band[band == 0] = False
                    
                    # Mask clouds                    
                    band[cloud_mask] = False
                    
                    accumulated_bands_list.append(band)          
                    
            # Create arrays from our cloud-masked and no-cloud-masked band lists
            accumulated_bands_arr = np.array(accumulated_bands_list)

            # Calculate the NDVI array and append to list
            ndvi_arr = common.calculate_NDVI(accumulated_bands_arr)
            ndvi_df = ndvi_df.append({"year": year, 
                                    "fname": file_basename, 
                                    "RGB_arr": accumulated_bands_arr, 
                                    "NDVI_arr": ndvi_arr, 
                                    "landsat_ver": landsat_version_number,
                                    "valid_vals": band[band != False].size
                                    }, 
                                    ignore_index=True)

    # Metadata with pandas is unfortunate; would move to something that handles metadata more robustly like xarray
    # but there's additional overhead/complexity with that. Don't copy this dataframe otherwise this metadata will
    # disappear in the copy.
    ndvi_df.affine = landsat_affine
    ndvi_df.crs = landsat_crs

    return ndvi_df


def generate_dndvi(ndvi_df, avalanche_overlap_shape):
    shapefile_below_threshold = avalanche_overlap_shape[
        (avalanche_overlap_shape['height_bucket'] < maximum_elevation_threshold) & 
        (avalanche_overlap_shape['height_bucket'] > minimum_elevation_threshold)]

    mean_below_thresh = pd.DataFrame()

    ndvi_year = ndvi_df.groupby(by="year")
    for year, group in ndvi_year:
        ndvi_vals = np.array(group['NDVI_arr'].values.tolist())
        annual_ndvi_array = np.nanmax(ndvi_vals, axis=0)
        ndvi_below_elevation_thresh = common.rasterstats_grouped_by_height(shapefile_below_threshold, 
                                                                        annual_ndvi_array, 
                                                                        ndvi_df.affine, 
                                                                        "mean")
        mean_below_thresh = mean_below_thresh.append(
            {
                "year": year,
                "mean_NDVI_in_slide": ndvi_below_elevation_thresh
                                        .replace([np.inf, -np.inf], np.nan)['mean_avalanche']
                                        .mean(), 
                "mean_NDVI_out_of_slide": ndvi_below_elevation_thresh
                                            .replace([np.inf, -np.inf], np.nan)['mean_no_avalanche']
                                            .mean(), 
                "snow_depth": common.snowfall_data_df
                                .loc[common.snowfall_data_df['Year'] == int(year), "Total"]
                                .iat[0]
            },
            ignore_index=True
            )
    mean_below_thresh['mean_dNDVI_in_slide'] = mean_below_thresh['mean_NDVI_in_slide'].diff()
    mean_below_thresh['mean_dNDVI_out_of_slide'] = mean_below_thresh['mean_NDVI_out_of_slide'].diff()
    
    return mean_below_thresh

def generate_avalanche_shapes(ndvi_crs):
    # Generate a single shapefile that contains the union of the
    # avalanche path and the elevation buckets for our entire study area
    # This step takes forever when you run it for the first 
    # time and if the geojson isn't available on disk
    return common.generate_unioned_avalanche_overlay(ndvi_crs)


def ndvi_analysis(ndvi_df, avalanche_overlap_shape):
    # The rgb image that has the most valid values to be used as the background for plotting
    best_rgb = ndvi_df.loc[ndvi_df['valid_vals'].idxmax()]['RGB_arr']

    # Create a 3-d array and take the mean over the 0th dimension (time)
    ndvi_vals = np.array(ndvi_df['NDVI_arr'].values.tolist())
    mean_ndvi_array = np.nanmax(ndvi_vals, axis=0)
    
    # This is the stat we are using in our zonal stats - taking the spatial mean
    stat = "mean"
    slide_paths_elev_buckets = avalanche_overlap_shape[~(pd.isna(avalanche_overlap_shape['avalanche_id']))]
    ndvi_slide_paths = common.get_zonal_stats_dataframe(slide_paths_elev_buckets, 
                                                        mean_ndvi_array, 
                                                        ndvi_df.affine, 
                                                        stat)
    _, _ = common.plot_rgb_and_vector(best_rgb, 
                                      ndvi_df.crs,
                                      ndvi_slide_paths,
                                      "Maximum NDVI in Slide Path Height Intervals\n(Landsat Fig. 1)", 
                                      "Imagery: Landsat, 2008-2018, " + \
                                      "Avalanche Shapes: Utah Automated Geographic Reference Center",
                                      vmax=1,
                                      vmin=-1,
                                      color=stat)

    # Find the deviation from the slide path NDVI and the remainder of the height bin
    # plot the result with a colormap
    ndvi_elevation_buckets = common.rasterstats_grouped_by_height(avalanche_overlap_shape, 
                                                                    mean_ndvi_array, 
                                                                    ndvi_df.affine, 
                                                                    stat)
    slide_paths_elev_buckets["NDVI_deviation"] = np.nan

    # Loop through the different elevation buckets
    for _, row in ndvi_elevation_buckets.iterrows():

        # Isolate just the slide paths in this elevation bucket
        is_in_height_bucket = slide_paths_elev_buckets['height_bucket'] == row['height_bucket']

        # Subtract the mean no-avalanche NDVI from the equivalent elevation bucket within the slide paths 
        slide_paths_elev_buckets.loc[is_in_height_bucket, "NDVI_deviation"] = \
            ndvi_slide_paths[stat] - row[stat + '_no_avalanche']

    _, _ = common.plot_rgb_and_vector(best_rgb, 
                                    ndvi_df.crs,
                                    slide_paths_elev_buckets,
                                    "Deviation From Typical Elevation NDVI\n(Landsat Fig. 2)", 
                                    "Imagery: Landsat Avalanche Shapes: Utah Automated Geographic Reference Center",
                                    vmax=.25,
                                    vmin=-.25,
                                    color="NDVI_deviation")

    common.plot_bar(ndvi_elevation_buckets[ndvi_elevation_buckets['height_bucket'] != 0], 
                "height_bucket", 
                "Elevation (meters)", 
                ['mean_avalanche', 'mean_no_avalanche'], 
                "NDVI", 
                "Maximum NDVI in Avalanche-Prone Areas vs Low Avalanche-Risk Areas\n" + \
                "maximum of %d Landsat datasets\n" % len(ndvi_df) + \
                "(Landsat Fig. 3)", 
                "Landsat, 2008-2018",
                series_names=['Within Avalanche Paths', 
                                'Outside Avalanche Paths'])


def dndvi_analysis(dndvi_df_below_altitude_thresh):
    # First row is na since we don't have a previous NDVI to compare it to.  Drop this.
    dndvi_df_below_altitude_thresh_no_na = dndvi_df_below_altitude_thresh.dropna(axis=0)
    dndvi_df_below_altitude_thresh_no_na.set_index("year")
    ax1 = common.plot_bar(dndvi_df_below_altitude_thresh_no_na, 
                        'year', 
                        'Year', 
                        ['mean_dNDVI_in_slide','mean_dNDVI_out_of_slide'], 
                        "Mean dNDVI Between Years\n(positive indicates growth)", 
                        "dNDVI Below %s Meters and above %s Meters vs Snowfall\n(Landsat Fig. 5)"
                        % (maximum_elevation_threshold, minimum_elevation_threshold), 
                        "Imagery Data: Landsat 2008-2018\n" + \
                        "Snowfall Data: Utah Department of Transportation\n" + \
                        "Avalanche information: Utah Automated Geographic Reference Center", 
                        series_names=["Annual dNDVI In Slide Paths", "Annual dNDVI Outside Slide Paths"],
                        display_plot=False)
    ax2 = ax1.twinx()
    dndvi_df_below_altitude_thresh_no_na['snow_depth'].plot(x='year', ax=ax2)
    ax2.set_ylabel('Total Snowfall in Prior Winter (inches)', fontsize=22)
    plt.show()


def calculate_maximum_diff(ndvi_df):
    # Get absolute max dNDVI for each pixel in our study period
    combined_arr = np.array(ndvi_df['NDVI_arr'].values.tolist())
    absolute_max_diff_ndvi = np.nanmax(combined_arr, axis=0) - np.nanmin(combined_arr, axis=0)
    _ = common.plot_array_and_vector(absolute_max_diff_ndvi, 
                                    ndvi_df.crs,
                                    common.avalanche_shapes_object,
                                    "Maximum Absolute Difference in NDVI, 2008-2018\n(Landsat Fig. 4)", 
                                    "Imagery: Landsat, 2008-2018",
                                    vmax=2,
                                    vmin=0,
                                    cmap_array='OrRd')