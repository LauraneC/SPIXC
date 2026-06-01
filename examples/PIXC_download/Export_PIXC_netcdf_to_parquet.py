import sys
from pixcdust.readers.netcdf import NcSimpleReader
import geopandas as gpd
import os
import warnings

warnings.filterwarnings("ignore")

# Get NetCDF file path from command line
ncfile = sys.argv[1]
output_dir = sys.argv[2]
gdf_geom_name = sys.argv[3]
gdf_geom = gpd.read_file(gdf_geom_name)



try:

    conditions = {}

    ds_PIXC_nc = NcSimpleReader(
        path=ncfile,
        variables=[
            'time', 'height', 'sig0', 'classification', 'water_frac',
            'pixel_area', 'phase_noise_std', 'dheight_dphase', 'geoid',
            'solid_earth_tide', 'load_tide_fes', 'pole_tide', 'geolocation_qual', "phase_unwrapping_region",
            "layover_impact", "inc", "ancillary_surface_classification_flag", "range_index", "azimuth_index",
            "eff_num_rare_looks", "eff_num_medium_looks","xtrk_dist"],
        area_of_interest=gdf_geom,
        conditions=conditions,
    )
    ds_PIXC_nc.open_mfdataset(orbit_info=True)

    ds_PIXC_ggp = ds_PIXC_nc.to_geodataframe()

    parquet_filename = os.path.join(output_dir, f'{os.path.basename(ncfile).replace(".nc", ".parquet")}')
    ds_PIXC_ggp.to_parquet(parquet_filename, index=False, engine='pyarrow')


except Exception as e:
    print(f"Error processing {ncfile}: {e}")
    sys.exit(1)