from pixcdust.downloaders.hydroweb_next import PixCDownloader
import geopandas as gpd
from datetime import datetime
import glob
import subprocess
import os
import pandas as pd
import polars as pl

gdf_geom_file_name = "/cnrm/cen/micro_ondes/NO_SAVE/charriel/Joux/area_joux.gpkg" #name of area where to download the data
filename_crash_file = "/cnrm/cen/micro_ondes/NO_SAVE/charriel/Joux/crash_files.txt" #name of file where to put the crashed filed
path_data = '/cnrm/cen/micro_ondes/NO_SAVE/charriel/Joux/' #name of the path where to save the data

dates = (datetime(2023,1,1),datetime(2026,3,25))


output_dir = os.path.join(path_data, 'gpd_withoutfiltering')
path_netcdf = os.path.join(path_data, 'netcdf')

# ============================================================================
# Download PIXC as NetCDF
# ============================================================================
os.makedirs(path_netcdf, exist_ok=True)
#
gdf_geom = gpd.read_file(gdf_geom_file_name)

pixcdownloader = PixCDownloader(
    gdf_geom,
    dates,
    verbose=1,
    path_download=path_netcdf
    )
pixcdownloader.search_download()

# ============================================================================
# Export them as csv files with selected variables
# ============================================================================

nc_files = glob.glob(path_data + 'netcdf/*/*.nc')
os.makedirs(output_dir, exist_ok=True)
crashed_files = []
for ncfile in nc_files:
    print(f"Processing: {ncfile}")
    try:
        result = subprocess.run(
            ['python3', 'Export_PIXC_netcdf_to_csv.py', ncfile, output_dir, gdf_geom_file_name],
            check=True
        )
    except subprocess.CalledProcessError as e:
        try:
            result = subprocess.run(
                ['python3', 'Export_PIXC_netcdf_to_csv.py', ncfile, output_dir, gdf_geom_file_name],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ Crashed on {ncfile} with exit code {e.returncode}")
            crashed_files.append(ncfile)

# Optionally, save the crashed file paths to a text file
if crashed_files:
    with open(filename_crash_file, "w") as f:
        for file in crashed_files:
            f.write(f"{file}\n")

    print(f"\n⚠️ {len(crashed_files)} files crashed. Paths saved to {filename_crash_file}.")
else:
    print("\n✅ All files processed successfully.")

# ============================================================================
# Merge them into one parquet file
# ============================================================================
csv_files = glob.glob(os.path.join(output_dir, "*.parquet"))

# Read and concatenate all files
df_combined = pd.concat((pd.read_parquet(file) for file in csv_files), ignore_index=True)

df_combined.to_parquet(f"{path_data}/combined.parquet")
