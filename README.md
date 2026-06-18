
# Installation

## Environment installation

    micromamba create -n swot_env -f env_swot.yml poetry
    pip install -U eodag==2.12.1 packaging
    git clone https://github.com/SWOT-community/PixCDust.git
    cd PixCDust
    poetry install


## Eodag configuration, to access hydroweb-next

Follow these steps:

1a. Generate an API-Key from Hydroweb.next portal in your user settings https://hydroweb.next.theia-land.fr/

1b. Carefully store your API-Key

For example for Laurane:
5okXEHjoarWR1VQH7NB3AnU1d0KY2mPuvHyDlXkBQ1ZdTdFvbY

- either in your eodag configuration file (usually ~/.config/eodag/eodag.yml, automatically generated the first time you use eodag) in auth/credentials/apikey="PLEASE_CHANGE_ME"

- or in an environment variable `export EODAG__HYDROWEB_NEXT__AUTH__CREDENTIALS__APIKEY="PLEASE_CHANGE_ME"`

2. You can change download directory by modifying the variable path_out. By default, current path is used.


Example of eodag.yml (usually in ~/.config/eodag/eodag.yml):

hydroweb_next:

    priority: # Lower value means lower priority (Default: 1)

    search:  # Search parameters configuration

    auth:

        credentials:

            apikey: ifXSoDIYaRQWmuDGuGGJCFaWnnMCzW

    download:

        outputs_prefix: 

## If needed, you can install RiverObs

    git clone https://github.com/SWOTAlgorithms/RiverObs.git
    pip install --use-pep517 -e 

## In case of ModuleNotFoundError: No module named 'zarr.meta'
    micromamba install -c conda-forge zarr==2.13.3

# Structure of the code

The code is divided into 3 main class objects:
- data_field : to manipulate in situ data, stored in a csv file
- swot_processing : to manipulate PIXC data, stored in a csv or parquet file using pandas
- swot_processing_fast : to manipulate PIXC data, stored in a csv or parquet file using polars

# Examples 

Several examples are provided in these three folders:
- PIXC_download : script framework used to download PIXC data as .parquet filed using PIXCDust
- PIXC_process : notebook to process PIXC data 
- SP_download_and_process : notebook to load and filter SP data



