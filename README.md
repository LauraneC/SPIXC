
# Environment installation

    micromamba create -n swot_env -f env_swot.yml poetry
    pip install -U eodag==2.12.1 packaging
    git clone https://github.com/SWOT-community/PixCDust.git
    cd PixCDust
    poetry install


# Eodag configuration, to access hydroweb-next

Follow these steps:

1. If not already done, install EODAG and packaging latest version using `pip install -U eodag==2.12.1 packaging` or `conda update eodag packaging`

2a. Generate an API-Key from Hydroweb.next portal in your user settings https://hydroweb.next.theia-land.fr/

2b. Carefully store your API-Key

For example for Laurane:
Ix2Xrzhfin3EXRkecPmhw6dkXDi9UTHlzU3Yq5ZKQB6FgOdGps

- either in your eodag configuration file (usually ~/.config/eodag/eodag.yml, automatically generated the first time you use eodag) in auth/credentials/apikey="PLEASE_CHANGE_ME"

- or in an environment variable `export EODAG__HYDROWEB_NEXT__AUTH__CREDENTIALS__APIKEY="PLEASE_CHANGE_ME"`

3. You can change download directory by modifying the variable path_out. By default, current path is used.

4. You are all set, run this script `python download_SWOT_Level-2_HR_Raster_-_100m.py`


Example of eodag.yml (usually in ~/.config/eodag/eodag.yml):

hydroweb_next:

    priority: # Lower value means lower priority (Default: 1)

    search:  # Search parameters configuration

    auth:

        credentials:

            apikey: ifXSoDIYaRQWmuDGuGGJCFaWnnMCzW

    download:

        outputs_prefix: 

# If needed install RiverObs

    git clone https://github.com/SWOTAlgorithms/RiverObs.git
    pip install --use-pep517 -e 

# in case of ModuleNotFoundError: No module named 'zarr.meta'

    micromamba install -c conda-forge zarr==2.13.3