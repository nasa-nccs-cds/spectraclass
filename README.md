spectraclass
===============================

Jupyterlab workbench supporting visual exploration and classification of high dimensional sensor data.

Conda CPU Environment Setup
---------------

    > conda create --prefix /explore/nobackup/projects/ilab/conda/envs/spectraclass python=3.10 mamba
    > conda create --name spectraclass -c conda-forge python=3.10 mamba

    > conda activate spectraclass
    > pip3 install 'holoviews[all]' gdal pythreejs geoviews torch torchvision
    > mamba install -c conda-forge scikit-learn hvplot rasterio dask numba rioxarray bottleneck

    > conda create -n spectraclass-hvplot -c pyviz -c conda-forge -c pytorch -c nodefaults hvplot gdal geoviews pytorch::pytorch torchvision panel datashader xarray pandas geopandas dask numba streamz networkx intake intake-xarray intake-parquet s3fs scipy scikit-learn spatialpandas pooch rasterio fiona plotly matplotlib jupyterlab

Installation
------------

    $ git clone https://github.com/nasa-nccs-cds/spectraclass.git
    $ cd spectraclass
    $ pip install .

Jupyter Lab Startup
------------

    > conda activate spectraclass
    > cd {PREFIX}/spectraclass/notebooks
    > jupyter-lab

Image Index Creation
--------------------

For example, with DESIS data:

>> gdaltindex -t_srs EPSG:32618 image_index_srs.shp *-SPECTRAL_IMAGE.tif

When actively developing your extension, build Jupyter Lab with the command:

    $ jupyter lab --watch

This takes a minute or so to get started, but then automatically rebuilds JupyterLab when your javascript changes.

Note on first `jupyter lab --watch`, you may need to touch a file to get Jupyter Lab to open.

