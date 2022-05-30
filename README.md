spectraclass
===============================

Jupyterlab workbench supporting visual exploration and classification of high dimensional sensor data.

Conda CPU Environment Setup
---------------

    > conda create -n spectraclass -c conda-forge jupyterlab=3 "ipykernel>=6" xeus-python
    > conda activate spectraclass
    > conda install -c conda-forge mamba
    > mamba install -c conda-forge pyepsg ipysheet pytorch pythreejs nb_conda_kernels nodejs jupyterlab jupyterlab_server ipywidgets ipympl matplotlib mplcursors pythreejs numpy xarray rasterio scipy scikit-learn dask netcdf4 scikit-image numba gdal owslib rioxarray cartopy shapely bottleneck geopandas keras tensorflow

The x-ray application requires the following additional packages:

    > mamba install -c conda-forge jupyter_bokeh

Installation
------------

    $ git clone https://github.com/nasa-nccs-cds/spectraclass.git
    $ cd spectraclass
    $ python setup.py install

Image Index Creation
--------------------

For example, with DESIS data:

>> gdaltindex -t_srs EPSG:32618 image_index_srs.shp *-SPECTRAL_IMAGE.tif

When actively developing your extension, build Jupyter Lab with the command:

    $ jupyter lab --watch

This takes a minute or so to get started, but then automatically rebuilds JupyterLab when your javascript changes.

Note on first `jupyter lab --watch`, you may need to touch a file to get Jupyter Lab to open.

