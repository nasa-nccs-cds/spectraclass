spectraclass
===============================

Jupyterlab workbench supporting visual exploration and classification of high dimensional sensor data.

Conda CPU Environment Setup
---------------

    > conda create -n spectraclass python=3.7
    > conda activate spectraclass
    > conda install -c conda-forge mamba
    > mamba install -c conda-forge nb_conda_kernels nodejs jupyterlab jupyterlab_server ipywidgets ipympl matplotlib mplcursors pythreejs contextily kdtree numpy pynndescent xarray rasterio umap-learn scipy scikit-learn dask netcdf4 scikit-image gdal owslib pyepsg rioxarray cartopy shapely bottleneck geopandas utm keras tensorflow
    > mamba install pytorch -c pytorch

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

