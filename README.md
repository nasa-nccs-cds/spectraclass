spectraclass
===============================

Jupyterlab workbench supporting visual exploration and classification of high dimensional sensor data.

Conda CPU Environment Setup
---------------

    > conda create --prefix /explore/nobackup/projects/ilab/conda/envs/spectraclass -c conda-forge python=3.9 mamba
    > conda create --name spectraclass.hv -c pyviz -c conda-forge python=3.9 mamba holoviews geopandas geoviews hvplot

    > conda activate spectraclass.hv
    > mamba install -c pyviz -c conda-forge ipympl jupytext pyepsg ipysheet tensorflow h5py nb_conda_kernels nodejs jupyterlab jupyterlab_server rasterio dask netcdf4 scikit-image numba owslib rioxarray bottleneck  
    > pip install pythreejs

Installation
------------

    $ git clone https://github.com/nasa-nccs-cds/spectraclass.git
    $ cd spectraclass
    $ pip install .

Image Index Creation
--------------------

For example, with DESIS data:

>> gdaltindex -t_srs EPSG:32618 image_index_srs.shp *-SPECTRAL_IMAGE.tif

When actively developing your extension, build Jupyter Lab with the command:

    $ jupyter lab --watch

This takes a minute or so to get started, but then automatically rebuilds JupyterLab when your javascript changes.

Note on first `jupyter lab --watch`, you may need to touch a file to get Jupyter Lab to open.

