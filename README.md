spectraclass
===============================

Jupyterlab workbench supporting visual exploration and classification of high dimensional sensor data.

Conda CPU Environment Setup
---------------

    > conda create --prefix /explore/nobackup/projects/ilab/conda/envs/spectraclass  -c conda-forge -c nodefaults python=3.10 mamba
    > conda create --name spectraclass3 -c conda-forge python=3.10 mamba  

    > conda activate spectraclass
    > mamba install -c pytorch -c conda-forge -c nodefaults gdal pytorch::pytorch torchvision 
    > pip3 install 'holoviews[all]' geoviews hvplot rasterio rioxarray jupyterlab pythreejs scikit-learn

    > mamba install -c pyviz -c conda-forge holoviews jupyterlab hvplot
    > mamba install -c pytorch -c conda-forge -c nodefaults pytorch::pytorch torchvision 
    > mamba install -c conda-forge -c nodefaults gdal scikit-learn rasterio rioxarray

    > conda install -c pyviz holoviews 
    > conda install -c conda-forge hvplot jupyterlab
    > conda install pytorch::pytorch torchvision -c pytorch
    > conda install -c conda-forge gdal scikit-learn xarray scikit-image rioxarray
    > pip install pythreejs



    > conda create --name spectraclass -c conda-forge python=3.10 mamba 

    > mamba install -c pyviz -c conda-forge holoviews geoviews 
    > mamba install -c conda-forge jupyterlab hvplot
    > mamba install -c pytorch -c conda-forge pytorch::pytorch torchvision 
    > mamba install -c conda-forge gdal scikit-learn xarray scikit-image rioxarray
    > pip install pythreejs



    > conda create --name spectraclass -c conda-forge python=3.9
    > conda activate spectraclass
    > conda install -c pyviz holoviews bokeh hvplot geoviews
    > conda install -c conda-forge jupyterlab
    > conda install pytorch::pytorch torchvision -c pytorch
    > conda install -c conda-forge scikit-learn scikit-image affine
    > pip install pythreejs

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

