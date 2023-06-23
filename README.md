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



    > conda create --name spectraclass -c conda-forge python=3.10 
    > conda activate spectraclass

    > conda install -c pyviz -c conda-forge holoviews bokeh hvplot geoviews gdal rasterio jupyterlab ipykernel
    > conda install -c pyviz -c conda-forge holoviews bokeh hvplot geoviews gdal rasterio pytorch jupyterlab ipykernel

    > conda install pytorch::pytorch torchvision -c pytorch
    > conda install -c conda-forge scikit-learn scikit-image 
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

ADAPT install notes
--------------------
    > mkdir ~/.spectraclass/config/neon
    > cp /home/tpmaxwel/.spectraclass/config/neon/agb.py ~/.spectraclass/config/neon



