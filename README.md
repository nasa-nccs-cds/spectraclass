spectraclass
===============================

Jupyterlab workbench supporting visual exploration and classification of high dimensional sensor data.

Conda CPU Setup
---------------
   
    > conda create --name spectraclass
    > conda activate spectraclass
    > conda install -c conda-forge nodejs jupyterlab jupytext ipywidgets ipycanvas ipyevents itkwidgets qgrid numpy pynndescent xarray jupyter_bokeh rasterio umap-learn scipy scikit-learn toml keras tensorflow rioxarray numba dask netcdf4 zarr toolz scikit-image
    > jupyter labextension install @jupyter-widgets/jupyterlab-manager itk-jupyter-widgets qgrid2 @bokeh/jupyter_bokeh
    > npm i @jupyterlab/apputils

Conda GPU Setup
---------------

    > conda create -n spectraclass -c rapidsai -c nvidia -c conda-forge  -c defaults rapids python cudatoolkit nodejs jupyterlab jupytext ipywidgets ipycanvas ipyevents itkwidgets qgrid jupyter_bokeh netcdf4 keras tensorflow-gpu
    > conda activate spectraclass
    > jupyter labextension install @jupyter-widgets/jupyterlab-manager itk-jupyter-widgets qgrid2 @bokeh/jupyter_bokeh

Installation
------------

    $ git clone https://github.com/nasa-nccs-cds/spectraclass.git
    $ cd spectraclass
    $ python setup.py install

When actively developing your extension, build Jupyter Lab with the command:

    $ jupyter lab --watch

This takes a minute or so to get started, but then automatically rebuilds JupyterLab when your javascript changes.

Note on first `jupyter lab --watch`, you may need to touch a file to get Jupyter Lab to open.

