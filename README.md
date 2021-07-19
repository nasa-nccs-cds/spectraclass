spectraclass
===============================

Jupyterlab workbench supporting visual exploration and classification of high dimensional sensor data.

Conda CPU Versioned Setup
---------------

    > conda create --name spectraclass
    > conda activate spectraclass

##### Jupyterlab 2
    > conda install -c conda-forge nb_conda_kernels nodejs jupyterlab=2.2.9 jupytext  jupyterlab_server ipywidgets ipycanvas ipyevents itkwidgets qgrid ipympl numpy gdal shapely pynndescent xarray rasterio umap-learn scipy scikit-learn toml jupyter_bokeh keras tensorflow rioxarray numba dask netcdf4 tornado matplotlib utm scikit-image
    > jupyter labextension install @jupyter-widgets/jupyterlab-manager ipycanvas ipyevents ipywidgets itkwidgets qgrid2 jupyter-matplotlib @bokeh/jupyter_bokeh

##### Jupyterlab 3
    > conda install -c conda-forge nb_conda_kernels nodejs jupyterlab jupytext  jupyterlab_server ipywidgets ipycanvas ipyevents itkwidgets ipympl numpy gdal shapely pynndescent xarray rasterio umap-learn scipy scikit-learn toml jupyter_bokeh keras tensorflow rioxarray numba dask netcdf4 tornado matplotlib utm scikit-image

Conda GPU Setup
---------------

    > conda create -n spectraclass -c rapidsai -c nvidia -c conda-forge  -c defaults rapids python cudatoolkit nodejs jupyterlab jupytext ipywidgets ipycanvas ipyevents itkwidgets qgrid jupyter_bokeh netcdf4 keras tensorflow-gpu rioxarray
    > conda activate spectraclass
    > jupyter labextension install @jupyter-widgets/jupyterlab-manager itk-jupyter-widgets qgrid2 @bokeh/jupyter_bokeh
    
OR
--

    > conda create -n spectraclass-r17 -c rapidsai-nightly -c nvidia -c conda-forge -c defaults python rapids=0.17 cudatoolkit=10.2 nodejs jupyterlab jupytext ipywidgets ipycanvas ipyevents itkwidgets qgrid jupyter_bokeh netcdf4 keras tensorflow-gpu rioxarray
    > conda activate spectraclass-r17
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

