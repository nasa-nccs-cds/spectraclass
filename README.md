spectraclass
===============================

Jupyterlab workbench supporting visual exploration and classification of high dimensional sensor data.

Conda CPU Versioned Setup
---------------

    > conda create --name spectraclass
    > conda activate spectraclass
    > conda install -y clang_osx-64 clangxx_osx-64 gfortran_osx-64    # (osx only)
    > conda install -c rusty1s -c conda-forge nb_conda_kernels nodejs jupyterlab jupytext  jupyterlab_server ipywidgets ipycanvas ipyevents itkwidgets ipympl numpy gdal shapely pynndescent xarray rasterio umap-learn scipy scikit-learn toml jupyter_bokeh keras tensorflow rioxarray numba dask netcdf4 matplotlib utm scikit-image pytorch-geometric pytorch
    > conda install -c pyviz hvplot

Installation
------------

    $ git clone https://github.com/nasa-nccs-cds/spectraclass.git
    $ cd spectraclass
    $ python setup.py install

When actively developing your extension, build Jupyter Lab with the command:

    $ jupyter lab --watch

This takes a minute or so to get started, but then automatically rebuilds JupyterLab when your javascript changes.

Note on first `jupyter lab --watch`, you may need to touch a file to get Jupyter Lab to open.

