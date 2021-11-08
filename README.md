spectraclass
===============================

Jupyterlab workbench supporting visual exploration and classification of high dimensional sensor data.

Conda CPU Versioned Setup
---------------

    > conda create --name spectraclass
    > conda activate spectraclass
    > conda install -c conda-forge conda nb_conda_kernels nodejs jupyterlab jupytext jupyterlab_server ipywidgets ipycanvas ipyevents itkwidgets numpy pynndescent xarray rasterio umap-learn scipy scikit-learn dask netcdf4 keras tensorflow gdal owslib pyepsg hvplot bokeh regionmask rioxarray cartopy shapely pandas geoviews hvplot utm bottleneck geopandas

   > conda install -c conda-forge nb_conda_kernels jupyterlab jupytext jupyterlab_server ipywidgets ipycanvas ipyevents  pynndescent xarray rasterio umap-learn scipy scikit-learn dask netcdf4 scikit-image gdal owslib pyepsg rioxarray cartopy shapely bottleneck geopandas geoviews hvplot utm
Legacy:

    > conda install -c rusty1s -c conda-forge conda nb_conda_kernels nodejs jupyterlab jupytext jupyterlab_server  
                        ipycanvas ipyevents itkwidgets ipympl numpy gdal geos shapely kdtree pynndescent xarray  
                        umap-learn scipy scikit-learn toml jupyter_bokeh keras tensorflow rioxarray numba dask   
                        utm proj4 pyproj pyepsg owslib basemap ipywidgets scikit-image pytorch-geometric pytorch
                        torchmetrics  matplotlib hvplot bokeh basemap-data-hires rasterio netcdf4
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

