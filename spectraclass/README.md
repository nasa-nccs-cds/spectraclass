# astrolab
Jupyterlab workbench supporting visual exploration and classification of astronomical xray and light curve data.

#### Create conda env + jupyterlab with extensions
   
    > conda create --name astrolab
    > conda activate astrolab
    > conda install -c conda-forge nodejs jupyterlab jupytext ipywidgets ipycanvas ipyevents qgrid numpy pynndescent xarray jupyter_bokeh rasterio umap-learn scipy scikit-learn toml keras tensorflow rioxarray numba dask netcdf4 zarr toolz scikit-image
    > jupyter labextension install @jupyter-widgets/jupyterlab-manager  ipycanvas ipyevents qgrid2 @bokeh/jupyter_bokeh 
    > npm i @jupyterlab/apputils

