from __future__ import print_function
from setuptools import setup, find_packages

name = 'spectraclass'
LONG_DESCRIPTION = 'Jupyterlab workbench supporting visual exploration and classification of high dimensional sensor data.'
version = "0.1"

setup_args = dict(
    name=name,
    version=version,
    description=LONG_DESCRIPTION,
    include_package_data=True,
    install_requires=[ 'ipywidgets>=7.0.0',  ],
    packages=find_packages(),
    zip_safe=False,
    author='Thomas Maxwell',
    author_email='thomas.maxwell@nasa.gov',
    url='https://github.com/nasa-nccs-cds/spectraclass',
    data_files=[ ],
    keywords=[  'ipython',  'jupyter',  'widgets',  ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Framework :: IPython',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Graphics',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)

setup(**setup_args)



