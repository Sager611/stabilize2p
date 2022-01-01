import os
import setuptools
from subprocess import call
from distutils.command.install import install


def read(fname):
    # retrieves ``fname`` file contents as a string
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(
    name='stabilize2p',
    version='0.1',
    license='GPLv2',
    description='Registration tools for 2-photon imaging',
    url='',  # TODO: add github url
    long_description=read('README.rst'),
    keywords=['deformation', 'registration', 'imaging', 'dnn', '2-photon'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Operating System :: Linux',
    ],

    packages=['stabilize2p'],  # folders with the packages' modules
    python_requires='>=3.8',
    install_requires=[
        'tifffile',
        'jupyterlab',
        'numpy',
        'matplotlib',
        'opencv-python',
        'scipy',
        'pystackreg',
        # tensorflow >= 2.4.0 is automatically built with cuda support for Nvidia GPUs
        # 'tensorflow>=2.4.1',
        'tensorflow>=2.8.0rc0',
        'voxelmorph'
    ]
)