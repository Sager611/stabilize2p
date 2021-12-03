stabilize2p
============

Different approaches to stabilize 2-photon imaging video.

Requirements
------------

`make <https://www.gnu.org/software/make/>`_ should be installed.

If you want to install tensorflow with Nvidia GPU support you have to install the `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_ and `cuDNN <https://developer.nvidia.com/cudnn>`_. 
Instructions are system-dependent. Otherwise, if you have `Anaconda <https://www.anaconda.com/>`_ installed, you can install them through:

.. code:: shell

    conda install -c conda-forge cudatoolkit cudnn

Installation
------------

Run:

.. code:: shell

    make install
    pip install -e .

Please create in the project root folder a ``data/`` link pointing to the directory
with the dataset. For example:

.. code:: shell

    $ ln -s /path/to/data "${PWD}/data"
    $ vdir -ph data/
    total 2.5M
    drwxrwxrwx 1 admin admin 256K Sep  4 11:23 200901_G23xU1/
    drwxrwxrwx 1 admin admin 256K Sep  5 20:52 200908_G23xU1/
    drwxrwxrwx 1 admin admin 256K Sep  6 05:04 200909_G23xU1/
    drwxrwxrwx 1 admin admin 256K Sep  6 14:11 200910_G23xU1/
    drwxrwxrwx 1 admin admin 256K Sep  7 17:37 200929_G23xU1/
    drwxrwxrwx 1 admin admin 256K Sep  7 22:52 200930_G23xU1/
    drwxrwxrwx 1 admin admin 256K Sep  8 02:19 201002_G23xU1/

Additional scripts
------------------

The ``bin/`` folder contains scripts you may find useful to deal with
the dataset.

To run these scripts you need to `install stabilize2p
first <#installation>`__.

Scripts:

-  raw2tiff: shell script to transform raw 2-photon video to a TIFF file
-  pystackreg: shell script to apply pystackreg method to a tiff file to stabilize the video
-  register.py: Voxelmorph's
   `register.py <https://github.com/voxelmorph/voxelmorph/blob/dev/scripts/tf/register.py>`__.
   Used to load a model and register an image.
-  train-voxelmorph.py: train a Voxelmorph model using a pool of files. Check ``train-voxelmorph.py --help`` for more information.
