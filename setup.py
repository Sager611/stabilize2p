import os
import setuptools
from subprocess import call
from distutils.command.build import build

BASEPATH = os.path.dirname(os.path.abspath(__file__))
# STABILIZE2P_PATH = os.path.join(BASEPATH, 'modules')


class Stabilize2pBuild(build):
    def run(self):
        def install_github_deps():
            # install github dependency packages through the Makefile
            cmd = [
                'make',
                'github_install'
            ]
            call(cmd, cwd=BASEPATH)
        
        self.execute(install_conda_env, [], 'Installing GitHub packages')


def read(fname):
    # retrieves ``fname`` file contents as a string
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(
    name='stabilize2p',
    version='0.1',
    license='GPLv2',
    description='Registration tools for 2-photon imaging',
    url='',
    long_description=read('README.rst'),
    keywords=['deformation', 'registration', 'imaging', 'dnn', '2-photon'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],

    packages=['modules'],  # folders with the packages' modules
    python_requires='>=3.6',
    install_requires=[
        'tifffile',
        'jupyterlab',
        'numpy',
        'matplotlib',
        'scipy',
        'pystackreg',
        # tensorflow >= 2.4.0 is automatically built with cuda support for Nvidia GPUs
        'tensorflow-gpu>=2.4.1',
        'voxelmorph'
    ],
    cmdclass={
        'build': Stabilize2pBuild
    }
)