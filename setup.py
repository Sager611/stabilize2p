import setuptools

setuptools.setup(
    name='stabilize-2p',
    version='0.1',
    license='Apache 2.0',
    description='Image Registration with Convolutional Networks',
    url='',
    keywords=['deformation', 'registration', 'imaging', 'dnn', '2-photon'],
    packages=setuptools.find_packages(),
    python_requires='>=3.9',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'voxelmorph',
    ]
)