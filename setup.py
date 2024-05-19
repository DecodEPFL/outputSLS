#!/usr/bin/env python
import setuptools


setuptools.setup(
    name='outputSLS',
    version='1.0',
    url='https://github.com/DecodEPFL/outputSLS',
    license='CC-BY-4.0 License',
    author='Clara Galimberti',
    author_email='clara.galimberti@epfl.ch',
    description='Boosting the performance of nonlinear systems through NN-based output-feedback controllers',
    packages=setuptools.find_packages(),
    install_requires=['torch>=2.1.0',
                      'numpy>=1.26.0',
                      'matplotlib==3.6.2'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.10',
)
