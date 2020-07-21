#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(
    name="adam_visual_perception",
    version='0.1.0',
    author="Mikayel Samvelyan",
    author_email="mikayel@samvelyan.com",
    url="https://github.com/isi-vista/adam-visual-perception",
    packages=[
        "adam_visual_perception",
    ],
    python_requires=">=3.6",
    install_requires=[
        "sacred>=0.8.1",
        "numpy>=1.16.2",
        "pandas>=0.24.2",
        "moviepy>=1.0.1",
        "matplotlib>=3.0.3",
        "imutils>=0.5.3",
        "dlib>=19.19.0",
        "keras==2.4.2",
        "tensorflow==2.2.0",
    ],
    scripts=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
