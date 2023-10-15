import os

from setuptools import find_packages, setup

with open("README.md", encoding="utf8") as file:
    long_description = file.read()

setup(
    author="Maximilian Mekiska",
    author_email="maxmekiska@gmail.com",
    url="https://github.com/maxmekiska/imbrium",
    description="Standard and Hybrid Deep Learning Multivariate-Multi-Step & Univariate-Multi-Step Time Series Forecasting.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    name="imbrium",
    version="2.0.1",
    packages=find_packages(include=["imbrium", "imbrium.*"]),
    install_requires=[
        "setuptools >= 41.0.0",
        "keras-core >=0.1.5, <0.2.0",
        "tensorflow >=2.13.0, <2.14.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=["machinelearning", "keras", "deeplearning", "timeseries", "forecasting"],
    python_rquieres=">= 3.8.0, <= 3.11.0",
)
