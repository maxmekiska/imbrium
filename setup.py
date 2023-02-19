import os

from setuptools import find_packages, setup

with open("README.md", encoding="utf8") as file:
    long_description = file.read()

setup(
    author="Maximilian Mekiska",
    author_email="maxmekiska@gmail.com",
    url="https://github.com/maxmekiska/Imbrium",
    description="Standard and Hybrid Deep Learning Multivariate-Multi-Step & Univariate-Multi-Step Time Series Forecasting.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    name="imbrium",
    version="1.0.2",
    packages=find_packages(include=["imbrium", "imbrium.*"]),
    install_requires=[
        "setuptools >= 41.0.0",
        "tensorflow >= 2.11.0, < 2.12.0",
        "matplotlib >= 3.5.0, < 3.7.0",
        "pandas >= 1.3.3, < 1.6.0",
        "scikit-learn >= 1.0, < 1.3.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=["machinelearning", "keras", "deeplearning", "timeseries", "forecasting"],
    python_rquieres=">= 3.7.0, < 3.11.0",
)
