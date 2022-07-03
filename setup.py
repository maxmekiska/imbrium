from setuptools import setup, find_packages
import os

with open('README.md', encoding="utf8") as file:
    long_description = file.read()

setup(
    author="Maximilian Mekiska",
    author_email="maxmekiska@gmail.com",
    url="https://github.com/maxmekiska/Imbrium",
    description="Standard and Hybrid Deep Learning Multivariate-Multi-Step & Univariate-Multi-Step Time Series Forecasting.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    name="Imbrium",
    version="0.1.1",
    packages = find_packages(include=["imbrium", "imbrium.*"]),
    install_requires=[
        "tensorflow==2.9.1",
        "scikit-learn==0.21.3",
        "matplotlib==3.5.2",
        "numpy==1.21.6",
        "pandas==0.25.1",
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
    ],
    keywords = ["machineleaning", "keras", "deeplearning", "timeseries", "forecasting"],
    python_rquieres=">=3.7"
)
