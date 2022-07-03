from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    author="Maximilian Mekiska",
    author_email="maxmekiska@gmail.com",
    url="https://github.com/maxmekiska/Imbrium",
    description="Standard and Hybrid Deep Learning Multivariate-Multi-Step & Univariate-Multi-Step Time Series Forecasting.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    name="Imbrium",
    version="0.1.0",
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
