import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="timemachines",
    version="1.0.0",
    description="DEPRECATED: use `skaters` instead. This is a thin shim re-exporting `laplace` from skaters.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/microprediction/skaters",
    author="microprediction",
    author_email="peter.cotton@microprediction.com",
    license="MIT",
    classifiers=[
        "Development Status :: 7 - Inactive",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    packages=["timemachines"],
    python_requires=">=3.10",
    include_package_data=True,
    install_requires=["skaters>=0.10.0"],
)
