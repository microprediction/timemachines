import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="timemachines",
    version="0.0.1",
    description="Really simple pure function time series prediction models",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/microprediction/pointy",
    author="microprediction",
    author_email="pcotton@intechinvestments.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["timemachines"],
    test_suite='pytest',
    tests_require=['pytest'],
    include_package_data=True,
    install_requires=["pathlib","microconventions","sklearn","divinity"],
    entry_points={
        "console_scripts": [
            "pointy=pointy.__main__:main",
        ]
    },
)
