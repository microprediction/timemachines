import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="timemachines",
    version="0.2.6",
    description="Bringing together popular time series packages, and popular optmization packages for hyper-param selection.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/microprediction/timemachines",
    author="microprediction",
    author_email="pcotton@intechinvestments.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["timemachines","timemachines.optimizers","timemachines.skaters","timemachines.stochastictests","timemachines.data"],
    test_suite='pytest',
    tests_require=['pytest'],
    include_package_data=True,
    install_requires=["wheel","pathlib","numpy>=1.16.5","importlib-metadata>=1.7.0","microconventions>0.5.0","getjson","sklearn",
                      "divinity","pmdarima","hyperopt","scipy",
                      "poap","pySOT","funcy","pydlm","optuna","deap","ax-platform","sigopt","statsmodels","tdigest","platypus-opt",
                      "pymoo","ratings","nevergrad","swarmlib","momentum","pandas",
                      "fbprophet"],
    entry_points={
        "console_scripts": [
            "timemachines=timemachines.__main__:main",
        ]
    },
)
