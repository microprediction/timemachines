import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="timemachines",
    version="0.2.11",
    description="Popular time series and optimization packages, with a simple, consistent functional interface.",
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
    packages=["timemachines","timemachines.optimizers","timemachines.skaters","timemachines.stochastictests","timemachines.data",
              "timemachines.skaters.components","timemachines.skaters.simple","timemachines.skaters.proph","timemachines.skaters.divine",
              "timemachines.skaters.dlm","timemachines.skaters.pmd",'timemachines.skaters.utilities','timemachines.skaters.components',
              'timemachines.data'],
    test_suite='pytest',
    tests_require=['pytest','microprediction'],
    include_package_data=True,
    install_requires=["wheel","pathlib","numpy>=1.19.5","pandas","importlib-metadata>=1.7.0","microconventions>0.5.0","getjson","sklearn",
                      "divinity","pmdarima","hyperopt","scipy",
                      "poap","pySOT","funcy","pydlm","optuna","deap","ax-platform","sigopt","statsmodels","tdigest","platypus-opt",
                      "pymoo","ratings","nevergrad","swarmlib","momentum","pandas",
                      "fbprophet","ratings>=0.0.9","landscapes"],
    entry_points={
        "console_scripts": [
            "timemachines=timemachines.__main__:main",
        ]
    },
)
