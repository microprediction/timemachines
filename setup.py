import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="timemachines",
    version="0.0.19",
    description="Time series models represented as pure functions with SKATER convention.",
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
    packages=["timemachines","timemachines.optimizers","timemachines.skaters"],
    test_suite='pytest',
    tests_require=['pytest'],
    include_package_data=True,
    install_requires=["wheel","pathlib","numpy>=1.16.5","importlib-metadata>=1.7.0","microconventions>0.5.0","sklearn","divinity","pmdarima","hyperopt","scipy",
                      "poap","pySOT","funcy","pydlm","optuna","deap","ax-platform","sigopt","statsmodels","tdigest","platypus-opt",
                      "pymoo"],
    entry_points={
        "console_scripts": [
            "timemachines=timemachines.__main__:main",
        ]
    },
)
