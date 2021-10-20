import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="timemachines",
    version="0.14.7",
    description="Evaluation and standardization of popular time series packages",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/microprediction/timemachines",
    author="microprediction",
    author_email="peter.cotton@microprediction.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["timemachines",
              "timemachines.skaters",
              "timemachines.inclusion",
              "timemachines.skaters.divine",
              "timemachines.skaters.dlm",
              "timemachines.skaters.flux",
              "timemachines.skaters.pmd",
              "timemachines.skaters.tsa",
              "timemachines.skaters.proph",
              "timemachines.skaters.nproph",
              "timemachines.skaters.simple",
              "timemachines.skaters.orbt",
              "timemachines.skaters.bats",
              "timemachines.skaters.rvr",
              "timemachines.skaters.sk",
              "timemachines.skaters.elo",
              "timemachines.skaters.smdk",
              "timemachines.skaters.gk",
              "timemachines.skaters.drts",
              "timemachines.skaters.mrln",
              "timemachines.skaters.kts",
              "timemachines.skaters.ats",
              "timemachines.skaters.suc",
              "timemachines.skatertools",
              "timemachines.skatertools.comparison",
              "timemachines.skatertools.components",
              "timemachines.skatertools.composition",
              "timemachines.skatertools.ensembling",
              "timemachines.skatertools.data",
              "timemachines.skatertools.evaluation",
              "timemachines.skatertools.tuning",
              "timemachines.skatertools.utilities",
              "timemachines.skatertools.visualization",
              "timemachines.skatertools.batch",
              "timemachines.skatertools.recommendations"],
    test_suite='pytest',
    tests_require=['pytest','microprediction','scikit-learn','scipy'],
    include_package_data=True,
    install_requires=["wheel","pathlib","numpy>=1.19.5","importlib-metadata>=1.7.0",
                      "microconventions>=0.5.1","getjson",'pytz',
                       "convertdate>=2.2.0","momentum>=0.2.1","requests"],
    entry_points={
        "console_scripts": [
            "timemachines=timemachines.__main__:main",
        ]
    },
)
