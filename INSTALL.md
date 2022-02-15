![](https://github.com/microprediction/timemachines/blob/main/images/download_time.png)

Don't worry this will all go smoothly. Two suggested paths:

    1. Use only venv and pip
    2. Use miniconda

I definitely suggest the latter if you are an M1 user

# Option 1) Install on python virtual environment
See the reasonable intro to venv [here](https://medium.com/swlh/how-to-setup-your-python-projects-1eb5108086b1) if you are not familiar. Skip to next section if using mac silicon (new M1). Example of creating new env:

    mkdir virtual-envs
    cd virtual-envs
    python -m venv myenv 
    source myenv/bin/activate 
    
I suggest that life is less frustrating if you patiently install popular prerequisites one by one

    pip install --upgrade pip
    pip install --upgrade wheel
    pip install --upgrade numpy
    pip install --upgrade timemachines
    pip install --upgrade joblib
    pip install --upgrade numba
    pip install --upgrade scipy 
    pip install --upgrade scikit-learn 
 
so you can see if you run into difficulty. If desperate, you can maybe scrape by without the last two as they are only used for metrics. You'll still have a slew of speedy home-spun forecasting functions such as the 
[simple](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/simple) models. But, if you want to make a reasonably large number of models (skaters) available to yourself then you'll need most of these and also:

    pip install --upgrade statsmodels
    pip install --upgrade tensorflow
    pip install --upgrade torch
    pip install --upgrade pandas
    pip install --upgrade cython
    
It is hardest to avoid using statsmodels as just about every other package wraps statsmodels.tsa As an aside you may get better performance by first installing tensorflow following the [instructions](https://www.tensorflow.org/install) and perhaps reading this [thread](https://stackoverflow.com/questions/66092421/how-to-rebuild-tensorflow-with-the-compiler-flags). 

Next decide how badly you want prophet, or other timeseries packages that wrap it. If so:

     pip install --upgrade prophet
    
On some systems pystan is flaky, thus also prophet, thus also things wrapping prophet. You'll need an older pystan (unless things have changed). Maybe read my [review of prophet](https://www.microprediction.com/blog/prophet) before spending too much install agony there. It is an extrapolation library, really, not a timeseries forecasting library.   

Finally, it is time to install the packages you wish to employ. You may wish to first check the [Elo ratings](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/univariate-k_003.html) to get a vague idea of accuracy and speed, and which packages you wish to install. But here are some suggestions, ordered by approximate easy of install rather than performance.  
    
    pip install --upgrade river
    pip install --upgrade git+https://github.com/oseiskar/simdkalman
    pip install --upgrade pydlm
    pip install --upgrade divinity
    pip install --upgrade pmdarima
    pip install --upgrade u8darts        (does not include prophet, or...)
    pip install --upgrade darts          (does include prophet)
    
Here it might pay to read the [darts install guide](https://github.com/unit8co/darts#installation-guide) for advice on libomp and other troubleshooting. In particular if lightgbm gives you headaches, then maybe resort to the miniconda route. See notes on conda install -c conda-forge lightgbm in the next section.

Moving on:
    
    pip install --upgrade sktime
    
If that fails, try

    export SKTIME_NO_OPENMP=true
    pip install --upgrade sktime
    
Continuing...
    
    pip install --upgrade tbats
    pip install --upgrade successor
    pip install --upgrade orbit-ml
    pip install --upgrade neuralprophet
    pip install --upgrade greykite
    
You might get this warning:

    sktime 0.9.0 requires statsmodels<=0.12.1
    
Proceeding..
    
    pip install --upgrade salesforce-merlion
    pip install --upgrade pycaret-ts-alpha
    
Finally, do this only if you wish to have greater ability to pull exogenous data in:

    pip install --upgrade microprediction

# Option 2. Install with (mini) conda 

If you are a conda person, or use Mac silicon and are not especially brave, then: 

    brew install miniforge

(you can get homebrew [here](https://brew.sh/)). You have other options [here](https://github.com/conda-forge/miniforge) for installing miniforge. Then make a new conda env:

    conda create -n myenv
    conda activate myenv 
    
and proceed cautiously: 
    
    conda install numpy
    conda install scipy 
    conda install scikit-learn
    conda install pandas
    conda install statsmodels
    conda install matplotlib
    pip install timemachines

Then proceed, using miniconda where possible and pip if conda doesn't have it yet. For instance if installing a package that needs lightgbm is causing a headache then try:  

    conda install -c conda-forge lightgbm
    
After exhausting proceed as above as see how far you get with timeseries packages

     pip install --upgrade darts
     pip install --upgrade river 
    
and so on (see list above). Again, the apple silicon (m1) install situation is particularly fluid. Maybe see [this thread](https://stackoverflow.com/questions/65745683/how-to-install-scipy-on-apple-silicon-arm-m1) for ideas and 
keep open the possibility of the --no-use-pep517 option.
 
    pip install whatever --no-use-pep517

    
## Install troubleshooting remark on colab
To clean out old versions of pytz etc I sometimes use:

    !pip uninstall numpy -y
    
in colab notebooks. However hopefully that won't be required by the time you read this. 
    

