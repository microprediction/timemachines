Skaters powered by sktime

This is also a reasonable template for creation of skaters from batch-style forecasting routines. 
You might want to C&P this one if incorporating other libraries. 

-1. Locate a python package that you think should be exposed to timemachines
 0. Grok the package and maybe contribute to https://github.com/microprediction/timeseries-notebooks while at it. 
 1. Choose a short PREFIX that isn't exactly the same as the library (here PREFIX='sk', obviously)
 2. Write PREFIXinclusion.py           (pretty trivial)
 3. Write PREFIXwrappers.py            (expose the batch functionality in a simple way)
 4. Write PREFIX<category1>.py         (skaters using timemachines.skatertools.batch.empiricalbatchskater) 
 5. Write PREFIX<category2>.py         (skaters using timemachines.skatertools.batch.empiricalbatchskater) 
 6. Collate all skaters in allPREFIXskaters.py       
 7. Modify timemachines.skaters.allskaters 
 8. Modify setup.py to include the path 
 9. Write tests.PREFIX...
 10. Run pytest 
 
 
 
 