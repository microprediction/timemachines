
## Contribution pattern:  new batch-style time-series package

Do you have a favourite Python package for time-series analysis that you would like to see included in the funnel? There's some discussion
on this [LI post](https://www.linkedin.com/posts/petercotton_timeseries-timeseriesanalysis-forecasting-activity-6844102671906037760-bGtu)

### Likely contributor

- You maintain a time-series package, or just really like one. 
- The package exposes a way of predicting time-series in an offline (sometimes tabular) fashion (as compared with online, incremental)

### When not to use this pattern

If the package is a state-space model, or some other incremental method, you'll not want to go down this path. 

### How to contribute 

1. Look at the example provided by [sk skaters directory](https://github.com/microprediction/timemachines/tree/main/timemachines/skaters/sk) and try to grok
2. If step (1) fails, ask questions on slack (invite [here](https://www.microprediction.com/knowledge-center))
3. Grok the package you think should be in. Create an example colab notebook (like [examples here](https://github.com/microprediction/timeseries-notebooks) that uses the package to be sure you understand the conventions adopted.  
 4. Choose a short PREFIX that isn't exactly the same as the library (here PREFIX='sk', obviously)
 5. Write PREFIXinclusion.py           (in the same style as [skinclusion.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/sk/skinclusion.py))
 6. Write PREFIXwrappers.py            (expose the batch functionality in the same style as [skwrappers.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/sk/skwrappers.py))

Thus far this has exposed the 3rd party library in an offline manner (i.e. expects fit to be called each data point). Now you need to turn this into a skater. 

 7. Read or re-read the main [README.md](https://github.com/microprediction/timemachines) that explains the skater signature
 8. Read the short pithy example [skautoarima](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/sk/skautoarima.py) and notice how it makes use of [batchskater.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skatertools/batch/batchskater.py) that does all the work for you.
 9. Create one or more skater files similar to [skautoarima](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/sk/skautoarima.py)
 10. For the convenience of others, adopt the style of putting a mini-test at the bottom, as with [[skautoarima](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/sk/skautoarima.py)]
 
Next you need to register a list of fully autonomous skaters
 
 9. Create a file like [allskskaters](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/sk/allskskaters.py)      
 
Write unit tests

 10. Cut and paste the example unit test [test_sk_random_skater](https://github.com/microprediction/timemachines/blob/main/tests/sk/test_sk_random_skater.py)

Run pytest and the individual test repeatedly. When it is working well you can include it in the master list:
 
 11. Modify [localskaters.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/localskaters.py) 
 10. Modify [README.md](https://github.com/microprediction/timemachines/blob/main/README.md) to include a suggestion to users to pip install the new requirements

## More help / discussion

As noted, I try to jump on a Google Meet twice a week and the details are in the microprediction [knowledge center](https://www.microprediction.com/knowledge-center). My arrival rate is higher on Fridays than Tuesdays :)

I'm not so good at scheduling calls outside of these times and frankly that tends to be counter to my productivity anyway. So just jump on some Tuesday night or 
Friday noon if you are keen to contribute to this package, or anything else that relates to open source community prediction. 


![](https://i.imgur.com/l14hKmr.jpg)
