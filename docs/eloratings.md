# Elo ratings
View as [web page](https://microprediction.github.io/timemachines/eloratings)  or [source](https://github.com/microprediction/timemachines/blob/main/docs/eloratings.md).


### Retrieving top rated models
See suggestion utility [code](https://github.com/microprediction/timemachines/blob/main/timemachines/skatertools/recommendations/suggestions.py). For example:

     from timemachines.skatertools.recommendations.suggestions import top_rated_models
     my_skaters = top_rated_models(k=3, n=15, max_seconds=1,category='univariate')
     print([f.__name__ for f in my_skaters])


## Interpretation of Elo ratings:

Using an F-factor of 1000, one can interpret Elo rating differences as the probability that one model will outperform the other, as measured by root mean square error, when
tasked with fifty consecutive k-step ahead forecasts. Both models are supplied 400 prior data points to warm up on. If the errors are within one percent of each other, a draw is declared. 
The probability of the weaker model winning is

       P(win) = 1. / (1 + 10 ** (rating difference / 1000))

Consult the [eloformulas.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skatertools/comparison/eloformulas.py) and the script
[skatereloupdate.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skatertools/comparison/skaterelo.py) in the timemachines package if you seek more details. 
Constants are set in the script [update_skater_elo_ratings](https://github.com/microprediction/timeseries-elo-ratings/blob/main/update_skater_elo_ratings.py) and change from time to time.  


### Articles about Elo ratings

- [The Only Prediction Function You'll Ever Need?](https://microprediction.medium.com/the-only-prediction-function-youll-ever-need-fe2ae42eaff0)

 

-+- 

Documentation [map](https://microprediction.github.io/timemachines/map.html)
 
  


![skating](https://i.imgur.com/elu5muO.png)
