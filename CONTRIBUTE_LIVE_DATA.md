
## Contribution pattern:  new live data

Add live data that feeds the Elo ratings 

### Likely contributor

- You have an interesting source of live data. 
- Or you want something predicted


### How to contribute
  
  - Publish live data on an ongoing basis
    - This can help make the [Elo ratings](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/univariate-k_003.html) better.
    - However only "live" data is ideal, to prevent algorithms from memorizing
    - See [instructions for publishing data](https://www.microprediction.com/get-predictions) at www.microprediction.com (maybe jump to 
    https://www.microprediction.com/python-4). Basically what you need to do is set up a cron job or similar that periodically grabs a live
    data point and publishes it. First:

    
           pip install microprediction
           from microprediction import new_key
           write_key = new_key(difficulty=12)  # <--- Takes a long time, sorry
           
Then your regular job can do the following: 
         
           from microprediction import MicroWriter
           writer = MicroWriter(write_key=write_key)
           writer.set(name='my_own_stream.json',value=3.14157)  # <--- Must end in .json 

This will create a stream like [airport short term parking](https://www.microprediction.org/stream_dashboard.html?stream=airport-ewr-short-term_parking_a_b_c) and a bunch of
hungry time-series algorithms will come to it. The remainder of this note deals only with skater creation. 

## More help / discussion

As noted, I try to jump on a Google Meet twice a week and the details are in the microprediction [knowledge center](https://www.microprediction.com/knowledge-center). My arrival rate is higher on Fridays than Tuesdays :)

I'm not so good at scheduling calls outside of these times and frankly that tends to be counter to my productivity anyway. So just jump on some Tuesday night or 
Friday noon if you are keen to contribute to this package, or anything else that relates to open source community prediction. 


![](https://i.imgur.com/l14hKmr.jpg)
