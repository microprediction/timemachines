# Time-machines contributors guide

You may ask yourself, "Well, how did I get here?" And you may ask yourself, "How do I work this?". And you may find yourself behind the wheel of a large automobile. 

But enough 80's rock. Chances are you're here because you reached out to connect on Linked-In and you have some manner of time-series or quantitative interest so I sent you an invite. And now you find yourself in a conspiracy aimed at upending artisan data science by unleashing a billion little reward-seeking algorithms into a micro-economy, commoditizing repeated quantitative tasks of all kinds, and making data scientists look like fools. That might be a better idea than waiting for DeepMind to to something ([according to me](https://www.microprediction.com/blog/reward), anyway).   

Hello, we hope you are interested in helping us create free open short term prediction of anything for anyone. 

## Goals 

   - To make available a large slew of fully autonomous univariate forecasting functions with sequence-to-sequence signature ("skaters")
   - To evaluate them continuously over fresh, live, diverse real-world timeseries in two ways:
        * Computation of [Elo ratings](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/overall.html)
        * Incorporation in "crawlers" than operate in real-time and predict time-series at www.microprediction.org where they play this [game](https://www.microprediction.com/blog/intro), and in doing so, provide free high quality short-term prediction to anyone, anywhere, for any reason.
   - To create more accurate autonomous time-series models by stacking, composition and so forth. 
   - To create more fast autonomous time-series models (as discussed [here](https://www.microprediction.com/blog/fast))
   - To address otherwise thorny issues like [defining anomaly detection](https://www.microprediction.com/blog/anomaly) in a way that isn't circular. 
  
## Contribution patterns

Little things
   - Follow, clap, heckle on [medium](https://microprediction.medium.com/), [linked-in](https://www.linkedin.com/company/65109690)
   - Star, fork, watch [timemachines](https://github.com/microprediction/timemachines)
   - See [good first issues](https://github.com/microprediction/timemachines/issue)

Contributing compute:
   1. Cut and paste a bash command to drive the default "crawler". See [CONTRIBUTE_COMPUTE_LOCAL_ONE_LINE.md](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE_COMPUTE_LOCAL_ONE_LINE.md). Run a Python script directly if you prefer. See [CONTRIBUTE_COMPUTE_LOCAL](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE_COMPUTE_LOCAL.md). For example, run a Python script on a PythonAnywhere account that drives a "crawler". See [CONTRIBUTE_COMPUTE_PA](
 https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE_COMPUTE_PA.md)
   2. Cut and paste a bash command to burn some rare Memorable Unique Identifiers, and donate them. See [CONTRIBUTE_COMPUTE_MUIDS.md](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE_COMPUTE_MUIDS.md)
 
Contributing compute, but also brains: 
   3. Create any kind of crawler you like in any language (though Python is easier). Then improve it. See the [knowledge center](https://www.microprediction.com/knowledge-center) tutorials. 
   
Adding time-series functionality (skaters) to timemachines:
   5. See [CONTRIBUTE_BATCH_STYLE_MODELS](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE_BATCH_STYLE_MODELS.md) to add new functionality using non-incremental methods.
   6. See [CONTRIBUTE_ONLINE_STYLE_MODELS](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE_ONLINE_STYLE_MODELS.md) to add new functionality using incremental methods.
   

## Friday chats

  - Grab the slack invite from the [knowledge center](https://www.microprediction.com/knowledge-center) or convey your email to me in some way (info@microprediction.com will do)
  - Turn up to one of the informal chats we have every Friday noon EST. 
  - If that timezone is bad, maybe Tue 8pm EST. 
  
But if you are shy that's fine too. I look forward to your pull requests, or seeing you on the leaderboard. Crawling can be completely anonymous, by the way. 
