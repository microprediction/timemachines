# Time-machines contributors guide

This project hopes to make it easier for humans, and also autonomous creatures, to get a rough idea of what time-series package or technique might be applicable to their domain. If you wish to help with this search problem, there are easy and more involved ways to help.  

   - [CONTRIBUTE COLAB NOTEBOOK](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE_COLAB_NOTEBOOK.md) if you like a package (pretty easy).
   - [CONTRIBUTE_BATCH_STYLE_MODELS](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE_BATCH_STYLE_MODELS.md) to add new functionality using non-incremental methods.
   - [CONTRIBUTE_ONLINE_STYLE_MODELS](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE_ONLINE_STYLE_MODELS.md) to add new functionality using incremental methods.

It that seems daunting, read on. 

## New here? 

You may ask yourself, "Well, how did I get here?" And you may ask yourself, "How do I work this?". And you may find yourself behind the wheel of a large automobile. 

<img src="https://github.com/microprediction/timemachines/blob/main/images/talking_heads.jpeg" alt="drawing" width="650"/>

But enough 80's rock. Chances are you're here because you reached out to connect on Linked-In, and you have some manner of time-series or quantitative interest, so I sent you an invite. Stop what you are doing. Open this [notebook](https://github.com/microprediction/microprediction/blob/master/submission_examples_die/first_submission.ipynb) and run it. The [README](https://github.com/microprediction) will make more sense, and perhaps too the notion of collective autonomous prediction.   

## Specific goals 
The strategy here:   

   1. Packaging a slew of fully autonomous univariate forecasting functions:
        * With a simple state-machine style signature ("[skaters](https://github.com/microprediction/timemachines)")
        * Drawing on whatever useful open-source Python packages can be found (and there's a [lot of them](https://www.microprediction.com/blog/popular-timeseries-packages)) 
        * Stacking, composing and otherwise exploiting existing skaters. 
        * Computation of [Elo ratings](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/overall.html)
        * ... so that [Fast Python Timeseries Forecasting](https://www.microprediction.com/blog/fast) might become the norm. 
        
   2. Creating "crawlers" and other programs than operate in real-time and predict time-series at www.microprediction.org where:
        * They play this [game](https://www.microprediction.com/blog/intro), and in doing so, 
        * Provide free high quality short-term prediction to anyone, anywhere, for any reason.
 
   3. Demonstrating creative use of on-tap community prediction such as:
        * Multi-level autonomous crowd-sourcing where things feed back to the next step (see [crypto examples](https://github.com/microprediction/microprediction/blob/master/stream_examples_crypto/README.md)).
        * Otherwise inventive use of general purpose prediction, conditional prediction, and prediction of ancilliary quantities to achieve intelligent systems in surprising ways. A book on the topic will be published by MIT Press Fall 2022.  
        * Attacking otherwise thorny issues like [defining anomaly detection](https://www.microprediction.com/blog/anomaly) in a way that isn't circular. 
        * Driving investment returns for clients of Intech Investments, the project sponsor. After all if it helps with what might be the hardest problem of all (or at least the most competitive) it is a no-brainer that this will work elsewhere. 


# Contribution Patterns

Can an open-source conspiracy upend artisan data science by unleashing a billion little reward-seeking algorithms and commoditizing repeated quantitative tasks of all kinds? It sure might be fun to find out, and possible a better idea than waiting for DeepMind to solve general intelligence, according to a wise man (see my unsolicited views on the notion that "Reward is Enough" [here](https://www.microprediction.com/blog/reward)). 

It's nice if people follow, clap, share, heckle on [medium](https://microprediction.medium.com/), [linked-in](https://www.linkedin.com/company/65109690) if that helps bring in contributors. Thanks. I suppose you can star, fork, watch [timemachines](https://github.com/microprediction/timemachines) or even sign this tongue-in-cheek [petition](https://www.change.org/p/towards-data-science-have-towards-data-science-publish-an-article-critical-of-facebook-software) - unless you want a job at Facebook or Towards Data Science, some day :) But here's how you can really help, even if you are new to open source...

## Creating colab notebooks illustrating the use of Python timeseries packages
It helps speed the creation of autonomous algorithms, and Elo ratings, to have example notebooks for python time-series packages

   0. See [good first issues](https://github.com/microprediction/timemachines/issues).
      Or search the same link for "Create colab notebook"

It's also not a bad way to familiarize yourself with packages that might be useful. No need to limit yourself to the ones in the issues. Anything that can predict k-steps ahead is fair game. See the [long list of packages](https://www.microprediction.com/blog/popular-timeseries-packages)

## Running scripts

Contributing compute:
   1. Cut and paste a bash command to drive the default "crawler". See [CONTRIBUTE_COMPUTE_LOCAL_ONE_LINE.md](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE_COMPUTE_LOCAL_ONE_LINE.md). Run a Python script directly if you prefer. See [CONTRIBUTE_COMPUTE_LOCAL](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE_COMPUTE_LOCAL.md). Or run a Python script on a PythonAnywhere account that drives a "crawler". See [CONTRIBUTE_COMPUTE_PA](
 https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE_COMPUTE_PA.md)
   2. Cut and paste a bash command to burn some rare Memorable Unique Identifiers, and donate them. See [CONTRIBUTE_COMPUTE_MUIDS.md](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE_COMPUTE_MUIDS.md)
 
 ## Contribution to crawler creation  

   3. Create any kind of Python crawler. Run it. Improve it. Repeat. See the [knowledge center](https://www.microprediction.com/knowledge-center) tutorials.
   4. Create any kind of crawler, not in Python. There's less support for that, but see the [public api](https://www.microprediction.com/public-api) and Google search (for "microprediction client Julia", for example, or "micropredciction client typescript). 

## Contribution to the timemachines package
Open issues:

   5. See [good first issues](https://github.com/microprediction/timemachines/issues)

New package inclusion and approaches

   6. See [CONTRIBUTE_BATCH_STYLE_MODELS](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE_BATCH_STYLE_MODELS.md) to add new functionality using non-incremental methods.
   7. See [CONTRIBUTE_ONLINE_STYLE_MODELS](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE_ONLINE_STYLE_MODELS.md) to add new functionality using incremental methods.
   
## Contribution of live data
Add live data that feeds the Elo ratings, and live contests too. 

   8. See [CONTRIBUTE_LIVE_DATA.md](https://github.com/microprediction/timemachines/blob/main/CONTRIBUTE_LIVE_DATA.md)

# Friday chats

  - Grab the slack invite from the [knowledge center](https://www.microprediction.com/knowledge-center) or convey your email to me in some way (info@microprediction.com will do)
  - Turn up to one of the informal chats we have every Friday noon EST. 
  - If that timezone is bad, maybe Tue 8pm EST. 
  
But if you are shy that's fine too. I look forward to your pull requests, or seeing you on the leaderboard. Crawling can be completely anonymous, by the way. 

## Career advice? 

Some fraction of you were asking about career advice. There are people in the microprediction slack who can probably give better advice than me. Hassle them, but mine would be:

 - Take the time to learn how to contribute to open-source and do all your hobby projects in the open, on GitHub.  
 - Read the [Mathematics subject classification](https://en.wikipedia.org/wiki/Mathematics_Subject_Classification) and slowly, over time, familiarize yourself with the key seminal tricks in each area. Even if you expect to spend most of your time in [4.2.1](https://en.wikipedia.org/wiki/Computer_science#Artificial_intelligence) this will give you angles on problems that other's don't have.  

I fear my other advice mostly overlaps with platitudes. 
