# Why?
View as [web page](https://microprediction.github.io/timemachines/why). 

### Why skater functions?

Oh here comes the justification. 

   1. **Simple k-step ahead forecasts in functional style** I mean that's obvious. There are no "models" here requiring setup or estimation, only stateless functions as noted:
       
          x, x_std, s = f(y,s,k)
         
and if you need, 

          x, x_std, s = f(y,s,k,a,t,e,r)
       
The s-k-a-t-e-r functions take on the responsibility of incremental estimation, so you don't have to. Some skaters are computationally efficient in this respect, whereas others are drawn from traditional packages intended for batch/offline work, and can be quite slow when called repeatedly. But they are here because it is necessary to compare the accuracy of fast and slow algorithms, even if the latter might not suit your production volumetrics. 

   2. **Ongoing, incremental, empirical evaluation**. Again, see the [leaderboards](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/univariate-k_003.html) produced by
    the accompanying repository [timeseries-elo-ratings](https://github.com/microprediction/timeseries-elo-ratings). Assessment is always out of sample and uses *live*, constantly updating real-world data 
     from [microprediction.org](https://www.microprediction.org/browse_streams.html).   
    
   3. **Terse stacking, ensembling, boosting and other combining** of models. The function form makes it easy to do this, usually with one or two lines of code (again, see [thinking.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/simple/thinking.py) for an illustration, 
   or [prophetskaterscomposed.py](https://github.com/microprediction/timemachines/blob/main/timemachines/skaters/proph/prophskaterscomposed.py)).

   4. **Simplified deployment**. There is no state, other that that explicitly returned to the caller. For skaters relying only on the timemachines and river packages (the fast ones), the state is a pure Python dictionary trivially converted to JSON and back (for instance in a web application). See the [FAQ](https://github.com/microprediction/timemachines/blob/main/FAQ.md) for a little more discussion.   

So there you have it. No classes. Few dataframes. Hopefully little ceremony.   


-+- 

Documentation [map](https://microprediction.github.io/timemachines/map.html)
 
  


![skating](https://i.imgur.com/elu5muO.png)
