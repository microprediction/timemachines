# Ensembling skaters
View as [web page](https://microprediction.github.io/timemachines/ensembling)  or [source](https://github.com/microprediction/timemachines/blob/main/docs/ensembling.md).


### Precision weighted ensembles
Create a new skater *f* that is a precision weighted ensemble of moving averages as follows:

     from timemachines.skaters.simple.movingaverage import EMA_BASIC_SKATERS
     def f(y :Y_TYPE, s:dict, k:int =1, a:A_TYPE =None, t:T_TYPE =None, e:E_TYPE =None)->([float] , Any , Any):
         return precision_weighted_ensemble_factory(fs=EMA_BASIC_SKATERS,y=y,s=s,k=k,a=a,t=t,e=e,r=0.5)

Modify the *r* parameter to make the ensemble more evenly weighted, or less. 


### Implementation

- See the [ensembling](https://github.com/microprediction/timemachines/tree/main/timemachines/skatertools/ensembling) utilities. 


### Articles about ensembling skaters

- [The Only Prediction Function You'll Ever Need?](https://microprediction.medium.com/the-only-prediction-function-youll-ever-need-fe2ae42eaff0)

 

-+- 

Documentation [map](https://microprediction.github.io/timemachines/map.html)
 
  


![skating](https://i.imgur.com/elu5muO.png)
