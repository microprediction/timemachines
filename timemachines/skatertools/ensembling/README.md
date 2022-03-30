# Ensembles, stacking, experts, portfolios, et cetera 


The ensemblefactory takes a list of skaters *fs* and also a meta-model *g*. It prepares a vector of the constituent model's 
predictions and passes them to the meta-model. 
 
   - If 'empirical_std' is set to True, the constituent model's predictions will be interleaved 
   with their std error estimates. See precisionweightedskater for an example of this style of use
 
### Trust constituent's errors? 

Note the *trust* argument in the ensemble factory. 
 
   1. If set to true, the ensemble will use the constituent models' own estimates of their errors.
   2. If set to false, the ensemble factory will create a separate parade object for each model, to untrustingly 
     construct independent estimates of the errors. 
     

 
 
  


