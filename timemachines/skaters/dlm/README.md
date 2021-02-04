## PyDLM skaters


### PyDLM Advantages 

- Fast, incremental using HW discounts 
- Nice design makes adding factors easy


### Disadvantages

- State serialization to JSON is not obvious



### Why two? 

- dlm_exogenous  currently works for k=1 only,
- dlm_univariate works for k>=1 

TODO: Refactor former so we don't need latter. 