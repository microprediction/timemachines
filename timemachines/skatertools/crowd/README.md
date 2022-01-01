### Micro-managing patterns

Some utilities for creating stacks benefit from exogenous input (e.g. other algorithms recommending
which skaters to use next and being rewarded for doing so).

For the really long explanation see the book 
"Microprediction: Building an Open AI Network" (2022, but who knows when you are reading this)

A word of caution, the state will be serializable with json.dumps() in general, but to reinflate
you need to use the state utility provided (the lottery-type mechanisms are objects derived
from dict). 

