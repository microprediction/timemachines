
# PMDArima skaters

pmdarima is the second most downloaded time series package.
Here we create a simple skater function encompassing some of the functionality of this library.

As with all skaters, the intent is that you can cycle through observations as follows:

      for yi, ai in zip(y, a):
         x, x_std, s = pmd_exog(y=yi, s=s, k=7, a=ai)

where each x and s_std are length 7, in this example.