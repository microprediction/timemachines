import numpy as np

# Quick and dirty time-series stitching
# This should really use matrix profiles or motif algos or whatever. 

def find_similar_time_series(stitched,time_series_list, h=5):
    """
    """
    volatilities = [ np.std(np.diff(ts[:10*h])) for ts in time_series_list]
    levels = [ np.mean(ts[:3*h]) for ts in time_series_list]
    target_vol =   np.std(np.diff(stitched[-10*h:]))
    target_level = np.mean(stitched[-3*h:])
    distances = [np.mean(np.abs(stitched[-h:] - ts[:h])) for ts in time_series_list]
    
    # Calculate the score of each time series as a combination of the distance and the volatility
    scores = [1.0/(1.0 + d) * 10.0/(1.0 + np.abs(v - target_vol)+np.abs(l-target_level) ) for d, v,l in zip(distances, volatilities, levels)]
    
    # Find the time series with the highest score
    max_score_index = np.argmax(scores)

    return max_score_index

    

def stitch_time_series(time_series_list, n_stitches:int=None, h:int=5):
    """
         n_stitches    The number of timeseries
         h             The overlap
    """
    if n_stitches is None:
       n_stitches = int(len(time_series_list)/2+0.6)

    n_iter = max(n_stitches-1,len(time_series_list)-1)
    ndx = 0 
    stitched = time_series_list[ndx]
    del time_series_list[ndx]

    for i in range(n_iter):
        ndx = find_similar_time_series(stitched, time_series_list, h=h)
        selected_ts = time_series_list[ndx]
        ts = time_series_list[ndx]
        del time_series_list[ndx]
        compromise = [ (t+s)/2 for t,s in zip(ts[-h:],selected_ts[:h])]
        for i in range(h):
           stitched[-i] = compromise[-i]
        stitched = list(stitched) + list(selected_ts)    
    return stitched


if __name__=='__main__':
      # Generate some random time series
      time_series_list = []
      for i in range(15):
          ts = np.random.normal(size=100)*np.random.rand()
          time_series_list.append(ts)

      # Stitch the time series together

      stitched_ts = stitch_time_series(time_series_list,h=6)

      import matplotlib.pyplot as plt
      # Plot the stitched time series
      plt.plot(stitched_ts)
      plt.xlabel('Time')
      plt.ylabel('Value')
      plt.title('Stitched Fake Time Series')
      plt.show()
