
# Facebook Prophet skaters

The most popular time series package, so it seems like it should be included if only for comparison 

- But not incremental. See [this thread](https://github.com/facebook/prophet/issues/46), to wit:
*The online learning used by some sklearn models is pretty fundamentally different from how Stan models are fit, I don't think we are going to have a partial_fit like that in the future.* 

- The generative model is unlikely to work well for general time-series. See my analysis in the blog article titled [Is Facebook Prophet the Time-Series Messiah, or Just a Very Naughty Boy? ](https://www.microprediction.com/blog/prophet)

### Skater implementation

- Probably too many lines of code. 
- Predates some utilities for skater creation
- The sktime (sk) or tbats (bats) skaters are simpler examples to mimic

### Methodology 

See the [paper](https://peerj.com/preprints/3190/). 

    y(t) = g(t) + s(t) + h(t) + eps 
    
where g(t) is trend intended to model non-periodic changes, s(t) is periodic, and h(t) captures holiday
effects. The trend model is a logistic growth model whose parameters can have changepoints. The seasonality
component is, morally speaking, a low pass filter. 

### Hyper-parameters
Copied from the very helpful discussion [here](https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning)


#### changepoint_prior_scale: 
This is probably the most impactful parameter. It determines the flexibility of the trend, and in particular how much the trend changes at the trend changepoints. As described in this documentation, if it is too small, the trend will be underfit and variance that should have been modeled with trend changes will instead end up being handled with the noise term. If it is too large, the trend will overfit and in the most extreme case you can end up with the trend capturing yearly seasonality. The default of 0.05 works for many time series, but this could be tuned; a range of [0.001, 0.5] would likely be about right. Parameters like this (regularization penalties; this is effectively a lasso penalty) are often tuned on a log scale.

#### seasonality_prior_scale: 
This parameter controls the flexibility of the seasonality. Similarly, a large value allows the seasonality to fit large fluctuations, a small value shrinks the magnitude of the seasonality. The default is 10., which applies basically no regularization. That is because we very rarely see overfitting here (there’s inherent regularization with the fact that it is being modeled with a truncated Fourier series, so it’s essentially low-pass filtered). A reasonable range for tuning it would probably be [0.01, 10]; when set to 0.01 you should find that the magnitude of seasonality is forced to be very small. This likely also makes sense on a log scale, since it is effectively an L2 penalty like in ridge regression.

#### holidays_prior_scale: 
This controls flexibility to fit holiday effects. Similar to seasonality_prior_scale, it defaults to 10.0 which applies basically no regularization, since we usually have multiple observations of holidays and can do a good job of estimating their effects. This could also be tuned on a range of [0.01, 10] as with seasonality_prior_scale.

#### seasonality_mode: 
Options are ['additive', 'multiplicative']. Default is 'additive', but many business time series will have multiplicative seasonality. This is best identified just from looking at the time series and seeing if the magnitude of seasonal fluctuations grows with the magnitude of the time series (see the documentation here on multiplicative seasonality), but when that isn’t possible, it could be tuned.

Maybe tune?

#### changepoint_range: 
This is the proportion of the history in which the trend is allowed to change. This defaults to 0.8, 80% of the history, meaning the model will not fit any trend changes in the last 20% of the time series. This is fairly conservative, to avoid overfitting to trend changes at the very end of the time series where there isn’t enough runway left to fit it well. With a human in the loop, this is something that can be identified pretty easily visually: one can pretty clearly see if the forecast is doing a bad job in the last 20%. In a fully-automated setting, it may be beneficial to be less conservative. It likely will not be possible to tune this parameter effectively with cross validation over cutoffs as described above. The ability of the model to generalize from a trend change in the last 10% of the time series will be hard to learn from looking at earlier cutoffs that may not have trend changes in the last 10%. So, this parameter is probably better not tuned, except perhaps over a large number of time series. In that setting, [0.8, 0.95] may be a reasonable range.
