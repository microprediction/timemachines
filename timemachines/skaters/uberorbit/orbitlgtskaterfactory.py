import pandas as pd
from timemachines.skatertools.utilities.conventions import wrap
from timemachines.skaters.uberorbit.orbitinclusion import using_orbit, LGTMAP, LGTAggregated, LGTFull


if using_orbit:

    def orbit_lgt_skater_factory(y, s, k, a=None, t=None, e=None, seasonality=None):
        assert seasonality is not None

        y = wrap(y)

        if not s.get('y'):
            s = {'p': {},  # parade
                 'y': list(),  # historical y
                 'k': k}
        else:
            # Assert immutability of k, dimensions of y,a
            if s['y']:
                assert len(y) == len(s['y'][0])
                assert k == s['k']

        if y is None or (e is not None and e < 0):
            return None, None, s
        else:
            s['y'].append(y)

            if e is None:
                burn_in_period = 410
            else:
                burn_in_period = int(e)

            if len(s['y']) > burn_in_period:
                y_arg = s['y'][k:]
                x, x_std = run_orbit_lgt(s['y'], k, seasonality)
            else:
                x = [y[0]] * k
                x_std = [1] * k

        return x, x_std, s

    def run_orbit_lgt(data,k,seasonality):

        #It seems uberorbit needs a two column dataframe of an index column and
        #measuremnt column in order to work. It also seems to need enough
        #space in the dataframe to insert the values it predicts.
        df = pd.DataFrame(data).reset_index().rename(columns={0:"response_col",'index':"date_col"})
        for i in range(k):
            df2 = pd.DataFrame([[len(df),0]], columns=["date_col","response_col"])
            df = df.append(df2, ignore_index=True)

        #"test/train" split
        test_size = k
        train_df = df[:-test_size]
        test_df = df[-test_size:]

        #run the model
        if isinstance(seasonality, int) or seasonality == None:
            lgt = LGTMAP(
                response_col="response_col",
                date_col="date_col",
                seasonality=seasonality
            )
        else:
            print("The seasonality given was not an int or None type. Try again...")
            return
        lgt.fit(df=train_df)

        #get the prediction(s)
        predicted_df = lgt.predict(df=test_df)

        #list containing predictions 1 through k steps out
        x = list(predicted_df.prediction)
        #standard deviaiton list (this is technically 2 std)
        x_std = [a / 2 for a in list(predicted_df.prediction_95 - predicted_df.prediction_5)]

        return x, x_std


    def orbit_lgt_12(y,s,k,a=None, t=None,e=None):
        return orbit_lgt_skater_factory(y=y, s=s, k=k, a=a,t=t,e=e, seasonality=12)


    def orbit_lgt_24(y,s,k,a=None, t=None,e=None):
        return orbit_lgt_skater_factory(y, s, k, a=a,t=t,e=e, seasonality=24)

