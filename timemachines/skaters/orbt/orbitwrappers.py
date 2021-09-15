import pandas as pd

from timemachines.skaters.orbt.orbitinclusion import using_orbit

if using_orbit:
    from orbit.models.lgt import LGTMAP, LGTAggregated, LGTFull

    def orbit_lgt_iskater(y, k, a, t, seasonality):
        """
           Run orbit over a fixed history and extract predictions
        """
        # It seems that orbit needs a two column dataframe of an index column and
        # measurement column in order to work. It also seems to need enough
        # space in the dataframe to insert the values it predicts.
        df = pd.DataFrame(y).reset_index().rename(columns={0: "response_col", 'index': "date_col"})
        for i in range(k):
            df2 = pd.DataFrame([[len(df), 0]], columns=["date_col", "response_col"])
            df = df.append(df2, ignore_index=True)

        # "test/train" split
        test_size = k
        train_df = df[:-test_size]
        test_df = df[-test_size:]

        # run the model
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

        # get the prediction(s)
        predicted_df = lgt.predict(df=test_df)

        # list containing predictions 1 through k steps out
        x = list(predicted_df.prediction)
        # standard deviaiton list (this is technically 2 std)
        x_std = [a / 2 for a in list(predicted_df.prediction_95 - predicted_df.prediction_5)]

        return x, x_std