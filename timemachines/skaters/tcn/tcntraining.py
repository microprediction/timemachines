# Creating surrogate models

from timemachines.skaters.tcn.tcninclusiontraining import using_tcntraining
from timemachines.skatertools.data.live import using_micro

if using_tcntraining and using_micro:

    from timemachines.skatertools.data.live import random_surrogate_data
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential
    from tcn import TCN
    from sklearn.metrics import mean_squared_error
    from tf2onnx.keras2onnx_api import convert_keras


    def train_tcn_surrogate(f, model=None, k=1, n_real=3, n_samples=200, n_warm = 500, n_input=20, verbose=False,
                            n_iterations=100, n_tile=10, n_models=5,
                            include_str ='electricity',
                            exclude_str='~',
                            name_stem='tcn_surrogate_'):
        """
         :returns [

         This function takes a skater f and does the following:
              - retrieves live data history for streams,
              - runs the skater to create sequence-to-target training data
                       (n_input lags, only one output which is the k-step ahead prediction)
              - augments the data (n_tile copies with random linear scaling)
              - initiates training of a TCN model
              - serializes at most n_models ONNX models as training progresses

        :param f:   skater
        :param k:   number of steps ahead
        :param n_real: number of real world time-series to use
        :param include_str: str to match to time series
        :param exclude_str: str to match to time series for exclusion
        :return:   list of ONNX models
        """

        def saving_criterion(low_error_surrogate_true,error_surrogate_true,error_model_true):
            """  Determines whether the partially trained model is worth saving as an ensemble candidate
            :param low_error_surrogate_true:   lowest running surrogate model prediction error
            :param error_surrogate_true:       surrogate prediction error
            :param error_model_true:           original model prediction error
            :return:
            """
            return (error_surrogate_true < error_model_true) and (error_surrogate_true < 1.05*low_error_surrogate_true)


        import numpy as np
        if verbose:
            print('Creating training data')
        x_train_one, y_train_one, y_true_one = random_surrogate_data(f=f, k=k, n_real=n_real, n_samples=n_samples,
                                                          n_warm = n_warm, n_input=n_input, verbose=verbose,
                                                         include_str=include_str, exclude_str=exclude_str)

        x_train = np.tile(x_train_one,(n_tile,1,1))
        y_train = np.tile(y_true_one,n_tile)
        n = np.shape(x_train)[0]
        for i in range(n):
            random_scale = np.random.exponential()
            for j in range(n_input):
                x_train[i,j,0]  = (x_train[i,j,0]+0.001*np.random.randn())*random_scale
            y_train[i] = (y_train[i]+0.001*np.random.randn())*random_scale

        if model is None:
            tcn_layer = TCN(input_shape=(n_input, 1),dilations=tuple([1 for _ in range(n_input)]))
            tcn_layer = TCN(input_shape=(n_input, 1))
            model = Sequential([
                tcn_layer,
                Dense(1,activation='linear')
            ])
            model.compile(optimizer='adam', loss='mse')

        onnx_models = list()
        low_error_surrogate_true = 100000000
        for iter_no in range(n_iterations):
            model.fit(x_train, y_train, epochs=10, verbose=0)
            y_hat_one = model.predict(x_train_one)
            error_model_surrogate = mean_squared_error(y_hat_one, y_train_one)
            error_surrogate_true = mean_squared_error(y_hat_one, y_true_one)
            error_model_true = mean_squared_error(y_train_one, y_true_one)
            if verbose:
                print({'surrogate-model': error_model_surrogate,
                       'ratio':error_model_surrogate/error_model_true,
                       'surrogate-true': error_surrogate_true,
                       'model-true': error_model_true})
            if saving_criterion(low_error_surrogate_true=low_error_surrogate_true,
                                error_surrogate_true=error_surrogate_true,
                                error_model_true=error_model_true):
                print('  serializing')
                onnx_model_name = name_stem+str(len(onnx_models))
                onnx_model = convert_keras(model=model,name=onnx_model_name, doc_string='TCN model created by tcn_surrogate_train')
                onnx_models.append(onnx_model)
            if len(onnx_models)>=n_models:
                break
            low_error_surrogate_true = min(low_error_surrogate_true, error_surrogate_true)

        return onnx_models


if __name__=='__main__':
    assert using_tcntraining
    assert using_micro
    from timemachines.skaters.elo.eloensembles import elo_fastest_residual_precision_ensemble
    from timemachines.skaters.tsa.tsaensembles import tsa_precision_d0_ensemble
    from timemachines.skaters.simple.trivial import trivial_last_value
    from timemachines.skaters.simple.movingaverage import quickly_moving_average
    f = tsa_precision_d0_ensemble
    from keras.layers import LSTM


    n_lags = 40
    n_units = 16
    model = Sequential()
    model.add(LSTM(n_units, input_shape=(n_lags,1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    onnx_models = train_tcn_surrogate(f=f, model=model, k=1, n_real=25, n_samples=300, n_warm = 500, n_tile = 1,
                                n_input=n_lags, verbose=True, n_iterations=10000, n_models=30000,
                                      include_str='electricity')
    # Check inference with ONNX runtime
    if len(onnx_models):
        onnx_model = onnx_models[0]
        onnx_model_as_byte_string = onnx_model.SerializeToString()
        from onnxruntime import InferenceSession
        import numpy as np
        session = InferenceSession(onnx_model_as_byte_string)
        example_input = np.random.randn(1, n_lags, 1).astype(np.float32)
        if False:
            got = session.run(None, {'tcn_input': example_input}) # <-- need to fix input name
            y_hat = got[0][0][0]