from timemachines.skaters.proph.prophiskaterfactory import using_prophet
if using_prophet:
    from timemachines.skaters.proph.prophiskaterfactory import prophet_iskater_factory
    from timemachines.skatertools.utilities.conventions import wrap
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import random
    import math
    import os
    from pprint import pprint

    PLOTS_PATH = os.path.dirname(os.path.realpath(__file__)).replace('timemachines/timemachines','timemachines/gallery')


    #  Just some scripts that try to investigate some curious behaviour of prophet, and performance
    #  comparison against dead-simple averaging of recent values
    #  TODO: Move this to timemachines-testing


    def is_opinonated(y, forecast:pd.DataFrame,k:int, n_recent:int, multiple:float)->bool:
        """ Check if the forecast is far from any recent values, and thus "opinionated"

        :param y:          data used to fit
        :param forecast:   dataframe produced by prophet fitting
        :param m:          fitted facebook prophet model
        :param k:          number of steps ahead
        :return:
        """
        if isinstance(y[0], float):
            y = [wrap(yj) for yj in y]
        y0 = [yj[0] for yj in y]

        for j in range(1,k+1):
           j_std = np.nanstd(np.diff(y0[-k-50:-k],j))
           recent_ys = y0[-(k+n_recent):-(k+1)]
           upper = np.max(recent_ys)+multiple*j_std*math.sqrt(j)+0.1
           lower = np.min(recent_ys)-multiple*j_std*math.sqrt(j)-0.1
           j_x = forecast['yhat'].values[-(1+k-j)]
           if j_x>upper or j_x<lower:
               deviation = abs(j_x-upper)
               print(deviation)
               return True

        return False


    def next_opinionated_forecast(n_train,k,n_recent, multiple,name=None):
        while True:
            try:
                from microprediction import MicroReader
                n_obs = 0
                while n_obs<1000:
                    mr = MicroReader()
                    if name is None:
                        names = mr.get_stream_names()
                        random_name = random.choice(names)
                    else:
                        random_name = name
                    lag_values, lag_times = mr.get_lagged_values_and_times(name=random_name,count=2000)
                    y = list(reversed(lag_values))
                    t = list(reversed(lag_times))
                    n_obs = len(y)
            except ImportError:
                from timemachines.skatertools.data import hospital
                y = hospital()
                t = [15*60*i for i in range(len(y))]
                name = 'hospital'

            for i in [100*j for j in range(10)]:
                print('Looking at ' + random_name + ' ' + str(i) + '/1000')
                if len(y)>i+2*k+n_train:
                    y_fit = y[i:i + n_train]
                    t_fit = t[i:i + n_train]
                    y_hats, _, forecast, m = prophet_iskater_factory(y=y_fit, k=k, t=t_fit)
                    if is_opinonated(y=y_fit, forecast=forecast,k=k, n_recent=n_recent, multiple=multiple):
                        y_3avg = np.mean(y[i + n_train-3:i + n_train])  # avg of last three points
                        y_true_mean = np.mean( y[i+n_train+k-1:i+n_train+k+1] ) # avg of 3 future points
                        error = (y_hats[-1]-y_true_mean)/abs(0.01+y_3avg)
                        avg_error = (y_3avg-y_true_mean)/abs(0.01+y_3avg)
                        return forecast, m, random_name, error, avg_error, y[i+n_train:i+n_train+k]
            print(random_name+' is okay.')

        return None, None, None


    def plot_next_optionated_forecast(k, n_train, n_recent, multiple, name=None):
        forecast, m, name, err, lv_err, y_next = next_opinionated_forecast(k=k, n_train=n_train,name=name,n_recent=n_recent, multiple=multiple)

        if forecast is not None:
            m.plot(forecast)
            plt.title(name.replace('.json', ''))
            plt.xlabel('Error = ' + str(round(err,3)) + ', Last value err =' + str(round(lv_err,3)))
            plt.plot(forecast['ds'].values[-k:],y_next,'r+')
            plt.pause(5)
            plt.tight_layout()
            rel_error = int(10000*abs((0.01+err)/(0.01+lv_err)))
            jpg_path = PLOTS_PATH+os.path.sep+'n_train='+str(n_train)+'_k_'+str(k)+'_multiple_'+str(round(multiple,1))
            try:
                os.mkdir(jpg_path)
            except FileExistsError:
                pass
            jpg_file = jpg_path+os.path.sep + 'rel='+str(rel_error)+'_name='+name.replace('.json', '.jpg')
            plt.savefig(jpg_file)
            plt.close()
            print(jpg_file)

        return err, lv_err, jpg_path


if __name__=='__main__':
    from momentum import var_init, var_update
    import json
    lv = var_init()
    pr = var_init()
    params = {'multiple':1.5,'n_recent':5,'n_train':500,'k':20}
    while True:
        err, lv_err, jpg_path = plot_next_optionated_forecast(**params)
        report = params.copy()
        report['simple_average'] = var_update(lv,lv_err)
        report['prophet'] = var_update(pr,err)
        pprint(report)
        report_file = jpg_path+'_summary.json'
        with open(report_file,'wt') as f:
            json.dump(report,f)
