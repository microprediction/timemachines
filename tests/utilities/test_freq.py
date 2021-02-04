from timemachines.skaters.utilities.arrivals import infer_freq_from_epoch,  RECENT_SECONDS,naive_datetime_to_epoch, epoch_to_naive_datetime


def test_date_stuff():
    seconds = [1,5,7,10]
    dts = epoch_to_naive_datetime(seconds)
    sec_back = naive_datetime_to_epoch(dts)
    dts_back = epoch_to_naive_datetime(sec_back)
    assert all([t1==t2 for t1,t2 in zip(dts,dts_back)])
    assert all([int(s1)==int(s2) for s1,s2 in zip(seconds, sec_back)])


def test_inference():
    # https://github.com/pandas-dev/pandas/blob/master/pandas/tseries/frequencies.py
    expected = {'S':[RECENT_SECONDS + j for j in range(200)],
                '5S':[RECENT_SECONDS + 5 * j for j in range(200)],
                'T':[RECENT_SECONDS + 60 * j for j in range(200)],
                '5T':[RECENT_SECONDS + 5 * 60 * j for j in range(200)]} # 5 minutes
    for expected_freq, t in expected.items():
        freq = infer_freq_from_epoch(t)
        assert freq==expected_freq


if __name__=='__main__':
    test_date_stuff()