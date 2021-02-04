
# Implements the old Matlab tic/toc which is still my favourite interface
# That style makes is somewhat invasive - though the odds of s['chronometer'] being used for
# another purpose seem slim

import time
from pprint import pprint


def tick(s):
    if not s.get('chronometer'):
        s['chronometer'] = {'tick':time.time()}
    else:
        s['chronometer'].update({'tick':time.time()})


def tocks(s):
    """ Returns a copy of reported timings """
    return dict([(k,v) for k,v in s['chronometer'].items() if k not in ['tick']])


def tock(s,label:str='invocation',reset=True, accumulate=True):
    assert label not in ['tick','tock']
    assert s.get('chronometer') and s.get('chronometer').get('tick'),'Must call tick before tock'
    elapsed = time.time()-s['chronometer']['tick']
    if accumulate and s['chronometer'].get(label):
        elapsed += s['chronometer'][label]
    s['chronometer'][label] = elapsed
    if reset:
        s['chronometer']['tick']=time.time()
    return s


def lap(s,label:str):
    return tock(s=s,label=label,reset=False)


if __name__=='__main__':
    s = dict()
    tick(s)
    time.sleep(1)
    tock(s)
    pprint(s)
    time.sleep(1)
    tock(s,'fit')
    pprint(s)