from timemachines.skaters.pmd.pmdinclusion import using_pmd
if using_pmd:
    from timemachines.skaters.pmd.pmdskaters import pmd_exogenous
    from timemachines.skatertools.data import hospital_with_exog
    from pmdarima.arima import ARIMA


    # So the task here is to serialize and re-inflate the arima model, and thus all the skater state, to JSON
    # It may be a lost cause, but by all means prove me wrong.


    def example_pmd():
        s = {}
        y, a = hospital_with_exog(k=3)
        x = [ pmd_exogenous(y=yj,s=s,k=3,a=aj) for yj,aj in zip(y[:500],a) ]
        return s


    def arima_res_to_dict(arima_res):
        state = arima_res.__dict__
        return state


    def pmd_to_dict(pmd):
        pmd['model'] = pmd['model'].__getstate__()
        pmd['model']['arima_res_'] = arima_res_to_dict(pmd['model']['arima_res_'])
        return pmd


    def pmd_from_dict(pmd):
        pmd['model']['arima_res_'] = ''


if __name__=='__main__':
    assert using_pmd,'pip install pmdarima'
    pmd = example_pmd()
    model = pmd['model']
    model1 = ARIMA(**model.get_params())
    prms = model.__dict__['arima_res_'].__dict__['_results'].params




