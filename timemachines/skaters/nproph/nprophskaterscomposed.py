from timemachines.skaters.nproph.nprophetinclusion import using_neuralprophet, NeuralProphet
if using_neuralprophet:
    from timemachines.skatertools.utilities.conventions import Y_TYPE, A_TYPE, E_TYPE, T_TYPE
    from timemachines.skatertools.composition.residualshypocratic import quickly_moving_hypocratic_residual_factory
    from timemachines.skaters.nproph.nprophetskaters import nprophet_p1, nprophet_p2, nprophet_p3, nprophet_p5, nprophet_p8


    def nprophet_p1_hypocratic(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
        """ Chase residuals, somewhat cautiously using, quickly moving average """
        return quickly_moving_hypocratic_residual_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=None, f=nprophet_p1)


    def nprophet_p2_hypocratic(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
        """ Chase residuals, somewhat cautiously using, quickly moving average """
        return quickly_moving_hypocratic_residual_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=None, f=nprophet_p2)


    def nprophet_p3_hypocratic(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
        """ Chase residuals, somewhat cautiously using, quickly moving average """
        return quickly_moving_hypocratic_residual_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=None, f=nprophet_p3)


    def nprophet_p5_hypocratic(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
        """ Chase residuals, somewhat cautiously using, quickly moving average """
        return quickly_moving_hypocratic_residual_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=None, f=nprophet_p5)


    def nprophet_p8_hypocratic(y: Y_TYPE, s: dict, k: int, a: A_TYPE = None, t: T_TYPE = None, e: E_TYPE = None):
        """ Chase residuals, somewhat cautiously using, quickly moving average """
        return quickly_moving_hypocratic_residual_factory(y=y, s=s, k=k, a=a, t=t, e=e, r=None, f=nprophet_p8)

    NPROPHET_SKATERS_COMPOSED = [nprophet_p1_hypocratic, nprophet_p2_hypocratic, nprophet_p3_hypocratic,
                                 nprophet_p5_hypocratic, nprophet_p8_hypocratic ]
else:
    NPROPHET_SKATERS_COMPOSED = []



if __name__ == '__main__':
    assert using_neuralprophet,'pip install neuralprophet'
    from timemachines.skatertools.data.real import hospital_with_exog
    from timemachines.skatertools.visualization.priorplot import prior_plot
    import matplotlib.pyplot as plt
    k = 1
    y, a = hospital_with_exog(k=k, n=450, offset=True)
    f = nprophet_p1_hypocratic
    err2 = prior_plot(f=f, k=k, y=y, n=450, n_plot=50)
    print(err2)
    plt.show()
    pass
