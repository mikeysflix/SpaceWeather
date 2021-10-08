from naive_lag_correlations import *

def example_lag_correlations(with_parameterization):
    np.random.seed(0)
    vs = np.random.lognormal(mean=5.3, sigma=1.2, size=1000)
    lag_ks = lag_ks=np.arange(151, dtype=int)
    nresamples = 10
    if with_parameterization:
        extreme_values = np.arange(500, 1501, dtype=int)
        lag_corr = NaiveLagCorrelationsParameterization(
            vs=vs,
            extreme_parameter='parameter id as type <str>',
            extreme_condition='greater than')
        lag_corr(
            extreme_values=extreme_values,
            lag_ks=lag_ks,
            nresamples=nresamples)
    else:
        extreme_values = (700, 800, 1000)
        lag_corr = NaiveLagCorrelations(
            vs=vs,
            extreme_parameter='parameter id as type <str>',
            extreme_condition='greater than')
        lag_corr(
            extreme_values=extreme_values,
            lag_ks=lag_ks,
            nresamples=nresamples)
    print(lag_corr)

example_lag_correlations(with_parameterization=True)
example_lag_correlations(with_parameterization=False)

##
