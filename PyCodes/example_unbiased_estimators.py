from unbiased_estimators import *

def example_unbiased_estimators(data_id):
    """
    data_id:
        'evens' or 'uniform'
    """

    if data_id == 'evens':
        vs = np.arange(1, 100).astype(int)
        extreme_indices = np.arange(1, 6).astype(int)
        value_to_replace = None
    elif data_id == 'uniform':
        vs = np.random.uniform(low=0, high=500, size=int(1e5))
        extreme_indices = np.arange(4, 12).astype(int)
        value_to_replace = -1
    else:
        raise ValueError("invalid data_id: {}".format(data_id))
    unbiased_estimators = UnbiasedEstimators(
        vs=vs,
        extreme_parameter='parameter id as type<str>',
        extreme_indices=extreme_indices)
    unbiased_estimators(
        prms_guess=None,
        nresamples=1000,
        nshuffles=3,
        with_replacement=False,
        value_to_replace=value_to_replace,
        ddof=0,
        alpha_histogram_kwargs={
            'nbins' : 20},
        intercept_histogram_kwargs={
            'nbins' : 20},
        theta_histogram_kwargs={
            'nbins' : 20},
        scale='local',
        method='Nelder-Mead')
    print(unbiased_estimators)

example_unbiased_estimators('evens')
example_unbiased_estimators('uniform')



##
