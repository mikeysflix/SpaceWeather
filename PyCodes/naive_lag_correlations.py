from optimization_methods import *
from resampling_configuration import *

class NaiveLagCorrelations(ResamplingConfiguration):

    def __init__(self, vs, extreme_parameter, extreme_condition='greater than'):
        super().__init__()
        self.vs = vs
        self.extreme_parameter = extreme_parameter
        self.extreme_condition = extreme_condition
        self.extreme_operator = ConditionMapping().comparisons[extreme_condition]
        self._extreme_values = OrderedDict()
        self._lag_ks = None

    @property
    def extreme_values(self):
        return self._extreme_values

    @property
    def lag_ks(self):
        return self._lag_ks

    def __repr__(self):
        return 'NaiveExtremalTimeLagCorrelation(%r, %r, %r)' % (
            self.vs,
            self.extreme_parameter,
            self.extreme_condition)

    def __str__(self):
        extreme_values = list(self.extreme_values.keys())
        lowest_extreme_value, highest_extreme_value = extreme_values[0], extreme_values[-1]
        s = '\n ** Naive Lag-Correlations **\n %s extreme-values: %s \n' % (
            len(extreme_values),
            extreme_values)
        for extreme_value in (lowest_extreme_value, highest_extreme_value):
            s += '\n results via extreme_value=%s \n\n .. k=%s ==> lambda = %s \n .. k=%s ==> lambda = %s \n' % (
                extreme_value,
                self.lag_ks[0],
                self.extreme_values[extreme_value]['original lambda'][0],
                self.lag_ks[-1],
                self.extreme_values[extreme_value]['original lambda'][-1])
        return s

    def update_lag_ks(self, lag_ks):
        if not isinstance(lag_ks, np.ndarray):
            raise ValueError("invalid type(lag_ks): {}".format(type(lag_ks)))
        if lag_ks.size > self.vs.size:
            raise ValueError("len(lag_ks)={} should not exceed len(vs)={}".format(lag_ks.size, self.vs.size))
        if np.max(lag_ks) > self.vs.size:
            raise ValueError("max(lag_ks)={} should not exceed len(vs)={}".format(lag_ks.size, self.vs.size))
        self._lag_ks = lag_ks

    def get_lams(self, extreme_value, vs):
        binary_bins = self.extreme_operator(vs, extreme_value)
        mirror_bins = binary_bins.reshape((-1, 1))
        mesh = binary_bins * mirror_bins
        lams = [np.trace(mesh, offset=k) / (vs.size - k) for k in self.lag_ks]
        return np.array(lams)

    def update_extreme_value(self, extreme_value):
        binary_bins = self.extreme_operator(self.vs, extreme_value)
        constant_counts = np.mean(binary_bins)
        og_lams = self.get_lams(
            extreme_value=extreme_value,
            vs=self.vs)
        re_lams = []
        for i in range(self.nresamples):
            self.resample_indices()
            time_resampled_lams = self.get_lams(
                extreme_value=extreme_value,
                vs=self.vs[self.resampled_indices])
            re_lams.append(time_resampled_lams)
        re_lams = np.array(re_lams)
        if self.nresamples > 0:
            df_lams = og_lams - np.mean(re_lams, axis=0)
        else:
            df_lams = np.full(og_lams.shape, np.nan)
        self._extreme_values[extreme_value] = {
            'constant counts' : constant_counts,
            'original lambda' : og_lams / constant_counts,
            'time-randomized lambda' : re_lams / constant_counts,
            'difference lambda' : df_lams / constant_counts}

    def __call__(self, extreme_values, lag_ks, nresamples=100, nshuffles=3, with_replacement=False):
        self.update_lag_ks(
            lag_ks=lag_ks)
        self.update_resampling_criteria(
            nresamples=nresamples,
            nshuffles=nshuffles,
            with_replacement=with_replacement)
        for extreme_value in extreme_values:
            self.update_resampling_indices(
                n=self.vs.size)
            self.update_extreme_value(
                extreme_value=extreme_value)

class NaiveLagCorrelationsParameterization(NaiveLagCorrelations):

    def __init__(self, vs, extreme_parameter, extreme_condition='greater than'):
        super().__init__(
            vs=vs,
            extreme_parameter=extreme_parameter,
            extreme_condition=extreme_condition)
        self._parameterization = dict()

    @property
    def parameterization(self):
        return self._parameterization

    def __repr__(self):
        return 'NaiveLagCorrelationParameterization(%r, %r, %r)' % (
            self.vs,
            self.extreme_parameter,
            self.extreme_condition)

    def __str__(self):
        extreme_values = list(self.extreme_values.keys())
        lowest_extreme_value, highest_extreme_value = extreme_values[0], extreme_values[-1]
        s = '\n ** Naive Lag-Correlations Parameterization **\n %s extreme-values: %s \n' % (
            len(extreme_values),
            extreme_values)
        for extreme_value in (lowest_extreme_value, highest_extreme_value):
            s += '\n results via extreme_value=%s \n\n .. k=%s ==> lambda = %s \n .. k=%s ==> lambda = %s \n' % (
                extreme_value,
                self.lag_ks[0],
                self.extreme_values[extreme_value]['original lambda'][0],
                self.lag_ks[-1],
                self.extreme_values[extreme_value]['original lambda'][-1])
        return s

    def update_parameterization(self):
        vs = []
        Z_og, Z_df = [], []
        for extreme_value, lam_result in self.extreme_values.items():
            vs.append(extreme_value)
            Z_og.append(lam_result['original lambda'])
            Z_df.append(lam_result['difference lambda'])
        X, Y = np.meshgrid(
            np.array(vs),
            self.lag_ks)
        self._parameterization = {
            'X' : X,
            'Y' : Y,
            'Z' : {
                'original lambda' : np.array(Z_og).T,
                'difference lambda' : np.array(Z_df).T}}

    def __call__(self, extreme_values, lag_ks, nresamples=100, nshuffles=3, with_replacement=False):
        self.update_lag_ks(
            lag_ks=lag_ks)
        self.update_resampling_criteria(
            nresamples=nresamples,
            nshuffles=nshuffles,
            with_replacement=with_replacement)
        for extreme_value in extreme_values:
            self.update_resampling_indices(
                n=self.vs.size)
            self.update_extreme_value(
                extreme_value=extreme_value)
        self.update_parameterization()


##
