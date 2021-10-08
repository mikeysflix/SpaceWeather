from optimization_methods import *
from resampling_configuration import *
# import datetime

class GeneralizedLagCorrelations(ResamplingConfiguration):

    def __init__(self, vs, dts, extreme_parameter, extreme_condition='greater than'):
        super().__init__()
        self.vs = vs
        self.dts = dts
        self.extreme_parameter = extreme_parameter
        self.extreme_condition = extreme_condition
        self.extreme_operator = ConditionMapping().comparisons[extreme_condition]
        self._extreme_values = OrderedDict()
        self._lag_ks = None
        self._time_step = None
        self._edges = None

    @property
    def extreme_values(self):
        return self._extreme_values

    @property
    def lag_ks(self):
        return self._lag_ks

    @property
    def time_step(self):
        return self._time_step

    @property
    def edges(self):
        return self._edges

    def __repr__(self):
        return 'GeneralizedLagCorrelations(%r, %r, %r, %r)' % (
            self.vs,
            self.dts,
            self.extreme_parameter,
            self.extreme_condition)

    def __str__(self):
        extreme_values = list(self.extreme_values.keys())
        lowest_extreme_value, highest_extreme_value = extreme_values[0], extreme_values[-1]
        s = '\n ** Generalized Lag-Correlations **\n %s extreme-values: %s \n' % (
            len(extreme_values),
            extreme_values)
        for extreme_value in (lowest_extreme_value, highest_extreme_value):
            s += '\n results via extreme_value=%s \n\n original mesh-matrix:\n %s \n\n .. k=%s ==> lambda = %s \n .. k=%s ==> lambda = %s \n' % (
                extreme_value,
                self.extreme_values[extreme_value]['original mesh-matrix'],
                self.lag_ks[0],
                self.extreme_values[extreme_value]['original lambda'][0],
                self.lag_ks[-1],
                self.extreme_values[extreme_value]['original lambda'][-1])
        return s

    def update_edges(self, edges=None, nbins=None, wbin=None, midpoints=None, bin_widths=None, lbin=None, rbin=None, time_step='second', time_scale='relative'):
        histogram = TemporalHistogram(data={'datetime' : self.dts})
        histogram.update_edges(
            edges=edges,
            nbins=nbins,
            wbin=wbin,
            midpoints=midpoints,
            bin_widths=bin_widths,
            lbin=lbin,
            rbin=rbin,
            time_step=time_step,
            time_scale=time_scale)
        self._time_step = histogram.time_step
        self._edges = histogram.edges

    def update_lag_ks(self, lag_ks):
        if not isinstance(lag_ks, np.ndarray):
            raise ValueError("invalid type(lag_ks): {}".format(type(lag_ks)))
        if lag_ks.size > self.edges.size - 1:
            raise ValueError("len(lag_ks)={} should not exceed number of edges={}".format(lag_ks.size))
        if np.max(lag_ks) > self.edges.size - 1:
            raise ValueError("max(lag_ks)={} should not exceed maximum bin-edge".format(lag_ks.size))
        self._lag_ks = lag_ks

    def get_lams(self, extreme_value, data):
        histogram = TemporalHistogram(
            data=data)
        histogram.update_edges(
            edges=self.edges)
        histogram.update_counts()
        mat = np.zeros((histogram.counts.size, histogram.counts.size))
        mat[:, :] = histogram.counts + histogram.counts.reshape((-1, 1))
        loc = np.where(histogram.counts == 0)[0]
        mat[loc, :] = 0
        mat[:, loc] = 0
        _lams = [np.trace(mat, offset=k) / (histogram.counts.size - k)
            for k in self.lag_ks]
        lams = np.array(_lams) / np.mean(histogram.counts)
        return lams, histogram, mat

    def update_extreme_value(self, extreme_value):
        loc = self.extreme_operator(self.vs, extreme_value)
        original_data = {
            'datetime' : np.copy(self.dts[loc])}
        original_lams, original_histogram, original_mat = self.get_lams(
            extreme_value=extreme_value,
            data=original_data)
        indices = np.where(loc)[0]
        self.update_resampling_indices(
            # n=indices.size)
            n=loc.size)
        time_randomized_lams = []
        for i in range(self.nresamples):
            self.resample_indices()
            re_vs = np.copy(self.vs[self.resampled_indices])
            vloc = self.extreme_operator(re_vs, extreme_value)
            re_data = {
                'datetime' : np.copy(self.dts[vloc])}
            re_lams, _, _ = self.get_lams(
                extreme_value=extreme_value,
                data=re_data)
            time_randomized_lams.append(re_lams)
        time_randomized_lams = np.array(time_randomized_lams)
        if self.nresamples > 0:
            difference_lams = original_lams - np.mean(time_randomized_lams, axis=0)
        else:
            difference_lams = np.copy(original_lams)
            difference_lams[:] = np.nan
        self._extreme_values[extreme_value] = {
            'original histogram' : original_histogram,
            'original mesh-matrix' : original_mat,
            'original lambda' : original_lams,
            'time-randomized lambda' : time_randomized_lams,
            'difference lambda' : difference_lams}

    def __call__(self, extreme_values, lag_ks, nresamples=100, nshuffles=3, with_replacement=False, edges=None, nbins=None, wbin=None, midpoints=None, bin_widths=None, lbin=None, rbin=None, time_step='second', time_scale='relative'):
        self.update_edges(
            edges=edges,
            nbins=nbins,
            wbin=wbin,
            midpoints=midpoints,
            bin_widths=bin_widths,
            lbin=lbin,
            rbin=rbin,
            time_step=time_step,
            time_scale=time_scale)
        self.update_lag_ks(
            lag_ks=lag_ks)
        self.update_resampling_criteria(
            nresamples=nresamples,
            nshuffles=nshuffles,
            with_replacement=with_replacement)
        for extreme_value in extreme_values:
            self.update_extreme_value(
                extreme_value=extreme_value)

class GeneralizedLagCorrelationsParameterization(GeneralizedLagCorrelations):

    def __init__(self, vs, dts, extreme_parameter, extreme_condition='greater than'):
        super().__init__(
            vs=vs,
            dts=dts,
            extreme_parameter=extreme_parameter,
            extreme_condition=extreme_condition)
        self._parameterization = dict()

    @property
    def parameterization(self):
        return self._parameterization

    def __repr__(self):
        return 'GeneralizedLagCorrelationsParameterization(%r, %r, %r, %r)' % (
            self.vs,
            self.dts,
            self.extreme_parameter,
            self.extreme_condition)

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
        self._parameterization.update({
            'X' : X,
            'Y' : Y,
            'Z' : {
                'original lambda' : Z_og,
                'difference lambda' : Z_df}})

    def __call__(self, extreme_values, lag_ks, nresamples=100, nshuffles=3, with_replacement=False, edges=None, nbins=None, wbin=None, midpoints=None, bin_widths=None, lbin=None, rbin=None, time_step='second', time_scale='relative'):
        self.update_edges(
            edges=edges,
            nbins=nbins,
            wbin=wbin,
            midpoints=midpoints,
            bin_widths=bin_widths,
            lbin=lbin,
            rbin=rbin,
            time_step=time_step,
            time_scale=time_scale)
        self.update_lag_ks(
            lag_ks=lag_ks)
        self.update_resampling_criteria(
            nresamples=nresamples,
            nshuffles=nshuffles,
            with_replacement=with_replacement)
        for extreme_value in extreme_values:
            self.update_extreme_value(
                extreme_value=extreme_value)
        self.update_parameterization()


def example(with_parameterization=False):
    np.random.seed(0)
    vs = np.random.lognormal(mean=5.3, sigma=1.2, size=1000)
    dts = np.array([datetime.datetime(1999, 10, 30) + datetime.timedelta(hours=i) for i in range(vs.size)])
    lag_ks = lag_ks=np.arange(151, dtype=int)
    nresamples = 10
    if with_parameterization:
        extreme_values = np.arange(500, 1501, dtype=int)
        lag_corr = GeneralizedLagCorrelationParameterization(
            vs,
            dts,
            'extreme parameter')
    else:
        extreme_values = (700, 800, 1000)
        lag_corr = GeneralizedLagCorrelations(
            vs,
            dts,
            'extreme parameter')
    lag_corr(
        extreme_values=extreme_values,
        lag_ks=lag_ks,
        nresamples=nresamples,
        lbin=dts[0],
        rbin=dts[-1],
        wbin=1,
        time_step='hour')
    print(lag_corr)

# example(with_parameterization=True)
# example(with_parameterization=False)









#
