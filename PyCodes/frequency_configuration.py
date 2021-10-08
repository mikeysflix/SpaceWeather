from optimization_methods import *

class TemporalFrequencyConfiguration():

    def __init__(self, data, extreme_parameter, temporal_histogram=None):
        super().__init__()
        self.data = data
        self.extreme_parameter = extreme_parameter
        if temporal_histogram is None:
            self._temporal_histogram = TemporalHistogram(
                data=data)
        else:
            self._temporal_histogram  = temporal_histogram
        self._temporal_statistics = None

    @property
    def temporal_histogram(self):
        return self._temporal_histogram

    @property
    def temporal_statistics(self):
        return self._temporal_statistics

    def __call__(self, edges=None, nbins=None, wbin=None, midpoints=None, bin_widths=None, lbin=None, rbin=None, time_step='second', time_scale='relative', threshold=None, merge_condition='less than', squeeze_trails=False, squeeze_leads=False, tol=0, extra_counts=False, bias=False, fisher=False, ddof=0):
        if self.temporal_histogram.edges is None:
            self._temporal_histogram.update_edges(
                edges=edges,
                nbins=nbins,
                wbin=wbin,
                midpoints=midpoints,
                bin_widths=bin_widths,
                lbin=lbin,
                rbin=rbin,
                time_step=time_step,
                time_scale=time_scale)
        if self.temporal_histogram.counts is None:
            self._temporal_histogram.update_counts(
                threshold=threshold,
                merge_condition=merge_condition,
                squeeze_trails=squeeze_trails,
                squeeze_leads=squeeze_leads,
                tol=tol,
                extra_counts=extra_counts)
        temporal_statistics = StatisticsConfiguration(vs=self.temporal_histogram.counts)
        temporal_statistics.update_statistics(
            bias=bias,
            fisher=fisher,
            ddof=ddof,
            nan_policy='omit')
        self._temporal_statistics = temporal_statistics

class MixedFrequencyConfiguration(TemporalFrequencyConfiguration):

    def __init__(self, data, extreme_parameter):
        super().__init__(
            data=data,
            extreme_parameter=extreme_parameter,
            temporal_histogram=None)
        self._parameter_histogram = Histogram(data[extreme_parameter])
        self._parameter_statistics = None
        self._mixed_histogram = None

    @property
    def parameter_histogram(self):
        return self._parameter_histogram

    @property
    def parameter_statistics(self):
        return self._parameter_statistics

    @property
    def mixed_histogram(self):
        return self._mixed_histogram

    def __call__(self, parameter_bin_kwargs, temporal_bin_kwargs=None, threshold=None, merge_condition='less than', squeeze_trails=False, squeeze_leads=False, tol=0, extra_counts=False, bias=False, fisher=False, ddof=0):
        ## initialize parameter histogram
        self._parameter_histogram.update_edges(**parameter_bin_kwargs)
        self._parameter_histogram.update_counts(
            threshold=threshold,
            merge_condition=merge_condition,
            squeeze_trails=squeeze_trails,
            squeeze_leads=squeeze_leads,
            tol=tol,
            extra_counts=extra_counts)
        ## initialize parameter statistics
        parameter_statistics = StatisticsConfiguration(vs=self.parameter_histogram.counts)
        parameter_statistics.update_statistics(
            bias=bias,
            fisher=fisher,
            ddof=ddof,
            nan_policy='omit')
        self._parameter_statistics = parameter_statistics
        ## initialize temporal histogram
        if self.temporal_histogram.edges is None:
            self._temporal_histogram.update_edges(**temporal_bin_kwargs)
        if self.temporal_histogram.counts is None:
            self._temporal_histogram.update_counts(
                threshold=threshold,
                merge_condition=merge_condition,
                squeeze_trails=squeeze_trails,
                squeeze_leads=squeeze_leads,
                tol=tol,
                extra_counts=extra_counts)
        ## initialize temporal statistics
        temporal_statistics = StatisticsConfiguration(vs=self.temporal_histogram.counts)
        temporal_statistics.update_statistics(
            bias=bias,
            fisher=fisher,
            ddof=ddof,
            nan_policy='omit')
        self._temporal_statistics = temporal_statistics
        ## update bi-dimensional counts
        xy_counts, x_edges, y_edges = np.histogram2d(
            x=self.temporal_histogram.vs,
            y=self.parameter_histogram.vs,
            bins=(self.temporal_histogram.edges, self.parameter_histogram.edges))
        self._mixed_histogram = {
            'x edges' : x_edges,
            'y edges' : y_edges,
            'counts' : xy_counts}







##
