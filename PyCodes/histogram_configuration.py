import numpy as np
import scipy.stats as SPstats # kurtosis, skew, lognorm
from scipy.integrate import quad
from temporal_configuration import *

class StatisticsConfiguration(TemporalConfiguration):

    def __init__(self, vs):
        super().__init__()
        self.vs = np.copy(vs)
        self._skew = None
        self._kurtosis = None
        self._standard_deviation = None
        self._standard_error = None
        self._mean = None
        self._median = None
        self._maximum = None
        self._minimum = None
        self._rolling_windows = None
        self._rolling_window_id = None
        self._rolling_statistics = dict()

    @property
    def skew(self):
        return self._skew

    @property
    def kurtosis(self):
        return self._kurtosis

    @property
    def standard_deviation(self):
        return self._standard_deviation

    @property
    def standard_error(self):
        return self._standard_error

    @property
    def mean(self):
        return self._mean

    @property
    def median(self):
        return self._median

    @property
    def maximum(self):
        return self._maximum

    @property
    def minimum(self):
        return self._minimum

    @property
    def rolling_windows(self):
        return self._rolling_windows

    @property
    def rolling_window_id(self):
        return self._rolling_window_id

    @property
    def rolling_statistics(self):
        return self._rolling_statistics

    @staticmethod
    def get_midpoints(edges):
        return 0.5 * (edges[1:] + edges[:-1])

    def update_rolling_windows(self, window_size, overlap=False):
        shape = self.vs.shape[:-1] + (self.vs.shape[-1] - window_size + 1, window_size)
        strides = self.vs.strides + (self.vs.strides[-1],)
        result = np.lib.stride_tricks.as_strided(self.vs, shape=shape, strides=strides)
        if overlap:
            self._rolling_windows = result
            self._rolling_window_id = 'overlap'
        else:
            self._rolling_windows = np.copy(result[::window_size, :])
            self._rolling_window_id = 'non-overlap'

    @staticmethod
    def get_skew(vs, bias=False, nan_policy='propagate', **kwargs):
        return SPstats.skew(vs, bias=bias, nan_policy=nan_policy, **kwargs)

    @staticmethod
    def get_kurtosis(vs, fisher=False, bias=False, nan_policy='propagate', **kwargs):
        return SPstats.kurtosis(vs, fisher=fisher, bias=bias, nan_policy=nan_policy, **kwargs)

    @staticmethod
    def get_standard_deviation(vs, ddof=0, **kwargs):
        return np.nanstd(vs, ddof=ddof, **kwargs)

    @staticmethod
    def get_standard_error_of_mean(vs, ddof=0, nan_policy='propagate', **kwargs):
        return sem(vs, ddof=ddof, nan_policy=nan_policy, **kwargs)

    @staticmethod
    def get_mean(vs, **kwargs):
        return np.nanmean(vs, **kwargs)

    @staticmethod
    def get_median(vs, **kwargs):
        return np.nanmedian(vs, **kwargs)

    @staticmethod
    def get_maximum(vs, **kwargs):
        return np.nanmax(vs, **kwargs)

    @staticmethod
    def get_minimum(vs, **kwargs):
        return np.nanmin(vs, **kwargs)

    def update_skew(self, bias=False, nan_policy='propagate', **kwargs):
        self._skew = self.get_skew(
            vs=self.vs,
            bias=bias,
            nan_policy=nan_policy,
            **kwargs)

    def update_kurtosis(self, fisher=False, bias=False, nan_policy='propagate', **kwargs):
        self._kurtosis = self.get_kurtosis(
            vs=self.vs,
            fisher=fisher,
            bias=bias,
            nan_policy=nan_policy,
            **kwargs)

    def update_standard_deviation(self, ddof=0, **kwargs):
        self._standard_deviation = self.get_standard_deviation(
            vs=self.vs,
            ddof=ddof,
            **kwargs)

    def update_standard_error_of_mean(self, ddof=0, nan_policy='propagate', **kwargs):
        self._standard_error = self.get_standard_error_of_mean(
            vs=self.vs,
            ddof=ddof,
            nan_policy=nan_policy,
            **kwargs)

    def update_mean(self, **kwargs):
        self._mean = self.get_mean(
            vs=self.vs,
            **kwargs)

    def update_median(self, **kwargs):
        self._median = self.get_median(
            vs=self.vs,
            **kwargs)

    def update_maximum(self, **kwargs):
        self._maximum = self.get_maximum(
            vs=self.vs,
            **kwargs)

    def update_minimum(self, **kwargs):
        self._minimum = self.get_minimum(
            vs=self.vs,
            **kwargs)

    def update_statistics(self, bias=False, fisher=False, ddof=0, nan_policy='propagate'):
        self.update_skew(bias, nan_policy)
        self.update_kurtosis(fisher, bias, nan_policy)
        self.update_standard_deviation(ddof)
        self.update_standard_error_of_mean(ddof, nan_policy)
        self.update_mean()
        self.update_median()
        self.update_minimum()
        self.update_maximum()

    def dispatch_func(self, statistic_id):
        fmap = {
            'skew' : self.get_skew,
            'kurtosis' : self.get_kurtosis,
            'mean' : self.get_mean,
            'median' : self.get_median,
            'standard deviation' : self.get_standard_deviation,
            'standard error of mean' : self.get_standard_error_of_mean,
            'maximum' : self.get_maximum,
            'minimum' : self.get_minimum}
        return fmap[statistic_id]

    def update_rolling_statistics(self, statistic_ids, **kwargs):
        if not isinstance(statistic_ids, (tuple, list, np.ndarray)):
            statistic_ids = [statistic_ids]
        result = dict()
        for statistic_id in statistic_ids:
            f = self.dispatch_func(statistic_id)
            result[statistic_id] = f(vs=self.rolling_windows, axis=1, **kwargs)
        self._rolling_statistics.update(result)

class Histogram(StatisticsConfiguration):

    def __init__(self, vs, hbias='left'):
        super().__init__(
            vs=vs)
        if hbias not in ('left', 'right'):
            raise ValueError("invalid hbias: {}".format(hbias))
        self.hbias = hbias
        self.verify_id = 'Histogram'
        self._edges = None
        self._counts = None
        self._midpoints = None
        self._bin_widths = None
        self._normalization_constant = None
        self._normalized_counts = None
        self._cumulative_counts = None
        self._normalized_cumulative_counts = None
        self._probability_density = None
        self._bin_threshold = None
        self._merge_condition = None
        self._mesh = None

    def __repr__(self):
        return 'Histogram(%r, %r)' % (
            self.vs,
            self.hbias)

    def __str__(self):
        return '\n ** Histogram ** \n\n .. %s edges = %s - %s \n .. %s midpoints = %s - %s \n .. %s counts \n' % (
            self.edges.size,
            self.edges[0],
            self.edges[-1],
            self.midpoints.size,
            self.midpoints[0],
            self.midpoints[-1],
            self.counts.size)

    @property
    def edges(self):
        return self._edges

    @property
    def counts(self):
        return self._counts

    @property
    def midpoints(self):
        return self._midpoints

    @property
    def bin_widths(self):
        return self._bin_widths

    @property
    def normalization_constant(self):
        return self._normalization_constant

    @property
    def normalized_counts(self):
        return self._normalized_counts

    @property
    def cumulative_counts(self):
        return self._cumulative_counts

    @property
    def normalized_cumulative_counts(self):
        return self._normalized_cumulative_counts

    @property
    def probability_density(self):
        return self._probability_density

    @property
    def bin_threshold(self):
        return self._bin_threshold

    @property
    def merge_condition(self):
        return self._merge_condition

    @property
    def mesh(self):
        return self._mesh

    @staticmethod
    def get_bin_widths(edges):
        return np.diff(edges)

    @staticmethod
    def from_midpoint_to_edges(midpoints, bin_widths):
        # left = midpoints[0] - 0.5 * bin_widths[0]
        # non_left_edges = midpoints + 0.5 * bin_widths
        # edges = np.concatenate([left, non_left_edges], axis=0)
        # return edges
        edges = [midpoints[0] - 0.5 * bin_widths[0]]
        for midpoint, width in zip(midpoints, bin_widths):
            current_edge = midpoint + 0.5 * width
            edges.append(current_edge)
        return np.array(edges)

    @staticmethod
    def get_normalized_counts(counts):
        return counts / np.sum(counts)

    @staticmethod
    def get_cumulative_counts(counts):
        return np.cumsum(counts)

    def get_normalized_cumulative_counts(self, counts):
        normalized_counts = self.get_normalized_counts(counts)
        return self.get_cumulative_counts(normalized_counts)

    def get_normalization_constant(self, counts):
        return np.sum(self.bin_widths * counts)

    def get_probability_density(self, counts):
        normalization_constant = self.get_normalization_constant(counts)
        return counts / normalization_constant

    def update_edges(self, edges=None, nbins=None, wbin=None, midpoints=None, bin_widths=None, criteria=None, lbin=None, rbin=None):
        if edges is None:
            number_conditions = (nbins or wbin)
            container_conditions = ((midpoints is not None) or (bin_widths is not None))
            if (number_conditions) and (container_conditions):
                raise ValueError("input nbins or wbin (not both), or midpoints and bin_widths")
            if number_conditions:
                if (nbins and wbin):
                    raise ValueError("input nbins or wbin; not both")
                if criteria:
                    raise ValueError("input nbins or wbin or criteria")
                emin = np.nanmin(self.vs) if lbin is None else lbin
                emax = np.nanmax(self.vs) if rbin is None else rbin
                if nbins:
                    edges = np.linspace(emin, emax, nbins)
                else:
                    edges = np.arange(emin, emax+wbin, wbin)
                self._midpoints = self.get_midpoints(edges)
                self._bin_widths = self.get_bin_widths(edges)
            elif container_conditions:
                if criteria:
                    raise ValueError("input nbins or wbin or criteria")
                if not isinstance(midpoints, (tuple, list, np.ndarray)):
                    raise ValueError("invalid type(midpoints): {}".format(midpoints))
                if not isinstance(bin_widths, (tuple, list, np.ndarray)):
                    raise ValueError("invalid type(bin_widths): {}".format(bin_widths))
                midpoints = np.array(midpoints)
                bin_widths = np.array(bin_widths)
                if midpoints.size != bin_widths.size:
                    raise ValueError("number of midpoints={} should be the same as the number of bin_widths={}".format(midpoints.size, bin_widths.size))
                edges = self.from_midpoint_to_edges(
                    midpoints=midpoints,
                    bin_widths=bin_widths)
                self._midpoints = midpoints
                self._bin_widths = bin_widths
            else:
                if criteria is None:
                    raise ValueError("invalid criteria: {}".format(criteria))
                emin = np.nanmin(self.vs) if lbin is None else lbin
                emax = np.nanmax(self.vs) if rbin is None else rbin
                edges = np.histogram_bin_edges(
                    self.vs,
                    bins=criteria,
                    range=(emin, emax))
                self._midpoints = self.get_midpoints(edges)
                self._bin_widths = self.get_bin_widths(edges)
        elif isinstance(edges, (tuple, list, np.ndarray)):
            edges = np.array(edges)
            self._midpoints = self.get_midpoints(edges)
            self._bin_widths = self.get_bin_widths(edges)
        else:
            raise ValueError("invalid type(edges): {}".format(edges))
        self._edges = edges

    def update_extra_counts(self, counts):
        self._normalized_counts = self.get_normalized_counts(counts)
        self._cumulative_counts = self.get_cumulative_counts(counts)
        self._normalized_cumulative_counts = self.get_normalized_cumulative_counts(counts)
        self._normalization_constant = self.get_normalization_constant(counts)
        self._probability_density = self.get_probability_density(counts)

    def squeeze_interval(self, squeeze_trails=False, squeeze_leads=False, tol=0, extra_counts=False):
        if any([squeeze_trails, squeeze_leads]):
            counts = np.copy(self.counts)
            edges = np.copy(self.edges)
            count_indices = np.zeros(counts.size, dtype=bool)
            edge_indices = np.zeros(edges.size, dtype=bool)
            condition = (counts > tol)
            loc = np.where(condition)[0]
            try:
                if all([squeeze_trails, squeeze_leads]):
                    i, j = loc[0], loc[-1]
                else:
                    if squeeze_trails:
                        i, j = 0, loc[-1] # loc[-1], counts.size-1
                    else:
                        i, j = loc[0], counts.size-1
                if i == j:
                    ## counts of one bin
                    count_indices[i] = True
                    ## left/right edges of one bin
                    edge_indices[i] = True
                    edge_indices[i+1] = True
                else:
                    i_to_j = np.arange(i, j+1).astype(int)
                    count_indices[i_to_j] = True
                    edge_indices[i_to_j] = True
                    edge_indices[j+1] = True
                counts = np.copy(counts[count_indices])
                edges = np.copy(edges[edge_indices])
                self.update_edges(
                    edges=edges)
                self.update_counts(
                    counts=counts,
                    extra_counts=extra_counts)
            except IndexError:
                if extra_counts:
                    self.update_extra_counts(
                        counts=counts)
        else:
            if extra_counts:
                self.update_extra_counts(
                    counts=counts)

    def apply_threshold_to_bin_frequency(self, threshold=5, merge_condition='less than', extra_counts=False):
        if (self.edges is None) or (self.counts is None):
            raise ValueError("edges and counts must both be initialized before applying a bin-threshold")
        if merge_condition not in ('less than', 'less than or equal'):
            raise ValueError("invalid merge_condition: {}".format(merge_condition))
        operand = self.comparisons[merge_condition]
        if not isinstance(threshold, (int, float)):
            raise ValueError("invalid type(threshold): {}".format(type(threshold)))
        if threshold < 1:
            raise ValueError("threshold should be at least one")
        largest_count = np.max(self.counts)
        if operand(largest_count, threshold):
            raise ValueError("largest bin-count={} is {} than threshold={}".format(largest_count, merge_condition, threshold))
        ## apply threshold (assumes only one mode)
        og_edges, og_counts = np.copy(self.edges), np.copy(self.counts)
        iloc = np.where(og_counts > 0)[0]
        if np.any(operand(og_counts, threshold)):
            xpeak = np.argmax(og_counts)
            ## merge left->right towards peak (inclusive)
            i = iloc[0]
            left_counts = []
            left_edges = [og_edges[i]]
            while i <= xpeak:
                cc, ee = og_counts[i], og_edges[i+1]
                while operand(cc, threshold):
                    i += 1
                    cc += og_counts[i]
                    ee = og_edges[i+1]
                left_counts.append(cc)
                left_edges.append(ee)
                i += 1
            ## merge right->left towards peak (exclusive)
            i = iloc[-1]
            right_edges = []
            right_counts = []
            while i > xpeak:
                cc, ee = og_counts[i], og_edges[i+1]
                while operand(cc, threshold):
                    i -= 1
                    cc += og_counts[i]
                    ee = og_edges[i+1]
                right_counts.append(cc)
                right_edges.append(ee)
                i -= 1
            ## update frequency
            re_counts = np.array(left_counts + right_counts[::-1])
            re_edges = np.array(left_edges + right_edges[::-1])
            re_edges[-1] = og_edges[iloc[-1]+1]
            if re_edges.size != re_counts.size +1:
                raise ValueError("something went wrong; check algorithm")
            self.update_edges(
                edges=re_edges)
            self.update_counts(
                counts=re_counts,
                extra_counts=extra_counts)
            self._bin_threshold = threshold
            self._merge_condition = merge_condition
            if extra_counts:
                self.update_extra_counts(
                    counts=re_counts)
        else:
            if extra_counts:
                self.update_extra_counts(
                    counts=re_counts)

    def update_counts(self, counts=None, threshold=None, merge_condition='less than', squeeze_trails=False, squeeze_leads=False, tol=0, extra_counts=False, verify_counts=False):
        if counts is None:
            if self.hbias == 'left':
                counts, _ = np.histogram(self.vs, self.edges)
            else:
                counts = np.zeros(len(self.edges) - 1, dtype=int)
                for idx, val in zip(*np.unique(np.searchsorted(self.edges, self.vs, side='left'), return_counts=True)):
                    counts[idx - 1] = val
        else:
            if isinstance(counts, (tuple, list, np.ndarray)):
                counts = np.array(counts)
            if self.edges.size - counts.size != 1:
                raise ValueError("{} edges are not compatible with {} counts".format(self.edges.size, counts.size))
        if verify_counts:
            if not np.any(counts > 0):
                raise ValueError("all counts are zero")
        self._counts = counts
        if extra_counts:
            self.update_extra_counts(counts)
        if threshold is not None:
            self.apply_threshold_to_bin_frequency(
                threshold=threshold,
                merge_condition=merge_condition,
                extra_counts=extra_counts)
        if any([squeeze_trails, squeeze_leads]):
            self.squeeze_interval(
                squeeze_trails=squeeze_trails,
                squeeze_leads=squeeze_leads,
                tol=tol,
                extra_counts=extra_counts)

class TemporalHistogram(Histogram):

    def __init__(self, data, hbias='left'):
        self.data = data
        super().__init__(
            vs=np.copy(data['datetime']),
            hbias=hbias)
        self.verify_id = 'Temporal Histogram'
        self._time_step = None
        self._time_scale = None
        self._step_size = None
        self._duration = None

    @property
    def time_step(self):
        return self._time_step

    @property
    def time_scale(self):
        return self._time_scale

    @property
    def step_size(self):
        return self._step_size

    @property
    def duration(self):
        return self._duration

    @staticmethod
    def get_midpoints(edges):
        result = []
        for right, left in zip(edges[1:], edges[:-1]):
            midpoint = left + 0.5 * (right - left)
            result.append(midpoint)
        return np.array(result)

    @staticmethod
    def from_midpoint_to_edges(midpoints, bin_widths, time_step):
        kw = {'{}s'.format(time_step) : 0.5 * bin_widths[0]}
        edges = [midpoints[0] - relativedelta(**kw)]
        for midpoint, width in zip(midpoints, bin_widths):
            kw = {'{}s'.format(time_step) : 0.5 * width}
            current_edge = midpoint + relativedelta(**kw)
            edges.append(current_edge)
        return np.array(edges)

    def get_bin_widths(self, edges, time_step='second'):
        width = []
        for right, left in zip(edges[1:], edges[:-1]):
            delta = (right - left).total_seconds()
            width.append(delta)
        width = np.array(width)
        if time_step != 'second':
            if time_step in ('second', 'minute', 'hour', 'day', 'week'):
                width /= self.relative_time_scales[time_step]
            elif time_step in ('month', 'year', 'solar cycle'):
                pass
            else:
                raise ValueError("invalid time_step: {}".format(time_step))
        return width

    def get_absolute_edge_locations(self, wbin, time_step):
        if wbin < 1:
            raise ValueError("wbin (step_size) should be at least one for 'absolute' time_scale")
        if time_step in ('solar cycle', 'year'):
            condition = (np.diff(self.data[time_step]) >= wbin)
        elif time_step in ('month', 'day', 'hour', 'minute', 'second'):
            if wbin > 1:
                raise ValueError("not yet implemented")
            dmonths = np.diff(self.data['month'])
            dyears = np.diff(self.data['year'])
            if time_step == 'month':
                condition = ((dyears > 0) | (np.abs(dmonths) >= wbin))
            elif time_step == 'day':
                ddays = np.diff(self.data['day'])
                condition = ((dyears > 0) | (np.abs(dmonths) > 0) | (np.abs(ddays) >= wbin))
            elif time_step == 'hour':
                ddays = np.diff(self.data['day'])
                dhours = np.diff(self.data['hour'])
                condition = ((dyears > 0) | (np.abs(dmonths) > 0) | (np.abs(ddays) > 0) | (np.abs(dhours) >= wbin))
            elif time_step == 'minute':
                ddays = np.diff(self.data['day'])
                dhours = np.diff(self.data['hour'])
                dminutes = np.diff(self.data['minute'])
                condition = ((dyears > 0) | (np.abs(dmonths) > 0) | (np.abs(ddays) > 0) | (np.abs(dhours) > 0) | (np.abs(dminutes) >= wbin))
            else: ## 'second'
                ddays = np.diff(self.data['day'])
                dhours = np.diff(self.data['hour'])
                dminutes = np.diff(self.data['minute'])
                dseconds = np.diff(self.data['second'])
                condition = ((dyears > 0) | (np.abs(dmonths) > 0) | (np.abs(ddays) > 0) | (np.abs(dhours) > 0) | (np.abs(dminutes) > 0) | (np.abs(dseconds) >= wbin))
        else:
            raise ValueError("invalid time_step for absolute edges: {}".format(time_step))
        indices = np.where(condition)[0] + 1
        if len(indices) <= 1:
            raise ValueError("not enough indices for {} datetime edges".format(len(indices)))
        return indices

    def get_absolute_edges(self, wbin, time_step):
        indices = self.get_absolute_edge_locations(
            wbin=wbin,
            time_step=time_step)
        edges = np.copy(self.data['datetime'][indices])
        return edges

    def get_relative_edges(self, emin, emax, wbin, time_step):
        edges = []
        if time_step == 'solar cycle':
            cycle_bounds = []
            deltas = []
            for i, cycle_num in enumerate(np.unique(self.data['solar cycle'])):
                full_bounds = np.copy(self.solar_cycles[cycle_num]['full'])
                if i == 0:
                    cycle_bounds.append(full_bounds[0])
                cycle_bounds.append(full_bounds[1])
                deltas.append((full_bounds[1] - full_bounds[0]).total_seconds())
            if wbin == 1:
                edges.extend(cycle_bounds)
            elif wbin < 1:
                raise ValueError("not yet implemented")
            elif wbin > 1:
                raise ValueError("not yet implemented")
        else:
            current_edge = emin
            kws = {'{}s'.format(time_step) : wbin}
            step = relativedelta(**kws)
            edges = [emin]
            while current_edge < emax:
                current_edge += step
                edges.append(current_edge)
            if edges[-1] > emax:
                del edges[-1]
                if edges[-1] < emax:
                    edges.append(emax)
        return np.array(edges)

    def update_edges(self, edges=None, nbins=None, wbin=None, midpoints=None, bin_widths=None, lbin=None, rbin=None, time_step='second', time_scale='relative'):
        if edges is None:
            number_conditions = (nbins or wbin)
            container_conditions = ((midpoints is not None) or (bin_widths is not None))
            if (number_conditions) and (container_conditions):
                raise ValueError("input nbins or wbin (not both), or midpoints and bin_widths")
            if (number_conditions or container_conditions):
                if number_conditions:
                    if (nbins and wbin):
                        raise ValueError("input nbins or wbin; not both")
                    emin = np.nanmin(self.vs) if lbin is None else lbin
                    emax = np.nanmax(self.vs) if rbin is None else rbin
                    if nbins is None:
                        nbins_flag = False
                        if time_scale == 'relative':
                            edges = self.get_relative_edges(
                                emin=emin,
                                emax=emax,
                                wbin=wbin,
                                time_step=time_step)
                        elif time_scale == 'absolute':
                            edges = self.get_absolute_edges(
                                wbin=wbin,
                                time_step=time_step)
                        else:
                            raise ValueError("invalid time_scale: {}".format(time_scale))
                        self._midpoints = self.get_midpoints(edges)
                        self._bin_widths = self.get_bin_widths(edges, time_step=time_step)
                        self._step_size = wbin
                    else:
                        nbins_flag = True
                        duration = (emax - emin).total_seconds()
                        wbin = duration / nbins
                else: # elif container_conditions:
                    nbins_flag = False
                    if not isinstance(midpoints, (tuple, list, np.ndarray)):
                        raise ValueError("invalid type(midpoints): {}".format(midpoints))
                    if not isinstance(bin_widths, (tuple, list, np.ndarray)):
                        raise ValueError("invalid type(bin_widths): {}".format(bin_widths))
                    midpoints = np.array(midpoints)
                    bin_widths = np.array(bin_widths)
                    if midpoints.size != bin_widths.size:
                        raise ValueError("number of midpoints={} should be the same as the number of bin_widths={}".format(midpoints.size, bin_widths.size))
                    edges = self.from_midpoint_to_edges(
                        midpoints=midpoints,
                        bin_widths=bin_widths,
                        time_step=time_step)
                    self._midpoints = midpoints
                    self._bin_widths = bin_widths
            else:
                raise ValueError("invalid inputs")
        elif isinstance(edges, (tuple, list, np.ndarray)):
            nbins_flag = False
            edges = np.array(edges)
            self._midpoints = self.get_midpoints(edges)
            self._bin_widths = self.get_bin_widths(edges)
        else:
            raise ValueError("invalid type(edges): {}".format(edges))
        if nbins_flag:
            try:
                _wbin = wbin / self.relative_time_scales[time_step]
                self.update_edges(
                    edges=None,
                    nbins=None,
                    wbin=_wbin,
                    midpoints=None,
                    bin_widths=None,
                    lbin=lbin,
                    rbin=rbin,
                    time_step=time_step,
                    time_scale=time_scale)
            except:
                self.update_edges(
                    edges=None,
                    nbins=None,
                    wbin=wbin,
                    midpoints=None,
                    bin_widths=None,
                    lbin=lbin,
                    rbin=rbin,
                    time_step='second',
                    time_scale=time_scale)
        else:
            self._edges = edges
            self._time_step = time_step
            self._time_scale = time_scale
            self._duration = None
        # ## FIX ME
        # if time_step == 'solar cycle':
        #     duration = None
        # else:
        #     duration = None
        #     # kw = {'{}s'.format(time_step) : edges[-1] - edges[0]}
        #     # duration = relativedelta(**kw)\
        # ## FIX ME

    # def update_edges(self, edges=None, nbins=None, wbin=None, midpoints=None, bin_widths=None, lbin=None, rbin=None, time_step='second', time_scale='relative'):
    #     """
    #
    #     """
    #     if edges is None:
    #         number_conditions = (nbins or wbin)
    #         container_conditions = ((midpoints is not None) or (bin_widths is not None))
    #         if (number_conditions) and (container_conditions):
    #             raise ValueError("input nbins or wbin (not both), or midpoints and bin_widths")
    #         if (number_conditions or container_conditions):
    #             if number_conditions:
    #                 if (nbins and wbin):
    #                     raise ValueError("input nbins or wbin; not both")
    #                 emin = np.nanmin(self.vs) if lbin is None else lbin
    #                 emax = np.nanmax(self.vs) if rbin is None else rbin
    #                 if nbins is not None:
    #                     duration = (emax - emin).total_seconds()
    #                     wbin = duration / nbins
    #                     try:
    #                         _wbin = wbin / self.relative_time_scales[time_step]
    #                         self.update_edges(
    #                             edges=None,
    #                             nbins=None,
    #                             wbin=_wbin,
    #                             midpoints=None,
    #                             bin_widths=None,
    #                             lbin=lbin,
    #                             rbin=rbin,
    #                             time_step=time_step,
    #                             time_scale=time_scale)
    #                     except:
    #                         self.update_edges(
    #                             edges=None,
    #                             nbins=None,
    #                             wbin=wbin,
    #                             midpoints=None,
    #                             bin_widths=None,
    #                             lbin=lbin,
    #                             rbin=rbin,
    #                             time_step='second',
    #                             time_scale=time_scale)
    #                 else:
    #                     if time_scale == 'relative':
    #                         edges = self.get_relative_edges(
    #                             emin=emin,
    #                             emax=emax,
    #                             wbin=wbin,
    #                             time_step=time_step)
    #                     elif time_scale == 'absolute':
    #                         edges = self.get_absolute_edges(
    #                             wbin=wbin,
    #                             time_step=time_step)
    #                     else:
    #                         raise ValueError("invalid time_scale: {}".format(time_scale))
    #                     self._midpoints = self.get_midpoints(edges)
    #                     self._bin_widths = self.get_bin_widths(edges, time_step=time_step)
    #                     self._step_size = wbin
    #             else: # elif container_conditions:
    #                 if not isinstance(midpoints, (tuple, list, np.ndarray)):
    #                     raise ValueError("invalid type(midpoints): {}".format(midpoints))
    #                 if not isinstance(bin_widths, (tuple, list, np.ndarray)):
    #                     raise ValueError("invalid type(bin_widths): {}".format(bin_widths))
    #                 midpoints = np.array(midpoints)
    #                 bin_widths = np.array(bin_widths)
    #                 if midpoints.size != bin_widths.size:
    #                     raise ValueError("number of midpoints={} should be the same as the number of bin_widths={}".format(midpoints.size, bin_widths.size))
    #                 edges = self.from_midpoint_to_edges(
    #                     midpoints=midpoints,
    #                     bin_widths=bin_widths,
    #                     time_step=time_step)
    #                 self._midpoints = midpoints
    #                 self._bin_widths = bin_widths
    #         else:
    #             raise ValueError("invalid inputs")
    #     elif isinstance(edges, (tuple, list, np.ndarray)):
    #         edges = np.array(edges)
    #         self._midpoints = self.get_midpoints(edges)
    #         self._bin_widths = self.get_bin_widths(edges)
    #     else:
    #         raise ValueError("invalid type(edges): {}".format(edges))
    #
    #     ## FIX ME
    #     if time_step == 'solar cycle':
    #         duration = None
    #     else:
    #         duration = None
    #         # kw = {'{}s'.format(time_step) : edges[-1] - edges[0]}
    #         # duration = relativedelta(**kw)\
    #     ## FIX ME
    #
    #     self._edges = edges
    #     self._time_step = time_step
    #     self._time_scale = time_scale
    #     self._duration = duration

















##
