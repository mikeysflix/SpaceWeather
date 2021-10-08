from optimization_methods import *

class InterExceedanceDistribution(ConditionMapping):

    def __init__(self, data, extreme_parameter, extreme_condition):
        super().__init__()
        self.event_searcher = EventSearcher(data)
        self.data = data
        self.extreme_parameter = extreme_parameter
        self.extreme_condition = extreme_condition
        self._extreme_values = OrderedDict()

    @property
    def extreme_values(self):
        return self._extreme_values

    def __repr__(self):
        return 'InterExceedanceDistribution(%r, %r, %r)' % (self.data, self.extreme_parameter, self.extreme_condition)

    def __str__(self):
        extreme_values = list(self.extreme_values.keys())
        lowest_extreme_value, highest_extreme_value = extreme_values[0], extreme_values[-1]
        nevents_all = np.sum(self.data['is event']) # self.data[self.extreme_parameter].size
        s = '\n ** Inter-Exceedance Distribution **\n %s extreme-values: %s \n' % (
            len(extreme_values),
            extreme_values)
        for extreme_value in (lowest_extreme_value, highest_extreme_value):
            inter_exceedance_result = self.extreme_values[extreme_value]
            nevents_ext = inter_exceedance_result['nevents']
            histograms = inter_exceedance_result['histograms']
            inter_exceedance_histogram = histograms['inter-exceedance']
            if 'inverse-transform sample' in list(histograms.keys()):
                inverse_transform_histogram = histograms['inverse-transform sample']
                s += '\n results via extreme_value=%s \n\n .. number of events = %s \n .. number of extreme events (number of inter-exceedance times +1) = %s \n .. mean(inter-exceedance times) = %s \n .. mean(inverse-transform sample) = %s \n .. standard_deviation(inter-exceedance times) = %s \n .. standard_deviation(inverse-transform sample) = %s \n' % (
                    extreme_value,
                    nevents_all,
                    nevents_ext,
                    inter_exceedance_histogram.mean,
                    inverse_transform_histogram.mean,
                    inter_exceedance_histogram.standard_deviation,
                    inverse_transform_histogram.standard_deviation)
            else:
                s += '\n results via extreme_value=%s \n\n .. number of events = %s \n .. number of extreme events (number of inter-exceedance times +1) = %s \n .. mean(inter-exceedance times) = %s \n .. standard_deviation(inter-exceedance times) = %s \n' % (
                    extreme_value,
                    nevents_all,
                    nevents_ext,
                    inter_exceedance_histogram.mean,
                    inter_exceedance_histogram.standard_deviation)
        return s

    def update_extreme_value(self, extreme_value, include_inverse_transform_sample=False, edges=None, nbins=None, wbin=None, midpoints=None, bin_widths=None, criteria=None, lbin=None, rbin=None, hbias='left', squeeze_leads=False, squeeze_trails=False, tol=0, bias=False, fisher=False, ddof=0, nan_policy='propagate'):
        extreme_events, _ = self.event_searcher.search_events(
            parameters=self.extreme_parameter,
            conditions=self.extreme_condition,
            values=extreme_value)
        inter_exceedance_times = np.diff(extreme_events['elapsed'])
        inter_exceedance_histogram = Histogram(
            vs=inter_exceedance_times,
            hbias=hbias)
        inter_exceedance_histogram.update_statistics(
            bias=bias,
            fisher=fisher,
            ddof=ddof,
            nan_policy=nan_policy)
        inter_exceedance_histogram.update_edges(
            nbins=nbins,
            wbin=wbin,
            midpoints=midpoints,
            bin_widths=bin_widths,
            criteria=criteria,
            lbin=lbin,
            rbin=rbin)
        inter_exceedance_histogram.update_counts(
            squeeze_leads=squeeze_leads,
            squeeze_trails=squeeze_trails,
            tol=tol)
        result = {
            'extreme events' : extreme_events,
            'inter-exceedance times' : inter_exceedance_times,
            'nevents' : inter_exceedance_times.size + 1,
            'histograms' : dict()}
        result['histograms']['inter-exceedance'] = inter_exceedance_histogram
        if include_inverse_transform_sample:
            x = np.random.uniform(size=inter_exceedance_times.size)
            y = np.log(1/x) * np.mean(inter_exceedance_times) / np.mean(np.log(1/x))
            inverse_transform_histogram = Histogram(
                vs=y,
                hbias=inter_exceedance_histogram.hbias)
            inverse_transform_histogram.update_statistics(
                bias=bias,
                fisher=fisher,
                ddof=ddof,
                nan_policy=nan_policy)
            inverse_transform_histogram.update_edges(
                edges=np.copy(inter_exceedance_histogram.edges))
            inverse_transform_histogram.update_counts(
                squeeze_leads=squeeze_leads,
                squeeze_trails=squeeze_trails,
                tol=tol)
            result['histograms']['inverse-transform sample'] = inverse_transform_histogram
        self._extreme_values[extreme_value] = result

    def __call__(self, extreme_values, include_inverse_transform_sample=False, edges=None, nbins=None, wbin=None, midpoints=None, bin_widths=None, criteria=None, lbin=None, rbin=None, hbias='left', squeeze_leads=False, squeeze_trails=False, tol=0, bias=False, fisher=False, ddof=0, nan_policy='propagate'):
        if not isinstance(extreme_values, (tuple, list, np.ndarray)):
            extreme_values = [extreme_values]
        for extreme_value in extreme_values:
            self.update_extreme_value(
                extreme_value=extreme_value,
                include_inverse_transform_sample=include_inverse_transform_sample,
                edges=edges,
                nbins=nbins,
                wbin=wbin,
                midpoints=midpoints,
                bin_widths=bin_widths,
                criteria=criteria,
                lbin=lbin,
                rbin=rbin,
                hbias=hbias,
                squeeze_leads=squeeze_leads,
                squeeze_trails=squeeze_trails,
                tol=tol,
                bias=bias,
                fisher=fisher,
                ddof=ddof,
                nan_policy=nan_policy)










##
