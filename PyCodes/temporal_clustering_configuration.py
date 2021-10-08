from inter_exceedance_distribution import *

class TemporalClusteringConfiguration(InterExceedanceDistribution):

    def __init__(self, data, extreme_parameter, extreme_condition):
        super().__init__(
            data=data,
            extreme_parameter=extreme_parameter,
            extreme_condition=extreme_condition)
        self._bias_ids = None
        self._cluster_ids = None

    @property
    def bias_ids(self):
        return self._bias_ids

    @property
    def cluster_ids(self):
        return self._cluster_ids

    def __str__(self):
        extreme_values = list(self.extreme_values.keys())
        lowest_extreme_value, highest_extreme_value = extreme_values[0], extreme_values[-1]
        s = '\n ** Temporal Clustering **\n %s extreme-values: %s \n' % (
            len(extreme_values),
            extreme_values)
        for extreme_value in (lowest_extreme_value, highest_extreme_value):
            s += '\n results via extreme_value=%s \n' % (
                extreme_value)
            temporal_clustering_result = self.extreme_values[extreme_value]['temporal clustering']
            for bias_id, data in temporal_clustering_result.items():
                all_cluster_searcher = data['all clusters']['cluster searcher']
                lone_cluster_searcher = data['lone clusters']['cluster searcher']
                non_lone_cluster_searcher = data['non-lone clusters']['cluster searcher']
                try:
                    s += '\n .. cluster-bias = %s \n .. theta = %s \n .. time_threshold = %s \n .. number of extreme events from lone clusters = %s \n .. number of extreme events from non-lone clusters = %s \n .. number of extreme events from all clusters = %s \n .. number of lone clusters = %s \n .. number of non-lone clusters = %s \n .. number of all clusters = %s \n' % (
                        bias_id,
                        data['moment estimator'],
                        data['time threshold'],
                        lone_cluster_searcher.events['is event'].size,
                        non_lone_cluster_searcher.events['is event'].size,
                        all_cluster_searcher.events['is event'].size,
                        len(lone_cluster_searcher.clusters['is event']),
                        len(non_lone_cluster_searcher.clusters['is event']),
                        len(all_cluster_searcher.clusters['is event']))
                except:
                    pass
                s += self.get_cluster_information(
                    extreme_value=extreme_value,
                    bias_ids=bias_id,
                    cluster_ids=None,
                    n=3)
        return s

    @staticmethod
    def get_intra_times(clusters):
        intra_times = [np.diff(cluster) for cluster in clusters['elapsed']]
        return np.array(intra_times)

    @staticmethod
    def get_intra_durations(clusters):
        intra_durations = np.array([cluster[-1] - cluster[0] for cluster in clusters['elapsed']])
        return np.array(intra_durations)

    @staticmethod
    def get_inter_durations(clusters):
        inter_durations = np.array([
            curr_cluster[0] - prev_cluster[-1] for prev_cluster, curr_cluster in zip(clusters['elapsed'][:-1], clusters['elapsed'][1:])])
        return np.array(inter_durations)

    def get_cluster_information(self, extreme_value, bias_ids=None, cluster_ids=None, temporal_key='datetime', n=10):
        if n < 1:
            raise ValueError("invalid n: {}".format(n))
        if not isinstance(n, int):
            raise ValueError("invalid type(n): {}".format(type(n)))
        if extreme_value not in list(self.extreme_values.keys()):
            raise ValueError("invalid extreme_value: ".format(extreme_value))
        temporal_clustering_result = self.extreme_values[extreme_value]['temporal clustering']
        if bias_ids is None:
            bias_ids = np.copy(self.bias_ids)
        elif not isinstance(bias_ids, (tuple, list, np.ndarray)):
            bias_ids = [bias_ids]
        if cluster_ids is None:
            cluster_ids = np.copy(self.cluster_ids)
        elif not isinstance(cluster_ids, (tuple, list, np.ndarray)):
            cluster_ids = [cluster_ids]
        s = ''
        for bias_id in bias_ids:
            if bias_id not in list(temporal_clustering_result.keys()):
                raise ValueError("invalid bias_id in bias_ids: {}".format(bias_id))
            bias_result = temporal_clustering_result[bias_id]
            for cluster_id in cluster_ids:
                if cluster_id not in list(bias_result.keys()):
                    raise ValueError("invalid cluster_id in cluster_ids: {}".format(cluster_id))
                cluster_searcher = bias_result[cluster_id]['cluster searcher']
                number_of_clusters = len(cluster_searcher.clusters['is event'])
                s += "\n\n{} First and Last Clusters ({} Bias; {})\n\n".format(n, bias_id.title(), cluster_id.title())
                for i, (extreme_parameter_cluster_values, temporal_cluster_values) in enumerate(zip(cluster_searcher.clusters[self.extreme_parameter], cluster_searcher.clusters[temporal_key])):
                    if (i < n) or (i >= number_of_clusters - n):
                        s += "\n** Cluster #{}/{}\n\n .. DATE and TIME:\n{}\n\n .. {}:\n{}\n".format(
                            i + 1,
                            number_of_clusters,
                            temporal_cluster_values,
                            self.extreme_parameter.upper(),
                            extreme_parameter_cluster_values)
        return s

    def update_bias_ids(self, apply_first_order_bias=False, apply_threshold_bias=False, baseline_theta=None):
        bias_ids = []
        if apply_first_order_bias:
            bias_ids.append('first-order')
        if apply_threshold_bias:
            bias_ids.append('threshold')
        if baseline_theta is not None:
            bias_ids.append('baseline')
        if len(bias_ids) == 0:
            raise ValueError("at least one of the following inputs must be True: apply_first_order_bias, apply_threshold_bias; AND/OR, the following input should not be None: baseline_theta")
        self._bias_ids = tuple(bias_ids)

    def update_cluster_ids(self, include_all_clusters=False, include_lone_clusters=False, include_non_lone_clusters=False):
        cluster_ids = []
        if include_all_clusters:
            cluster_ids.append('all clusters')
        if include_lone_clusters:
            cluster_ids.append('lone clusters')
        if include_non_lone_clusters:
            cluster_ids.append('non-lone clusters')
        if len(cluster_ids) == 0:
            raise ValueError("at least one of the following inputs must be True: include_all_clusters, include_lone_clusters, include_non_lone_clusters")
        self._cluster_ids = tuple(cluster_ids)

    def update_moment_estimators(self, extreme_value, baseline_theta):
        result = self.extreme_values[extreme_value]
        inter_exceedance_times = result['inter-exceedance times']
        sq_sum = np.square(np.sum(inter_exceedance_times))
        sum_sq = np.sum(np.square(inter_exceedance_times))
        rev_one = inter_exceedance_times - 1
        rev_two = inter_exceedance_times - 2
        thetas = dict()
        if 'first-order' in self.bias_ids:
            result['temporal clustering']['first-order'] = {
                'moment estimator' : 2 * np.square(np.sum(rev_one)) / (inter_exceedance_times.size * np.sum(rev_one * rev_two))}
        if 'threshold' in self.bias_ids:
            result['temporal clustering']['threshold'] = {
                'moment estimator' : 2 * sq_sum / (inter_exceedance_times.size * sum_sq)}
        if 'baseline' in self.bias_ids:
            if (0 >= baseline_theta) or (1 <= baseline_theta):
                raise ValueError("invalid baseline_theta: {}; 0 < baseline_theta < 1".format(baseline_theta))
            result['temporal clustering']['baseline'] = {
                'moment estimator' : baseline_theta}
        self._extreme_values[extreme_value] = result

    def update_time_thresholds(self, extreme_value, time_threshold=None):
        if time_threshold is None:
            result = self.extreme_values[extreme_value]
            inter_exceedance_times, nevents = result['inter-exceedance times'], result['nevents']
            increasing_times = np.sort(inter_exceedance_times)
            for bias_id, data in result['temporal clustering'].items():
                i = int(data['moment estimator'] * nevents)
                try:
                    declustering_time = increasing_times[i]
                except:
                    declustering_time = np.nan
                result['temporal clustering'][bias_id]['time threshold'] = declustering_time
            self._extreme_values[extreme_value] = result
        else:
            for bias_id in list(self.extreme_values[extreme_value]['temporal clustering'].keys()):
                self.extreme_values[extreme_value]['temporal clustering'][bias_id]['time threshold'] = time_threshold

    def update_clusters(self, extreme_value, override_error=False):
        result = self.extreme_values[extreme_value]
        inter_exceedance_times, extreme_events = result['inter-exceedance times'], result['extreme events']
        cluster_size_operator = self.comparisons['less than']
        if override_error:
            for bias_id, data in result['temporal clustering'].items():
                declustering_time = data['time threshold']
                if np.isnan(declustering_time):
                    for cluster_id in self.cluster_ids:
                        result['temporal clustering'][bias_id][cluster_id] = {
                            'cluster searcher' : ClusterSearcher({
                                key : np.array([]) for key in list(self.data.keys())})}
                else:
                    loc = cluster_size_operator(declustering_time, inter_exceedance_times)
                    indices = np.where(loc)[0] + 1
                    all_clusters = {key : np.array(np.split(value, indices))
                        for key, value in extreme_events.items()}
                    all_searcher = ClusterSearcher(all_clusters)
                    if 'all clusters' in self.cluster_ids:
                        result['temporal clustering'][bias_id]['all clusters'] = {
                            'cluster searcher' : all_searcher}
                    if 'lone clusters' in self.cluster_ids:
                        try:
                            lone_extreme_events, _ = all_searcher.search_clusters(
                                **self.lone_cluster_kwargs)
                            result['temporal clustering'][bias_id]['lone clusters'] = {
                                'cluster searcher' : ClusterSearcher(lone_extreme_events)}
                        except:
                            result['temporal clustering'][bias_id]['lone clusters'] = {
                                'cluster searcher' : ClusterSearcher(
                                    {key : np.array([]) for key in list(self.data.keys())})}
                    if 'non-lone clusters' in self.cluster_ids:
                        try:
                            non_lone_extreme_events, _ = all_searcher.search_clusters(
                                **self.non_lone_cluster_kwargs)
                            result['temporal clustering'][bias_id]['non-lone clusters'] = {
                                'cluster searcher' : ClusterSearcher(non_lone_extreme_events)}
                        except:
                            result['temporal clustering'][bias_id]['non-lone clusters'] = {
                                'cluster searcher' : ClusterSearcher(
                                    {key : np.array([]) for key in list(self.data.keys())})}
        else:
            for bias_id, data in result['temporal clustering'].items():
                declustering_time = data['time threshold']
                if np.isnan(declustering_time):
                    raise ValueError("cannot initialize clusters using NaN time-threshold")
                loc = cluster_size_operator(declustering_time, inter_exceedance_times)
                indices = np.where(loc)[0] + 1
                all_clusters = {key : np.array(np.split(value, indices))
                    for key, value in extreme_events.items()}
                all_searcher = ClusterSearcher(all_clusters)
                if 'all clusters' in self.cluster_ids:
                    result['temporal clustering'][bias_id]['all clusters'] = {
                        'cluster searcher' : all_searcher}
                if 'lone clusters' in self.cluster_ids:
                    lone_extreme_events, _ = all_searcher.search_clusters(
                        parameters='cluster size',
                        conditions='equal',
                        values=1)
                    result['temporal clustering'][bias_id]['lone clusters'] = {
                        'cluster searcher' : ClusterSearcher(lone_extreme_events)}
                if 'non-lone clusters' in self.cluster_ids:
                    non_lone_extreme_events, _ = all_searcher.search_clusters(
                        parameters='cluster size',
                        conditions='greater than',
                        values=1)
                    result['temporal clustering'][bias_id]['non-lone clusters'] = {
                        'cluster searcher' : ClusterSearcher(non_lone_extreme_events)}
        self._extreme_values[extreme_value] = result

    def update_cluster_times_and_durations(self, extreme_value):
        result = self.extreme_values[extreme_value]
        for bias_id, data in result['temporal clustering'].items():
            for cluster_id in self.cluster_ids:
                cluster_searcher = data[cluster_id]['cluster searcher']
                result['temporal clustering'][bias_id][cluster_id]['intra-cluster times'] = {
                    'values' : self.get_intra_times(
                        clusters=cluster_searcher.clusters)}
                result['temporal clustering'][bias_id][cluster_id]['intra-cluster durations'] = {
                    'values' : self.get_intra_durations(
                        clusters=cluster_searcher.clusters)}
                result['temporal clustering'][bias_id][cluster_id]['inter-cluster durations'] = {
                    'values' : self.get_inter_durations(
                        clusters=cluster_searcher.clusters)}
        self._extreme_values[extreme_value] = result

    def update_relative_statistics(self, extreme_value):
        result = self.extreme_values[extreme_value]
        for bias_id, data in result['temporal clustering'].items():
            for cluster_id in self.cluster_ids:
                cluster_searcher = data[cluster_id]['cluster searcher']
                initial_cluster_sizes = np.array([cluster.size
                    for cluster in cluster_searcher.clusters['elapsed']])
                cluster_sizes, size_counts = np.unique(initial_cluster_sizes, return_counts=True)
                nevents = cluster_sizes * size_counts
                rel_prob = nevents / np.sum(nevents)
                result['temporal clustering'][bias_id][cluster_id]['relative statistics'] = {
                    'cluster size' : cluster_sizes,
                    'number of clusters' : size_counts,
                    'number of events' : nevents,
                    'relative probability' : rel_prob}
        self._extreme_values[extreme_value] = result

    def update_histogram_statistics_of_cluster_times_and_durations(self, extreme_value, time_difference_id, edges=None, nbins=None, wbin=None, midpoints=None, bin_widths=None, criteria=None, lbin=None, rbin=None, squeeze_trails=False, squeeze_leads=False, tol=0, hbias='left', bias=False, fisher=False, ddof=0, only_statistics=False):
        result = self.extreme_values[extreme_value]
        for bias_id, data in result['temporal clustering'].items():
            for cluster_id in self.cluster_ids:
                time_differences = data[cluster_id][time_difference_id]['values']
                time_threshold = data['time threshold']
                if time_difference_id == 'intra-cluster times':
                    try:
                        flat_differences = np.concatenate(time_differences, axis=0)
                    except:
                        if time_differences.size != 0:
                            raise ValueError("something went wrong\n\n .. time_differences ({}):\n{}\n".format(type(time_differences), time_differences))
                        flat_differences = np.array([])
                    histogram = Histogram(
                        vs=flat_differences,
                        hbias=hbias)
                elif time_difference_id in ('intra-cluster durations', 'inter-cluster durations'):
                    histogram = Histogram(
                        vs=time_differences,
                        hbias=hbias)
                else:
                    raise ValueError("invalid time_difference_id: {}".format(time_difference_id))
                if histogram.vs.size > 0:
                    histogram.update_statistics(
                        bias=bias,
                        fisher=fisher,
                        ddof=ddof,
                        nan_policy='omit')
                    if not only_statistics:
                        histogram.update_edges(
                            edges=edges,
                            nbins=nbins,
                            wbin=wbin,
                            midpoints=midpoints,
                            bin_widths=bin_widths,
                            criteria=criteria,
                            lbin=lbin,
                            rbin=rbin)
                        histogram.update_counts(
                            squeeze_leads=squeeze_leads,
                            squeeze_trails=squeeze_trails,
                            tol=tol)
                result['temporal clustering'][bias_id][cluster_id][time_difference_id]['histogram'] = histogram
        self._extreme_values[extreme_value] = result

    def update_histogram_statistics_of_intra_cluster_times(self, extreme_value, edges=None, nbins=None, wbin=None, midpoints=None, bin_widths=None, criteria=None, lbin=None, rbin=None, squeeze_trails=False, squeeze_leads=False, tol=0, hbias='left', bias=False, fisher=False, ddof=0, only_statistics=False):
        self.update_histogram_statistics_of_cluster_times_and_durations(
            extreme_value=extreme_value,
            time_difference_id='intra-cluster times',
            edges=edges,
            nbins=nbins,
            wbin=wbin,
            midpoints=midpoints,
            bin_widths=bin_widths,
            criteria=criteria,
            lbin=lbin,
            rbin=rbin,
            squeeze_trails=squeeze_trails,
            squeeze_leads=squeeze_leads,
            tol=tol,
            hbias=hbias,
            bias=bias,
            fisher=fisher,
            ddof=ddof,
            only_statistics=only_statistics)

    def update_histogram_statistics_of_intra_cluster_durations(self, extreme_value, edges=None, nbins=None, wbin=None, midpoints=None, bin_widths=None, criteria=None, lbin=None, rbin=None, squeeze_trails=False, squeeze_leads=False, tol=0, hbias='left', bias=False, fisher=False, ddof=0, only_statistics=False):
        self.update_histogram_statistics_of_cluster_times_and_durations(
            extreme_value=extreme_value,
            time_difference_id='intra-cluster durations',
            edges=edges,
            nbins=nbins,
            wbin=wbin,
            midpoints=midpoints,
            bin_widths=bin_widths,
            criteria=criteria,
            lbin=lbin,
            rbin=rbin,
            squeeze_trails=squeeze_trails,
            squeeze_leads=squeeze_leads,
            tol=tol,
            hbias=hbias,
            bias=bias,
            fisher=fisher,
            ddof=ddof,
            only_statistics=only_statistics)

    def update_histogram_statistics_of_inter_cluster_durations(self, extreme_value, edges=None, nbins=None, wbin=None, midpoints=None, bin_widths=None, criteria=None, lbin=None, rbin=None, squeeze_trails=False, squeeze_leads=False, tol=0, hbias='left', bias=False, fisher=False, ddof=0, only_statistics=False):
        self.update_histogram_statistics_of_cluster_times_and_durations(
            extreme_value=extreme_value,
            time_difference_id='inter-cluster durations',
            edges=edges,
            nbins=nbins,
            wbin=wbin,
            midpoints=midpoints,
            bin_widths=bin_widths,
            criteria=criteria,
            lbin=lbin,
            rbin=rbin,
            squeeze_trails=squeeze_trails,
            squeeze_leads=squeeze_leads,
            tol=tol,
            hbias=hbias,
            bias=bias,
            fisher=fisher,
            ddof=ddof,
            only_statistics=only_statistics)

class TemporalClustering(TemporalClusteringConfiguration):

    def __init__(self, data, extreme_parameter, extreme_condition):
        super().__init__(
            data=data,
            extreme_parameter=extreme_parameter,
            extreme_condition=extreme_condition)

    def __repr__(self):
        return 'TemporalClustering(%r, %r, %r)' % (
            self.data,
            self.extreme_parameter,
            self.extreme_condition)

    def __call__(self, extreme_values, apply_first_order_bias=False, apply_threshold_bias=False, include_all_clusters=False, include_lone_clusters=False, include_non_lone_clusters=False, baseline_theta=None, intra_cluster_times_histogram_kwargs=None, intra_cluster_durations_histogram_kwargs=None, inter_cluster_durations_histogram_kwargs=None, time_threshold=None):
        self.update_bias_ids(
            apply_first_order_bias=apply_first_order_bias,
            apply_threshold_bias=apply_threshold_bias,
            baseline_theta=baseline_theta)
        self.update_cluster_ids(
            include_all_clusters=include_all_clusters,
            include_lone_clusters=include_lone_clusters,
            include_non_lone_clusters=include_non_lone_clusters)
        for extreme_value in extreme_values:
            extreme_events, _ = self.event_searcher.search_events(
                parameters=self.extreme_parameter,
                conditions=self.extreme_condition,
                values=extreme_value)
            inter_exceedance_times = np.diff(extreme_events['elapsed'])
            self._extreme_values[extreme_value] = {
                'extreme events' : extreme_events,
                'inter-exceedance times' : inter_exceedance_times,
                'nevents' : inter_exceedance_times.size + 1,
                'temporal clustering' : dict()}
            self.update_moment_estimators(
                extreme_value=extreme_value,
                baseline_theta=baseline_theta)
            self.update_time_thresholds(
                extreme_value=extreme_value,
                time_threshold=time_threshold)
            self.update_clusters(
                extreme_value=extreme_value,
                override_error=False)
            self.update_cluster_times_and_durations(
                extreme_value=extreme_value)
            if intra_cluster_times_histogram_kwargs is not None:
                self.update_histogram_statistics_of_intra_cluster_times(
                    extreme_value=extreme_value,
                    only_statistics=False,
                    **intra_cluster_times_histogram_kwargs)
            if intra_cluster_durations_histogram_kwargs is not None:
                self.update_histogram_statistics_of_intra_cluster_durations(
                    extreme_value=extreme_value,
                    only_statistics=False,
                    **intra_cluster_durations_histogram_kwargs)
            if inter_cluster_durations_histogram_kwargs is not None:
                self.update_histogram_statistics_of_inter_cluster_durations(
                    extreme_value=extreme_value,
                    only_statistics=False,
                    **inter_cluster_durations_histogram_kwargs)
            self.update_relative_statistics(
                extreme_value=extreme_value)

class TemporalClusteringParameterization(TemporalClusteringConfiguration):

    def __init__(self, data, extreme_parameter, extreme_condition):
        super().__init__(
            data=data,
            extreme_parameter=extreme_parameter,
            extreme_condition=extreme_condition)
        self._parameterization = dict()

    @property
    def parameterization(self):
        return self._parameterization

    def __repr__(self):
        return 'TemporalClusteringParameterization(%r, %r, %r)' % (
            self.data,
            self.extreme_parameter,
            self.extreme_condition)

    def update_parameterization(self):
        cluster_id_items = {
            'intra-cluster times' : [],
            'intra-cluster durations' : [],
            'inter-cluster durations' : [],
            'cluster searcher' : []}
        bias_id_items = {
            'moment estimator' : [],
            'time threshold' : [],
            'lone clusters' : deepcopy(cluster_id_items),
            'non-lone clusters' : deepcopy(cluster_id_items),
            'all clusters' : deepcopy(cluster_id_items)}
        bias_mapping = {
            'first-order' : deepcopy(bias_id_items),
            'threshold' : deepcopy(bias_id_items),
            'baseline' : deepcopy(bias_id_items)}
        extreme_values = []
        for extreme_value, result in self.extreme_values.items():
            extreme_values.append(extreme_value)
            for bias_id, data in result['temporal clustering'].items():
                for key in ('moment estimator', 'time threshold'):
                    bias_mapping[bias_id][key].append(data[key])
                for cluster_id in self.cluster_ids:
                    for key in list(cluster_id_items.keys()):
                        bias_mapping[bias_id][cluster_id][key].append(data[cluster_id][key])
        parameterization = {
            'extreme values' : np.array(extreme_values),
            'bias-mapping' : bias_mapping}
        self._parameterization = parameterization

    def get_parameterized_cluster_configuration_quantity(self, quantity, bias_id):
        result = dict()
        if quantity in ('moment estimator', 'time threshold'):
            arr = self.parameterization['bias-mapping'][bias_id][quantity]
        else:
            raise ValueError("invalid quantity: {}".format(quantity))
        result['values'] = np.array(arr)
        return result

    def get_parameterized_time_difference_quantity(self, quantity, bias_id, cluster_id, bias=False, fisher=False, ddof=0):
        if cluster_id is None:
            raise ValueError("cluster_id = {} is not compatible with quantity = {}".format(cluster_id, quantity))
        result = dict()
        if quantity in ('intra-cluster times', 'intra-cluster durations', 'inter-cluster durations'):
            for statistic_id in ('mean', 'median', 'maximum', 'minimum', 'standard deviation', 'standard error', 'skew', 'kurtosis'):
                arr = []
                for time_difference in self.parameterization['bias-mapping'][bias_id][cluster_id][quantity]:
                    histogram = time_difference['histogram']
                    substring = statistic_id.replace('-', '_').replace(' ', '_')
                    value = getattr(histogram, substring)
                    if value is None:
                        arr.append(np.nan)
                    else:
                        arr.append(value)
                result[statistic_id] = np.array(arr)
        else:
            raise ValueError("invalid quantity: {}".format(quantity))
        return result

    def get_parameterized_population_quantity(self, quantity, bias_id, cluster_id, bias=False, fisher=False, ddof=0):
        if cluster_id is None:
            raise ValueError("cluster_id = {} is not compatible with quantity = {}".format(cluster_id, quantity))
        if quantity in ('number of extreme events', 'number of clusters', 'cluster size'):
            result = dict()
            cluster_searchers = self.parameterization['bias-mapping'][bias_id][cluster_id]['cluster searcher']
            if quantity == 'number of extreme events':
                arr = []
                for searcher in cluster_searchers:
                    arr.append(searcher.events['is event'].size)
                result['values'] = np.array(arr)
            elif quantity == 'number of clusters':
                arr = []
                for searcher in cluster_searchers:
                    arr.append(len(searcher.clusters['is event']))
                result['values'] = np.array(arr)
            else: # quantity == 'cluster size'
                result = dict()
                for statistic_id in ('mean', 'median', 'maximum', 'minimum', 'standard deviation', 'standard error', 'skew', 'kurtosis'):
                    result[statistic_id] = []
                for searcher in cluster_searchers:
                    # vs = searcher.clusters[quantity]
                    vs = np.array([cluster.size
                        for cluster in searcher.clusters['is event']])
                    if vs.size > 0:
                        statistics_configuration = StatisticsConfiguration(
                            vs=vs)
                        statistics_configuration.update_statistics(
                            bias=bias,
                            fisher=fisher,
                            ddof=ddof,
                            nan_policy='omit')
                        for statistic_id in ('mean', 'median', 'maximum', 'minimum', 'standard deviation', 'standard error', 'skew', 'kurtosis'):
                            substring = statistic_id.replace('-', '_').replace(' ', '_')
                            value = getattr(statistics_configuration, substring)
                            result[statistic_id].append(value)
                    else:
                        for statistic_id in ('mean', 'median', 'maximum', 'minimum', 'standard deviation', 'standard error', 'skew', 'kurtosis'):
                            result[statistic_id].append(np.nan)
                result = {key : np.array(value) for key, value in result.items()}
        else:
            raise ValueError("invalid quantity: {}".format(quantity))
        return result

    def get_parameterized_data_quantity(self, quantity, bias_id, cluster_id, bias=False, fisher=False, ddof=0):
        if quantity in list(self.data.keys()):
            cluster_searchers = self.parameterization['bias-mapping'][bias_id][cluster_id]['cluster searcher']
            result = dict()
            for statistic_id in ('mean', 'median', 'maximum', 'minimum', 'standard deviation', 'standard error', 'skew', 'kurtosis'):
                result[statistic_id] = []
            for searcher in cluster_searchers:
                try:
                    vs = np.concatenate(searcher.clusters[quantity], axis=0)
                    statistics_configuration = StatisticsConfiguration(
                        vs=vs)
                    statistics_configuration.update_statistics(
                        bias=bias,
                        fisher=fisher,
                        ddof=ddof,
                        nan_policy='omit')
                    for statistic_id in ('mean', 'median', 'maximum', 'minimum', 'standard deviation', 'standard error', 'skew', 'kurtosis'):
                        substring = statistic_id.replace('-', '_').replace(' ', '_')
                        value = getattr(statistics_configuration, substring)
                        result[statistic_id].append(value)
                except:
                    for statistic_id in ('mean', 'median', 'maximum', 'minimum', 'standard deviation', 'standard error', 'skew', 'kurtosis'):
                        result[statistic_id].append(np.nan)
                    # vs = np.array([np.nan])
            result = {key : np.array(_value) for key, _value in result.items()}
        else:
            raise ValueError("invalid quantity: {}".format(quantity))
        return result

    def get_parameterized_quantity(self, quantity, bias_id, cluster_id=None, bias=False, fisher=False, ddof=0):
        if quantity in ('moment estimator', 'time threshold'):
            result = self.get_parameterized_cluster_configuration_quantity(
                quantity=quantity,
                bias_id=bias_id)
        elif quantity in ('intra-cluster times', 'intra-cluster durations', 'inter-cluster durations'):
            result = self.get_parameterized_time_difference_quantity(
                quantity=quantity,
                bias_id=bias_id,
                cluster_id=cluster_id,
                bias=bias,
                fisher=fisher,
                ddof=ddof)
        elif quantity in ('number of extreme events', 'number of clusters', 'cluster size'):
            result = self.get_parameterized_population_quantity(
                quantity=quantity,
                bias_id=bias_id,
                cluster_id=cluster_id,
                bias=bias,
                fisher=fisher,
                ddof=ddof)
        else:
            result = self.get_parameterized_data_quantity(
                quantity=quantity,
                bias_id=bias_id,
                cluster_id=cluster_id,
                bias=bias,
                fisher=fisher,
                ddof=ddof)
        return result

    def __call__(self, extreme_values, apply_first_order_bias=False, apply_threshold_bias=False, include_all_clusters=False, include_lone_clusters=False, include_non_lone_clusters=False, baseline_theta=None, intra_cluster_times_histogram_kwargs=None, intra_cluster_durations_histogram_kwargs=None, inter_cluster_durations_histogram_kwargs=None, time_threshold=None):
        self.update_bias_ids(
            apply_first_order_bias=apply_first_order_bias,
            apply_threshold_bias=apply_threshold_bias,
            baseline_theta=baseline_theta)
        self.update_cluster_ids(
            include_all_clusters=include_all_clusters,
            include_lone_clusters=include_lone_clusters,
            include_non_lone_clusters=include_non_lone_clusters)
        for extreme_value in extreme_values:
            extreme_events, _ = self.event_searcher.search_events(
                parameters=self.extreme_parameter,
                conditions=self.extreme_condition,
                values=extreme_value)
            inter_exceedance_times = np.diff(extreme_events['elapsed'])
            self._extreme_values[extreme_value] = {
                'extreme events' : extreme_events,
                'inter-exceedance times' : inter_exceedance_times,
                'nevents' : inter_exceedance_times.size + 1,
                'temporal clustering' : dict()}
            self.update_moment_estimators(
                extreme_value=extreme_value,
                baseline_theta=baseline_theta)
            self.update_time_thresholds(
                extreme_value=extreme_value,
                time_threshold=time_threshold)
            self.update_clusters(
                extreme_value=extreme_value,
                override_error=True)
            self.update_cluster_times_and_durations(
                extreme_value=extreme_value)
            ##
            ##
            self.update_histogram_statistics_of_intra_cluster_times(
                extreme_value=extreme_value,
                only_statistics=True)
            self.update_histogram_statistics_of_intra_cluster_durations(
                extreme_value=extreme_value,
                only_statistics=True)
            self.update_histogram_statistics_of_inter_cluster_durations(
                extreme_value=extreme_value,
                only_statistics=True)
            ##
            ##
            # if intra_cluster_times_histogram_kwargs is not None:
            #     self.update_histogram_statistics_of_intra_cluster_times(
            #         extreme_value=extreme_value,
            #         only_statistics=False,
            #         **intra_cluster_times_histogram_kwargs)
            # if intra_cluster_durations_histogram_kwargs is not None:
            #     self.update_histogram_statistics_of_intra_cluster_durations(
            #         extreme_value=extreme_value,
            #         only_statistics=False,
            #         **intra_cluster_durations_histogram_kwargs)
            # if inter_cluster_durations_histogram_kwargs is not None:
            #     self.update_histogram_statistics_of_inter_cluster_durations(
            #         extreme_value=extreme_value,
            #         only_statistics=False,
            #         **inter_cluster_durations_histogram_kwargs)
            ##
            ##
            self.update_relative_statistics(
                extreme_value=extreme_value)
        self.update_parameterization()




##
