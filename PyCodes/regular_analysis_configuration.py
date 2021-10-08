from unbiased_estimators import *
from temporal_clustering_configuration import *
from naive_lag_correlations import *
from visual_configuration import *

class RegularAnalaysisConfiguration(VisualConfiguration):

    def __init__(self, series, savedir=None):
        super().__init__(savedir=savedir)
        if isinstance(series, list):
            self.series = series
        else:
            self.series = [series]
        self.n = len(self.series)

    def add_inter_exceedances(self, extreme_values, extreme_condition='greater than', extreme_parameter=None, series_indices=None, **kwargs):
        if series_indices is None:
            for i in range(self.n):
                self.add_inter_exceedances(
                    extreme_parameter=extreme_parameter,
                    extreme_condition=extreme_condition,
                    extreme_values=extreme_values,
                    series_indices=i,
                    **kwargs)
        elif isinstance(series_indices, (tuple, list, np.ndarray)):
            for i in series_indices:
                self.add_inter_exceedances(
                    extreme_parameter=extreme_parameter,
                    extreme_condition=extreme_condition,
                    extreme_values=extreme_values,
                    series_indices=i,
                    **kwargs)
        elif isinstance(series_indices, int):
            if extreme_parameter is None:
                extreme_parameter = self.series[series_indices]['identifiers']['parameter id']
            inter_exceedances = InterExceedanceDistribution(
                data=self.series[series_indices]['data'],
                extreme_parameter=extreme_parameter,
                extreme_condition=extreme_condition)
            inter_exceedances(
                extreme_values=extreme_values,
                **kwargs)
            self.series[series_indices]['inter-exceedance'] = inter_exceedances
        else:
            raise ValueError("invalid type(series_indices): {}".format(type(series_indices)))

    def add_unbiased_estimators(self, extreme_indices=None, extreme_parameter=None, series_indices=None, **kwargs):
        if series_indices is None:
            for i in range(self.n):
                self.add_unbiased_estimators(
                    extreme_parameter=extreme_parameter,
                    extreme_indices=extreme_indices,
                    series_indices=i,
                    **kwargs)
        elif isinstance(series_indices, (tuple, list, np.ndarray)):
            for i in series_indices:
                self.add_unbiased_estimators(
                    extreme_parameter=extreme_parameter,
                    extreme_indices=extreme_indices,
                    series_indices=i,
                    **kwargs)
        elif isinstance(series_indices, int):
            if extreme_parameter is None:
                extreme_parameter = self.series[series_indices]['identifiers']['parameter id']
            unbiased_estimators = UnbiasedEstimators(
                vs=self.series[series_indices]['data'][extreme_parameter],
                extreme_parameter=extreme_parameter,
                extreme_indices=extreme_indices)
            unbiased_estimators(**kwargs)
            self.series[series_indices]['unbiased estimators'] = unbiased_estimators
        else:
            raise ValueError("invalid type(series_indices): {}".format(type(series_indices)))

    def add_temporal_clustering(self, extreme_values, extreme_condition='greater than', extreme_parameter=None, series_indices=None, **kwargs):
        if series_indices is None:
            for i in range(len(self.series)):
                self.add_temporal_clustering(
                    extreme_parameter=extreme_parameter,
                    extreme_condition=extreme_condition,
                    extreme_values=extreme_values,
                    series_indices=i,
                    **kwargs)
        elif isinstance(series_indices, (tuple, list, np.ndarray)):
            for i in series_indices:
                self.add_temporal_clustering(
                    extreme_parameter=extreme_parameter,
                    extreme_condition=extreme_condition,
                    extreme_values=extreme_values,
                    series_indices=i,
                    **kwargs)
        elif isinstance(series_indices, int):
            if extreme_parameter is None:
                extreme_parameter = self.series[series_indices]['identifiers']['parameter id']
            temporal_clustering = TemporalClustering(
                data=self.series[series_indices]['data'],
                extreme_parameter=extreme_parameter,
                extreme_condition=extreme_condition)
            temporal_clustering(
                extreme_values=extreme_values,
                **kwargs)
            self.series[series_indices]['temporal clustering'] = temporal_clustering
        else:
            raise ValueError("invalid type(series_indices): {}".format(type(series_indices)))

    def add_temporal_clustering_parameterization(self, extreme_values, extreme_condition='greater than', extreme_parameter=None, series_indices=None, **kwargs):
        if series_indices is None:
            for i in range(len(self.series)):
                self.add_temporal_clustering_parameterization(
                    extreme_parameter=extreme_parameter,
                    extreme_condition=extreme_condition,
                    extreme_values=extreme_values,
                    series_indices=i,
                    **kwargs)
        elif isinstance(series_indices, (tuple, list, np.ndarray)):
            for i in series_indices:
                self.add_temporal_clustering_parameterization(
                    extreme_parameter=extreme_parameter,
                    extreme_condition=extreme_condition,
                    extreme_values=extreme_values,
                    series_indices=i,
                    **kwargs)
        elif isinstance(series_indices, int):
            if extreme_parameter is None:
                extreme_parameter = self.series[series_indices]['identifiers']['parameter id']
            temporal_clustering_parameterization = TemporalClusteringParameterization(
                data=self.series[series_indices]['data'],
                extreme_parameter=extreme_parameter,
                extreme_condition=extreme_condition)
            temporal_clustering_parameterization(
                extreme_values=extreme_values,
                **kwargs)
            self.series[series_indices]['temporal clustering parameterization'] = temporal_clustering_parameterization
        else:
            raise ValueError("invalid type(series_indices): {}".format(type(series_indices)))

    def add_naive_lag_correlations(self, extreme_values, extreme_condition='greater than', extreme_parameter=None, series_indices=None, **kwargs):
        if series_indices is None:
            for i in range(len(self.series)):
                self.add_naive_lag_correlations(
                    extreme_parameter=extreme_parameter,
                    extreme_condition=extreme_condition,
                    extreme_values=extreme_values,
                    series_indices=i,
                    **kwargs)
        elif isinstance(series_indices, (tuple, list, np.ndarray)):
            for i in series_indices:
                self.add_naive_lag_correlations(
                    extreme_parameter=extreme_parameter,
                    extreme_condition=extreme_condition,
                    extreme_values=extreme_values,
                    series_indices=i,
                    **kwargs)
        elif isinstance(series_indices, int):
            if extreme_parameter is None:
                extreme_parameter = self.series[series_indices]['identifiers']['parameter id']
            naive_lag_correlations = NaiveLagCorrelations(
                vs=self.series[series_indices]['data'][extreme_parameter],
                extreme_parameter=extreme_parameter,
                extreme_condition=extreme_condition)
            naive_lag_correlations(
                extreme_values=extreme_values,
                **kwargs)
            self.series[series_indices]['naive lag correlations'] = naive_lag_correlations
        else:
            raise ValueError("invalid type(series_indices): {}".format(type(series_indices)))

    def add_naive_lag_correlations_parameterization(self, extreme_values, extreme_condition='greater than', extreme_parameter=None, series_indices=None, **kwargs):
        if series_indices is None:
            for i in range(len(self.series)):
                self.add_naive_lag_correlations_parameterization(
                    extreme_parameter=extreme_parameter,
                    extreme_condition=extreme_condition,
                    extreme_values=extreme_values,
                    series_indices=i,
                    **kwargs)
        elif isinstance(series_indices, (tuple, list, np.ndarray)):
            for i in series_indices:
                self.add_naive_lag_correlations_parameterization(
                    extreme_parameter=extreme_parameter,
                    extreme_condition=extreme_condition,
                    extreme_values=extreme_values,
                    series_indices=i,
                    **kwargs)
        elif isinstance(series_indices, int):
            if extreme_parameter is None:
                extreme_parameter = self.series[series_indices]['identifiers']['parameter id']
            naive_lag_correlations_parameterization = NaiveLagCorrelationsParameterization(
                vs=self.series[series_indices]['data'][extreme_parameter],
                extreme_parameter=extreme_parameter,
                extreme_condition=extreme_condition)
            naive_lag_correlations_parameterization(
                extreme_values=extreme_values,
                **kwargs)
            self.series[series_indices]['naive lag correlations parameterization'] = naive_lag_correlations_parameterization
        else:
            raise ValueError("invalid type(series_indices): {}".format(type(series_indices)))

    def write_to_file(self):
        raise ValueError("not yet implemented; try viewing table in mean-time")
        for series in self.series:
            s = ""
            s += "\n .. SERIES IDENTIFIERS\n"
            for key, value in series['identifiers'].items():
                s += "\n{}:\n\t{}\n".format(key, value)
            for key, cls in series.items():
                if key not in ('identifiers', 'data'):
                    s += str(cls)
            ...


class RegularAnalaysis(RegularAnalaysisConfiguration):

    def __init__(self, series, savedir=None):
        super().__init__(
            series=series,
            savedir=savedir)
        self.cls = RegularAnalaysis

    def subview_histogram_of_inter_exceedance_times(self, ax, series, extreme_value, layout, elapsed_unit, show_inverse_transform, i, facecolor, sample_color, xs, ys):
        inter_exceedance_histogram = series['inter-exceedance'].extreme_values[extreme_value]['histograms']['inter-exceedance']
        xs.append(inter_exceedance_histogram.edges[0])
        xs.append(inter_exceedance_histogram.edges[-1])
        ys.append(np.max(inter_exceedance_histogram.counts) * 1.125)
        nevents_label = '${:,}$ Events'.format(
            series['inter-exceedance'].extreme_values[extreme_value]['nevents'])
        if (layout != 'overlay') or (self.n == 1):
            inter_exceedance_alpha = 0.5
            inverse_transform_alpha = 0.5
            if i == 0:
                inter_exceedance_label = 'Distribution of\nInter-Exceedance Times'
                inverse_transform_label = 'Inverse-Transform Sample\nof Exponential Distribution'
            else:
                inter_exceedance_label = None
                inverse_transform_label = None
        else:
            if show_inverse_transform:
                inter_exceedance_alpha = 1 / (2 * self.n)
                inverse_transform_alpha = 1 / (2 * self.n)
            else:
                inter_exceedance_alpha = 1 / self.n
            solar_cycles_label = TemporalConfiguration().get_solar_cycles_label(series['data'])
            inter_exceedance_label = 'Inter-Exceedance Times\nvia {} from {}'.format(
                nevents_label,
                solar_cycles_label)
            if i == 0:
                inverse_transform_label = 'Inverse-Transform Sample\nof Exponential Distribution'
            else:
                inverse_transform_label = None
        if show_inverse_transform and ((layout != 'overlay') or (self.n == 1)):
            inverse_transform_histogram = series['inter-exceedance'].extreme_values[extreme_value]['histograms']['inverse-transform sample']
            xs.append(inverse_transform_histogram.edges[0])
            xs.append(inverse_transform_histogram.edges[-1])
            ys.append(np.max(inverse_transform_histogram.counts) * 1.125)
            ax.bar(
                inter_exceedance_histogram.midpoints,
                inter_exceedance_histogram.counts,
                width=inter_exceedance_histogram.bin_widths,
                color=facecolor,
                label=inter_exceedance_label,
                alpha=inter_exceedance_alpha)
            ax.bar(
                inverse_transform_histogram.midpoints,
                inverse_transform_histogram.counts,
                width=inverse_transform_histogram.bin_widths,
                color=sample_color,
                label=inverse_transform_label,
                alpha=inverse_transform_alpha)
        else:
            ax.bar(
                inter_exceedance_histogram.midpoints,
                inter_exceedance_histogram.counts,
                width=inter_exceedance_histogram.bin_widths,
                color=facecolor,
                label=inter_exceedance_label,
                alpha=inter_exceedance_alpha)
        if (layout != 'overlay') or (self.n == 1):
            text_box = ax.text(
                0.95,
                0.95,
                nevents_label,
                fontsize=self.textsize,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)
            text_box.set_bbox(
                dict(
                    facecolor='silver',
                    alpha=0.25,
                    edgecolor='k'))
        self.apply_grid(ax)
        return xs, ys

    def view_histogram_of_inter_exceedance_times(self, extreme_values=None, show_inverse_transform=False, facecolors=('darkorange', 'green', 'purple', 'steelblue'), sample_color='k', sharex=False, sharey=False, collapse_x=False, collapse_y=False, figsize=None, save=False, layout='single'):
        if layout is None:
            permutable_layouts = ['single']
            if self.n > 1:
                permutable_layouts.append('overlay')
            if 2 <= self.n < 4:
                permutable_layouts.append('horizontal')
            elif (self.n > 2) and (self.n % 2 == 0):
                permutable_layouts.append('square')
            self.view_layout_permutations(
                f=self.view_histogram_of_inter_exceedance_times,
                layouts=permutable_layouts,
                extreme_values=extreme_values,
                show_inverse_transform=show_inverse_transform,
                facecolors=facecolors,
                sample_color=sample_color,
                sharex=sharex,
                sharey=sharey,
                collapse_x=collapse_x,
                collapse_y=collapse_y,
                figsize=figsize,
                save=save)
        elif layout == 'single':
            for series in self.series:
                cls = deepcopy(self.cls)
                visualizer = cls(
                    series=[series],
                    savedir=self.savedir)
                visualizer.view_histogram_of_inter_exceedance_times(
                    extreme_values=extreme_values,
                    show_inverse_transform=show_inverse_transform,
                    facecolors=facecolors,
                    sample_color=sample_color,
                    sharex=sharex,
                    sharey=sharey,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y,
                    figsize=figsize,
                    save=save,
                    layout='overlay')
        else:
            if extreme_values is None:
                extreme_values = set(self.series[0]['inter-exceedance'].extreme_values)
                if self.n > 1:
                    for i in range(1, self.n):
                        extreme_values = extreme_values.intersection(
                            set(self.series[i]['inter-exceedance'].extreme_values))
                extreme_values = np.sort(list(extreme_values))
                if extreme_values.size == 0:
                    raise ValueError("could not find extreme values common to all series")
            elif not isinstance(extreme_values, (tuple, list, np.ndarray)):
                extreme_values = [extreme_values]
            if not isinstance(facecolors, (tuple, list, np.ndarray)):
                facecolors = [facecolors]
            nc = len(facecolors)
            if nc < self.n:
                raise ValueError("{} facecolors for {} series".format(nc, self.n))
            for extreme_value in extreme_values:
                shared_xlabel = 'Inter-Exceedance Times'
                shared_ylabel = 'Frequency'
                xs, ys = [], [0]
                time_labels, extreme_labels, alt_extreme_labels = [], [], []
                handles, labels = [], []
                ## get figure and axes
                kws = self.get_number_of_figure_rows_and_columns(self.n, layout)
                fig, axes = plt.subplots(figsize=figsize, **kws)
                ## initialize plot
                if layout == 'overlay':
                    for i, (series, facecolor) in enumerate(zip(self.series, facecolors)):
                        elapsed_unit = series['identifiers']['elapsed unit']
                        time_label = "{} [{}]".format(
                            shared_xlabel,
                            elapsed_unit)
                        extreme_label = self.get_extreme_label(
                            extreme_parameter=series['inter-exceedance'].extreme_parameter,
                            extreme_condition=series['inter-exceedance'].extreme_condition,
                            extreme_value=extreme_value,
                            parameter_mapping=series['parameter mapping'],
                            unit_mapping=series['unit mapping'])
                        alt_extreme_label = self.get_generalized_extreme_label(
                            extreme_parameter=series['inter-exceedance'].extreme_parameter,
                            extreme_condition=series['inter-exceedance'].extreme_condition,
                            extreme_value=extreme_value,
                            parameter_mapping=series['parameter mapping'],
                            unit_mapping=series['unit mapping'],
                            generalized_parameter_mapping=series['generalized parameter mapping'])
                        time_labels.append(time_label)
                        extreme_labels.append(extreme_label)
                        alt_extreme_labels.append(alt_extreme_label)
                        xs, ys = self.subview_histogram_of_inter_exceedance_times(
                            ax=axes,
                            series=series,
                            extreme_value=extreme_value,
                            layout=layout,
                            elapsed_unit=elapsed_unit,
                            show_inverse_transform=show_inverse_transform,
                            i=i,
                            facecolor=facecolor,
                            sample_color=sample_color,
                            xs=xs,
                            ys=ys)
                    if self.n == 1:
                        axes.set_title(
                            self.series[0]['identifiers']['series id'],
                            fontsize=self.titlesize)
                    if self.is_same_elements(elements=time_labels, s='', n=self.n):
                        shared_xlabel = '{}'.format(time_labels[0])
                    axes.set_xlabel(shared_xlabel, fontsize=self.labelsize)
                    axes.set_ylabel(shared_ylabel, fontsize=self.labelsize)
                    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator())
                    axes.yaxis.set_minor_locator(ticker.AutoMinorLocator())
                    handles, labels = axes.get_legend_handles_labels()
                    axes = np.array([axes])
                    textcolor = 'k'
                else:
                    for i, (ax, series) in enumerate(zip(axes.ravel(), self.series)):
                        elapsed_unit = series['identifiers']['elapsed unit']
                        time_label = "{} [{}]".format(
                            shared_xlabel,
                            elapsed_unit)
                        extreme_label = self.get_extreme_label(
                            extreme_parameter=series['inter-exceedance'].extreme_parameter,
                            extreme_condition=series['inter-exceedance'].extreme_condition,
                            extreme_value=extreme_value,
                            parameter_mapping=series['parameter mapping'],
                            unit_mapping=series['unit mapping'])
                        alt_extreme_label = self.get_generalized_extreme_label(
                            extreme_parameter=series['inter-exceedance'].extreme_parameter,
                            extreme_condition=series['inter-exceedance'].extreme_condition,
                            extreme_value=extreme_value,
                            parameter_mapping=series['parameter mapping'],
                            unit_mapping=series['unit mapping'],
                            generalized_parameter_mapping=series['generalized parameter mapping'])
                        time_labels.append(time_label)
                        extreme_labels.append(extreme_label)
                        alt_extreme_labels.append(alt_extreme_label)
                        xs, ys = self.subview_histogram_of_inter_exceedance_times(
                            ax=ax,
                            series=series,
                            extreme_value=extreme_value,
                            layout=layout,
                            elapsed_unit=elapsed_unit,
                            show_inverse_transform=show_inverse_transform,
                            i=i,
                            facecolor=facecolors[0],
                            sample_color=sample_color,
                            xs=xs,
                            ys=ys)
                        ax.set_xlabel(
                            time_label,
                            fontsize=self.labelsize)
                        ax.set_ylabel(
                            shared_ylabel,
                            fontsize=self.labelsize)
                        ax.xaxis.set_minor_locator(
                            ticker.AutoMinorLocator())
                        ax.yaxis.set_minor_locator(
                            ticker.AutoMinorLocator())
                        ax.set_title(
                            series['identifiers']['series id'],
                            fontsize=self.titlesize)
                        _handles, _labels = ax.get_legend_handles_labels()
                        handles.extend(_handles)
                        labels.extend(_labels)
                hspace = 0.425 if 'vertical' in layout else 0.3
                fig.subplots_adjust(hspace=hspace)
                textcolor = True
                ## update axes
                if self.is_same_elements(elements=time_labels, s='', n=self.n):
                    shared_xlabel = '{}'.format(time_labels[0])
                self.share_axes(
                    axes=axes,
                    layout=layout,
                    xs=xs,
                    ys=ys,
                    sharex=sharex,
                    sharey=sharey,
                    xticks=True,
                    yticks=True,
                    xlim=True,
                    ylim=True,
                    xlabel=shared_xlabel,
                    ylabel=shared_ylabel,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y)
                ## auto-correct grid
                if (self.n > 1) and (layout == 'overlay'):
                    self.apply_grid(axes[0])
                fig.align_ylabels()
                ## update title
                s = 'Distribution of Inter-Exceedance Times'
                if show_inverse_transform:
                    s = '{} & Corresponding Inverse-Transform Sample'.format(s)
                if (self.n > 1) and (layout == 'overlay'):
                    axes[0].set_title(s, fontsize=self.titlesize)
                else:
                    fig.suptitle(s, fontsize=self.titlesize)
                ## show legend
                nhandles = len(handles)
                ncol = nhandles if nhandles > 1 else None
                if self.is_same_elements(elements=extreme_labels, s='', n=self.n):
                    leg_title = "{}".format(extreme_labels[0])
                else:
                    if self.is_same_elements(elements=alt_extreme_labels, s='', n=self.n):
                        leg_title = "{}".format(alt_extreme_labels[0])
                    else:
                        leg_title = None
                self.subview_legend(
                    fig=fig,
                    ax=axes.ravel()[0],
                    handles=handles,
                    labels=labels,
                    textcolor=textcolor,
                    facecolor='silver',
                    bottom=0.2,
                    ncol=ncol,
                    title=leg_title)
                ## show / save
                if save:
                    savename = 'RegularAnalysis_Inter-Exceedance_Histogram'
                    if show_inverse_transform:
                        savename = '{}_withInverseSample'.format(
                            savename)
                    for series in self.series:
                        cycle_nums = np.unique(series['data']['solar cycle'])
                        if cycle_nums.size == 1:
                            cycle_id = "SC-{}".format(cycle_nums[0])
                        else:
                            cycle_id = "SC" + "-".join(cycle_nums.astype(str))
                        savename = '{}_{}_{}_{}'.format(
                            savename,
                            cycle_id,
                            series['identifiers']['event type'].replace(' ', '-'),
                            series['identifiers']['parameter id'].replace(' ', '-'))
                    savename = '{}_EX-{}_{}'.format(
                        savename,
                        extreme_value,
                        layout)
                    savename = savename.replace(' ', '_')
                else:
                    savename = None
                self.display_image(fig, savename=savename)

    def subview_histogram_of_unbiased_estimators_parameter(self, ax, series, parameter, facecolor, symbol, xs, ys, show_skew=False, show_kurtosis=False, show_mean=False, show_median=False):
        histogram = series['unbiased estimators'].histograms[parameter]
        label = '{}\n${:,}$ resamples'.format(
            series['identifiers']['series id'],
            series['unbiased estimators'].nresamples)
        if show_skew:
            skew_label = 'skew({}) $=$ ${:0.2f}$'.format(symbol, histogram.skew)
            label = '{}\n{}'.format(
                label,
                skew_label)
        if show_kurtosis:
            kurtosis_label = 'kurtosis({}) $=$ ${:0.2f}$'.format(symbol, histogram.kurtosis)
            label = '{}\n{}'.format(
                label,
                kurtosis_label)
        if show_mean:
            mean_label = 'mean({}) $=$ ${:0.2f}$'.format(symbol, histogram.mean)
            label = '{}\n{}'.format(
                label,
                mean_label)
        if show_median:
            median_label = 'median({}) $=$ ${:0.2f}$'.format(symbol, histogram.median)
            label = '{}\n{}'.format(
                label,
                median_label)
        ax.bar(
            histogram.midpoints,
            histogram.counts,
            width=histogram.bin_widths,
            facecolor=facecolor,
            label=label,
            alpha=1/self.n)
        ## update max/min bounds for xy-axes
        xs.append(histogram.edges[0])
        xs.append(histogram.edges[-1])
        ys.append(np.max(histogram.counts) * 1.25)
        return xs, ys

    def view_histogram_of_unbiased_estimators_parameter(self, parameters=None, show_skew=False, show_kurtosis=False, show_mean=False, show_median=False, facecolors=('darkorange', 'steelblue', 'green', 'purple'), sharex=False, sharey=False, collapse_x=False, collapse_y=False, figsize=None, save=False, layout='single'):
        allowed_parameters = ('alpha', 'intercept', 'theta')
        if not isinstance(parameters, (tuple, list, np.ndarray)):
            parameters = [parameters]
        for parameter in parameters:
            if parameter not in allowed_parameters:
                raise ValueError("invalid parameter: {}".format(parameter))
            if layout is None:
                permutable_layouts = ['single']
                if self.n > 1:
                    permutable_layouts.append('overlay')
                if 2 <= self.n < 4:
                    permutable_layouts.append('horizontal')
                elif (self.n > 2) and (self.n % 2 == 0):
                    permutable_layouts.append('square')
                self.view_layout_permutations(
                    f=self.view_histogram_of_unbiased_estimators_parameter,
                    layouts=permutable_layouts,
                    parameters=parameter,
                    show_skew=show_skew,
                    show_kurtosis=show_kurtosis,
                    show_mean=show_mean,
                    show_median=show_median,
                    facecolors=facecolors,
                    sharex=sharex,
                    sharey=sharey,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y,
                    figsize=figsize,
                    save=save)
            elif layout == 'single':
                for series in self.series:
                    cls = deepcopy(self.cls)
                    visualizer = cls(
                        series=[series],
                        savedir=self.savedir)
                    visualizer.view_histogram_of_unbiased_estimators_parameter(
                        parameters=parameter,
                        show_skew=show_skew,
                        show_kurtosis=show_kurtosis,
                        show_mean=show_mean,
                        show_median=show_median,
                        facecolors=facecolors,
                        sharex=sharex,
                        sharey=sharey,
                        collapse_x=collapse_x,
                        collapse_y=collapse_y,
                        figsize=figsize,
                        save=save,
                        layout='overlay')
            else:
                ## verify user inputs
                if isinstance(facecolors, str):
                    facecolors = [facecolors]
                nc = len(facecolors)
                if nc < self.n:
                    raise ValueError("{} facecolors for {} series".format(nc, self.n))
                symbols_mapping = {
                    'alpha' : r'$\hat\alpha$',
                    'intercept' : r'$\hat{C}$',
                    'theta' : r'$\hat\theta$'}
                symbol = symbols_mapping[parameter]
                ## initialize bounds per ax (sharex, sharey)
                xs, ys = [], [0]
                shared_xlabel = '{}'.format(symbol)
                shared_ylabel = 'Frequency'
                ## get figure and axes
                kws = self.get_number_of_figure_rows_and_columns(self.n, layout)
                fig, axes = plt.subplots(figsize=figsize, **kws)
                if layout == 'overlay':
                    for series, facecolor in zip(self.series, facecolors):
                        xs, ys = self.subview_histogram_of_unbiased_estimators_parameter(
                            ax=axes,
                            series=series,
                            parameter=parameter,
                            facecolor=facecolor,
                            symbol=symbol,
                            xs=xs,
                            ys=ys,
                            show_skew=show_skew,
                            show_kurtosis=show_kurtosis,
                            show_mean=show_mean,
                            show_median=show_median)
                    if self.n == 1:
                        axes.set_title(self.series[0]['identifiers']['series id'], fontsize=self.titlesize)
                    axes.set_xlabel(shared_xlabel, fontsize=self.labelsize)
                    axes.set_ylabel(shared_ylabel, fontsize=self.labelsize)
                    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator())
                    axes.yaxis.set_minor_locator(ticker.AutoMinorLocator())
                    handles, labels = axes.get_legend_handles_labels()
                    axes = np.array([axes])
                    textcolor = 'k'
                else:
                    handles, labels = [], []
                    for ax, series, facecolor in zip(axes.ravel(), self.series, facecolors):
                        xs, ys = self.subview_histogram_of_unbiased_estimators_parameter(
                            ax=ax,
                            series=series,
                            parameter=parameter,
                            facecolor=facecolor,
                            symbol=symbol,
                            xs=xs,
                            ys=ys,
                            show_skew=show_skew,
                            show_kurtosis=show_kurtosis,
                            show_mean=show_mean,
                            show_median=show_median)
                        ax.set_xlabel(shared_xlabel, fontsize=self.labelsize)
                        ax.set_ylabel(shared_ylabel, fontsize=self.labelsize)
                        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
                        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
                        ax.set_title(series['identifiers']['series id'], fontsize=self.titlesize)
                        _handles, _labels = ax.get_legend_handles_labels()
                        handles.extend(_handles)
                        labels.extend(_labels)
                    fig.subplots_adjust(hspace=0.3)
                    textcolor = True
                ## update axes
                xlim = [0, 1] if parameter == 'theta' else True
                self.share_axes(
                    axes=axes,
                    layout=layout,
                    xs=xs,
                    ys=ys,
                    sharex=sharex,
                    sharey=sharey,
                    xticks=True,
                    yticks=True,
                    xlim=xlim,
                    ylim=True,
                    xlabel=shared_xlabel,
                    ylabel=shared_ylabel,
                    yfmt='{x:,.0f}',
                    collapse_x=collapse_x,
                    collapse_y=collapse_y)
                fig.align_ylabels()
                for ax in axes.ravel():
                    self.apply_grid(ax)
                ## update title
                title_mapping = {
                    'alpha' : r'Tail-Parameter $\hat\alpha$',
                    'intercept' : r'Cluster-Parameter $\hat{C}$',
                    'theta' : r'Extremal Index $\hat\theta$'}
                s = 'Histogram of {}'.format(title_mapping[parameter])
                if (self.n > 1) and (layout == 'overlay'):
                    axes[0].set_title(s, fontsize=self.titlesize)
                else:
                    fig.suptitle(s, fontsize=self.titlesize)
                ## show legend
                ncol = self.n if self.n in np.arange(2, 7).astype(int) else None
                self.subview_legend(
                    fig=fig,
                    ax=axes.ravel()[0],
                    handles=handles,
                    labels=labels,
                    facecolor='silver',
                    textcolor=textcolor,
                    bottom=0.2,
                    ncol=ncol)
                ## show / save
                if save:
                    savename = 'RegularAnalysis_UnbiasedEstimators_Histogram_{}'.format(parameter)
                    for series in self.series:
                        cycle_nums = np.unique(series['data']['solar cycle'])
                        if cycle_nums.size == 1:
                            cycle_id = "SC-{}".format(cycle_nums[0])
                        else:
                            cycle_id = "SC" + "-".join(cycle_nums.astype(str))
                        savename = '{}_{}_{}_{}'.format(
                            savename,
                            cycle_id,
                            series['identifiers']['event type'].replace(' ', '-'),
                            series['identifiers']['parameter id'].replace(' ', '-'))
                    savename = '{}_{}'.format(
                        savename,
                        layout)
                    savename = savename.replace(' ', '_')
                else:
                    savename = None
                self.display_image(fig, savename=savename)

    def subview_max_spectrum(self, ax, series, sample_num, show_points, show_fit, show_standard_deviation, show_standard_error, i, pts_color, fit_color, std_color, ste_color, show_linear_scale, show_log_scale, xs, ys):
        unbiased_estimators = series['unbiased estimators']
        pts_label = 'Max Spectrum' if i == 0 else None
        fit_label = 'Power-Law' if i == 0 else None
        std_label = 'Standard\nDeviation' if i == 0 else None
        ste_label = 'Standard Error\nof the Mean' if i == 0 else None
        if show_linear_scale:
            if show_points:
                ax.scatter(
                    unbiased_estimators.js,
                    unbiased_estimators.max_spectra[sample_num]['yj initial'],
                    s=5,
                    marker='.',
                    color=pts_color,
                    label=pts_label)
                xs.append(unbiased_estimators.js[0])
                xs.append(unbiased_estimators.js[-1])
                ys.append(np.min(unbiased_estimators.max_spectra[sample_num]['yj initial']))
                ys.append(np.max(unbiased_estimators.max_spectra[sample_num]['yj initial']))
            if show_fit:
                ax.plot(
                    unbiased_estimators.xj,
                    unbiased_estimators.max_spectra[sample_num]['yj fit'],
                    color=fit_color,
                    label=fit_label)
                xs.append(unbiased_estimators.xj[0])
                xs.append(unbiased_estimators.xj[-1])
                ys.append(np.min(unbiased_estimators.max_spectra[sample_num]['yj fit']))
                ys.append(np.max(unbiased_estimators.max_spectra[sample_num]['yj fit']))
            if show_standard_deviation:
                ax.errorbar(
                    unbiased_estimators.js,
                    unbiased_estimators.max_spectra[sample_num]['yj initial'],
                    unbiased_estimators.max_spectra[sample_num]['standard deviation'],
                    alpha=0.7,
                    capsize=5,
                    ecolor=std_color,
                    fmt='none',
                    label=std_label)
                xs.append(unbiased_estimators.js[0])
                xs.append(unbiased_estimators.js[-1])
                ys.append(np.min(unbiased_estimators.max_spectra[sample_num]['yj initial']) - np.min(unbiased_estimators.max_spectra[sample_num]['standard deviation']))
                ys.append(np.max(unbiased_estimators.max_spectra[sample_num]['yj initial']) + np.max(unbiased_estimators.max_spectra[sample_num]['standard deviation']))
            if show_standard_error:
                ax.errorbar(
                    unbiased_estimators.js,
                    unbiased_estimators.max_spectra[sample_num]['yj initial'],
                    unbiased_estimators.max_spectra[sample_num]['standard error'],
                    alpha=0.7,
                    capsize=5,
                    ecolor=ste_color,
                    fmt='none',
                    label=ste_label)
                xs.append(unbiased_estimators.js[0])
                xs.append(unbiased_estimators.js[-1])
                ys.append(np.min(unbiased_estimators.max_spectra[sample_num]['yj initial']) - np.min(unbiased_estimators.max_spectra[sample_num]['standard error']))
                ys.append(np.max(unbiased_estimators.max_spectra[sample_num]['yj initial']) + np.max(unbiased_estimators.max_spectra[sample_num]['standard error']))
        else:
            if show_points:
                ax.scatter(
                    2 ** unbiased_estimators.js,
                    2 ** unbiased_estimators.max_spectra[sample_num]['yj initial'],
                    s=5,
                    marker='.',
                    color=pts_color,
                    label=pts_label)
                xs.append(2 ** unbiased_estimators.js[0])
                xs.append(2 ** unbiased_estimators.js[-1])
                ys.append(2 ** np.min(unbiased_estimators.max_spectra[sample_num]['yj initial']))
                ys.append(2 ** np.max(unbiased_estimators.max_spectra[sample_num]['yj initial']))
            if show_fit:
                ax.plot(
                    2 ** unbiased_estimators.xj,
                    2 ** unbiased_estimators.max_spectra[sample_num]['yj fit'],
                    color=fit_color,
                    label=fit_label)
                xs.append(2 ** unbiased_estimators.xj[0])
                xs.append(2 ** unbiased_estimators.xj[-1])
                ys.append(2 ** np.min(unbiased_estimators.max_spectra[sample_num]['yj fit']))
                ys.append(2 ** np.max(unbiased_estimators.max_spectra[sample_num]['yj fit']))
            if show_standard_deviation:
                ax.errorbar(
                    2 ** unbiased_estimators.js,
                    2 ** unbiased_estimators.max_spectra[sample_num]['yj initial'],
                    2 ** unbiased_estimators.max_spectra[sample_num]['standard deviation'],
                    alpha=0.7,
                    capsize=5,
                    ecolor=std_color,
                    fmt='none',
                    label=std_label)
                xs.append(2 ** unbiased_estimators.js[0])
                xs.append(2 ** unbiased_estimators.js[-1])
                ys.append(2 ** (np.min(unbiased_estimators.max_spectra[sample_num]['yj initial']) - np.min(unbiased_estimators.max_spectra[sample_num]['standard deviation'])))
                ys.append(2 ** (np.max(unbiased_estimators.max_spectra[sample_num]['yj initial']) + np.max(unbiased_estimators.max_spectra[sample_num]['standard deviation'])))
            if show_standard_error:
                ax.errorbar(
                    2 ** unbiased_estimators.js,
                    2 ** unbiased_estimators.max_spectra[sample_num]['yj initial'],
                    2 ** unbiased_estimators.max_spectra[sample_num]['standard error'],
                    alpha=0.7,
                    capsize=5,
                    ecolor=ste_color,
                    fmt='none',
                    label=ste_label)
                xs.append(2 ** unbiased_estimators.js[0])
                xs.append(2 ** unbiased_estimators.js[-1])
                ys.append(2 ** (np.min(unbiased_estimators.max_spectra[sample_num]['yj initial']) - np.min(unbiased_estimators.max_spectra[sample_num]['standard error'])))
                ys.append(2 ** (np.max(unbiased_estimators.max_spectra[sample_num]['yj initial']) + np.max(unbiased_estimators.max_spectra[sample_num]['standard error'])))
        if (show_linear_scale and show_log_scale):
            sc_label = '{}'.format(series['identifiers']['series id'])
            # fit_label = r'$\alpha$ $=$ ${:.2f}$'.format(analysis.unbiased_estimators.alphas[sample_num])
            fit_label = r'mean($\alpha$) $=$ ${:.2f}$'.format(unbiased_estimators.histograms['alpha'].mean)
            text_label = '{}\n{}'.format(sc_label, fit_label)
        else:
            ax.set_title(series['identifiers']['series id'], fontsize=self.titlesize)
            text_label = r'mean($\alpha$) $=$ ${:.2f}$'.format(unbiased_estimators.histograms['alpha'].mean)
        text_box = ax.text(
            0.05,
            0.95,
            text_label,
            fontsize=self.textsize,
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes)
        text_box.set_bbox(dict(facecolor='silver', alpha=0.25, edgecolor='k'))
        return xs, ys

    def view_max_spectrum(self, sample_num=0, show_points=False, show_fit=False, show_standard_deviation=False, show_standard_error=False, show_linear_scale=False, show_log_scale=False, pts_color='gray', fit_color='k', std_color='darkorange', ste_color='steelblue', sharex=False, sharey=False, collapse_x=False, collapse_y=False, figsize=None, save=False, layout='single'):
        if layout is None:
            permutable_layouts = ['single']
            if 2 <= self.n < 4:
                permutable_layouts.append('horizontal')
            elif (self.n > 2) and (self.n % 2 == 0):
                permutable_layouts.append('square')
            self.view_layout_permutations(
                f=self.view_max_spectrum,
                layouts=permutable_layouts,
                sample_num=sample_num,
                show_points=show_points,
                show_fit=show_fit,
                show_standard_deviation=show_standard_deviation,
                show_standard_error=show_standard_error,
                show_linear_scale=show_linear_scale,
                show_log_scale=show_log_scale,
                pts_color=pts_color,
                fit_color=fit_color,
                std_color=std_color,
                ste_color=ste_color,
                sharex=sharex,
                sharey=sharey,
                collapse_x=collapse_x,
                collapse_y=collapse_y,
                figsize=figsize,
                save=save)
        elif layout == 'single':
            for series in self.series:
                cls = deepcopy(self.cls)
                visualizer = cls(
                    series=[series],
                    savedir=self.savedir)
                visualizer.view_max_spectrum(
                sample_num=sample_num,
                show_points=show_points,
                show_fit=show_fit,
                show_standard_deviation=show_standard_deviation,
                show_standard_error=show_standard_error,
                show_linear_scale=show_linear_scale,
                show_log_scale=show_log_scale,
                pts_color=pts_color,
                fit_color=fit_color,
                std_color=std_color,
                ste_color=ste_color,
                sharex=sharex,
                sharey=sharey,
                collapse_x=collapse_x,
                collapse_y=collapse_y,
                figsize=figsize,
                save=save,
                layout='overlay')
        else:
            ## verify user inputs
            if not any([show_points, show_fit, show_standard_deviation, show_standard_error]):
                raise ValueError("set at least one of the following inputs to True: show_points, show_fit, show_standard_deviation, show_standard_error")
            if not any([show_linear_scale, show_log_scale]):
                raise ValueError("set at least one of the following inputs to True: show_linear_scale, show_log_scale")
            if not isinstance(sample_num, int):
                raise ValueError("invalid type(sample_num): {}".format(type(sample_num)))
            ## initialize bounds and units per ax (sharex, sharey)
            xs, ys = [], []
            time_unit_labels, parameter_unit_labels = [], []
            handles, labels = [], []
            if (show_linear_scale and show_log_scale):
                basex = None
                basey = None
                mirror_basex = 2
                mirror_basey = 2
                xlabel = r'$j$'
                ylabel = r'$Y(j)$'
                mirror_xlabel = r'$2^j$'
                mirror_ylabel = r'$2^{Y(j)}$'
                xfmt = None
                yfmt = None
                mirror_xfmt = '{x:,.0f}'
                mirror_yfmt = '{x:,.0f}'
            else:
                mirror_basex = None
                mirror_basey = None
                mirror_xlabel = None
                mirror_ylabel = None
                if show_linear_scale:
                    basex = None
                    basey = None
                    xlabel = r'$j$'
                    ylabel = r'$Y(j)$'
                    xfmt = None
                    yfmt = None
                    mirror_xfmt = None
                    mirror_yfmt = None
                else:
                    basex = 2
                    basey = 2
                    xlabel = r'$2^j$'
                    ylabel = r'$2^{Y(j)}$'
                    xfmt = '{x:,.0f}'
                    yfmt = '{x:,.0f}'
                    mirror_xfmt = None
                    mirror_yfmt = None
            ## get figure and axes
            kws = self.get_number_of_figure_rows_and_columns(self.n, layout)
            fig, axes = plt.subplots(figsize=figsize, **kws)
            ## initialize plot
            if layout == 'overlay':
                if self.n != 1:
                    raise ValueError("layout='overlay' for this method will only work for one series")
                axes = np.array([axes])
            for i, (ax, series) in enumerate(zip(axes.ravel(), self.series)):
                time_unit_label = series['identifiers']['elapsed unit']
                parameter_unit_label = series['unit mapping'][series['unbiased estimators'].extreme_parameter]
                time_unit_labels.append(time_unit_label)
                parameter_unit_labels.append(parameter_unit_label)
                xs, ys = self.subview_max_spectrum(
                    ax=ax,
                    series=series,
                    sample_num=sample_num,
                    show_points=show_points,
                    show_fit=show_fit,
                    show_standard_deviation=show_standard_deviation,
                    show_standard_error=show_standard_error,
                    i=i,
                    pts_color=pts_color,
                    fit_color=fit_color,
                    std_color=std_color,
                    ste_color=ste_color,
                    show_linear_scale=show_linear_scale,
                    show_log_scale=show_log_scale,
                    xs=xs,
                    ys=ys)
                ax.set_xlabel(xlabel, fontsize=self.labelsize)
                ax.set_ylabel(ylabel, fontsize=self.labelsize)
                if i == 0:
                    _handles, _labels = ax.get_legend_handles_labels()
                    handles.extend(_handles)
                    labels.extend(_labels)
            if layout == 'overlay':
                textcolor = 'k'
            else:
                hspace = 0.425 if 'vertical' in layout else 0.3
                fig.subplots_adjust(hspace=hspace)
                textcolor = True
            if show_log_scale:
                ys = np.array(ys)
                nonpositive_loc = (ys <= 0)
                if np.any(nonpositive_loc):
                    positive_loc = np.invert(nonpositive_loc)
                    ys = ys[positive_loc].tolist()
                    ys.append(0.1)
            ## update axes
            if self.is_same_elements(elements=time_unit_labels, s='', n=self.n):
                if (show_log_scale and not show_linear_scale):
                    shared_xlabel = '{} [{}]'.format(xlabel, time_unit_labels[0])
                else:
                    shared_xlabel = r'{} [$log_2$ {}]'.format(xlabel, time_unit_labels[0])
                if mirror_xlabel is not None:
                    shared_mirror_xlabel = '{} [{}]'.format(mirror_xlabel, time_unit_labels[0])
                else:
                    shared_mirror_xlabel = None
            else:
                shared_xlabel = '{}'.format(xlabel)
                shared_mirror_xlabel = '{}'.format(mirror_xlabel) if mirror_xlabel is not None else None
            if self.is_same_elements(elements=parameter_unit_labels, s='', n=self.n):
                if (show_log_scale and not show_linear_scale):
                    shared_ylabel = '{} [{}]'.format(ylabel, parameter_unit_labels[0])
                else:
                    shared_ylabel = r'{} [$log_2$ {}]'.format(ylabel, parameter_unit_labels[0])
                if mirror_ylabel is not None:
                    shared_mirror_ylabel = '{} [{}]'.format(mirror_ylabel, parameter_unit_labels[0])
                else:
                    shared_mirror_ylabel = None
            else:
                shared_ylabel = '{}'.format(ylabel)
                shared_mirror_ylabel = '{}'.format(mirror_ylabel) if mirror_ylabel is not None else None
            self.share_axes(
                axes=axes,
                layout=layout,
                xs=xs,
                ys=ys,
                sharex=sharex,
                sharey=sharey,
                xlim=True,
                ylim=True,
                xticks=True,
                yticks=True,
                xfmt=xfmt,
                yfmt=yfmt,
                xlabel=shared_xlabel,
                ylabel=shared_ylabel,
                basex=basex,
                basey=basey,
                collapse_x=collapse_x,
                collapse_y=collapse_y)
            if (show_linear_scale and show_log_scale):
                self.share_mirror_axes(
                    axes=axes,
                    layout=layout,
                    sharex=sharex,
                    sharey=sharey,
                    xlim=True,
                    ylim=True,
                    xticks=True,
                    yticks=True,
                    xticklabels=lambda x : 2**x,
                    yticklabels=lambda x : 2**x,
                    xfmt='{:,.0f}',
                    yfmt='{:,.0f}',
                    xlabel=shared_mirror_xlabel,
                    ylabel=shared_mirror_ylabel,
                    xcolor='grey',
                    ycolor='grey',
                    collapse_x=collapse_x,
                    collapse_y=collapse_y)
            for ax in axes.ravel():
                self.apply_grid(ax)
            fig.align_ylabels()
            ## update title
            s = 'Max Spectrum Power-Law'
            if (self.n > 1) and (layout == 'overlay'):
                axes[0].set_title(s, fontsize=self.titlesize)
            else:
                fig.suptitle(s, fontsize=self.titlesize)
            ## show legend
            self.subview_legend(
                fig=fig,
                ax=axes.ravel()[0],
                handles=handles,
                labels=labels,
                facecolor='silver',
                textcolor=textcolor,
                bottom=0.2,
                ncol=None)
            ## show / save
            if save:
                savename = 'RegularAnalysis_UnbiasedEstimators_MaxSpectrum_{}'.format(sample_num)
                if show_points:
                    savename = '{}_OG'.format(savename)
                if show_fit:
                    savename = '{}_POWERLAW'.format(savename)
                if show_standard_deviation:
                    savename = '{}_STD'.format(savename)
                if show_standard_error:
                    savename = '{}_STE'.format(savename)
                if show_linear_scale:
                    savename = '{}_LIN'.format(savename)
                if show_log_scale:
                    savename = '{}_LOG'.format(savename)
                for series in self.series:
                    cycle_nums = np.unique(series['data']['solar cycle'])
                    if cycle_nums.size == 1:
                        cycle_id = "SC-{}".format(cycle_nums[0])
                    else:
                        cycle_id = "SC" + "-".join(cycle_nums.astype(str))
                    savename = '{}_{}_{}_{}'.format(
                        savename,
                        cycle_id,
                        series['identifiers']['event type'].replace(' ', '-'),
                        series['identifiers']['parameter id'].replace(' ', '-'))
                savename = '{}_{}'.format(
                    savename,
                    layout)
                savename = savename.replace(' ', '_')
            else:
                savename = None
            self.display_image(fig, savename=savename)

    def subview_point_estimators_of_extremal_index(self, ax, series, i, mean_color, median_color, bounds_color, xs, ys):
        markers = ('o', '*', '_', '_')
        statistic_ids = ('mean', 'median', 'maximum', 'minimum')
        facecolors = (mean_color, median_color, bounds_color, bounds_color)
        unbiased_estimators = series['unbiased estimators']
        for statistic_id, marker, facecolor in zip(statistic_ids, markers, facecolors):
            if facecolor is not None:
                alpha = 0.5 if statistic_id in ('mean', 'median') else 1
                if i == 0:
                    if statistic_id in ('mean', 'median'):
                        label = r'{}$(\hat\theta_j)$'.format(statistic_id)
                    elif statistic_id == 'maximum':
                        label = r'min/max$(\hat\theta_j)$'
                    else:
                        label = None
                else:
                    label = None
                statistic_values = unbiased_estimators.point_estimators[statistic_id]
                lower_y_bound = np.nanmin(statistic_values) * 0.875
                upper_y_bound = np.nanmax(statistic_values) * 1.125
                ys.append(lower_y_bound)
                ys.append(upper_y_bound)
                ax.scatter(
                    unbiased_estimators.xj,
                    statistic_values,
                    marker=marker,
                    color=facecolor,
                    label=label,
                    s=5,
                    alpha=alpha)
            ax.set_title(series['identifiers']['series id'], fontsize=self.titlesize)
        xs.append(unbiased_estimators.xj[0] - 0.5)
        xs.append(unbiased_estimators.xj[-1] + 0.5)
        return xs, ys

    def view_point_estimators_of_extremal_index(self, show_full_scale=False, mean_color='darkorange', median_color='steelblue', bounds_color='k', sharex=False, sharey=False, collapse_x=False, collapse_y=False, figsize=None, save=False, layout='single'):
        if layout is None:
            permutable_layouts = ['single']
            if 2 <= self.n < 4:
                permutable_layouts.append('horizontal')
            elif (self.n > 2) and (self.n % 2 == 0):
                permutable_layouts.append('square')
            self.view_layout_permutations(
                f=self.view_point_estimators_of_extremal_index,
                layouts=permutable_layouts,
                show_full_scale=show_full_scale,
                mean_color=mean_color,
                median_color=median_color,
                bounds_color=bounds_color,
                sharex=sharex,
                sharey=sharey,
                collapse_x=collapse_x,
                collapse_y=collapse_y,
                figsize=figsize,
                save=save)
        elif layout == 'single':
            for series in self.series:
                cls = deepcopy(self.cls)
                visualizer = cls(
                    series=[series],
                    savedir=self.savedir)
                visualizer.view_point_estimators_of_extremal_index(
                    show_full_scale=show_full_scale,
                    mean_color=mean_color,
                    median_color=median_color,
                    bounds_color=bounds_color,
                    sharex=sharex,
                    sharey=sharey,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y,
                    figsize=figsize,
                    save=save,
                    layout='overlay')
        else:
            ## initialize bounds per ax (sharex, sharey)
            xs, ys = [], []
            time_unit_labels = []
            handles, labels = [], []
            shared_xlabel = r'Time-Scale $j$'
            shared_ylabel = r'$\hat\theta(j)$'
            ## get figure and axes
            kws = self.get_number_of_figure_rows_and_columns(self.n, layout)
            fig, axes = plt.subplots(figsize=figsize, **kws)
            if layout == 'overlay':
                if self.n != 1:
                    raise ValueError("layout='overlay' for this method will only work for one series")
                axes = np.array([axes])
            for i, (ax, series) in enumerate(zip(axes.ravel(), self.series)):
                time_unit_label = series['identifiers']['elapsed unit']
                time_unit_labels.append(time_unit_label)
                xs, ys = self.subview_point_estimators_of_extremal_index(
                    ax=ax,
                    series=series,
                    i=i,
                    mean_color=mean_color,
                    median_color=median_color,
                    bounds_color=bounds_color,
                    xs=xs,
                    ys=ys)
                ax.set_xlabel('{} [{}]'.format(shared_xlabel, time_unit_label, fontsize=self.labelsize))
                ax.set_ylabel(shared_ylabel, fontsize=self.labelsize)
                ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
                ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
                self.apply_grid(ax)
                if i == 0:
                    _handles, _labels = ax.get_legend_handles_labels()
                    handles.extend(_handles)
                    labels.extend(_labels)
            if layout == 'overlay':
                textcolor = 'k'
            else:
                hspace = 0.425 if 'vertical' in layout else 0.3
                fig.subplots_adjust(hspace=hspace)
                textcolor = True
            ## update axes
            if show_full_scale:
                ys.extend([0, 1])
            if self.is_same_elements(elements=time_unit_labels, s='', n=self.n):
                shared_xlabel = r'{} [$log_2$ {}]'.format(shared_xlabel, time_unit_labels[0])
            self.share_axes(
                axes=axes,
                layout=layout,
                xs=xs,
                ys=ys,
                sharex=sharex,
                sharey=sharey,
                xticks=True,
                yticks=True,
                xlim=True,
                ylim=True,
                xlabel=shared_xlabel,
                ylabel=shared_ylabel,
                collapse_x=collapse_x,
                collapse_y=collapse_y)
            fig.align_ylabels()
            ## update title
            s = r'Point Estimators of Extremal Index $\hat\theta_{j}$'
            fig.suptitle(s, fontsize=self.titlesize)
            ## show legend
            nhandles = len(handles)
            ncol = nhandles if nhandles > 1 else None
            self.subview_legend(
                fig=fig,
                ax=axes.ravel()[0],
                handles=handles,
                labels=labels,
                facecolor='silver',
                textcolor=textcolor,
                bottom=0.2,
                ncol=ncol)
            ## show / save
            if save:
                savename = 'RegularAnalysis_UnbiasedEstimators_PointEstimators'
                if show_full_scale:
                    savename = '{}_FullScale'.format(savename)
                for series in self.series:
                    cycle_nums = np.unique(series['data']['solar cycle'])
                    if cycle_nums.size == 1:
                        cycle_id = "SC-{}".format(cycle_nums[0])
                    else:
                        cycle_id = "SC" + "-".join(cycle_nums.astype(str))
                    savename = '{}_{}_{}_{}'.format(
                        savename,
                        cycle_id,
                        series['identifiers']['event type'].replace(' ', '-'),
                        series['identifiers']['parameter id'].replace(' ', '-'))
                savename = '{}_{}'.format(
                    savename,
                    layout)
                savename = savename.replace(' ', '_')
            else:
                savename = None
            self.display_image(fig, savename=savename)

    def subview_histogram_of_cluster_timings(self, ax, series, extreme_value, layout, cluster_id, time_difference_id, bias_id, time_unit_label, i, facecolor, time_color, xs, ys):
        time_threshold = series['temporal clustering'].extreme_values[extreme_value]['temporal clustering'][bias_id]['time threshold']
        time_label_with_units = '$T_C$ $=$ ${}$ {}'.format(
            time_threshold,
            self.make_plural(time_unit_label))
        histogram = series['temporal clustering'].extreme_values[extreme_value]['temporal clustering'][bias_id][cluster_id][time_difference_id]['histogram']
        xs.append(time_threshold * 1.125)
        xs.append(histogram.edges[-1] * 1.125)
        ys.append(np.max(histogram.counts) * 1.125)
        if (layout != 'overlay') or (self.n == 1):
            alpha = 1
            show_critical_time = True
            if i == 0:
                time_label_without_units = 'De-Clustering Time-Threshold $T_C$'
                histogram_label = time_difference_id.title()
            else:
                time_label_without_units = None
                histogram_label = None
        else:
            alpha = 1 / self.n
            show_critical_time = False
            histogram_label = '{}\n{}\n{}'.format(
                series['identifiers']['series id'],
                time_difference_id.replace('-', '-cluster ').title(),
                time_label_with_units)
        ax.bar(
            histogram.midpoints,
            histogram.counts,
            width=histogram.bin_widths,
            color=facecolor,
            label=histogram_label,
            alpha=alpha)
        if show_critical_time:
            ax.axvline(
                x=time_threshold,
                color=time_color,
                label=time_label_without_units)
            text_box = ax.text(
                0.95,
                0.95,
                time_label_with_units,
                fontsize=self.textsize,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)
            text_box.set_bbox(dict(facecolor='silver', alpha=0.25, edgecolor='k'))
        if (layout != 'overlay') or (self.n == 1):
            ax.set_title(series['identifiers']['series id'], fontsize=self.titlesize)
        return xs, ys

    def view_histogram_of_cluster_timings(self, extreme_values, cluster_ids, time_difference_ids, show_first_order_bias=False, show_threshold_bias=False, show_baseline=False, facecolors=('steelblue', 'darkorange', 'purple', 'green'), time_color='darkorange', sharex=False, sharey=False, collapse_x=False, collapse_y=False, figsize=None, save=False, layout='single'):
        if layout is None:
            permutable_layouts = ['single']
            if self.n > 1:
                permutable_layouts.append('overlay')
            if 2 <= self.n <= 4:
                # permutable_layouts.append('horizontal')
                permutable_layouts.append('vertical')
            elif (self.n > 2) and (self.n % 2 == 0):
                permutable_layouts.append('square')
            self.view_layout_permutations(
                f=self.view_histogram_of_cluster_timings,
                layouts=permutable_layouts,
                extreme_values=extreme_values,
                cluster_ids=cluster_ids,
                time_difference_ids=time_difference_ids,
                show_first_order_bias=show_first_order_bias,
                show_threshold_bias=show_threshold_bias,
                show_baseline=show_baseline,
                facecolors=facecolors,
                time_color=time_color,
                sharex=sharex,
                sharey=sharey,
                collapse_x=collapse_x,
                collapse_y=collapse_y,
                figsize=figsize,
                save=save)
        elif layout == 'single':
            for series in self.series:
                cls = deepcopy(self.cls)
                visualizer = cls(
                    series=[series],
                    savedir=self.savedir)
                visualizer.view_histogram_of_cluster_timings(
                    extreme_values=extreme_values,
                    cluster_ids=cluster_ids,
                    time_difference_ids=time_difference_ids,
                    show_first_order_bias=show_first_order_bias,
                    show_threshold_bias=show_threshold_bias,
                    show_baseline=show_baseline,
                    facecolors=facecolors,
                    time_color=time_color,
                    sharex=sharex,
                    sharey=sharey,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y,
                    figsize=figsize,
                    save=save,
                    layout='overlay')
        else:
            ## verify inputs
            if isinstance(facecolors, str):
                facecolors = [facecolors]
            elif not isinstance(facecolors, (tuple, list, np.ndarray)):
                raise ValueError("invalid type(facecolors): {}".format(type(facecolors)))
            bias_ids = ('first-order', 'threshold', 'baseline')
            conditions = np.array([show_first_order_bias, show_threshold_bias, show_baseline])
            nconditions = np.sum(conditions)
            if nconditions < 1:
                raise ValueError("input at least one of the following: show_first_order_bias, show_threshold_bias, show_baseline")
            if not isinstance(cluster_ids, (tuple, list, np.ndarray)):
                cluster_ids = [cluster_ids]
            if isinstance(time_difference_ids, str):
                time_difference_ids = [time_difference_ids]
            elif not isinstance(time_difference_ids, (tuple, list, np.ndarray)):
                raise ValueError("invalid type(time_difference_ids): {}".format(type(time_difference_ids)))
            if extreme_values is None:
                extreme_values = set(self.series[0]['temporal clustering'].extreme_values)
                if self.n > 1:
                    for i in range(1, self.n):
                        extreme_values = extreme_values.intersection(
                            set(self.series[i]['temporal clustering'].extreme_values))
                extreme_values = np.sort(list(extreme_values))
                if extreme_values.size == 0:
                    raise ValueError("could not find extreme values common to all series")
            elif not isinstance(extreme_values, (tuple, list, np.ndarray)):
                extreme_values = [extreme_values]
            for extreme_value in extreme_values:
                for cluster_id in cluster_ids:
                    for time_difference_id in time_difference_ids:
                        for bias_id, bias_condition in zip(bias_ids, conditions):
                            if bias_condition:
                                shared_xlabel = time_difference_id.title()
                                shared_ylabel = 'Frequency'
                                xs, ys = [0], [0]
                                time_unit_labels = []
                                extreme_labels, alt_extreme_labels = [], []
                                handles, labels = [], []
                                ## get figure and axes
                                kws = self.get_number_of_figure_rows_and_columns(self.n, layout)
                                fig, axes = plt.subplots(figsize=figsize, **kws)
                                ## initialize plot
                                if layout == 'overlay':
                                    nc = len(facecolors)
                                    if nc < self.n:
                                        raise ValueError("{} facecolors for {} series".format(nc, self.n))
                                    for i, (series, facecolor) in enumerate(zip(self.series, facecolors)):
                                        time_unit_label = series['identifiers']['elapsed unit']
                                        extreme_label = self.get_extreme_label(
                                            extreme_parameter=series['temporal clustering'].extreme_parameter,
                                            extreme_condition=series['temporal clustering'].extreme_condition,
                                            extreme_value=extreme_value,
                                            parameter_mapping=series['parameter mapping'],
                                            unit_mapping=series['unit mapping'])
                                        alt_extreme_label = self.get_generalized_extreme_label(
                                            extreme_parameter=series['temporal clustering'].extreme_parameter,
                                            extreme_condition=series['temporal clustering'].extreme_condition,
                                            extreme_value=extreme_value,
                                            parameter_mapping=series['parameter mapping'],
                                            unit_mapping=series['unit mapping'],
                                            generalized_parameter_mapping=series['generalized parameter mapping'])
                                        time_unit_labels.append(time_unit_label)
                                        extreme_labels.append(extreme_label)
                                        alt_extreme_labels.append(alt_extreme_label)
                                        xs, ys = self.subview_histogram_of_cluster_timings(
                                            ax=axes,
                                            series=series,
                                            extreme_value=extreme_value,
                                            layout=layout,
                                            cluster_id=cluster_id,
                                            time_difference_id=time_difference_id,
                                            bias_id=bias_id,
                                            time_unit_label=time_unit_label,
                                            i=i,
                                            facecolor=facecolor,
                                            time_color=time_color,
                                            xs=xs,
                                            ys=ys)
                                    axes.set_xlabel(shared_xlabel, fontsize=self.labelsize)
                                    axes.set_ylabel(shared_ylabel, fontsize=self.labelsize)
                                    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator())
                                    axes.yaxis.set_minor_locator(ticker.AutoMinorLocator())
                                    handles, labels = axes.get_legend_handles_labels()
                                    axes = np.array([axes])
                                    textcolor = 'k'
                                else:
                                    for i, (ax, series) in enumerate(zip(axes.ravel(), self.series)):
                                        time_unit_label = series['identifiers']['elapsed unit']
                                        extreme_label = self.get_extreme_label(
                                            extreme_parameter=series['temporal clustering'].extreme_parameter,
                                            extreme_condition=series['temporal clustering'].extreme_condition,
                                            extreme_value=extreme_value,
                                            parameter_mapping=series['parameter mapping'],
                                            unit_mapping=series['unit mapping'])
                                        alt_extreme_label = self.get_generalized_extreme_label(
                                            extreme_parameter=series['temporal clustering'].extreme_parameter,
                                            extreme_condition=series['temporal clustering'].extreme_condition,
                                            extreme_value=extreme_value,
                                            parameter_mapping=series['parameter mapping'],
                                            unit_mapping=series['unit mapping'],
                                            generalized_parameter_mapping=series['generalized parameter mapping'])
                                        time_unit_labels.append(time_unit_label)
                                        extreme_labels.append(extreme_label)
                                        alt_extreme_labels.append(alt_extreme_label)
                                        xs, ys = self.subview_histogram_of_cluster_timings(
                                            ax=ax,
                                            series=series,
                                            extreme_value=extreme_value,
                                            layout=layout,
                                            cluster_id=cluster_id,
                                            time_difference_id=time_difference_id,
                                            bias_id=bias_id,
                                            time_unit_label=time_unit_label,
                                            i=i,
                                            facecolor=facecolors[0],
                                            time_color=time_color,
                                            xs=xs,
                                            ys=ys)
                                        ax.set_xlabel('{} [{}]'.format(shared_xlabel, time_unit_label), fontsize=self.labelsize)
                                        ax.set_ylabel(shared_ylabel, fontsize=self.labelsize)
                                        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
                                        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
                                        _handles, _labels = ax.get_legend_handles_labels()
                                        handles.extend(_handles)
                                        labels.extend(_labels)
                                hspace = 0.425 if 'vertical' in layout else 0.3
                                fig.subplots_adjust(hspace=hspace)
                                ## update axes
                                if self.is_same_elements(elements=time_unit_labels, s='', n=self.n):
                                    shared_xlabel = r'{} [{}]'.format(shared_xlabel, time_unit_labels[0])
                                self.share_axes(
                                    axes=axes,
                                    layout=layout,
                                    xs=xs,
                                    ys=ys,
                                    sharex=sharex,
                                    sharey=sharey,
                                    xticks=True,
                                    yticks=True,
                                    xfmt='{x:,.1f}',
                                    yfmt='{x:,.1f}',
                                    xlim=True,
                                    ylim=True,
                                    xlabel=shared_xlabel,
                                    ylabel=shared_ylabel,
                                    collapse_x=collapse_x,
                                    collapse_y=collapse_y)
                                for ax in axes.ravel():
                                    self.apply_grid(ax)
                                fig.align_ylabels()
                                ## update title
                                s = 'Histogram of {} from {} (via {} Bias)'.format(time_difference_id.title(), cluster_id.title(), bias_id.title())
                                # s = 'Frequency of Events from {} (via {} Bias)'.format(cluster_id.title(), bias_id.title())
                                fig.suptitle(s, fontsize=self.titlesize)
                                ## show legend
                                nhandles = len(handles)
                                ncol = nhandles if nhandles > 1 else None
                                if self.is_same_elements(elements=extreme_labels, s='', n=self.n):
                                    leg_title = "{}".format(extreme_labels[0])
                                else:
                                    if self.is_same_elements(elements=alt_extreme_labels, s='', n=self.n):
                                        leg_title = "{}".format(alt_extreme_labels[0])
                                    else:
                                        leg_title = None
                                if leg_title is None:
                                    leg_title = '{}'.format(cluster_id.title())
                                else:
                                    leg_title = '{}\n{}'.format(cluster_id.title(), leg_title)
                                self.subview_legend(
                                    fig=fig,
                                    ax=axes.ravel()[0],
                                    handles=handles,
                                    labels=labels,
                                    textcolor=True,
                                    bottom=0.2,
                                    facecolor='silver',
                                    ncol=ncol,
                                    title=leg_title)
                                ## show / save
                                if save:
                                    savename = 'RegularAnalysis_TemporalClustering_Histogram_EX-{}_{}_{}_{}-BIAS'.format(
                                        extreme_value,
                                        time_difference_id.title().replace(' ', '-'),
                                        cluster_id,
                                        bias_id.title().replace(' ', '-'))
                                    for series in self.series:
                                        cycle_nums = np.unique(series['data']['solar cycle'])
                                        if cycle_nums.size == 1:
                                            cycle_id = "SC-{}".format(cycle_nums[0])
                                        else:
                                            cycle_id = "SC" + "-".join(cycle_nums.astype(str))
                                        savename = '{}_{}_{}_{}'.format(
                                            savename,
                                            cycle_id,
                                            series['identifiers']['event type'].replace(' ', '-'),
                                            series['identifiers']['parameter id'].replace(' ', '-'))
                                    savename = '{}_{}'.format(
                                        savename,
                                        layout)
                                    savename = savename.replace(' ', '_')
                                else:
                                    savename = None
                                self.display_image(fig, savename=savename)

    def view_histogram_of_intra_cluster_times(self, extreme_values=None, cluster_ids='non-lone clusters', show_first_order_bias=False, show_threshold_bias=False, show_baseline=False, facecolors=('steelblue', 'darkred', 'purple', 'green'), time_color='darkorange', sharex=False, sharey=False, collapse_x=False, collapse_y=False, figsize=None, save=False, layout='single'):
        self.view_histogram_of_cluster_timings(
            extreme_values=extreme_values,
            cluster_ids=cluster_ids,
            time_difference_ids='intra-cluster times',
            show_first_order_bias=show_first_order_bias,
            show_threshold_bias=show_threshold_bias,
            show_baseline=show_baseline,
            facecolors=facecolors,
            time_color=time_color,
            sharex=sharex,
            sharey=sharey,
            collapse_x=collapse_x,
            collapse_y=collapse_y,
            figsize=figsize,
            save=save,
            layout=layout)

    def view_histogram_of_intra_cluster_durations(self, extreme_values=None, cluster_ids='non-lone clusters', show_first_order_bias=False, show_threshold_bias=False, show_baseline=False, facecolors=('steelblue', 'darkred', 'purple', 'green'), time_color='darkorange', sharex=False, sharey=False, collapse_x=False, collapse_y=False, figsize=None, save=False, layout='single'):
        self.view_histogram_of_cluster_timings(
            extreme_values=extreme_values,
            cluster_ids=cluster_ids,
            time_difference_ids='intra-cluster durations',
            show_first_order_bias=show_first_order_bias,
            show_threshold_bias=show_threshold_bias,
            show_baseline=show_baseline,
            facecolors=facecolors,
            time_color=time_color,
            sharex=sharex,
            sharey=sharey,
            collapse_x=collapse_x,
            collapse_y=collapse_y,
            figsize=figsize,
            save=save,
            layout=layout)

    def view_histogram_of_inter_cluster_durations(self, extreme_values=None, cluster_ids='non-lone clusters', show_first_order_bias=False, show_threshold_bias=False, show_baseline=False, facecolors=('steelblue', 'darkred', 'purple', 'green'), time_color='darkorange', sharex=False, sharey=False, collapse_x=False, collapse_y=False, figsize=None, save=False, layout='single'):
        self.view_histogram_of_cluster_timings(
            extreme_values=extreme_values,
            cluster_ids=cluster_ids,
            time_difference_ids='inter-cluster durations',
            show_first_order_bias=show_first_order_bias,
            show_threshold_bias=show_threshold_bias,
            show_baseline=show_baseline,
            facecolors=facecolors,
            time_color=time_color,
            sharex=sharex,
            sharey=sharey,
            collapse_x=collapse_x,
            collapse_y=collapse_y,
            figsize=figsize,
            save=save,
            layout=layout)

    def subview_cluster_size_statistics(self, ax_top, ax_mid, ax_btm, series, extreme_value, cluster_id, bias_id, ylabels, i, cluster_freq_color, event_freq_color, rel_prob_color, xs, ys_btm, ys_mid, ys_top):
        cluster_statistics = series['temporal clustering'].extreme_values[extreme_value]['temporal clustering'][bias_id][cluster_id]['relative statistics']
        x = cluster_statistics['cluster size']
        xs.append(np.max(x))
        for j, (ax, ylabel, ys, facecolor) in enumerate(zip([ax_top, ax_mid, ax_btm], ylabels, (ys_top, ys_mid, ys_btm), (event_freq_color, cluster_freq_color, rel_prob_color))):
            y = cluster_statistics[ylabel]
            ys.append(np.max(y) * 1.125)
            ax.bar(
                x,
                y,
                width=1,
                color=facecolor,
                label=ylabel.title().replace(' Of ', ' of ') if i == 0 else None)
            if j in (0, 1):
                if j == 0:
                    text_box = ax.text(
                        0.95,
                        0.95,
                        '${:,}$ Extreme Events'.format(int(np.sum(y))),
                        fontsize=self.textsize,
                        horizontalalignment='right',
                        verticalalignment='top',
                        transform=ax.transAxes)
                else:
                    text_box = ax.text(
                        0.95,
                        0.95,
                        '${:,}$ Clusters'.format(int(np.sum(y))),
                        fontsize=self.textsize,
                        horizontalalignment='right',
                        verticalalignment='top',
                        transform=ax.transAxes)
                text_box.set_bbox(dict(facecolor='silver', alpha=0.25, edgecolor=facecolor))
            if j == 0:
                ax.set_title(series['identifiers']['series id'], fontsize=self.titlesize)
        return xs, ys_btm, ys_mid, ys_top

    def view_cluster_size_statistics(self, extreme_values=None, cluster_ids='non-lone clusters', show_first_order_bias=False, show_threshold_bias=False, show_baseline=False, cluster_freq_color='steelblue', event_freq_color='darkorange', rel_prob_color='green', sharex=False, sharey=False, collapse_x=False, collapse_y=False, figsize=None, save=False, layout='single'):
        if layout is None:
            permutable_layouts = ['single']
            if self.n > 1:
                permutable_layouts.append('square')
            self.view_layout_permutations(
                f=self.view_cluster_size_statistics,
                layouts=permutable_layouts,
                extreme_values=extreme_values,
                cluster_ids=cluster_ids,
                show_first_order_bias=show_first_order_bias,
                show_threshold_bias=show_threshold_bias,
                show_baseline=show_baseline,
                cluster_freq_color=cluster_freq_color,
                event_freq_color=event_freq_color,
                rel_prob_color=rel_prob_color,
                sharex=sharex,
                sharey=sharey,
                collapse_x=collapse_x,
                collapse_y=collapse_y,
                figsize=figsize,
                save=save)
        elif layout == 'single':
            for series in self.series:
                cls = deepcopy(self.cls)
                visualizer = cls(
                    series=[series],
                    savedir=self.savedir)
                visualizer.view_cluster_size_statistics(
                    extreme_values=extreme_values,
                    cluster_ids=cluster_ids,
                    show_first_order_bias=show_first_order_bias,
                    show_threshold_bias=show_threshold_bias,
                    show_baseline=show_baseline,
                    cluster_freq_color=cluster_freq_color,
                    event_freq_color=event_freq_color,
                    rel_prob_color=rel_prob_color,
                    sharex=sharex,
                    sharey=sharey,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y,
                    figsize=figsize,
                    save=save,
                    layout='vertical')
        else:
            ## verify inputs
            bias_ids = ('first-order', 'threshold', 'baseline')
            conditions = np.array([show_first_order_bias, show_threshold_bias, show_baseline])
            nconditions = np.sum(conditions)
            if nconditions < 1:
                raise ValueError("input at least one of the following: show_first_order_bias, show_threshold_bias, show_baseline")
            if layout == 'vertical':
                if self.n > 1:
                    raise ValueError("layout='vertical' for this method will only work for one series")
            elif layout != 'square':
                raise ValueError("invalid layout for this method: {}".format(layout))
            ylabels = ('number of events', 'number of clusters', 'relative probability')
            (ytop_label, ymid_label, ybtm_label) = ylabels
            if not isinstance(cluster_ids, (tuple, list, np.ndarray)):
                cluster_ids = [cluster_ids]
            if extreme_values is None:
                extreme_values = set(self.series[0]['temporal clustering'].extreme_values)
                if self.n > 1:
                    for i in range(1, self.n):
                        extreme_values = extreme_values.intersection(
                            set(self.series[i]['temporal clustering'].extreme_values))
                extreme_values = np.sort(list(extreme_values))
                if extreme_values.size == 0:
                    raise ValueError("could not find extreme values common to all series")
            elif not isinstance(extreme_values, (tuple, list, np.ndarray)):
                extreme_values = [extreme_values]
            for extreme_value in extreme_values:
                for cluster_id in cluster_ids:
                    for bias_id, bias_condition in zip(bias_ids, conditions):
                        if bias_condition:
                            extreme_labels, alt_extreme_labels = [], []
                            shared_xlabel = 'Cluster Size'
                            xs, ys_top, ys_mid, ys_btm = [0], [0], [0], [0]
                            handles, labels = [], []
                            ## get figure and axes
                            kws = {
                                'nrows' : len(ylabels),
                                'ncols' : self.n}
                            fig, axes = plt.subplots(figsize=figsize, **kws)
                            if layout == 'vertical':
                                if self.n != 1:
                                    raise ValueError("layout='vertical' for this method will only work for one series")
                                top_axes, mid_axes, btm_axes = np.array([axes[0]]), np.array([axes[1]]), np.array([axes[2]])
                            elif layout == 'square':
                                if self.n < 2:
                                    raise ValueError("layout='square' works for more than one series; try using layout='vertical' instead")
                                top_axes, mid_axes, btm_axes = axes[0, :], axes[1, :], axes[2, :]
                            else:
                                raise ValueError("invalid layout for this method: {}".format(layout))
                            ## initialize plot
                            for i, (ax_top, ax_mid, ax_btm, series) in enumerate(zip(top_axes.ravel(), mid_axes.ravel(), btm_axes.ravel(), self.series)):
                                extreme_label = self.get_extreme_label(
                                    extreme_parameter=series['temporal clustering'].extreme_parameter,
                                    extreme_condition=series['temporal clustering'].extreme_condition,
                                    extreme_value=extreme_value,
                                    parameter_mapping=series['parameter mapping'],
                                    unit_mapping=series['unit mapping'])
                                alt_extreme_label = self.get_generalized_extreme_label(
                                    extreme_parameter=series['temporal clustering'].extreme_parameter,
                                    extreme_condition=series['temporal clustering'].extreme_condition,
                                    extreme_value=extreme_value,
                                    parameter_mapping=series['parameter mapping'],
                                    unit_mapping=series['unit mapping'],
                                    generalized_parameter_mapping=series['generalized parameter mapping'])
                                extreme_labels.append(extreme_label)
                                alt_extreme_labels.append(alt_extreme_label)
                                xs, ys_btm, ys_mid, ys_top = self.subview_cluster_size_statistics(
                                    ax_top=ax_top,
                                    ax_mid=ax_mid,
                                    ax_btm=ax_btm,
                                    series=series,
                                    extreme_value=extreme_value,
                                    cluster_id=cluster_id,
                                    bias_id=bias_id,
                                    ylabels=ylabels,
                                    i=i,
                                    cluster_freq_color=cluster_freq_color,
                                    event_freq_color=event_freq_color,
                                    rel_prob_color=rel_prob_color,
                                    xs=xs,
                                    ys_btm=ys_btm,
                                    ys_mid=ys_mid,
                                    ys_top=ys_top)
                                for j, ax in enumerate((ax_top, ax_mid, ax_btm)):
                                    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
                                    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
                                    if i == 0:
                                        _handles, _labels = ax.get_legend_handles_labels()
                                        handles.extend(_handles)
                                        labels.extend(_labels)
                                    ax.set_xlabel(shared_xlabel, fontsize=self.labelsize)
                                    ax.set_ylabel(ylabels[j].title(), fontsize=self.labelsize)
                            ## update x-axes for all sub-plots
                            largest_cluster_size = np.max(xs)
                            minor_xticks = np.arange(np.max(xs)).astype(int)
                            if largest_cluster_size > 10:
                                major_xticks = np.array(minor_xticks[::5])
                            else:
                                minor_xticks = np.arange(np.max(xs)).astype(int)
                                major_xticks = np.array(minor_xticks[::2])
                            self.share_axes(
                                axes=axes,
                                layout=layout,
                                xs=xs,
                                ys=[],
                                sharex=True,
                                sharey=False,
                                xlim=[minor_xticks[0], minor_xticks[-1]],
                                xticks=[major_xticks, minor_xticks],
                                xfmt='{x:,.0f}',
                                xlabel=shared_xlabel.title(),
                                collapse_x=collapse_x)
                            ## share y- axes for top-row subplots
                            self.share_axes(
                                axes=top_axes,
                                layout='horizontal',
                                xs=xs,
                                ys=ys_top,
                                sharex=False,
                                sharey=True,
                                yticks=True,
                                yfmt='{x:,.0f}',
                                ylabel=ytop_label.title(),
                                ylim=True,
                                collapse_y=collapse_y)
                            ## share y- axes for mid-row subplots
                            self.share_axes(
                                axes=mid_axes,
                                layout='horizontal',
                                xs=xs,
                                ys=ys_mid,
                                sharex=False,
                                sharey=True,
                                yticks=True,
                                yfmt='{x:,.0f}',
                                ylabel=ymid_label.title(),
                                ylim=True,
                                collapse_y=collapse_y)
                            ## share y- axes for bottom-row subplots
                            self.share_axes(
                                axes=btm_axes,
                                layout='horizontal',
                                xs=xs,
                                ys=ys_btm,
                                sharex=False,
                                sharey=True,
                                yticks=True,
                                yfmt=None,
                                ylabel=ybtm_label.title(),
                                ylim=[0, 1],
                                collapse_y=collapse_y)
                            for ax in axes.ravel():
                                self.apply_grid(ax)
                            hspace = 0.425 if 'vertical' in layout else 0.3
                            fig.subplots_adjust(hspace=hspace)
                            fig.align_ylabels()
                            ## update title
                            s = 'Temporal Clusters: Relative Statistics (via {} Bias)'.format(bias_id.title())
                            fig.suptitle(s, fontsize=self.titlesize)
                            ## show legend
                            nhandles = len(handles)
                            ncol = nhandles if nhandles > 1 else None
                            if self.is_same_elements(elements=extreme_labels, s='', n=self.n):
                                leg_title = "{}".format(extreme_labels[0])
                            else:
                                if self.is_same_elements(elements=alt_extreme_labels, s='', n=self.n):
                                    leg_title = "{}".format(alt_extreme_labels[0])
                                else:
                                    leg_title = None
                            if leg_title is None:
                                leg_title = '{}'.format(cluster_id.title())
                            else:
                                leg_title = '{}\n{}'.format(cluster_id.title(), leg_title)
                            self.subview_legend(
                                fig=fig,
                                ax=axes.ravel()[0],
                                handles=handles,
                                labels=labels,
                                textcolor=True,
                                facecolor='silver',
                                bottom=0.2,
                                ncol=ncol,
                                title=leg_title)
                            ## show / save
                            if save:
                                savename = 'RegularAnalysis_TemporalClustering_RelStats_EX-{}_{}_{}-BIAS'.format(
                                    extreme_value,
                                    cluster_id.title().replace(' ', '-'),
                                    bias_id.title().replace(' ', '-'))
                                for series in self.series:
                                    cycle_nums = np.unique(series['data']['solar cycle'])
                                    if cycle_nums.size == 1:
                                        cycle_id = "SC-{}".format(cycle_nums[0])
                                    else:
                                        cycle_id = "SC" + "-".join(cycle_nums.astype(str))
                                    savename = '{}_{}_{}_{}'.format(
                                        savename,
                                        cycle_id,
                                        series['identifiers']['event type'].replace(' ', '-'),
                                        series['identifiers']['parameter id'].replace(' ', '-'))
                                savename = '{}_{}'.format(
                                    savename,
                                    layout)
                                savename = savename.replace(' ', '_')
                            else:
                                savename = None
                            self.display_image(fig, savename=savename)

    def view_alternating_clusters(self, parameter='cluster size', statistic='mean', extreme_values=None, cluster_ids='non-lone clusters', show_first_order_bias=False, show_threshold_bias=False, show_baseline=False, facecolors=('darkorange', 'darkgreen'), tfmt='%Y-%m', sharex=False, sharey=False, collapse_x=False, collapse_y=False, figsize=None, save=False, layout='single'):
        if layout is None:
            permutable_layouts = ['single']
            if (self.n > 1) and (self.n % 2 == 0):
                permutable_layouts.append('square')
            if self.n > 1:
                permutable_layouts.append('vertical')
                if self.n < 4:
                    permutable_layouts.append('horizontal')
            self.view_layout_permutations(
                f=self.view_alternating_clusters,
                layouts=permutable_layouts,
                parameter=parameter,
                statistic=statistic,
                extreme_values=extreme_values,
                cluster_ids=cluster_ids,
                show_first_order_bias=show_first_order_bias,
                show_threshold_bias=show_threshold_bias,
                show_baseline=show_baseline,
                facecolors=facecolors,
                tfmt=tfmt,
                sharex=sharex,
                sharey=sharey,
                collapse_x=collapse_x,
                collapse_y=collapse_y,
                figsize=figsize,
                save=save)
        elif layout == 'single':
            for series in self.series:
                cls = deepcopy(self.cls)
                visualizer = cls(
                    series=[series],
                    savedir=self.savedir)
                visualizer.view_alternating_clusters(
                    parameter=parameter,
                    statistic=statistic,
                    extreme_values=extreme_values,
                    cluster_ids=cluster_ids,
                    show_first_order_bias=show_first_order_bias,
                    show_threshold_bias=show_threshold_bias,
                    show_baseline=show_baseline,
                    facecolors=facecolors,
                    tfmt=tfmt,
                    sharex=sharex,
                    sharey=sharey,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y,
                    figsize=figsize,
                    save=save,
                    layout='overlay')
        else:
            ## verify inputs
            if (layout == 'overlay') and (self.n > 1):
                raise ValueError("{} is not a valid layout for this method".format(layout))
            if not isinstance(facecolors, (tuple, list, np.ndarray)):
                raise ValueError("invalid type(facecolors): {}".format(type(facecolors)))
            nc = len(facecolors)
            if nc != 2:
                raise ValueError("invalid number of facecolors: {}".format(nc))
            alternate_colors = np.array(facecolors)
            bias_ids = ('first-order', 'threshold', 'baseline')
            conditions = np.array([show_first_order_bias, show_threshold_bias, show_baseline])
            nconditions = np.sum(conditions)
            if nconditions < 1:
                raise ValueError("input at least one of the following: show_first_order_bias, show_threshold_bias, show_baseline")
            if not isinstance(cluster_ids, (tuple, list, np.ndarray)):
                cluster_ids = [cluster_ids]
            if extreme_values is None:
                extreme_values = set(self.series[0]['temporal clustering'].extreme_values)
                if self.n > 1:
                    for i in range(1, self.n):
                        extreme_values = extreme_values.intersection(
                            set(self.series[i]['temporal clustering'].extreme_values))
                extreme_values = np.sort(list(extreme_values))
                if extreme_values.size == 0:
                    raise ValueError("could not find extreme values common to all series")
            elif not isinstance(extreme_values, (tuple, list, np.ndarray)):
                extreme_values = [extreme_values]
            datetime_configuration = DateTimeConfiguration()
            for extreme_value in extreme_values:
                for cluster_id in cluster_ids:
                    for bias_id, bias_condition in zip(bias_ids, conditions):
                        if bias_condition:
                            xs, ys = [], [0]
                            shared_xlabel = 'Date'
                            extreme_labels, alt_extreme_labels, unit_labels = [], [], []
                            parameter_labels = []
                            handles, labels = dict(), dict()
                            ## get figure and axes
                            kws = self.get_number_of_figure_rows_and_columns(self.n, layout)
                            fig, axes = plt.subplots(figsize=figsize, **kws)
                            if layout == 'overlay':
                                axes = np.array([axes])
                            for i, (ax, series) in enumerate(zip(axes.ravel(), self.series)):
                                xdata, ydata, wdata = [], [], []
                                extreme_label = self.get_extreme_label(
                                    extreme_parameter=series['temporal clustering'].extreme_parameter,
                                    extreme_condition=series['temporal clustering'].extreme_condition,
                                    extreme_value=extreme_value,
                                    parameter_mapping=series['parameter mapping'],
                                    unit_mapping=series['unit mapping'])
                                alt_extreme_label = self.get_generalized_extreme_label(
                                    extreme_parameter=series['temporal clustering'].extreme_parameter,
                                    extreme_condition=series['temporal clustering'].extreme_condition,
                                    extreme_value=extreme_value,
                                    parameter_mapping=series['parameter mapping'],
                                    unit_mapping=series['unit mapping'],
                                    generalized_parameter_mapping=series['generalized parameter mapping'])
                                unit_label = series['unit mapping'][series['temporal clustering'].extreme_parameter]
                                if parameter == 'cluster size':
                                    parameter_labels.append('Cluster Size')
                                else:
                                    parameter_labels.append('{} [{}]'.format(
                                        parameter.title(),
                                        series['unit mapping'][parameter]))
                                extreme_labels.append(extreme_label)
                                alt_extreme_labels.append(alt_extreme_label)
                                unit_labels.append(unit_label)
                                cluster_searcher = series['temporal clustering'].extreme_values[extreme_value]['temporal clustering'][bias_id][cluster_id]['cluster searcher']
                                nclusters = len(cluster_searcher.clusters['datetime'])
                                nevents = cluster_searcher.events['datetime'].size
                                color_indices = np.zeros(nclusters).astype(int)
                                color_indices[1::2] = 1
                                mod_colors = alternate_colors[color_indices]
                                seconds_to_days = datetime_configuration.relative_time_scales['second'] / datetime_configuration.relative_time_scales['day']
                                xs.append(series['data']['datetime'][0])
                                xs.append(series['data']['datetime'][-1])
                                if parameter == 'cluster size':
                                    for dts in cluster_searcher.clusters['datetime']:
                                        bin_width = (dts[-1] - dts[0]).total_seconds() * seconds_to_days
                                        xdata.append(dts[0])
                                        wdata.append(bin_width)
                                        ydata.append(len(dts))
                                    ys.append(np.nanmax(ydata) * 1.125)
                                else:
                                    ymin_data, ymax_data = [], []
                                    f = StatisticsConfiguration([]).dispatch_func(statistic)
                                    statistics_configuration = StatisticsConfiguration([])
                                    for dts, parameter_values in zip(cluster_searcher.clusters['datetime'], cluster_searcher.clusters[parameter]):
                                        bin_width = (dts[-1] - dts[0]).total_seconds() * seconds_to_days
                                        xdata.append(dts[0])
                                        wdata.append(bin_width)
                                        ydata.append(f(parameter_values))
                                bar_handles = ax.bar(
                                    xdata,
                                    ydata,
                                    width=wdata,
                                    align='edge',
                                    color=mod_colors)
                                ax = self.subview_datetime_axis(
                                    ax=ax,
                                    axis='x',
                                    major_interval=12,
                                    minor_interval=1,
                                    sfmt=tfmt)
                                handles[i] = [tuple(bar_handles[:2])]
                                if self.n == 1:
                                    labels[i] = '${:,}$ Events from ${:,}$ Clusters'.format(nevents, nclusters)
                                else:
                                    labels[i] = '{}\n${:,}$ Events from ${:,}$ Clusters'.format(series['identifiers']['series id'], nevents, nclusters)
                                ax.set_xlabel(shared_xlabel, fontsize=self.labelsize)
                                ax.set_ylabel(parameter.title(), fontsize=self.labelsize)
                                ax.set_title(series['identifiers']['series id'], fontsize=self.titlesize)
                                ax.tick_params(axis='x', labelrotation=15)
                                ax.tick_params(axis='both', labelsize=8)
                                ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
                            if self.is_same_elements(elements=parameter_labels, s='', n=self.n):
                                shared_ylabel = parameter_labels[0]
                            else:
                                shared_ylabel = parameter.title()
                            ## update axes
                            self.share_axes(
                                axes=axes,
                                layout=layout,
                                xs=xs,
                                ys=ys,
                                sharex=sharex,
                                sharey=sharey,
                                xlim=[np.nanmin(xs), np.nanmax(xs)],
                                ylim=True,
                                xticks=False,
                                yticks=True,
                                xfmt=None,
                                yfmt='{x:,.1f}',
                                xlabel=shared_xlabel,
                                ylabel=shared_ylabel,
                                collapse_x=collapse_x,
                                collapse_y=collapse_y)
                            for ax in axes.ravel():
                                ax.tick_params(
                                    axis='x',
                                    which='both',
                                    labelsize=self.ticksize,
                                    rotation=15)
                                ax.tick_params(
                                    axis='y',
                                    which='both',
                                    labelsize=self.ticksize)
                            fig.align_ylabels()
                            for ax in axes.ravel():
                                self.apply_grid(ax)
                            ## update title
                            s = 'Chronological Clusters (via {} Bias)'.format(cluster_id.title(), bias_id.title())
                            fig.suptitle(s, fontsize=self.titlesize)
                            ## update legend
                            if self.is_same_elements(elements=extreme_labels, s='', n=self.n):
                                leg_title = "{}".format(extreme_labels[0])
                            else:
                                if self.is_same_elements(elements=alt_extreme_labels, s='', n=self.n):
                                    leg_title = "{}".format(alt_extreme_labels[0])
                                else:
                                    leg_title = None
                            if leg_title is None:
                                leg_title = '{}'.format(cluster_id.title())
                            else:
                                leg_title = '{}\n{}'.format(cluster_id.title(), leg_title)
                            fig.subplots_adjust(bottom=0.2, hspace=0.35)
                            self.subview_legend(
                                fig=fig,
                                ax=axes.ravel()[0],
                                handles=[tuple(handles[i][:2]) for i in range(self.n)],
                                labels=[labels[i] for i in range(self.n)],
                                ncol=self.n if self.n > 1 else None,
                                title=leg_title,
                                textcolor='k',
                                facecolor='silver',
                                edgecolor='k',
                                titlecolor='k',
                                handler_map={tuple : HandlerTuple(None)})
                            ## show / save
                            if save:
                                savename = 'RegularAnalysis_TemporalClustering_ChronoAlt_EX-{}_{}_{}-BIAS'.format(
                                    extreme_value,
                                    cluster_id,
                                    bias_id.title().replace(' ', '-'))
                                for series in self.series:
                                    cycle_nums = np.unique(series['data']['solar cycle'])
                                    if cycle_nums.size == 1:
                                        cycle_id = "SC-{}".format(cycle_nums[0])
                                    else:
                                        cycle_id = "SC" + "-".join(cycle_nums.astype(str))
                                    savename = '{}_{}_{}_{}'.format(
                                        savename,
                                        cycle_id,
                                        series['identifiers']['event type'].replace(' ', '-'),
                                        series['identifiers']['parameter id'].replace(' ', '-'))
                                savename = '{}_{}'.format(
                                    savename,
                                    layout)
                                savename = savename.replace(' ', '_')
                            else:
                                savename = None
                            self.display_image(fig, savename=savename)

    def view_cluster_parameterization_moment_estimators_and_time_thresholds(self, extreme_values=None, cluster_ids='non-lone clusters', show_first_order_bias=False, show_threshold_bias=False, show_baseline=False, threshold_color='steelblue', first_order_color='darkorange', baseline_color='green', sharex=False, sharey=False, collapse_x=False, collapse_y=False, figsize=None, save=False, layout='single'):
        if layout is None:
            permutable_layouts = ['single']
            if self.n > 1:
                permutable_layouts.append('square')
            self.view_layout_permutations(
                f=self.view_cluster_parameterization_moment_estimators_and_time_thresholds,
                layouts=permutable_layouts,
                extreme_values=extreme_values,
                cluster_ids=cluster_ids,
                show_first_order_bias=show_first_order_bias,
                show_threshold_bias=show_threshold_bias,
                show_baseline=show_baseline,
                threshold_color=threshold_color,
                first_order_color=first_order_color,
                baseline_color=baseline_color,
                sharex=sharex,
                sharey=sharey,
                collapse_x=collapse_x,
                collapse_y=collapse_y,
                figsize=figsize,
                save=save)
        elif layout == 'single':
            for series in self.series:
                cls = deepcopy(self.cls)
                visualizer = cls(
                    series=[series],
                    savedir=self.savedir)
                visualizer.view_cluster_parameterization_moment_estimators_and_time_thresholds(
                    extreme_values=extreme_values,
                    cluster_ids=cluster_ids,
                    show_first_order_bias=show_first_order_bias,
                    show_threshold_bias=show_threshold_bias,
                    show_baseline=show_baseline,
                    threshold_color=threshold_color,
                    first_order_color=first_order_color,
                    baseline_color=baseline_color,
                    sharex=sharex,
                    sharey=sharey,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y,
                    figsize=figsize,
                    save=save,
                    layout='vertical')
        else:
            ## verify inputs
            bias_ids = ('first-order', 'threshold', 'baseline')
            conditions = np.array([show_first_order_bias, show_threshold_bias, show_baseline])
            nconditions = np.sum(conditions)
            if nconditions < 1:
                raise ValueError("input at least one of the following: show_first_order_bias, show_threshold_bias, show_baseline")
            if layout == 'vertical':
                if self.n > 1:
                    raise ValueError("layout='vertical' for this method will only work for one series")
            elif layout != 'square':
                raise ValueError("invalid layout for this method: {}".format(layout))
            ylabels = (r'Moment Estimator $\hat\theta$', r'Time Threshold $T_C$')
            (ytop_label, ybtm_label) = ylabels
            if not isinstance(cluster_ids, (tuple, list, np.ndarray)):
                cluster_ids = [cluster_ids]
            if extreme_values is None:
                extreme_values = set(self.series[0]['temporal clustering parameterization'].extreme_values)
                if self.n > 1:
                    for i in range(1, self.n):
                        extreme_values = extreme_values.intersection(
                            set(self.series[i]['temporal clustering parameterization'].extreme_values))
                extreme_values = np.sort(list(extreme_values))
                if extreme_values.size == 0:
                    raise ValueError("could not find extreme values common to all series")
            elif not isinstance(extreme_values, (tuple, list, np.ndarray)):
                extreme_values = [extreme_values]
            for cluster_id in cluster_ids:
                ## get figure and axes
                kws = {
                    'nrows' : len(ylabels),
                    'ncols' : self.n}
                fig, axes = plt.subplots(figsize=figsize, **kws)
                if layout == 'vertical':
                    if self.n != 1:
                        raise ValueError("layout='vertical' for this method will only work for one series")
                    top_axes, btm_axes = np.array([axes[0]]), np.array([axes[1]])
                elif layout == 'square':
                    if self.n < 2:
                        raise ValueError("layout='square' works for more than one series; try using layout='vertical' instead")
                    top_axes, btm_axes = axes[0, :], axes[1, :]
                else:
                    raise ValueError("invalid layout for this method: {}".format(layout))
                for bias_id, bias_condition, bias_color in zip(bias_ids, conditions, (first_order_color, threshold_color, baseline_color)):
                    if bias_condition:
                        xs, ys_top, ys_btm = [np.min(extreme_values), np.max(extreme_values)], [0, 1], [0.1]
                        xlabels, ybtm_labels, unit_labels = [], [], []
                        handles, labels = [], []
                        ## initialize plot
                        for i, (ax_top, ax_btm, series) in enumerate(zip(top_axes.ravel(), btm_axes.ravel(), self.series)):
                            xlabel = "Extreme Values \n{} [{}]".format(
                                series['parameter mapping'][series['temporal clustering parameterization'].extreme_parameter],
                                series['unit mapping'][series['temporal clustering parameterization'].extreme_parameter])
                            unit_label = '{}'.format(series['unit mapping'][series['temporal clustering parameterization'].extreme_parameter])
                            elapsed_unit = series['identifiers']['elapsed unit']
                            _ybtm_label = "{} [{}]".format(
                                ybtm_label,
                                elapsed_unit)
                            xlabels.append(xlabel)
                            unit_labels.append(unit_label)
                            ybtm_labels.append(_ybtm_label)
                            thetas = series['temporal clustering parameterization'].get_parameterized_quantity(
                                quantity='moment estimator',
                                bias_id=bias_id)['values']
                            times = series['temporal clustering parameterization'].get_parameterized_quantity(
                                quantity='time threshold',
                                bias_id=bias_id)['values']
                            ys_btm.append(np.nanmin(times))
                            ys_btm.append(np.nanmax(times))
                            ax_top.plot(
                                extreme_values,
                                thetas,
                                color=bias_color,
                                alpha=0.7,
                                label='{} Bias'.format(bias_id.title()))
                            ax_btm.plot(
                                extreme_values,
                                times,
                                color=bias_color,
                                alpha=0.7)
                            ax_btm.set_xlabel(xlabel, fontsize=self.labelsize)
                            ax_top.set_ylabel(ytop_label, fontsize=self.labelsize)
                            ax_btm.set_ylabel(_ybtm_label, fontsize=self.labelsize)
                            ax_top.set_title(series['identifiers']['series id'], fontsize=self.titlesize)
                _handles, _labels = top_axes.ravel()[0].get_legend_handles_labels()
                handles.extend(_handles)
                labels.extend(_labels)
                ## update axes
                if self.is_same_elements(elements=xlabels, s='', n=self.n):
                    shared_xlabel = '{}'.format(xlabels[0])
                else:
                    if self.is_same_elements(elements=unit_labels, s='', n=self.n):
                        shared_xlabel = 'Extreme Values [{}]'.format(unit_labels[0])
                    else:
                        shared_xlabel = 'Extreme Values'
                if self.is_same_elements(elements=ybtm_labels, s='', n=self.n):
                    shared_ybtm_label = '{}'.format(ybtm_labels[0])
                else:
                    shared_ybtm_label = '{}'.format(ybtm_label)
                self.share_axes(
                    axes=top_axes,
                    layout='horizontal',
                    xs=xs,
                    ys=ys_top,
                    sharex=sharex,
                    sharey=sharey,
                    xticks=True,
                    yticks=True,
                    xlim=True,
                    ylim=True,
                    xlabel='',
                    ylabel=ytop_label,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y)
                self.share_axes(
                    axes=btm_axes,
                    layout='horizontal',
                    xs=xs,
                    ys=ys_btm,
                    basey=5,
                    sharex=sharex,
                    sharey=sharey,
                    xticks=True,
                    yticks=True,
                    xlim=True,
                    ylim=True,
                    xlabel=None,
                    ylabel=shared_ybtm_label,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y)
                self.share_axes(
                    axes=axes,
                    layout=layout,
                    xs=xs,
                    ys=[],
                    sharex=sharex,
                    sharey=False,
                    xticks=True,
                    yticks=False,
                    xlim=True,
                    ylim=False,
                    xlabel=shared_xlabel,
                    ylabel=None,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y)
                fig.subplots_adjust(hspace=0.225)
                fig.align_ylabels()
                ## update title
                s = r'Temporal Clusters: Moment Estimators & Time-Thresholds by Extreme Value'
                fig.suptitle(s, fontsize=self.titlesize)
                ## show legend
                nhandles = len(handles)
                ncol = nhandles if nhandles > 1 else None
                self.subview_legend(
                    fig=fig,
                    ax=axes.ravel()[0],
                    handles=handles,
                    labels=labels,
                    textcolor=True,
                    facecolor='silver',
                    bottom=0.2,
                    ncol=ncol,
                    title='Bias Comparison' if nhandles > 1 else None)
                ## show / save
                if save:
                    savename = 'RegularAnalysis_TemporalClusteringParameterization_Theta-Time_{}_{}-Bias'.format(
                        cluster_id.title().replace(' ', '-'),
                        "-and-".join([bias_id for bias_id, condition in zip(bias_ids, conditions) if condition]))
                    for series in self.series:
                        cycle_nums = np.unique(series['data']['solar cycle'])
                        if cycle_nums.size == 1:
                            cycle_id = "SC-{}".format(cycle_nums[0])
                        else:
                            cycle_id = "SC" + "-".join(cycle_nums.astype(str))
                        savename = '{}_{}_{}_{}'.format(
                            savename,
                            cycle_id,
                            series['identifiers']['event type'].replace(' ', '-'),
                            series['identifiers']['parameter id'].replace(' ', '-'))
                    savename = '{}_{}'.format(
                        savename,
                        layout)
                    savename = savename.replace(' ', '_')
                else:
                    savename = None
                self.display_image(fig, savename=savename)

    def view_cluster_parameterization_quantity(self, extreme_values=None, quantities=None, cluster_ids='non-lone clusters', show_first_order_bias=False, show_threshold_bias=False, show_baseline=False, show_min_and_max=False, central_statistic='mean', error_statistic='standard deviation', sharex=False, sharey=False, collapse_x=False, collapse_y=False, figsize=None, save=False, layout='single'):
        raise ValueError("debug me")
        if layout is None:
            permutable_layouts = ['single']
            if (self.n > 2) and (self.n % 2 == 0):
                permutable_layouts.append('square')
            self.view_layout_permutations(
                f=self.view_cluster_parameterization_quantity,
                layouts=permutable_layouts,
                extreme_values=extreme_values,
                quantities=quantities,
                cluster_ids=cluster_ids,
                show_first_order_bias=show_first_order_bias,
                show_threshold_bias=show_threshold_bias,
                show_baseline=show_baseline,
                show_min_and_max=show_min_and_max,
                central_statistic=central_statistic,
                error_statistic=error_statistic,
                sharex=sharex,
                sharey=sharey,
                collapse_x=collapse_x,
                collapse_y=collapse_y,
                figsize=figsize,
                save=save)
        elif layout == 'single':
            for series in self.series:
                cls = deepcopy(self.cls)
                visualizer = cls(
                    series=[series],
                    savedir=self.savedir)
                visualizer.view_cluster_parameterization_quantity(
                    extreme_values=extreme_values,
                    quantities=quantities,
                    cluster_ids=cluster_ids,
                    show_first_order_bias=show_first_order_bias,
                    show_threshold_bias=show_threshold_bias,
                    show_baseline=show_baseline,
                    show_min_and_max=show_min_and_max,
                    central_statistic=central_statistic,
                    error_statistic=error_statistic,
                    sharex=sharex,
                    sharey=sharey,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y,
                    figsize=figsize,
                    save=save,
                    layout='overlay')
        else:
            if (layout == 'overlay') and (self.n > 1):
                raise ValueError("layout='overlay' is invalid for this method since self.n > 1")
            available_quantities = ['intra-cluster times', 'intra-cluster durations', 'inter-cluster durations', 'number of extreme events', 'number of clusters', 'cluster size']
            quantities_without_statistics = ['number of extreme events', 'number of clusters']
            if quantities is None:
                quantities = tuple(available_quantities)
            elif isinstance(quantities, str):
                quantities = [quantities]
            elif not isinstance(quantities, (tuple, list, np.ndarray)):
                raise ValueError("invalid type(quantities): {}".format(type(quantities)))
            if central_statistic not in (None, 'mean', 'median'):
                raise ValueError("invalid central_statistic: {}".format(central_statistic))
            if error_statistic not in (None, 'standard deviation', 'standard error'):
                raise ValueError("invalid error_statistic: {}".format(error_statistic))
            if isinstance(cluster_ids, str):
                cluster_ids = [cluster_ids]
            elif not isinstance(cluster_ids, (tuple, list, np.ndarray)):
                raise ValueError("invalid type(cluster_ids): {}".format(type(cluster_ids)))
            if extreme_values is None:
                extreme_values = set(self.series[0]['temporal clustering parameterization'].extreme_values)
                if self.n > 1:
                    for i in range(1, self.n):
                        extreme_values = extreme_values.intersection(
                            set(self.series[i]['temporal clustering parameterization'].extreme_values))
                extreme_values = np.sort(list(extreme_values))
                if extreme_values.size == 0:
                    raise ValueError("could not find extreme values common to all series")
            elif not isinstance(extreme_values, (tuple, list, np.ndarray)):
                extreme_values = [extreme_values]
            for quantity in quantities:
                if quantity not in available_quantities:
                    raise ValueError("invalid quantity: {}".format(quantity))
                for cluster_id in cluster_ids:
                    for bias_id, show_bias in zip(['first-order', 'threshold', 'baseline'], [show_first_order_bias, show_threshold_bias, show_baseline]):
                        if show_bias:
                            xlabels, ylabels, time_unit_labels, parameter_unit_labels = [], [], [], []
                            xs, ys = [np.nanmin(extreme_values), np.nanmax(extreme_values)], [0]
                            ## get figure and axes
                            kws = self.get_number_of_figure_rows_and_columns(self.n, layout)
                            fig, axes = plt.subplots(figsize=figsize, **kws)
                            if not isinstance(axes, np.ndarray):
                                axes = np.array([axes])
                            for i, (ax, series) in enumerate(zip(axes.ravel(), self.series)):
                                time_unit_label = series['identifiers']['elapsed unit']
                                parameter_unit_label = '{}'.format(series['unit mapping'][series['temporal clustering parameterization'].extreme_parameter])
                                xlabel = "Extreme Values \n{} [{}]".format(
                                    series['parameter mapping'][series['temporal clustering parameterization'].extreme_parameter],
                                    series['unit mapping'][series['temporal clustering parameterization'].extreme_parameter])
                                ylabel = '{}'.format(quantity.title() if quantity not in ('intra-cluster times', 'intra-cluster durations', 'inter-cluster durations') else '{} [{}]'.format(quantity.title(), time_unit_label))
                                time_unit_labels.append(time_unit_label)
                                parameter_unit_labels.append(parameter_unit_label)
                                xlabels.append(xlabel)
                                ylabels.append(ylabel)
                                ax.set_title(series['identifiers']['series id'], fontsize=self.titlesize)
                                ax.set_xlabel(xlabel, fontsize=self.labelsize)
                                ax.set_ylabel(ylabel, fontsize=self.labelsize)
                                parameterized_data = series['temporal clustering parameterization'].get_parameterized_quantity(
                                    quantity=quantity,
                                    bias_id=bias_id,
                                    cluster_id=cluster_id,
                                    bias=False,
                                    fisher=False,
                                    ddof=0)
                                if quantity in quantities_without_statistics:
                                    ys.append(np.nanmin(parameterized_data['values']))
                                    ys.append(np.nanmax(parameterized_data['values']))
                                    ax.scatter(
                                        extreme_values,
                                        parameterized_data['values'],
                                        color='steelblue',
                                        label='{}'.format(quantity.title()),
                                        marker='.',
                                        s=2)
                                else:
                                    if central_statistic is not None:
                                        ax.scatter(
                                            extreme_values,
                                            parameterized_data[central_statistic],
                                            color='darkorange',
                                            label='{} {}'.format(central_statistic.title(), quantity.title()),
                                            marker='.',
                                            s=2)
                                    if error_statistic is not None:
                                        ys.append(np.nanmin(parameterized_data[central_statistic] + parameterized_data[error_statistic]))
                                        ys.append(np.nanmax(parameterized_data[central_statistic] + parameterized_data[error_statistic]))
                                        ys.append(np.nanmin(parameterized_data[central_statistic] - parameterized_data[error_statistic]))
                                        ys.append(np.nanmax(parameterized_data[central_statistic] - parameterized_data[error_statistic]))
                                        ax.errorbar(
                                            extreme_values,
                                            parameterized_data[central_statistic],
                                            parameterized_data[error_statistic],
                                            alpha=0.7,
                                            capsize=5,
                                            ecolor='steelblue',
                                            fmt='none',
                                            label=error_statistic.title())
                                    if show_min_and_max:
                                        ys.append(np.nanmin(parameterized_data['minimum']))
                                        ys.append(np.nanmax(parameterized_data['maximum']))
                                        ax.scatter(
                                            extreme_values,
                                            parameterized_data['minimum'],
                                            color='k',
                                            label='min/mmax {}'.format(quantity.title()),
                                            marker='_',
                                            s=2)
                                        ax.scatter(
                                            extreme_values,
                                            parameterized_data['maximum'],
                                            color='k',
                                            marker='_',
                                            s=2)
                            ## update axes
                            if self.is_same_elements(elements=xlabels, s='', n=self.n):
                                shared_xlabel = '{}'.format(xlabels[0])
                            else:
                                if self.is_same_elements(elements=parameter_unit_labels, s='', n=self.n):
                                    shared_xlabel = 'Extreme Values [{}]'.format(parameter_unit_labels[0])
                                else:
                                    shared_xlabel = 'Extreme Values'
                            if self.is_same_elements(elements=ylabels, s='', n=self.n):
                                shared_ylabel = '{}'.format(ylabels[0])
                            else:
                                if self.is_same_elements(elements=time_unit_labels, s='', n=self.n) and (quantity in ['intra-cluster times', 'intra-cluster durations', 'inter-cluster durations']):
                                    shared_ylabel = '{} [{}]'.format(quantity.title(), time_unit_labels[0])
                                else:
                                    shared_ylabel = '{}'.format(quantity.title())
                            self.share_axes(
                                axes=axes,
                                layout=layout,
                                xs=xs,
                                ys=ys,
                                sharex=sharex,
                                sharey=sharey,
                                xticks=True,
                                yticks=True,
                                xlim=True,
                                ylim=True,
                                xlabel=shared_xlabel,
                                ylabel=shared_ylabel,
                                collapse_x=collapse_x,
                                collapse_y=collapse_y)
                            fig.subplots_adjust(hspace=0.225)
                            fig.align_ylabels()
                            for ax in axes.ravel():
                                self.apply_grid(ax)
                            ## update title
                            s = r'Temporal Clusters: {} by Extreme Value'.format(quantity.title())
                            fig.suptitle(s, fontsize=self.titlesize)
                            ## show legend
                            handles, labels = axes.ravel()[0].get_legend_handles_labels()
                            nhandles = len(handles)
                            ncol = nhandles if nhandles > 1 else None
                            leg_title = '{} via {} Bias'.format(cluster_id.title(), bias_id.title())
                            self.subview_legend(
                                fig=fig,
                                ax=axes.ravel()[0],
                                handles=handles,
                                labels=labels,
                                textcolor=True,
                                facecolor='silver',
                                bottom=0.2,
                                ncol=ncol,
                                title=leg_title)
                            ## show / save
                            if save:
                                savename = 'RegularAnalysis_TemporalClusteringParameterization_{}_{}_{}-Bias'.format(
                                    quantity,
                                    cluster_id.title().replace(' ', '-'),
                                    bias_id.title())
                                if (central_statistic is not None) and (quantity not in quantities_without_statistics):
                                    savename = '{}_with{}'.format(savename, central_statistic.title().replace(' ', '-'))
                                if (error_statistic is not None) and (quantity not in quantities_without_statistics):
                                    savename = '{}_withST{}'.format(savename, error_statistic.replace('standard ', '').title().replace(' ', '-')[0])
                                if (show_min_and_max) and (quantity not in quantities_without_statistics):
                                    savename = '{}_withMinMax'.format(savename)
                                for series in self.series:
                                    cycle_nums = np.unique(series['data']['solar cycle'])
                                    if cycle_nums.size == 1:
                                        cycle_id = "SC-{}".format(cycle_nums[0])
                                    else:
                                        cycle_id = "SC" + "-".join(cycle_nums.astype(str))
                                    savename = '{}_{}_{}_{}'.format(
                                        savename,
                                        cycle_id,
                                        series['identifiers']['event type'].replace(' ', '-'),
                                        series['identifiers']['parameter id'].replace(' ', '-'))
                                savename = '{}_{}'.format(
                                    savename,
                                    layout)
                                savename = savename.replace(' ', '_')
                            else:
                                savename = None
                            self.display_image(fig, savename=savename)

    def view_cluster_size_table(self, extreme_values=None, cluster_ids='non-lone clusters', show_first_order_bias=False, show_threshold_bias=False, show_baseline=False, column_color='orange', facecolors=('peachpuff', 'bisque'), ddof=0, figsize=None, save=False, layout='single'):
        if layout is None:
            self.view_layout_permutations(
                f=self.view_analysis_table,
                layouts=['single'],
                extreme_values=extreme_values,
                cluster_ids=cluster_ids,
                show_first_order_bias=show_first_order_bias,
                show_threshold_bias=show_threshold_bias,
                show_baseline=show_baseline,
                row_color=row_color,
                column_color=column_color,
                facecolors=facecolors,
                ddof=ddof,
                figsize=figsize,
                save=save)
        elif layout == 'single':
            for series in self.series:
                cls = deepcopy(self.cls)
                visualizer = cls(
                    series=[series],
                    savedir=self.savedir)
                visualizer.view_cluster_size_table(
                    extreme_values=extreme_values,
                    cluster_ids=cluster_ids,
                    show_first_order_bias=show_first_order_bias,
                    show_threshold_bias=show_threshold_bias,
                    show_baseline=show_baseline,
                    column_color=column_color,
                    facecolors=facecolors,
                    ddof=ddof,
                    figsize=figsize,
                    save=save,
                    layout='overlay')
        else:
            if self.n > 1:
                raise ValueError("this method produces one figure per series; try using layout='single'")
            if layout != 'overlay':
                raise ValueError("invalid layout for this method: {}".format(layout))
            ## verify inputs
            bias_ids = ('first-order', 'threshold', 'baseline')
            conditions = np.array([show_first_order_bias, show_threshold_bias, show_baseline])
            nconditions = np.sum(conditions)
            if nconditions < 1:
                raise ValueError("input at least one of the following: show_first_order_bias, show_threshold_bias, show_baseline")
            if not isinstance(cluster_ids, (tuple, list, np.ndarray)):
                cluster_ids = [cluster_ids]
            if extreme_values is None:
                extreme_values = set(self.series[0]['temporal clustering'].extreme_values)
                if self.n > 1:
                    for i in range(1, self.n):
                        extreme_values = extreme_values.intersection(
                            set(self.series[i]['temporal clustering'].extreme_values))
                extreme_values = np.sort(list(extreme_values))
                if extreme_values.size == 0:
                    raise ValueError("could not find extreme values common to all series")
            elif not isinstance(extreme_values, (tuple, list, np.ndarray)):
                extreme_values = [extreme_values]
            for series in self.series:
                for extreme_value in extreme_values:
                    for cluster_id in cluster_ids:
                        for bias_id, bias_condition in zip(bias_ids, conditions):
                            if bias_condition:
                                cluster_searcher = series['temporal clustering'].extreme_values[extreme_value]['temporal clustering'][bias_id][cluster_id]['cluster searcher']
                                ## initialize table data
                                time_unit_label = series['identifiers']['elapsed unit']
                                column_labels = [
                                    'Cluster Size',
                                    'Number of Clusters',
                                    'Number of Extreme Events',
                                    'mean(Intra-Times)\n[{}]'.format(time_unit_label),
                                    'mean(Intra-Durations)\n[{}]'.format(time_unit_label),
                                    'mean(Inter-Durations)\n[{}]'.format(time_unit_label)]
                                cell_text = []
                                for cluster_size in np.unique(cluster_searcher.events['cluster size']):
                                    clusters_by_size, _ = cluster_searcher.search_clusters(
                                        parameters='cluster size',
                                        conditions='equal',
                                        values=cluster_size)
                                    nclusters = len(clusters_by_size['cluster size'])
                                    searcher_by_size = ClusterSearcher(
                                        clusters=clusters_by_size)
                                    nevents = searcher_by_size.events['cluster size'].size
                                    row = [
                                        '${:,}$'.format(
                                            cluster_size),
                                        '${:,}$'.format(
                                            nclusters),
                                        '${:,}$'.format(
                                            nevents)]
                                    intra_times = series['temporal clustering'].get_intra_times(
                                        clusters=clusters_by_size)
                                    flat_intra_times = np.concatenate(intra_times, axis=0)
                                    intra_durations = series['temporal clustering'].get_intra_durations(
                                        clusters=clusters_by_size)
                                    inter_durations = series['temporal clustering'].get_inter_durations(
                                        clusters=clusters_by_size)
                                    for time_values in (flat_intra_times, intra_durations, inter_durations):
                                        row.append('${:,.2f} ± {:,.2f}$'.format(
                                            np.nanmean(time_values),
                                            sem(time_values, ddof=ddof, nan_policy='omit')))
                                    cell_text.append(row)
                                cell_text = np.array(cell_text)
                                ## select cell colors
                                nrows, ncols = cell_text.shape
                                cell_colors = self.get_diagonal_table_colors(facecolors, nrows, ncols)
                                ## initialize plot
                                fig, ax = plt.subplots(figsize=figsize)
                                table = ax.table(
                                    colLabels=column_labels,
                                    cellText=cell_text,
                                    colColours=[column_color for col in range(ncols)],
                                    cellColours=cell_colors,
                                    loc='center',
                                    colLoc='center',
                                    cellLoc='center')
                                ## update table scale
                                self.autoformat_table(
                                    ax=ax,
                                    table=table)
                                ## show / save
                            if save:
                                savename = 'RegularAnalysis_Table-ClusterSize_EX-{}_{}_{}-BIAS'.format(
                                    extreme_value,
                                    cluster_id,
                                    bias_id.title().replace(' ', '-'))
                                for series in self.series:
                                    cycle_nums = np.unique(series['data']['solar cycle'])
                                    if cycle_nums.size == 1:
                                        cycle_id = "SC-{}".format(cycle_nums[0])
                                    else:
                                        cycle_id = "SC" + "-".join(cycle_nums.astype(str))
                                    savename = '{}_{}_{}_{}'.format(
                                        savename,
                                        cycle_id,
                                        series['identifiers']['event type'].replace(' ', '-'),
                                        series['identifiers']['parameter id'].replace(' ', '-'))
                                savename = '{}_{}'.format(
                                    savename,
                                    layout)
                                savename = savename.replace(' ', '_')
                            else:
                                savename = None
                            self.display_image(fig, savename=savename)

    def view_analysis_table(self, extreme_values=None, cluster_ids='non-lone clusters', show_first_order_bias=False, show_threshold_bias=False, show_baseline=False, row_color='orange', column_color='orange', facecolors=('peachpuff', 'bisque'), fmt="%Y-%m-%d %H:%M:%S", ddof=0, figsize=None, save=False, layout='single'):
        if layout is None:
            permutable_layouts = ['single']
            if self.n > 1:
                permutable_layouts.append('square')
            self.view_layout_permutations(
                f=self.view_analysis_table,
                layouts=permutable_layouts,
                extreme_values=extreme_values,
                cluster_ids=cluster_ids,
                show_first_order_bias=show_first_order_bias,
                show_threshold_bias=show_threshold_bias,
                show_baseline=show_baseline,
                row_color=row_color,
                column_color=column_color,
                facecolors=facecolors,
                fmt=fmt,
                ddof=ddof,
                figsize=figsize,
                save=save)
        elif layout == 'single':
            for series in self.series:
                cls = deepcopy(self.cls)
                visualizer = cls(
                    series=[series],
                    savedir=self.savedir)
                visualizer.view_analysis_table(
                    extreme_values=extreme_values,
                    cluster_ids=cluster_ids,
                    show_first_order_bias=show_first_order_bias,
                    show_threshold_bias=show_threshold_bias,
                    show_baseline=show_baseline,
                    row_color=row_color,
                    column_color=column_color,
                    facecolors=facecolors,
                    fmt=fmt,
                    ddof=ddof,
                    figsize=figsize,
                    save=save,
                    layout='overlay')
        else:
            ## verify inputs
            if self.n == 1:
                if layout != 'overlay':
                    raise ValueError("layout='overlay' for this method will only work for one series")
            else:
                if layout != 'square':
                    raise ValueError("invalid layout for this method: {}".format(layout))
            bias_ids = ('first-order', 'threshold', 'baseline')
            conditions = np.array([show_first_order_bias, show_threshold_bias, show_baseline])
            nconditions = np.sum(conditions)
            if nconditions < 1:
                raise ValueError("input at least one of the following: show_first_order_bias, show_threshold_bias, show_baseline")
            if not isinstance(cluster_ids, (tuple, list, np.ndarray)):
                cluster_ids = [cluster_ids]
            if extreme_values is None:
                extreme_values = set(self.series[0]['temporal clustering'].extreme_values)
                if self.n > 1:
                    for i in range(1, self.n):
                        extreme_values = extreme_values.intersection(
                            set(self.series[i]['temporal clustering'].extreme_values))
                extreme_values = np.sort(list(extreme_values))
                if extreme_values.size == 0:
                    raise ValueError("could not find extreme values common to all series")
            elif not isinstance(extreme_values, (tuple, list, np.ndarray)):
                extreme_values = [extreme_values]
            row_labels = [
                'Date',
                'Total Duration',
                'Number of Events',
                'Number of Extreme Events',
                'Extreme Event Criteria',
                r'mean(Minimum Extreme-Value Threshold)',
                r'median(Minimum Extreme-Value Threshold)',
                'Number of Resamples',
                r'mean($\hat\alpha$)',
                r'mean($\hat{C}$)',
                r'Extremal Index $\hat\theta$',
                r'Moment Estimator $\theta$',
                r'Time Threshold $T_C$',
                'Number of Extreme Events from Clusters',
                'Number of Clusters',
                r'Expected Cluster Size $\frac{1}{\theta}$',
                'mean(Cluster Size)',
                'mean(Intra-Times)',
                'mean(Intra-Durations)',
                'mean(Inter-Durations)']
            for cluster_id in cluster_ids:
                for extreme_value in extreme_values:
                    for bias_id, bias_condition in zip(bias_ids, conditions):
                        if bias_condition:
                            column_labels = []
                            cell_text = []
                            for series in self.series:
                                column_labels.append(series['identifiers']['series id'])
                                time_unit_label = series['identifiers']['elapsed unit']
                                parameter_unit_label = series['unit mapping'][series['temporal clustering'].extreme_parameter]
                                extreme_label = self.get_extreme_label(
                                    extreme_parameter=series['temporal clustering'].extreme_parameter,
                                    extreme_condition=series['temporal clustering'].extreme_condition,
                                    extreme_value=extreme_value,
                                    parameter_mapping=series['parameter mapping'],
                                    unit_mapping=series['unit mapping']).replace('Extreme-Value Threshold: ', '')
                                dts = series['data']['datetime']
                                elapsed = series['data']['elapsed']
                                cluster_searcher = series['temporal clustering'].extreme_values[extreme_value]['temporal clustering'][bias_id][cluster_id]['cluster searcher']
                                intra_times_histogram = series['temporal clustering'].extreme_values[extreme_value]['temporal clustering'][bias_id][cluster_id]['intra-cluster times']['histogram']
                                intra_durations_histogram = series['temporal clustering'].extreme_values[extreme_value]['temporal clustering'][bias_id][cluster_id]['intra-cluster durations']['histogram']
                                inter_durations_histogram = series['temporal clustering'].extreme_values[extreme_value]['temporal clustering'][bias_id][cluster_id]['inter-cluster durations']['histogram']
                                row = [
                                    r'${}$ - ${}$'.format(
                                        dts[0].strftime(fmt),
                                        dts[-1].strftime(fmt)),
                                    r'${:,}$ {}'.format(
                                        elapsed[-1] - elapsed[0],
                                        self.make_plural(time_unit_label)),
                                    r'${:,}$'.format(
                                        int(np.sum(series['data']['is event']))),
                                    r'${:,}$'.format(
                                        series['inter-exceedance'].extreme_values[extreme_value]['nevents']),
                                    r'{}'.format(extreme_label),
                                    r'${}$ {}'.format(
                                        series['unbiased estimators'].extreme_value_estimates['mean'], parameter_unit_label),
                                    r'${}$ {}'.format(
                                        series['unbiased estimators'].extreme_value_estimates['median'], parameter_unit_label),
                                    r'${:,}$'.format(
                                        series['unbiased estimators'].nresamples),
                                    r'${:.2f}$'.format(
                                        series['unbiased estimators'].histograms['alpha'].mean),
                                    r'${:.2f}$'.format(
                                        series['unbiased estimators'].histograms['intercept'].mean),
                                    r'${:.2f}$'.format(
                                        series['unbiased estimators'].histograms['theta'].mean),
                                    r'${:.2f}$'.format(
                                        series['temporal clustering'].extreme_values[extreme_value]['temporal clustering'][bias_id]['moment estimator']),
                                    r'${:,}$ {}'.format(
                                        series['temporal clustering'].extreme_values[extreme_value]['temporal clustering'][bias_id]['time threshold'],
                                        self.make_plural(time_unit_label)),
                                    r'${:,}$'.format(
                                        cluster_searcher.events['cluster size'].size),
                                    r'${:,}$'.format(
                                        len(cluster_searcher.clusters['cluster size'])),
                                    r'${:,.2f}$'.format(
                                        1 / series['temporal clustering'].extreme_values[extreme_value]['temporal clustering'][bias_id]['moment estimator']),
                                    r'${:,.2f}$'.format(
                                        np.mean([cluster.size for cluster in cluster_searcher.clusters['cluster size']])),
                                    r'${:,.2f} ± {:,.2f}$ {}'.format(
                                        intra_times_histogram.mean,
                                        intra_times_histogram.standard_error,
                                        self.make_plural(time_unit_label)),
                                    r'${:,.2f} ± {:,.2f}$ {}'.format(
                                        intra_durations_histogram.mean,
                                        intra_durations_histogram.standard_error,
                                        self.make_plural(time_unit_label)),
                                    r'${:,.2f} ± {:,.2f}$ {}'.format(
                                        inter_durations_histogram.mean,
                                        inter_durations_histogram.standard_error,
                                        self.make_plural(time_unit_label))]
                                cell_text.append(row)
                            cell_text = np.array(cell_text).T
                            ## select cell colors
                            nrows, ncols = len(row_labels), len(column_labels)
                            cell_colors = self.get_diagonal_table_colors(facecolors, nrows, ncols)
                            ## initialize plot
                            fig, ax = plt.subplots(figsize=figsize)
                            table = ax.table(
                                colLabels=column_labels,
                                rowLabels=row_labels,
                                cellText=cell_text,
                                colColours=[column_color for _ in range(ncols)],
                                rowColours=[row_color for _ in range(nrows)],
                                cellColours=cell_colors,
                                loc='center',
                                cellLoc='center')
                            ## update table scale
                            self.autoformat_table(
                                ax=ax,
                                table=table)
                            ## show / save
                            if save:
                                savename = 'RegularAnalysis_Table-Analysis_EX-{}_{}_{}-BIAS'.format(
                                    extreme_value,
                                    cluster_id,
                                    bias_id.title().replace(' ', '-'))
                                for series in self.series:
                                    cycle_nums = np.unique(series['data']['solar cycle'])
                                    if cycle_nums.size == 1:
                                        cycle_id = "SC-{}".format(cycle_nums[0])
                                    else:
                                        cycle_id = "SC" + "-".join(cycle_nums.astype(str))
                                    savename = '{}_{}_{}_{}'.format(
                                        savename,
                                        cycle_id,
                                        series['identifiers']['event type'].replace(' ', '-'),
                                        series['identifiers']['parameter id'].replace(' ', '-'))
                                savename = '{}_{}'.format(
                                    savename,
                                    layout)
                                savename = savename.replace(' ', '_')
                            else:
                                savename = None
                            self.display_image(fig, savename=savename)

    def subview_lag_correlations(self, ax, series, extreme_value, i, xs, ys, show_original=False, show_time_randomized=False, show_difference=False, facecolor='darkorange', resample_color='gray', diff_color='k'):
        lag_corr = series['naive lag correlations']
        xy_indices = (lag_corr.lag_ks != 0)
        if show_original:
            ax.plot(
                lag_corr.lag_ks[xy_indices],
                lag_corr.extreme_values[extreme_value]['original lambda'][xy_indices],
                color=facecolor,
                label='Original Data' if i == 0 else None,
                alpha=0.75)
            ys.append(np.nanmax(lag_corr.extreme_values[extreme_value]['original lambda'][xy_indices] * 1.125))
            ys.append(np.nanmin(lag_corr.extreme_values[extreme_value]['original lambda'][xy_indices]))
        if show_time_randomized:
            if i == 0:
                for j, time_randomized_lams in enumerate(lag_corr.extreme_values[extreme_value]['time-randomized lambda']):
                    ax.plot(
                        lag_corr.lag_ks[xy_indices],
                        time_randomized_lams[xy_indices],
                        color=resample_color,
                        label='Time-Randomized' if j == 0 else None,
                        alpha=1/np.sqrt(lag_corr.nresamples))
                    ys.append(np.nanmax(time_randomized_lams[xy_indices]) * 1.125)
                    ys.append(np.nanmin(time_randomized_lams[xy_indices]) * 1.125)
            else:
                for j, time_randomized_lams in enumerate(lag_corr.extreme_values[extreme_value]['time-randomized lambda']):
                    ax.plot(
                        lag_corr.lag_ks[xy_indices],
                        time_randomized_lams[xy_indices],
                        color=resample_color,
                        alpha=1/np.sqrt(lag_corr.nresamples))
                    ys.append(np.nanmax(time_randomized_lams[xy_indices]) * 1.125)
                    ys.append(np.nanmin(time_randomized_lams[xy_indices]) * 1.125)
        if show_difference:
            s = 'Difference'
            ax.plot(
                lag_corr.lag_ks[xy_indices],
                lag_corr.extreme_values[extreme_value]['difference lambda'][xy_indices],
                color=diff_color,
                label='Difference\nOriginal - mean(Time-Randomized)' if i == 0 else None,
                alpha=0.75)
            ys.append(np.nanmax(lag_corr.extreme_values[extreme_value]['difference lambda'][xy_indices] * 1.125))
            smallest_y = np.nanmin(lag_corr.extreme_values[extreme_value]['difference lambda'][xy_indices])
            if smallest_y < 0:
                ys.append(smallest_y * 1.125)
            else:
                ys.append(smallest_y / 1.125)
        xs.append(np.nanmin(lag_corr.lag_ks[xy_indices]))
        xs.append(np.nanmax(lag_corr.lag_ks[xy_indices]))
        return xs, ys

    def view_lag_correlations(self, extreme_values, show_original=False, show_time_randomized=False, show_difference=False, facecolor='darkorange', resample_color='gray', diff_color='k', sharex=False, sharey=False, collapse_x=False, collapse_y=False, figsize=None, save=False, layout='single'):
        if layout is None:
            permutable_layouts = ['single']
            if 2 <= self.n < 4:
                permutable_layouts.append('horizontal')
            elif (self.n > 2) and (self.n % 2 == 0):
                permutable_layouts.append('square')
            self.view_layout_permutations(
                f=self.view_lag_correlations,
                layouts=permutable_layouts,
                extreme_values=extreme_values,
                show_original=show_original,
                show_time_randomized=show_time_randomized,
                show_difference=show_difference,
                facecolor=facecolor,
                resample_color=resample_color,
                diff_color=diff_color,
                sharex=sharex,
                sharey=sharey,
                collapse_x=collapse_x,
                collapse_y=collapse_y,
                figsize=figsize,
                save=save)
        elif layout == 'single':
            for series in self.series:
                cls = deepcopy(self.cls)
                visualizer = cls(
                    series=[series],
                    savedir=self.savedir)
                visualizer.view_lag_correlations(
                    extreme_values=extreme_values,
                    show_original=show_original,
                    show_time_randomized=show_time_randomized,
                    show_difference=show_difference,
                    facecolor=facecolor,
                    resample_color=resample_color,
                    diff_color=diff_color,
                    sharex=sharex,
                    sharey=sharey,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y,
                    figsize=figsize,
                    save=save,
                    layout='overlay')
        else:
            ## autocorrect inputs
            conditions = np.array([show_original, show_time_randomized, show_difference])
            if np.sum(conditions) < 1:
                raise ValueError("input only one of the following to use this method: 'show_original', 'show_time_randomized', 'show_difference'")
            if extreme_values is None:
                extreme_values = set(self.series[0]['naive lag correlations'].extreme_values)
                if self.n > 1:
                    for i in range(1, self.n):
                        extreme_values = extreme_values.intersection(
                            set(self.series[i]['naive lag correlations'].extreme_values))
                extreme_values = np.sort(list(extreme_values))
                if extreme_values.size == 0:
                    raise ValueError("could not find extreme values common to all series")
            elif not isinstance(extreme_values, (tuple, list, np.ndarray)):
                extreme_values = [extreme_values]
            for extreme_value in extreme_values:
                xs, ys = [], [0]
                nresamples = []
                parameter_labels, alt_parameter_labels, time_labels = [], [], []
                shared_xlabel = 'Time-Lag $k$'
                shared_ylabel = r'Correlation $\lambda_{k}$'
                ## get figure and axes
                kws = self.get_number_of_figure_rows_and_columns(self.n, layout)
                fig, axes = plt.subplots(figsize=figsize, **kws)
                if layout == 'overlay':
                    if self.n != 1:
                        raise ValueError("layout='overlay' for this method will only work for one series")
                    axes = np.array([axes])
                handles, labels = [], []
                for i, (ax, series) in enumerate(zip(axes.ravel(), self.series)):
                    extreme_label = self.get_extreme_label(
                        extreme_parameter=series['naive lag correlations'].extreme_parameter,
                        extreme_condition=series['naive lag correlations'].extreme_condition,
                        extreme_value=extreme_value,
                        parameter_mapping=series['parameter mapping'],
                        unit_mapping=series['unit mapping'])
                    alt_extreme_label = self.get_generalized_extreme_label(
                        extreme_parameter=series['naive lag correlations'].extreme_parameter,
                        extreme_condition=series['naive lag correlations'].extreme_condition,
                        extreme_value=extreme_value,
                        parameter_mapping=series['parameter mapping'],
                        unit_mapping=series['unit mapping'],
                        generalized_parameter_mapping=series['generalized parameter mapping'])
                    elapsed_unit = series['identifiers']['elapsed unit']
                    time_label = '{} [{}]'.format(shared_xlabel, elapsed_unit)
                    parameter_labels.append(extreme_label)
                    alt_parameter_labels.append(alt_extreme_label)
                    time_labels.append(time_label)
                    nresamples.append(series['naive lag correlations'].nresamples)
                    xs, ys = self.subview_lag_correlations(
                        ax=ax,
                        series=series,
                        extreme_value=extreme_value,
                        i=i,
                        xs=xs,
                        ys=ys,
                        show_original=show_original,
                        show_time_randomized=show_time_randomized,
                        show_difference=show_difference,
                        facecolor=facecolor,
                        resample_color=resample_color,
                        diff_color=diff_color)
                    ax.set_ylabel(time_label, fontsize=self.labelsize)
                    ax.set_ylabel(shared_ylabel, fontsize=self.labelsize)
                    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
                    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
                    ax.set_title(series['identifiers']['series id'], fontsize=self.titlesize)
                    if i == 0:
                        _handles, _labels = ax.get_legend_handles_labels()
                        handles.extend(_handles)
                        labels.extend(_labels)
                fig.subplots_adjust(hspace=0.3)
                ## update axes
                if self.is_same_elements(elements=time_labels, s='', n=self.n):
                    shared_xlabel = '{}'.format(time_labels[0])
                # else:
                #     shared_xlabel = None
                self.share_axes(
                    axes=axes,
                    layout=layout,
                    xs=xs,
                    ys=ys,
                    sharex=sharex,
                    sharey=sharey,
                    xlim=True,
                    ylim=True,
                    xticks=True,
                    yticks=True,
                    xlabel=shared_xlabel,
                    ylabel=shared_ylabel,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y)
                for ax in axes.ravel():
                    self.apply_grid(ax)
                fig.align_ylabels()
                ## update title
                s = 'Lag-Correlation Analysis'
                fig.suptitle(s, fontsize=self.titlesize)
                ## show legend
                if self.is_same_elements(elements=parameter_labels, s='', n=self.n):
                    leg_title = '{}'.format(parameter_labels[0])
                elif self.is_same_elements(elements=alt_parameter_labels, s='', n=self.n):
                    leg_title = '{}'.format(alt_parameter_labels[0])
                else:
                    leg_title = None
                if self.is_same_elements(elements=nresamples, s='', n=self.n):
                    for j, label in enumerate(labels):
                        if label == 'Time-Randomized':
                            labels[j] = '{} Time-Randomized Samples'.format(nresamples[0])
                self.subview_legend(
                    fig=fig,
                    ax=axes.ravel()[0],
                    handles=handles,
                    labels=labels,
                    textcolor=True,
                    facecolor='silver',
                    bottom=0.2,
                    ncol=None,
                    title=leg_title)
                ## show / save
                if save:
                    savename = 'RegularAnalysis_NaiveLagCorr_EX-{}'.format(extreme_value)
                    if show_original:
                        savename = '{}_OG'.format(savename)
                    if show_time_randomized:
                        savename = '{}_tRAND'.format(savename)
                    if show_difference:
                        savename = '{}_DIFF'.format(savename)
                    for series in self.series:
                        cycle_nums = np.unique(series['data']['solar cycle'])
                        if cycle_nums.size == 1:
                            cycle_id = "SC-{}".format(cycle_nums[0])
                        else:
                            cycle_id = "SC" + "-".join(cycle_nums.astype(str))
                        savename = '{}_{}_{}_{}'.format(
                            savename,
                            cycle_id,
                            series['identifiers']['event type'].replace(' ', '-'),
                            series['identifiers']['parameter id'].replace(' ', '-'))
                    savename = '{}_{}'.format(
                        savename,
                        layout)
                    savename = savename.replace(' ', '_')
                else:
                    savename = None
                self.display_image(fig, savename=savename)

    def view_lag_correlations_parameterization(self, ndim, xparameter='time-lag k', yparameter='extreme value', zparameter='original lambda', cmap='Oranges', extremum_color='k', color_spacing='linear', levels=None, zfmt=None, azim=-60, elev=30, rstride=1, cstride=1, show_lines=False, show_colorbar=False, sharex=False, sharey=False, collapse_x=False, collapse_y=False, figsize=None, save=False, layout='single'):
        if layout is None:
            permutable_layouts = ['single']
            if 2 <= self.n < 4:
                permutable_layouts.append('horizontal')
            elif (self.n > 2) and (self.n % 2 == 0):
                permutable_layouts.append('square')
            self.view_layout_permutations(
                f=self.view_lag_correlations_parameterization,
                layouts=permutable_layouts,
                ndim=ndim,
                xparameter=xparameter,
                yparameter=yparameter,
                zparameter=zparameter,
                cmap=cmap,
                extremum_color=extremum_color,
                color_spacing=color_spacing,
                levels=levels,
                zfmt=zfmt,
                azim=azim,
                elev=elev,
                rstride=rstride,
                cstride=cstride,
                show_lines=show_lines,
                show_colorbar=show_colorbar,
                sharex=sharex,
                sharey=sharey,
                collapse_x=collapse_x,
                collapse_y=collapse_y,
                figsize=figsize,
                save=save)
        elif layout == 'single':
            for series in self.series:
                cls = deepcopy(self.cls)
                visualizer = cls(
                    series=[series],
                    savedir=self.savedir)
                visualizer.view_lag_correlations_parameterization(
                    ndim=ndim,
                    xparameter=xparameter,
                    yparameter=yparameter,
                    zparameter=zparameter,
                    cmap=cmap,
                    extremum_color=extremum_color,
                    color_spacing=color_spacing,
                    levels=levels,
                    zfmt=zfmt,
                    azim=azim,
                    elev=elev,
                    rstride=rstride,
                    cstride=cstride,
                    show_lines=show_lines,
                    show_colorbar=show_colorbar,
                    sharex=sharex,
                    sharey=sharey,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y,
                    figsize=figsize,
                    save=save,
                    layout='overlay')
        else:
            ## verify inputs
            if ndim not in (2, 3):
                raise ValueError("invalid ndim: {}".format(ndim))
            if layout == 'overlay':
                if self.n != 1:
                    raise ValueError("layout='overlay' for this method will only work for one series")
            if zfmt is None:
                zfmt = ticker.StrMethodFormatter('{x:,.3f}')
            if (xparameter == 'time-lag k' and yparameter == 'extreme value'):
                swap_xy = True
            elif (xparameter == 'extreme value' and yparameter == 'time-lag k'):
                swap_xy = False
            else:
                raise ValueError("invalid combination of xparameter='{}' and yparameter='{}'".format(xparameter, yparameter))
            if zparameter == 'difference lambda':
                shared_zlabel = r'Correlation Difference $\Delta \lambda_{k}$'
            elif zparameter == 'original lambda':
                shared_zlabel = r'Correlation $\lambda_{k}$'
            else:
                raise ValueError("invalid zparameter: {}".format(zparameter))
            ## get z-colorbar data
            zs = []
            for series in self.series:
                lag_corr = series['naive lag correlations parameterization']
                zs.append(np.nanmin(lag_corr.parameterization['Z'][zparameter]))
                zs.append(np.nanmax(lag_corr.parameterization['Z'][zparameter]))
            zs = np.array(zs)
            norm, fmt, vmin, vmax = self.get_colormap_configuration(
                z=zs,
                color_spacing=color_spacing,
                levels=levels,
                fmt=zfmt)
            ## initialize plot parameters
            xs, ys, zs = [], [], []
            handles, labels = [], []
            time_labels, parameter_labels, alt_parameter_labels = [], [], []
            nresamples = []
            shared_time_label = 'Time-Lag $k$'
            shared_zlabel = r'Correlation $\lambda_{k}$'
            ## get figure and axes
            kws = self.get_number_of_figure_rows_and_columns(self.n, layout)
            if ndim == 2:
                fig, axes = plt.subplots(figsize=figsize, **kws)
                if not isinstance(axes, np.ndarray):
                    axes = np.array([axes])
            else:
                fig, axes = self.get_dim3_figure_axes(figsize=figsize, **kws)
            temporary_ev_value = 545
            for ax, series in zip(axes.ravel(), self.series):
                lag_corr = series['naive lag correlations parameterization']
                extreme_label = self.get_extreme_label(
                    extreme_parameter=lag_corr.extreme_parameter,
                    extreme_condition=lag_corr.extreme_condition,
                    extreme_value=temporary_ev_value,
                    parameter_mapping=series['parameter mapping'],
                    unit_mapping=series['unit mapping']).replace(' Threshold: ', '').replace('{}'.format(temporary_ev_value), '')
                alt_extreme_label = self.get_generalized_extreme_label(
                    extreme_parameter=lag_corr.extreme_parameter,
                    extreme_condition=lag_corr.extreme_condition,
                    extreme_value=temporary_ev_value,
                    parameter_mapping=series['parameter mapping'],
                    unit_mapping=series['unit mapping'],
                    generalized_parameter_mapping=series['generalized parameter mapping']).replace(' Threshold: ', '').replace('{}'.format(temporary_ev_value), '')
                elapsed_unit = series['identifiers']['elapsed unit']
                time_label = '{} [{}]'.format(shared_time_label, elapsed_unit)
                parameter_labels.append(extreme_label)
                alt_parameter_labels.append(alt_extreme_label)
                time_labels.append(time_label)
                nresamples.append(series['naive lag correlations parameterization'].nresamples)
                X, Y = lag_corr.parameterization['X'], lag_corr.parameterization['Y']
                if swap_xy:
                    ax.set_xlabel(time_label, fontsize=self.labelsize)
                    ax.set_ylabel(extreme_label, fontsize=self.labelsize)
                    X, Y = Y, X
                else:
                    ax.set_xlabel(extreme_label, fontsize=self.labelsize)
                    ax.set_ylabel(time_label, fontsize=self.labelsize)
                if ndim == 2:
                    cbar_handle = self.subview_contour_space(
                        ax=ax,
                        X=X,
                        Y=Y,
                        Z=lag_corr.parameterization['Z'][zparameter],
                        norm=norm,
                        levels=levels,
                        cmap=cmap,
                        extremum_color=extremum_color,
                        show_fills=True,
                        show_lines=show_lines,
                        show_inline_labels=False,
                        scatter_args=None)
                else:
                    cbar_handle = self.subview_surface_space(
                        ax=ax,
                        X=X,
                        Y=Y,
                        Z=lag_corr.parameterization['Z'][zparameter],
                        norm=norm,
                        levels=levels,
                        cmap=cmap,
                        extremum_color=extremum_color,
                        show_lines=show_lines,
                        rstride=rstride,
                        cstride=cstride,
                        azim=azim,
                        elev=elev,
                        scatter_args=None)
                    ax.set_zlabel(shared_zlabel, fontsize=self.labelsize)
                    ax.zaxis.set_minor_locator(ticker.AutoMinorLocator())
                if show_colorbar:
                    self.subview_color_bar(
                        fig=fig,
                        ax=ax,
                        handle=cbar_handle,
                        title=r'$\lambda_{k}$',
                        levels=levels,
                        norm=norm,
                        extend='max',
                        orientation='vertical',
                        pad=0.1)
                _handles, _labels = ax.get_legend_handles_labels()
                handles.extend(_handles)
                labels.extend(_labels)
                ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
                ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
                ax.set_title(series['identifiers']['series id'], fontsize=self.titlesize)
                xs.append(np.min(lag_corr.parameterization['X']))
                xs.append(np.max(lag_corr.parameterization['X']))
                ys.append(np.min(lag_corr.parameterization['Y']))
                ys.append(np.max(lag_corr.parameterization['Y']))
            ## update axes
            hspace = 0.425 if 'vertical' in layout else 0.3
            fig.subplots_adjust(hspace=hspace)
            if self.is_same_elements(elements=parameter_labels, s='', n=self.n):
                shared_xlabel = '{}'.format(parameter_labels[0])
            else:
                if self.is_same_elements(elements=alt_parameter_labels, s='', n=self.n):
                    shared_xlabel = '{}'.format(alt_parameter_labels[0])
                else:
                    shared_xlabel = 'Extreme Parameter'
            if self.is_same_elements(elements=time_labels, s='', n=self.n):
                shared_ylabel = '{}'.format(time_labels[0])
            else:
                shared_ylabel = '{}'.format(shared_time_label)
            if swap_xy:
                shared_xlabel, shared_ylabel = shared_ylabel, shared_xlabel
                xs, ys = ys, xs
            if ndim == 2:
                self.share_axes(
                    axes=axes,
                    layout=layout,
                    xs=xs,
                    ys=ys,
                    sharex=sharex,
                    sharey=sharey,
                    xticks=True,
                    yticks=True,
                    xlim=True,
                    ylim=True,
                    xlabel=shared_xlabel,
                    ylabel=shared_ylabel,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y)
            else:
                self.share_dim3_axes(
                    axes=axes,
                    xs=xs,
                    ys=ys,
                    zs=zs,
                    xticks=True,
                    yticks=True,
                    zticks=True,
                    xlim=True,
                    ylim=True,
                    zlim=True,
                    zfmt=zfmt,
                    xlabel=shared_xlabel,
                    ylabel=shared_ylabel,
                    zlabel=shared_zlabel)
            if ndim == 2:
                for ax in axes.ravel():
                    self.apply_grid(ax)
            fig.align_ylabels()
            ## update title
            s = 'Lag-Correlation Parameterization'
            fig.suptitle(s, fontsize=self.titlesize)
            ## add legend

            ## show / save
            if save:
                savename = 'RegularAnalysis_NaiveLagCorr_Parameterization_{}_CS-{}'.format(
                    zparameter.title(),
                    color_spacing)
                if show_lines:
                    savename = '{}_withLINES'.format(savename)
                if show_colorbar:
                    savename = '{}_withCBAR'.format(savename)
                if swap_xy:
                    savename = '{}_swapXY'.format(savename)
                for series in self.series:
                    cycle_nums = np.unique(series['data']['solar cycle'])
                    if cycle_nums.size == 1:
                        cycle_id = "SC-{}".format(cycle_nums[0])
                    else:
                        cycle_id = "SC" + "-".join(cycle_nums.astype(str))
                    savename = '{}_{}_{}_{}'.format(
                        savename,
                        cycle_id,
                        series['identifiers']['event type'].replace(' ', '-'),
                        series['identifiers']['parameter id'].replace(' ', '-'))
                savename = '{}_{}'.format(
                    savename,
                    layout)
                savename = savename.replace(' ', '_')
            else:
                savename = None
            self.display_image(fig, savename=savename)

##
