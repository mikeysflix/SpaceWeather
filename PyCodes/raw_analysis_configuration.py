from visual_configuration import *
from parametric_distribution_configuration import *
from inter_exceedance_distribution import *
from frequency_configuration import *

class RawAnalaysisConfiguration(VisualConfiguration):

    def __init__(self, series, savedir=None):
        super().__init__(savedir=savedir)
        if isinstance(series, list):
            self.series = series
        else:
            self.series = [series]
        self.n = len(self.series)

    def add_normal_distribution(self, extreme_parameter=None, series_indices=None, **kwargs):
        if series_indices is None:
            for i in range(self.n):
                self.add_normal_distribution(
                    extreme_parameter=extreme_parameter,
                    series_indices=i,
                    **kwargs)
        elif isinstance(series_indices, (tuple, list, np.ndarray)):
            for i in series_indices:
                self.add_normal_distribution(
                    extreme_parameter=extreme_parameter,
                    series_indices=i,
                    **kwargs)
        elif isinstance(series_indices, int):
            if extreme_parameter is None:
                extreme_parameter = self.series[series_indices]['identifiers']['parameter id']
            normal_distribution = NormalDistributionConfiguration(
                data=self.series[series_indices]['data'],
                extreme_parameter=extreme_parameter)
            normal_distribution(**kwargs)
            self.series[series_indices]['normal distribution'] = normal_distribution
        else:
            raise ValueError("invalid type(series_indices): {}".format(type(series_indices)))

    def add_lognormal_distribution(self, extreme_parameter=None, series_indices=None, **kwargs):
        if series_indices is None:
            for i in range(self.n):
                self.add_lognormal_distribution(
                    extreme_parameter=extreme_parameter,
                    series_indices=i,
                    **kwargs)
        elif isinstance(series_indices, (tuple, list, np.ndarray)):
            for i in series_indices:
                self.add_lognormal_distribution(
                    extreme_parameter=extreme_parameter,
                    series_indices=i,
                    **kwargs)
        elif isinstance(series_indices, int):
            if extreme_parameter is None:
                extreme_parameter = self.series[series_indices]['identifiers']['parameter id']
            lognormal_distribution = LogNormalDistributionConfiguration(
                data=self.series[series_indices]['data'],
                extreme_parameter=extreme_parameter)
            lognormal_distribution(**kwargs)
            self.series[series_indices]['lognormal distribution'] = lognormal_distribution
            if lognormal_distribution.normal_distribution is not None:
                # if len(lognormal_distribution.normal_distribution.sub_series) > 0:
                #     lognormal_distribution.normal_distribution._sub_series = ...
                self.series[series_indices]['normal distribution'] = lognormal_distribution.normal_distribution
        else:
            raise ValueError("invalid type(series_indices): {}".format(type(series_indices)))

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

    def add_temporal_frequency(self, extreme_parameter=None, series_indices=None, **kwargs):
        if series_indices is None:
            for i in range(self.n):
                self.add_temporal_frequency(
                    extreme_parameter=extreme_parameter,
                    series_indices=i,
                    **kwargs)
        elif isinstance(series_indices, (tuple, list, np.ndarray)):
            for i in series_indices:
                self.add_temporal_frequency(
                    extreme_parameter=extreme_parameter,
                    series_indices=i,
                    **kwargs)
        elif isinstance(series_indices, int):
            if extreme_parameter is None:
                extreme_parameter = self.series[series_indices]['identifiers']['parameter id']
            if 'temporal histogram' in list(self.series[series_indices].keys()):
                temporal_histogram = self.series[series_indices]['temporal histogram']
            else:
                temporal_histogram = None
            temporal_frequency = TemporalFrequencyConfiguration(
                data=self.series[series_indices]['data'],
                extreme_parameter=extreme_parameter,
                temporal_histogram=temporal_histogram)
            temporal_frequency(**kwargs)
            self.series[series_indices]['temporal frequency'] = temporal_frequency
        else:
            raise ValueError("invalid type(series_indices): {}".format(type(series_indices)))

    def add_mixed_frequency(self, extreme_parameter=None, series_indices=None, **kwargs):
        if series_indices is None:
            for i in range(self.n):
                self.add_mixed_frequency(
                    extreme_parameter=extreme_parameter,
                    series_indices=i,
                    **kwargs)
        elif isinstance(series_indices, (tuple, list, np.ndarray)):
            for i in series_indices:
                self.add_mixed_frequency(
                    extreme_parameter=extreme_parameter,
                    series_indices=i,
                    **kwargs)
        elif isinstance(series_indices, int):
            if extreme_parameter is None:
                extreme_parameter = self.series[series_indices]['identifiers']['parameter id']
            mixed_frequency = MixedFrequencyConfiguration(
                data=self.series[series_indices]['data'],
                extreme_parameter=extreme_parameter)
            mixed_frequency(**kwargs)
            self.series[series_indices]['mixed frequency'] = mixed_frequency
        else:
            raise ValueError("invalid type(series_indices): {}".format(type(series_indices)))

class RawAnalaysis(RawAnalaysisConfiguration):

    def __init__(self, series, savedir=None):
        super().__init__(
            series=series,
            savedir=savedir)
        self.cls = RawAnalaysis
        self.period_map = {
            'solar cycle' : {
                'subperiod' : 'year',
                'ticks' : np.arange(15).astype(int)},
            'year' : {
                'subperiod' : 'month',
                'ticks' : np.arange(1, 13).astype(int)},
            'month' : {
                'subperiod' : 'day',
                'ticks' : np.arange(1, 32).astype(int)},
            'day' : {
                'subperiod' : 'hour',
                'ticks' : np.arange(1, 25).astype(int)},
            'hour' : {
                'subperiod' : 'minute',
                'ticks' : np.arange(1, 61).astype(int)},
            'minute' : {
                'subperiod' : 'second',
                'ticks' : np.arange(1, 61).astype(int)}}
        for period in list(self.period_map.keys()):
            if period == 'year':
                self.period_map[period]['ticklabels'] = np.array([calendar.month_abbr[tick] for tick in self.period_map[period]['ticks']])
            else:
                self.period_map[period]['ticklabels'] = np.copy(self.period_map[period]['ticks'])

    @staticmethod
    def get_exclusive_optimizer_id(show_chi_square=False, show_g_test=False, show_maximum_likelihood=False):
        conditions = np.array([show_chi_square, show_g_test, show_maximum_likelihood])
        optimizer_ids = ('chi square', 'g-test', 'maximum likelihood')
        if np.sum(conditions) != 1:
            raise ValueError("input only one of the following to use this method: 'show_chi_square', 'show_g_test', 'show_maximum_likelihood'")
        _conditions = iter(conditions)
        _optimizer_ids = iter(optimizer_ids)
        while True:
            current_condition = next(_conditions)
            current_id = next(_optimizer_ids)
            if current_condition:
                break
        return current_id

    @staticmethod
    def get_optimization_label(distribution_id, optimization_result, unit_label):
        if distribution_id == 'normal distribution':
            s = '{} {}\n{} {}\n{}'.format(
                optimization_result['labels']['mu'],
                unit_label,
                optimization_result['labels']['sigma'],
                unit_label,
                optimization_result['labels']['fun'])
        elif distribution_id == 'lognormal distribution':
            s = '{} {}\n{} {}\n{} {}\n{} {}\n{}'.format(
                optimization_result['labels']['mu'],
                unit_label,
                optimization_result['labels']['sigma'],
                unit_label,
                optimization_result['labels']['median'],
                unit_label,
                optimization_result['labels']['mode'],
                unit_label,
                optimization_result['labels']['fun'])
        else:
            raise ValueError("invalid distribution_id: {}".format(self.distribution_id))
        return s

    @staticmethod
    def autocorrect_error_space_configuration(extreme_parameter, event_type, optimizer_id, levels=None, color_spacing=None):
        if levels is None:
            if event_type == 'CME':
                if color_spacing is None:
                    color_spacing = 'log'
                available_parameters = []
                for possible_parameter in ('speed', 'linear speed', 'second order initial speed', 'second order final speed', 'second order 20R speed', r'$V_{CME}$', r'$V_{linear}$', r'$V_{20 R_{\odot}, i}$', r'$V_{20 R_{\odot}, f}$', r'$V_{20 R_{\odot}}$'):
                    available_parameters.append(possible_parameter)
                    available_parameters.append('log {}'.format(possible_parameter))
                if extreme_parameter in available_parameters:
                    if color_spacing == 'linear':
                        if optimizer_id == 'maximum likelihood':
                            error_levels = np.array([0, 1, 5, 10, 100, 1e3, 5e3, 1e4, 5e4, 7.5e4, 1e5, 2.5e5, 5e5, 7.5e5, 1e6, 2.5e6]).astype(int)
                            color_bar_levels = np.array([0, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7]).astype(int)
                            zfmt = ticker.StrMethodFormatter('{x:,.0f}')
                        elif optimizer_id == 'g-test':
                            error_levels = np.array([0, 1, 5, 10, 25, 50, 100, 250, 500, 750, 1e3, 2.5e3, 5e3, 7.5e3, 1e4]).astype(int)
                            color_bar_levels = np.array([0, 1, 10, 100, 1e3, 1e4]).astype(int)
                            zfmt = ticker.StrMethodFormatter('{x:,.0f}')
                        elif optimizer_id == 'chi square':
                            error_levels = np.array([0, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e10, 1e15, 1e20, 1e30, 1e50, 1e100, 1e150]).astype(int)
                            color_bar_levels = np.array([0, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e25, 1e100, 1e150]).astype(int)
                            zfmt = ticker.FormatStrFormatter('%.2e')
                    elif color_spacing == 'log':
                        if optimizer_id == 'maximum likelihood':
                            # error_levels = [1e-1, 1, 5, 10, 100, 1e3, 5e3, 1e4, 5e4, 7.5e4, 1e5, 2.5e5, 5e5, 7.5e5, 1e6, 2.5e6, 1e15]
                            # color_bar_levels = [1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e15]
                            error_levels = [1e4, 5e4, 7.5e4, 1e5, 2.5e5, 5e5, 7.5e5, 1e6, 2.5e6, 5e6, 1e7]
                            color_bar_levels = [1e4, 1e5, 1e6, 1e7]
                            # zfmt = ticker.StrMethodFormatter('{x:,f}')
                            zfmt = ticker.FormatStrFormatter('%.2e')
                        elif optimizer_id == 'g-test':
                            error_levels = [1e-2, 1e-1, 1, 2.5, 5, 10, 25, 50, 100, 250, 500, 750, 1e3, 2.5e3, 5e3, 7.5e3, 1e4]
                            color_bar_levels = [1e-2, 1e-1, 1, 10, 100, 1e3, 1e4]
                            zfmt = ticker.FormatStrFormatter('%.2e')
                        elif optimizer_id == 'chi square':
                            error_levels = np.geomspace(1e-3, 1e6, num=10).tolist() + [1e10, 1e15, 1e20, 1e30, 1e50, 1e100, 1e150]
                            color_bar_levels = [1e-3, 1, 1e5, 1e10, 1e25, 1e50, 1e75, 1e100, 1e125, 1e150]
                            zfmt = ticker.FormatStrFormatter('%.2e')
                    else:
                        raise ValueError("invalid color_spacing: {}".format(color_spacing))
                else:
                    raise ValueError("not yet implemented; this method only works for extreme_parameter == 'speed'")
            else:
                raise ValueError("not yet implemented; this method only works for event_type == 'CME'")
        else:
            error_levels = np.copy(levels)
            color_bar_levels = np.copy(levels)
            zfmt = ticker.StrMethodFormatter('{x:,f}')
        return color_spacing, error_levels, color_bar_levels, zfmt

    def get_mutually_common_distribution_ids(self):
        series_to_distribution_mapping = dict()
        for i, series in enumerate(self.series):
            series_to_distribution_mapping[i] = list()
            for distribution_id in ('normal distribution', 'lognormal distribution'):
                if distribution_id in list(series.keys()):
                    series_to_distribution_mapping[i].append(distribution_id)
        distribution_ids = set(series_to_distribution_mapping[0])
        if self.n > 1:
            for i in range(1, self.n):
                distribution_ids = distribution_ids.intersection(
                    set(series_to_distribution_mapping[i]))
        distribution_ids = np.sort(list(distribution_ids))
        if distribution_ids.size == 0:
            raise ValueError("could not find distribution_ids common to all series")
        return distribution_ids

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
                    savename = 'RawAnalysis_Inter-Exceedance_Histogram'
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

    def subview_random_variable(self, ax, series, distribution_id, layout, facecolor, i, xs, ys, show_chronological_distribution=False, show_empirical_accumulation=False, show_empirical_survival=False, alpha=1):
        show_values = np.array([show_chronological_distribution, show_empirical_accumulation, show_empirical_survival])
        nshow = np.sum(show_values)
        if nshow != 1:
            raise ValueError("only one of the following inputs should be True: show_chronological_distribution, show_empirical_accumulation, show_empirical_survival")
        parametric_distribution = series[distribution_id]
        _ymin = np.nanmin(parametric_distribution.us)
        if _ymin < 0:
            ys.append(_ymin * 1.125)
        else:
            ys.append(_ymin / 1.125)
        ys.append(np.nanmax(parametric_distribution.us) * 1.125)
        if show_chronological_distribution:
            x = series['data']['datetime']
            dts = date2num(x)
            xs.append(np.min(dts))
            xs.append(np.max(dts))
            if (layout == 'overlay' and self.n > 1):
                label = '{}'.format(series['identifiers']['series id'])
            else:
                if i == 0:
                    label = 'Chronological Distribution'
                else:
                    label = None
            ax.scatter(
                x,
                parametric_distribution.us,
                facecolor=facecolor,
                alpha=alpha,
                label=label,
                marker='.',
                s=3)
        elif show_empirical_accumulation:
            x = np.arange(parametric_distribution.us.size).astype(int) + 1
            xs.append(np.min(x))
            xs.append(np.max(x))
            if (layout == 'overlay' and self.n > 1):
                label = '{}'.format(series['identifiers']['series id'])
            else:
                if i == 0:
                    label = 'Empirical Cumulative Distribution'
                else:
                    label = None
            ax.scatter(
                x,
                parametric_distribution.vs,
                facecolor=facecolor,
                alpha=alpha,
                label=label,
                marker='.',
                s=3)
        else: # show_empirical_survival
            x = np.arange(parametric_distribution.us.size).astype(int) + 1
            xs.append(np.min(x))
            xs.append(np.max(x))
            if (layout == 'overlay' and self.n > 1):
                label = '{}'.format(series['identifiers']['series id'])
            else:
                if i == 0:
                    label = 'Empirical Survival Distribution'
                else:
                    label = None
            ax.scatter(
                x,
                parametric_distribution.vs[::-1],
                facecolor=facecolor,
                alpha=alpha,
                label=label,
                marker='.',
                s=3)
        return xs, ys

    def view_random_variable(self, distribution_ids=None, show_chronological_distribution=False, show_empirical_accumulation=False, show_empirical_survival=False, facecolors=('darkorange', 'steelblue', 'green', 'purple'), fmt='%Y-%m', sharex=False, sharey=False, collapse_x=False, collapse_y=False, figsize=None, save=False, layout='single'):
        if layout is None:
            permutable_layouts = ['single']
            if self.n > 1:
                permutable_layouts.append('overlay')
            if 2 <= self.n < 4:
                permutable_layouts.append('horizontal')
            elif (self.n > 2) and (self.n % 2 == 0):
                permutable_layouts.append('square')
            self.view_layout_permutations(
                f=self.view_random_variable,
                layouts=permutable_layouts,
                distribution_ids=distribution_ids,
                show_chronological_distribution=show_chronological_distribution,
                show_empirical_accumulation=show_empirical_accumulation,
                show_empirical_survival=show_empirical_survival,
                facecolors=facecolors,
                fmt=fmt,
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
                visualizer.view_random_variable(
                    distribution_ids=distribution_ids,
                    show_chronological_distribution=show_chronological_distribution,
                    show_empirical_accumulation=show_empirical_accumulation,
                    show_empirical_survival=show_empirical_survival,
                    facecolors=facecolors,
                    fmt=fmt,
                    sharex=sharex,
                    sharey=sharey,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y,
                    figsize=figsize,
                    save=save,
                    layout='overlay')
        else:
            ## verify inputs
            show_values = np.array([show_chronological_distribution, show_empirical_accumulation, show_empirical_survival])
            nshow = np.sum(show_values)
            if nshow < 1:
                raise ValueError("at least one of the following inputs should be True: show_chronological_distribution, show_empirical_accumulation, show_empirical_survival")
            if isinstance(facecolors, str):
                facecolors = [facecolors]
            if self.n > 1:
                nc = len(facecolors)
                if nc < self.n:
                    raise ValueError("{} facecolors for {} series".format(nc, self.n))
            if distribution_ids is None:
                distribution_ids = self.get_mutually_common_distribution_ids()
            elif isinstance(distribution_ids, str):
                distribution_ids = [distribution_ids]
            elif not isinstance(distribution_ids, (tuple, list, np.ndarray)):
                raise ValueError("invalid type(distribution_ids): {}".format(type(distribution_ids)))
            for distribution_id in distribution_ids:
                substrings = ('show_chronological_distribution', 'show_empirical_accumulation', 'show_empirical_survival')
                for substring, show_value in zip(substrings, show_values):
                    if show_value:
                        show_kwargs = {substring : show_value}
                        xs, ys = [], []
                        parameter_labels, alt_parameter_labels, unit_labels = [], [], []
                        ## get figure and axes
                        kws = self.get_number_of_figure_rows_and_columns(self.n, layout)
                        fig, axes = plt.subplots(figsize=figsize, **kws)
                        if layout == 'overlay':
                            alpha = 1 / self.n
                            if alpha < 0.4:
                                alpha = 0.4
                            for i, (series, facecolor) in enumerate(zip(self.series, facecolors)):
                                parameter_label = series['parameter mapping'][series[distribution_id].extreme_parameter]
                                alt_parameter_label = series['generalized parameter mapping'][series[distribution_id].extreme_parameter]
                                unit_label = series['unit mapping'][series[distribution_id].extreme_parameter]
                                if series[distribution_id].is_log_transformed:
                                    parameter_label = 'log {}'.format(parameter_label)
                                    alt_parameter_label = 'log {}'.format(alt_parameter_label)
                                    unit_label = 'log {}'.format(unit_label)
                                parameter_labels.append(parameter_label)
                                alt_parameter_labels.append(alt_parameter_label)
                                unit_labels.append(unit_label)
                                xs, ys = self.subview_random_variable(
                                    ax=axes,
                                    series=series,
                                    distribution_id=distribution_id,
                                    layout=layout,
                                    facecolor=facecolor,
                                    i=i,
                                    xs=xs,
                                    ys=ys,
                                    alpha=alpha,
                                    **show_kwargs)
                            if substring != 'show_chronological_distribution':
                                axes.xaxis.set_minor_locator(ticker.AutoMinorLocator())
                                axes.yaxis.set_minor_locator(ticker.AutoMinorLocator())
                            handles, labels = axes.get_legend_handles_labels()
                            textcolor = 'k'
                            if self.n == 1:
                                axes.set_title(self.series[0]['identifiers']['series id'], fontsize=self.titlesize)
                            axes = np.array([axes])
                        else:
                            handles, labels = [], []
                            for i, (ax, series) in enumerate(zip(axes.ravel(), self.series)):
                                parameter_label = series['parameter mapping'][series[distribution_id].extreme_parameter]
                                alt_parameter_label = series['generalized parameter mapping'][series[distribution_id].extreme_parameter]
                                unit_label = series['unit mapping'][series[distribution_id].extreme_parameter]
                                if series[distribution_id].is_log_transformed:
                                    parameter_label = 'log {}'.format(parameter_label)
                                    alt_parameter_label = 'log {}'.format(alt_parameter_label)
                                    unit_label = 'log {}'.format(unit_label)
                                parameter_labels.append(parameter_label)
                                alt_parameter_labels.append(alt_parameter_label)
                                unit_labels.append(unit_label)
                                xs, ys = self.subview_random_variable(
                                    ax=ax,
                                    series=series,
                                    distribution_id=distribution_id,
                                    layout=layout,
                                    facecolor=facecolors[0],
                                    i=i,
                                    xs=xs,
                                    ys=ys,
                                    alpha=0.8,
                                    **show_kwargs)
                                if substring != 'show_chronological_distribution':
                                    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
                                    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
                                ax.set_title(series['identifiers']['series id'], fontsize=self.titlesize)
                                _handles, _labels = ax.get_legend_handles_labels()
                                handles.extend(_handles)
                                labels.extend(_labels)
                            hspace = 0.425 if 'vertical' in layout else 0.3
                            fig.subplots_adjust(hspace=hspace)
                            textcolor = True
                        ## update axes
                        if self.is_same_elements(elements=parameter_labels, s='', n=self.n):
                            if self.is_same_elements(elements=unit_labels, s='', n=self.n):
                                shared_ylabel = '{} [{}]'.format(parameter_labels[0], unit_labels[0])
                            else:
                                shared_ylabel = '{}'.format(parameter_labels[0])
                        else:
                            if self.is_same_elements(elements=alt_parameter_labels, s='', n=self.n):
                                if self.is_same_elements(elements=unit_labels, s='', n=self.n):
                                    shared_ylabel = '{} [{}]'.format(alt_parameter_labels[0], unit_labels[0])
                                else:
                                    shared_ylabel = '{}'.format(alt_parameter_labels[0])
                            else:
                                if self.is_same_elements(elements=unit_labels, s='', n=self.n):
                                    shared_ylabel = '{} [{}]'.format(parameter_labels[0], unit_labels[0])
                                else:
                                    shared_ylabel = 'Random Variable'
                        if substring == 'show_chronological_distribution':
                            xfmt = None
                            shared_xlabel = 'Date'
                            xticks = False
                        else:
                            xfmt = '{x:,.0f}' if np.nanmax(xs) > 1000 else '{x:,.2f}'
                            shared_xlabel = 'Number of Events'
                            xticks = True
                        self.share_axes(
                            axes=axes,
                            layout=layout,
                            xs=xs,
                            ys=ys,
                            sharex=sharex,
                            sharey=sharey,
                            xticks=xticks,
                            yticks=True,
                            xfmt=xfmt,
                            yfmt='{x:,.0f}' if np.nanmax(ys) > 1000 else '{x:,.2f}',
                            xlim=True,
                            ylim=True,
                            xlabel=shared_xlabel,
                            ylabel=shared_ylabel,
                            collapse_x=collapse_x,
                            collapse_y=collapse_y)
                        fig.align_ylabels()
                        if substring == 'show_chronological_distribution':
                            for ax in axes.ravel():
                                ax = self.subview_datetime_axis(
                                    ax=ax,
                                    axis='x',
                                    major_interval=12,
                                    minor_interval=1,
                                    sfmt=fmt)
                                ax.tick_params(
                                    axis='x',
                                    which='both',
                                    labelsize=self.ticksize,
                                    rotation=15)
                        for ax in axes.ravel():
                            self.apply_grid(ax)
                        ## update legend
                        self.subview_legend(
                            fig=fig,
                            ax=axes.ravel()[0],
                            handles=handles,
                            labels=labels,
                            title='{}\n{}'.format(
                                distribution_id.title(),
                                substring.replace('show_', '').replace('_', ' ').title()),
                            bottom=0.2,
                            textcolor=textcolor,
                            facecolor='white',
                            edgecolor='k',
                            titlecolor='k',
                            ncol=None)
                        ## update title
                        # # event_types = set()
                        # # for series in self.series:
                        # #     event_types.add(series['identifiers']['event type'])
                        # # event_types = list(event_types)
                        # s = substring.replace('show_', '').replace('_', ' ').title() + '\n'
                        # # s += ""
                        #
                        # # if self.is_same_elements(elements=parameter_labels, s='', n=self.nseries):
                        # #     if self.is_same_elements(elements=event_types, s='', n=self.nseries):
                        # #         s += "Distribution of {} {}".format(event_types[0], self.make_plural(parameter_labels[0]))
                        # #     else:
                        # #         s += "Distribution of {}".format(self.make_plural(parameter_labels[0]))
                        # fig.suptitle(s, fontsize=self.titlesize)
                        ## show / save
                        if save:
                            savename = 'RawAnalysis_{}_{}'.format(
                                distribution_id.replace('_', '-').title().replace(' ', '_'),
                                substring.replace('show_', '').title().replace('_', '-').replace(' ', '_'))
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

    def view_exceedance_probability(self, distribution_ids=None, show_chi_square=False, show_g_test=False, show_maximum_likelihood=False, facecolor='b', sharex=False, sharey=False, collapse_x=False, collapse_y=False, figsize=None, save=False, layout='single'):
        if layout is None:
            permutable_layouts = ['single']
            if 2 <= self.n < 4:
                permutable_layouts.append('horizontal')
                permutable_layouts.append('vertical')
            elif (self.n > 2) and (self.n % 2 == 0):
                permutable_layouts.append('square')
            self.view_layout_permutations(
                f=self.view_exceedance_probability,
                layouts=permutable_layouts,
                distribution_ids=distribution_ids,
                show_chi_square=show_chi_square,
                show_g_test=show_g_test,
                show_maximum_likelihood=show_maximum_likelihood,
                facecolor=facecolor,
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
                visualizer.view_exceedance_probability(
                    distribution_ids=distribution_ids,
                    show_chi_square=show_chi_square,
                    show_g_test=show_g_test,
                    show_maximum_likelihood=show_maximum_likelihood,
                    facecolor=facecolor,
                    sharex=sharex,
                    sharey=sharey,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y,
                    figsize=figsize,
                    save=save,
                    layout='overlay')
        else:
            nfits = np.sum([show_chi_square, show_g_test, show_maximum_likelihood])
            if nfits < 1:
                raise ValueError("at least one of the following inputs must be True: show_chi_square, show_g_test, show_maximum_likelihood")
            if distribution_ids is None:
                distribution_ids = self.get_mutually_common_distribution_ids()
            elif isinstance(distribution_ids, str):
                distribution_ids = [distribution_ids]
            elif not isinstance(distribution_ids, (tuple, list, np.ndarray)):
                raise ValueError("invalid type(distribution_ids): {}".format(type(distribution_ids)))
            for distribution_id in distribution_ids:
                for show_id, show_value in zip(('show_chi_square', 'show_g_test', 'show_maximum_likelihood'), (show_chi_square, show_g_test, show_maximum_likelihood)):
                    if show_value:
                        kwargs = {show_id : show_value}
                        _optimizer_id = self.get_exclusive_optimizer_id(
                            **kwargs)
                        optimizer_id = _optimizer_id.replace(' ', '_').replace('-', '_')
                        shared_xlabel = 'Exceedance Probability'
                        xs, ys = [], []
                        handles, labels = [], []
                        parameter_labels, alt_parameter_labels, unit_labels = [], [], []
                        ## get figure and axes
                        kws = self.get_number_of_figure_rows_and_columns(self.n, layout)
                        fig, axes = plt.subplots(figsize=figsize, **kws)
                        ## initialize plot
                        if layout == 'overlay':
                            if self.n != 1:
                                raise ValueError("layout='overlay' for this method will only work for one series")
                            axes = np.array([axes])
                            textcolor = 'k'
                        else:
                            textcolor = True
                        for i, (ax, series) in enumerate(zip(axes.ravel(), self.series)):
                            parametric_distribution = series[distribution_id]
                            optimization_result = getattr(parametric_distribution, optimizer_id)
                            calculation_prms = optimization_result['calculation parameters']
                            parameter_label = series['parameter mapping'][parametric_distribution.extreme_parameter]
                            alt_parameter_label = series['generalized parameter mapping'][parametric_distribution.extreme_parameter]
                            unit_label = series['unit mapping'][parametric_distribution.extreme_parameter]
                            if parametric_distribution.is_log_transformed:
                                parameter_label = 'log {}'.format(parameter_label)
                                alt_parameter_label = 'log {}'.format(alt_parameter_label)
                                unit_label = 'log {}'.format(unit_label)
                            parameter_labels.append(parameter_label)
                            alt_parameter_labels.append(alt_parameter_label)
                            unit_labels.append(unit_label)
                            if distribution_id == 'normal distribution':
                                (mu, sigma) = calculation_prms
                                smqp = sm.ProbPlot(
                                    parametric_distribution.vs,
                                    loc=mu,
                                    scale=sigma,
                                    dist=SPstats.norm)
                                pp = smqp.probplot(
                                    parametric_distribution.vs,
                                    ax=ax,
                                    alpha=0.3,
                                    marker='o',
                                    markersize=2,
                                    markeredgecolor=facecolor,
                                    exceed=True)
                            elif distribution_id == 'lognormal distribution':
                                (mu, sigma) = calculation_prms
                                smqp = sm.ProbPlot(
                                    parametric_distribution.vs,
                                    loc=mu,
                                    scale=np.exp(mu),
                                    dist=SPstats.lognorm,
                                    distargs=(sigma,))
                                pp = smqp.probplot(
                                    parametric_distribution.vs,
                                    ax=ax,
                                    alpha=0.3,
                                    marker='o',
                                    markersize=2,
                                    markeredgecolor=facecolor,
                                    exceed=True)
                            else:
                                raise ValueError("invalid distribution_id: {}".format(distribution_id))
                            xs.extend(ax.get_xlim())
                            ys.extend(ax.get_ylim())
                            ax.set_xlabel(shared_xlabel, fontsize=self.labelsize)
                            ax.set_ylabel('{} [{}]'.format(parameter_label, unit_label), fontsize=self.labelsize)
                            ax.tick_params(axis='both', labelsize=7)
                            ax.grid(color='k', linestyle=':', alpha=0.3)
                            ax.set_title(series['identifiers']['series id'], fontsize=self.titlesize)
                            if i == 0:
                                prob_label = r'Exceedance Probability'
                                prob_handle = self.get_customized_scatter_handle(
                                    facecolor=facecolor,
                                    label=prob_label,
                                    marker='o',
                                    markersize=2,
                                    alpha=0.3)
                                handles.append(prob_handle)
                                labels.append(prob_label)
                        hspace = 0.425 if 'vertical' in layout else 0.475
                        fig.subplots_adjust(hspace=hspace)
                        ## update axes
                        yfmt = '{x:,.0f}'
                        if self.is_same_elements(elements=parameter_labels, s='', n=self.n):
                            if self.is_same_elements(elements=unit_labels, s='', n=self.n):
                                shared_ylabel = '{} [{}]'.format(parameter_labels[0], unit_labels[0])
                            else:
                                shared_ylabel = '{}'.format(parameter_labels[0])
                        else:
                            if self.is_same_elements(elements=alt_parameter_labels, s='', n=self.n):
                                if self.is_same_elements(elements=unit_labels, s='', n=self.n):
                                    shared_ylabel = '{} [{}]'.format(alt_parameter_labels[0], unit_labels[0])
                                else:
                                    shared_ylabel = '{}'.format(alt_parameter_labels[0])
                            else:
                                if self.is_same_elements(elements=unit_labels, s='', n=self.n):
                                    shared_ylabel = '{} [{}]'.format(parameter_labels[0], unit_labels[0])
                                else:
                                    shared_ylabel = 'Random Variable'
                        self.share_axes(
                            axes=axes,
                            layout=layout,
                            xs=xs,
                            ys=ys,
                            sharex=sharex,
                            sharey=sharey,
                            xticks=False,
                            yticks=False,
                            xfmt=None,
                            yfmt=yfmt,
                            # xlim=True,
                            xlim=[0.01, 100],
                            ylim=True,
                            xlabel=shared_xlabel,
                            ylabel=shared_ylabel,
                            collapse_x=collapse_x,
                            collapse_y=collapse_y)
                        fig.align_ylabels()
                        for ax in axes.ravel():
                            self.apply_grid(ax)
                        ## update legend
                        self.subview_legend(
                            fig=fig,
                            ax=axes.ravel()[0],
                            handles=handles,
                            labels=labels,
                            title='{}'.format(distribution_id.title()),
                            bottom=0.2,
                            textcolor=textcolor,
                            facecolor='white',
                            edgecolor='k',
                            titlecolor='k',
                            ncol=None)
                        ## update title
                        # s = 'Exceedance Probability'
                        # fig.suptitle(s, fontsize=self.titlesize)
                        ## show / save
                        if save:
                            savename = 'RawAnalysis_{}_ExceedanceProb'.format(
                                distribution_id.replace('_', '-').title().replace(' ', '_'))
                            if show_id == 'show_chi_square':
                                savename = '{}_CSQ'.format(savename)
                            elif show_id == 'show_g_test':
                                savename = '{}_GT'.format(savename)
                            elif show_id == 'show_maximum_likelihood':
                                savename = '{}_MLE'.format(savename)
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

    def view_qq(self, distribution_ids=None, show_chi_square=False, show_g_test=False, show_maximum_likelihood=False, quantile_color='b', line_color='r', sharex=False, sharey=False, collapse_x=False, collapse_y=False, figsize=None, save=False, layout='single'):
        if layout is None:
            permutable_layouts = ['single']
            if 2 <= self.n < 4:
                permutable_layouts.append('horizontal')
                permutable_layouts.append('vertical')
            elif (self.n > 2) and (self.n % 2 == 0):
                permutable_layouts.append('square')
            self.view_layout_permutations(
                f=self.view_qq,
                layouts=permutable_layouts,
                distribution_ids=distribution_ids,
                show_chi_square=show_chi_square,
                show_g_test=show_g_test,
                show_maximum_likelihood=show_maximum_likelihood,
                quantile_color=quantile_color,
                line_color=line_color,
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
                visualizer.view_qq(
                    distribution_ids=distribution_ids,
                    show_chi_square=show_chi_square,
                    show_g_test=show_g_test,
                    show_maximum_likelihood=show_maximum_likelihood,
                    quantile_color=quantile_color,
                    line_color=line_color,
                    sharex=sharex,
                    sharey=sharey,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y,
                    figsize=figsize,
                    save=save,
                    layout='overlay')
        else:
            nfits = np.sum([show_chi_square, show_g_test, show_maximum_likelihood])
            if nfits < 1:
                raise ValueError("at least one of the following inputs must be True: show_chi_square, show_g_test, show_maximum_likelihood")
            nfits = np.sum([show_chi_square, show_g_test, show_maximum_likelihood])
            if nfits < 1:
                raise ValueError("at least one of the following inputs must be True: show_chi_square, show_g_test, show_maximum_likelihood")
            if distribution_ids is None:
                distribution_ids = self.get_mutually_common_distribution_ids()
            elif isinstance(distribution_ids, str):
                distribution_ids = [distribution_ids]
            elif not isinstance(distribution_ids, (tuple, list, np.ndarray)):
                raise ValueError("invalid type(distribution_ids): {}".format(type(distribution_ids)))
            for distribution_id in distribution_ids:
                for show_id, show_value in zip(('show_chi_square', 'show_g_test', 'show_maximum_likelihood'), (show_chi_square, show_g_test, show_maximum_likelihood)):
                    if show_value:
                        kwargs = {show_id : show_value}
                        _optimizer_id = self.get_exclusive_optimizer_id(
                            **kwargs)
                        optimizer_id = _optimizer_id.replace(' ', '_').replace('-', '_')
                        shared_xlabel = 'Theoretical Quantiles'
                        shared_ylabel = 'Sample Quantiles'
                        xs, ys = [], []
                        handles, labels = [], []
                        parameter_labels, alt_parameter_labels, unit_labels = [], [], []
                        ## get figure and axes
                        kws = self.get_number_of_figure_rows_and_columns(self.n, layout)
                        fig, axes = plt.subplots(figsize=figsize, **kws)
                        ## initialize plot
                        if layout == 'overlay':
                            if self.n != 1:
                                raise ValueError("layout='overlay' for this method will only work for one series")
                            axes = np.array([axes])
                            textcolor = 'k'
                        else:
                            textcolor = True
                        line_style = '-' # '--'
                        fmt = '{}{}'.format(line_color, line_style)
                        for i, (ax, series) in enumerate(zip(axes.ravel(), self.series)):
                            parametric_distribution = series[distribution_id]
                            optimization_result = getattr(parametric_distribution, optimizer_id)
                            calculation_prms = optimization_result['calculation parameters']
                            parameter_label = series['parameter mapping'][parametric_distribution.extreme_parameter]
                            alt_parameter_label = series['generalized parameter mapping'][parametric_distribution.extreme_parameter]
                            unit_label = series['unit mapping'][parametric_distribution.extreme_parameter]
                            if series[distribution_id].is_log_transformed:
                                parameter_label = 'log {}'.format(parameter_label)
                                alt_parameter_label = 'log {}'.format(alt_parameter_label)
                                unit_label = 'log {}'.format(unit_label)
                            parameter_labels.append(parameter_label)
                            alt_parameter_labels.append(alt_parameter_label)
                            unit_labels.append(unit_label)
                            if distribution_id == 'normal distribution':
                                (mu, sigma) = calculation_prms
                                smqp = sm.ProbPlot(
                                    parametric_distribution.vs,
                                    loc=mu,
                                    scale=sigma,
                                    dist=SPstats.norm)
                                qq = smqp.qqplot(
                                    parametric_distribution.vs,
                                    line='45',
                                    ax=ax,
                                    alpha=0.3,
                                    marker='o',
                                    markersize=2,
                                    markeredgecolor=quantile_color)
                                sm.qqline(
                                    np.array(qq.axes).ravel()[i],
                                    line='45',
                                    fmt=fmt)
                            elif distribution_id == 'lognormal distribution':
                                (mu, sigma) = calculation_prms
                                smqp = sm.ProbPlot(
                                    parametric_distribution.vs,
                                    loc=mu,
                                    scale=np.exp(mu),
                                    dist=SPstats.lognorm,
                                    distargs=(sigma,))
                                qq = smqp.qqplot(
                                    parametric_distribution.vs,
                                    ax=ax,
                                    alpha=0.3,
                                    marker='o',
                                    markersize=2,
                                    markeredgecolor=quantile_color)
                                sm.qqline(
                                    np.array(qq.axes).ravel()[i],
                                    line='45',
                                    fmt=fmt)
                            else:
                                raise ValueError("invalid distribution_id: {}".format(distribution_id))
                            xs.extend(ax.get_xlim())
                            ys.extend(ax.get_ylim())
                            ax.set_xlabel(shared_xlabel, fontsize=self.labelsize)
                            ax.set_ylabel(shared_ylabel, fontsize=self.labelsize)
                            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
                            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
                            ax.tick_params(axis='both', labelsize=self.ticksize)
                            ax.set_title(series['identifiers']['series id'], fontsize=self.titlesize)
                            if i == 0:
                                quantile_label = 'Quantiles'
                                line_label = r'$45 \degree$ Line'
                                quantile_handle = self.get_customized_scatter_handle(
                                    facecolor=quantile_color,
                                    label=quantile_label,
                                    marker='o',
                                    markersize=2,
                                    alpha=0.3)
                                line_handle = self.get_customized_line_handle(
                                    facecolor=line_color,
                                    label=line_label)
                                handles.extend([quantile_handle, line_handle])
                                labels.extend([quantile_label, line_label])
                        hspace = 0.425 if 'vertical' in layout else 0.3
                        fig.subplots_adjust(hspace=hspace)
                        ## update axes
                        xfmt = '{x:,.0f}'
                        yfmt = '{x:,.0f}'
                        if self.is_same_elements(elements=parameter_labels, s='', n=self.n):
                            if self.is_same_elements(elements=unit_labels, s='', n=self.n):
                                shared_xlabel = '{}\n{} [{}]'.format(shared_xlabel, parameter_labels[0], unit_labels[0])
                                shared_ylabel = '{}\n{} [{}]'.format(shared_ylabel, parameter_labels[0], unit_labels[0])
                            else:
                                shared_xlabel = '{}\n{}'.format(shared_xlabel, parameter_labels[0])
                                shared_ylabel = '{}\n{}'.format(shared_ylabel, parameter_labels[0])
                        else:
                            if self.is_same_elements(elements=alt_parameter_labels, s='', n=self.n):
                                if self.is_same_elements(elements=unit_labels, s='', n=self.n):
                                    shared_xlabel = '{}\n{} [{}]'.format(shared_xlabel, alt_parameter_labels[0], unit_labels[0])
                                    shared_ylabel = '{}\n{} [{}]'.format(shared_ylabel, alt_parameter_labels[0], unit_labels[0])
                                else:
                                    shared_xlabel = '{}\n{}'.format(shared_xlabel, alt_parameter_labels[0])
                                    shared_ylabel = '{}\n{}'.format(shared_ylabel, alt_parameter_labels[0])
                            else:
                                if self.is_same_elements(elements=unit_labels, s='', n=self.n):
                                    shared_xlabel = '{}\n{} [{}]'.format(shared_xlabel, parameter_labels[0], unit_labels[0])
                                    shared_ylabel = '{}\n{} [{}]'.format(shared_ylabel, parameter_labels[0], unit_labels[0])
                        self.share_axes(
                            axes=axes,
                            layout=layout,
                            xs=xs,
                            ys=ys,
                            sharex=sharex,
                            sharey=sharey,
                            xticks=True,
                            yticks=True,
                            xfmt=xfmt,
                            yfmt=yfmt,
                            xlim=True,
                            ylim=True,
                            xlabel=shared_xlabel,
                            ylabel=shared_ylabel,
                            collapse_x=collapse_x,
                            collapse_y=collapse_y)
                        fig.align_ylabels()
                        for ax in axes.ravel():
                            self.apply_grid(ax)
                        ## update legend
                        self.subview_legend(
                            fig=fig,
                            ax=axes.ravel()[0],
                            handles=handles,
                            labels=labels,
                            title='{}'.format(distribution_id.title()),
                            bottom=0.2,
                            textcolor=textcolor,
                            facecolor='white',
                            edgecolor='k',
                            titlecolor='k',
                            ncol=None)
                        ## update title
                        s = 'Q-Q Plot'
                        fig.suptitle(s, fontsize=self.titlesize)
                        ## show / save
                        if save:
                            savename = 'RawAnalysis_{}_QQ'.format(
                                distribution_id.replace('_', '-').title().replace(' ', '_'))
                            if show_id == 'show_chi_square':
                                savename = '{}_CSQ'.format(savename)
                            elif show_id == 'show_g_test':
                                savename = '{}_GT'.format(savename)
                            elif show_id == 'show_maximum_likelihood':
                                savename = '{}_MLE'.format(savename)
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

    def subview_kernel_density_estimation(self, ax, series, distribution_id, density_id, histogram_id, kde_colors, kde_style, i, xs, ys, swap_xy=False):
        parametric_distribution = series[distribution_id]
        n = len(parametric_distribution.kernel_density_estimation)
        linestyles = itertools.cycle(['-', ':', '-.', '--'])
        _kde_colors = itertools.cycle(kde_colors)
        if kde_style == 'curve':
            alpha = 0.75
        elif kde_style == 'fill':
            alpha = 1 / n
        else:
            raise ValueError("invalid kde_style: {}".format(kde_style))
        for kernel_density_estimation, linestyle, facecolor in zip(parametric_distribution.kernel_density_estimation, linestyles, _kde_colors):
            x = kernel_density_estimation['x']
            if density_id == 'probability':
                y = kernel_density_estimation['y']
            elif density_id == 'observed':
                if histogram_id == 'original':
                    histogram = parametric_distribution.original_histogram
                elif histogram_id == 'threshold':
                    histogram = parametric_distribution.threshold_histogram
                else:
                    raise ValueError("invalid histogram_id: {}".format(histogram_id))
                y = kernel_density_estimation['y'] * histogram.normalization_constant
            else:
                raise ValueError("invalid density_id: {}; not yet implemented".format(density_id))
            if i == 0:
                label = 'Gaussian KDE (bandwidth = {})'.format(kernel_density_estimation['bandwidth'])
            else:
                label = None
            if swap_xy:
                if kde_style == 'curve':
                    ax.plot(
                        y,
                        x,
                        color=facecolor,
                        label=label,
                        linestyle=linestyle,
                        alpha=alpha)
                else: # elif kde_style == 'fill':
                    ax.fill_between(
                        y,
                        np.zeros(y.shape),
                        x,
                        color=facecolor,
                        alpha=alpha,
                        label=label)
            else:
                if kde_style == 'curve':
                    ax.plot(
                        x,
                        y,
                        color=facecolor,
                        label=label,
                        linestyle=linestyle,
                        alpha=alpha)
                else: # elif kde_style == 'fill':
                    ax.fill_between(
                        x,
                        np.zeros(y.shape),
                        y,
                        color=facecolor,
                        alpha=alpha,
                        label=label)
            xs.append(x[0])
            xs.append(x[-1])
            ys.append(0)
            ys.append(np.nanmax(y) * 1.125)
        return xs, ys

    def subview_distribution_histogram(self, ax, series, distribution_id, density_id, histogram_id, bar_color, step_color, i, xs, ys, alpha=1, swap_xy=False, separate_label=None):
        parametric_distribution = series[distribution_id]
        if histogram_id == 'original':
            histogram = parametric_distribution.original_histogram
            label = 'Histogram' if i == 0 else None
        elif histogram_id == 'threshold':
            histogram = parametric_distribution.threshold_histogram
            label = 'Thresholded Histogram' if i == 0 else None
        else:
            raise ValueError("invalid histogram_id: {}".format(histogram_id))
        if separate_label is not None:
            label = separate_label[:]
        if density_id == 'observed':
            hvalues = np.copy(histogram.counts)
        else: # if density_id == 'probability':
            hvalues = np.copy(histogram.probability_density)
        if swap_xy:
            if (bar_color is None) and (step_color is None):
                raise ValueError("input either 'bar_color' or 'step_color'")
            elif (bar_color is not None) and (step_color is None):
                ax.barh(
                    histogram.midpoints,
                    hvalues,
                    height=histogram.bin_widths,
                    color=bar_color,
                    alpha=alpha,
                    label=label)
            elif (bar_color is None) and (step_color is not None):
                ax.step(
                    hvalues,
                    histogram.midpoints,
                    color=step_color,
                    linewidth=0.5,
                    where='mid',
                    alpha=alpha,
                    label=label)
            elif (bar_color is not None) and (step_color is not None):
                ax.barh(
                    histogram.midpoints,
                    hvalues,
                    height=histogram.bin_widths,
                    color=bar_color,
                    edgecolor=step_color,
                    alpha=alpha,
                    label=label)
        else:
            if (bar_color is None) and (step_color is None):
                raise ValueError("input either 'bar_color' or 'step_color'")
            elif (bar_color is not None) and (step_color is None):
                ax.bar(
                    histogram.midpoints,
                    hvalues,
                    width=histogram.bin_widths,
                    color=bar_color,
                    alpha=alpha,
                    label=label)
            elif (bar_color is None) and (step_color is not None):
                ax.step(
                    histogram.midpoints,
                    hvalues,
                    color=step_color,
                    linewidth=0.5,
                    where='mid',
                    alpha=alpha,
                    label=label)
            elif (bar_color is not None) and (step_color is not None):
                ax.bar(
                    histogram.midpoints,
                    hvalues,
                    width=histogram.bin_widths,
                    color=bar_color,
                    edgecolor=step_color,
                    alpha=alpha,
                    label=label)
        xs.append(histogram.edges[0])
        xs.append(histogram.edges[-1])
        ys.append(np.nanmax(hvalues) * 1.125)
        return xs, ys, histogram

    def subview_optimized_fit(self, ax, series, distribution_id, optimization_metric, csq_color, gt_color, mle_color, i, xs, ys, show_chi_square=False, show_g_test=False, show_maximum_likelihood=False, alpha=1, swap_xy=False, separate_label=None):
        container_of_args = [
            show_chi_square,
            show_g_test,
            show_maximum_likelihood]
        container_of_kwargs = [
            dict(show_chi_square=show_chi_square),
            dict(show_g_test=show_g_test),
            dict(show_maximum_likelihood=show_maximum_likelihood)]
        container_of_colors = [
            csq_color,
            gt_color,
            mle_color]
        for args, kwargs, facecolor in zip(container_of_args, container_of_kwargs, container_of_colors):
            if args:
                parametric_distribution = series[distribution_id]
                _optimizer_id = self.get_exclusive_optimizer_id(**kwargs)
                optimizer_id = _optimizer_id.replace(' ', '_').replace('-', '_')
                optimization_result = getattr(parametric_distribution, optimizer_id)
                if optimization_metric in ('probability density', 'observed density', 'observed frequency'):
                    yfit = optimization_result[optimization_metric]
                else:
                    raise ValueError("invalid optimization_metric: {}".format(optimization_metric))
                if separate_label is None:
                    label = _optimizer_id.title() if i == 0 else None
                else:
                    label = '{} {}'.format(_optimizer_id.title(), separate_label)
                if swap_xy:
                    ax.plot(
                        yfit,
                        parametric_distribution.vs,
                        color=facecolor,
                        alpha=alpha,
                        label=label)
                else:
                    ax.plot(
                        parametric_distribution.vs,
                        yfit,
                        color=facecolor,
                        alpha=alpha,
                        label=label)
                xs.append(parametric_distribution.vs[0])
                xs.append(parametric_distribution.vs[-1])
                ys.append(np.nanmax(yfit) * 1.125)
        return xs, ys

    def subview_statistics(self, ax, series, distribution_id, unit_label, i, mu_color, sigma_color, median_color=None, mode_color=None, show_chi_square=False, show_g_test=False, show_maximum_likelihood=False, swap_xy=False, separate_label=None):
        parametric_distribution = series[distribution_id]
        _optimizer_id = self.get_exclusive_optimizer_id(
            show_chi_square=show_chi_square,
            show_g_test=show_g_test,
            show_maximum_likelihood=show_maximum_likelihood)
        optimizer_id = _optimizer_id.replace(' ', '_').replace('-', '_')
        optimization_result = getattr(parametric_distribution, optimizer_id)
        true_prms = optimization_result['true parameters']
        event_type = series['identifiers']['event type']
        nevents_label = '{:,} {}'.format(parametric_distribution.vs.size, self.make_plural(event_type))
        if distribution_id in ('normal distribution', 'lognormal distribution'):
            if swap_xy:
                yloc = 0.95
                verticalalignment = 'top'
                ax.axhline(
                    y=true_prms[0],
                    color=mu_color,
                    linestyle=':',
                    label=r'$\mu_{opt}$' if i == 0 else None,
                    alpha=0.7)
                ax.axhline(
                    y=true_prms[1],
                    color=sigma_color,
                    linestyle=':',
                    label=r'$\sigma_{opt}$' if i == 0 else None,
                    alpha=0.7)
                optimization_label = self.get_optimization_label(
                    distribution_id=distribution_id,
                    optimization_result=optimization_result,
                    unit_label=unit_label)
                horizontalalignment = 'right'
                xloc = 0.95
                if distribution_id == 'lognormal distribution':
                    additional_prms = optimization_result['additional parameters']
                    ax.axhline(
                        y=additional_prms['median'],
                        color=median_color,
                        linestyle=':',
                        label=r'$median_{opt}$' if i == 0 else None,
                        alpha=0.7)
                    ax.axhline(
                        y=additional_prms['mode'],
                        color=mode_color,
                        linestyle=':',
                        label=r'$mode_{opt}$' if i == 0 else None,
                        alpha=0.7)
                    horizontalalignment = 'right'
                    xloc = 0.95
            else:
                yloc = 0.95
                verticalalignment = 'top'
                ax.axvline(
                    x=true_prms[0],
                    color=mu_color,
                    linestyle=':',
                    label=r'$\mu_{opt}$' if i == 0 else None,
                    alpha=0.7)
                ax.axvline(
                    x=true_prms[1],
                    color=sigma_color,
                    linestyle=':',
                    label=r'$\sigma_{opt}$' if i == 0 else None,
                    alpha=0.7)
                optimization_label = self.get_optimization_label(
                    distribution_id=distribution_id,
                    optimization_result=optimization_result,
                    unit_label=unit_label)
                if distribution_id == 'normal distribution':
                    horizontalalignment = 'left'
                    xloc = 0.05
                else: ## lognormal
                    additional_prms = optimization_result['additional parameters']
                    ax.axvline(
                        x=additional_prms['median'],
                        color=median_color,
                        linestyle=':',
                        label=r'$median_{opt}$' if i == 0 else None,
                        alpha=0.7)
                    ax.axvline(
                        x=additional_prms['mode'],
                        color=mode_color,
                        linestyle=':',
                        label=r'$mode_{opt}$' if i == 0 else None,
                        alpha=0.7)
                    horizontalalignment = 'right'
                    xloc = 0.95
        else:
            raise ValueError("invalid distribution_id: {}".format(distribution_id))
        text_label = '{}\n{}'.format(nevents_label, optimization_label)
        if separate_label is not None:
            text_label = '{} {}'.format(text_label, separate_label)
        text_box = ax.text(
            xloc,
            yloc,
            text_label,
            fontsize=self.textsize,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            transform=ax.transAxes)
        text_box.set_bbox(dict(facecolor='gray', alpha=0.25, edgecolor='k'))

    def subview_confidence_interval(self, ax, series, distribution_id, optimization_metric, confidence_color, i, show_chi_square=False, show_g_test=False, show_maximum_likelihood=False, swap_xy=False):
        parametric_distribution = series[distribution_id]
        _optimizer_id = self.get_exclusive_optimizer_id(
            show_chi_square=show_chi_square,
            show_g_test=show_g_test,
            show_maximum_likelihood=show_maximum_likelihood)
        optimizer_id = _optimizer_id.replace(' ', '_').replace('-', '_')
        optimization_result = getattr(parametric_distribution, optimizer_id)
        true_prms = optimization_result['true parameters']
        if distribution_id in ('normal distribution', 'lognormal distribution'):
            mu, sigma = true_prms[0], true_prms[1]
            if optimization_metric in ('probability density', 'observed density', 'observed frequency'):
                yfit = optimization_result[optimization_metric]
            else:
                raise ValueError("invalid optimization_metric: {}".format(optimization_metric))
            js = list(range(1,4))
            alphas = np.cumsum(np.ones(len(js)) * 0.05)[::-1]
            for j, alpha in zip(js, alphas):
                if i == 0:
                    if j == 1:
                        label = r'$\mu$ $\pm$ $\sigma$'
                    else:
                        label = r'$\mu$ $\pm$ ${}\sigma$'.format(j)
                else:
                    label = None
                # alpha = 1/j
                if swap_xy:
                    ax.fill_betweenx(
                        parametric_distribution.vs,
                        yfit,
                        where=((mu - j * sigma) <= parametric_distribution.vs) & (parametric_distribution.vs <= (mu + j * sigma)),
                        color=confidence_color,
                        label=label,
                        alpha=alpha)
                else:
                    ax.fill_between(
                        parametric_distribution.vs,
                        yfit,
                        where=((mu - j * sigma) <= parametric_distribution.vs) & (parametric_distribution.vs <= (mu + j * sigma)),
                        color=confidence_color,
                        label=label,
                        alpha=alpha)
        else:
            raise ValueError("invalid distribution_id: {}".format(parametric_distribution.distribution_id))

    def subview_filled_tail(self, ax, series, distribution_id, optimization_metric, extreme_value, extreme_condition, tail_fill_color, i, show_chi_square=False, show_g_test=False, show_maximum_likelihood=False):
        if not isinstance(extreme_value, (float, int)):
            raise ValueError("invalid type(extreme_value): {}".format(type(extreme_value)))
        parametric_distribution = series[distribution_id]
        if parametric_distribution.is_log_transformed:
            ev = np.log(extreme_value)
        else:
            ev = extreme_value
        _optimizer_id = self.get_exclusive_optimizer_id(
            show_chi_square=show_chi_square,
            show_g_test=show_g_test,
            show_maximum_likelihood=show_maximum_likelihood)
        optimizer_id = _optimizer_id.replace(' ', '_').replace('-', '_')
        optimization_result = getattr(parametric_distribution, optimizer_id)
        if optimization_metric in ('probability density', 'observed density', 'observed frequency'):
            yfit = optimization_result[optimization_metric]
        else:
            raise ValueError("invalid optimization_metric: {}".format(optimization_metric))
        event_searcher = EventSearcher({parametric_distribution.extreme_parameter : parametric_distribution.vs})
        _, loc = event_searcher.search_events(
            parameters=parametric_distribution.extreme_parameter,
            conditions=extreme_condition,
            values=ev)
        indices = np.zeros(yfit.size, dtype=bool)
        indices[loc] = True
        ax.fill_between(
            parametric_distribution.vs,
            np.zeros(yfit.size),
            yfit,
            where=indices,
            color=tail_fill_color,
            label='Heavy Tail' if i == 0 else None,
            alpha=0.875)
        parametric_distribution = series[distribution_id]

    def subview_arrow_to_tail(self, ax, series, distribution_id, optimization_metric, extreme_value, tail_arrow_color, show_chi_square=False, show_g_test=False, show_maximum_likelihood=False):
        if not isinstance(extreme_value, (float, int)):
            raise ValueError("invalid type(extreme_value): {}".format(type(extreme_value)))
        parametric_distribution = series[distribution_id]
        if parametric_distribution.is_log_transformed:
            ev = np.log(extreme_value)
        else:
            ev = extreme_value
        _optimizer_id = self.get_exclusive_optimizer_id(
            show_chi_square=show_chi_square,
            show_g_test=show_g_test,
            show_maximum_likelihood=show_maximum_likelihood)
        optimizer_id = _optimizer_id.replace(' ', '_').replace('-', '_')
        optimization_result = getattr(parametric_distribution, optimizer_id)
        if optimization_metric in ('probability density', 'observed density', 'observed frequency'):
            yfit = optimization_result[optimization_metric]
        else:
            raise ValueError("invalid optimization_metric: {}".format(optimization_metric))
        if distribution_id == 'normal distribution':
            loc = np.argmin(np.abs(parametric_distribution.vs - ev))
            xy = (ev, yfit[loc])
            xytext = (ev / 2, yfit[loc] * 1.125)
        elif distribution_id == 'lognormal distribution':
            loc = np.argmin(np.abs(parametric_distribution.vs - ev))
            xy = (ev, yfit[loc])
            xytext = (ev * 1.25, yfit[loc])
        else:
            raise ValueError("invalid distribution_id: {}".format(distribution_id))
        arrowprops = {
            'arrowstyle': '->',
            'color' : tail_arrow_color}
        ax.annotate(
            'Heavy Tail',
            xy=xy,
            xytext=xytext,
            fontsize=self.textsize,
            arrowprops=arrowprops)

    def subview_rug(self, ax, series, distribution_id, rug_color, i, xs, ys):
        if len(ys) == 0:
            raise ValueError("ys must contain at least one element before running this method; try running this method after running a different method (kde, histogram, fit, etc)")
        parametric_distribution = series[distribution_id]
        ys.append(0)
        _ymin, _ymax = np.nanmin(ys), np.nanmax(ys)
        if _ymin >= 0:
            if _ymin > 1:
                yfactor = -1
                ys.append(-2)
                ys.append(_ymax * 1.125)
            else:
                yfactor = np.sqrt(_ymin) * -1
                ys.append(yfactor * 1.125)
                ys.append(_ymax * 1.125)
                ys.append(_ymax / 1.125)
        else:
            yfactor = 3 * _ymin / 2
            ys.append(2 * yfactor / 3)
            ys.append(_ymax * yfactor * 1.125 / _ymin)
        vt = np.ones(parametric_distribution.vs.size) * yfactor
        label = 'Rug' if i == 0 else None
        ax.scatter(
            parametric_distribution.vs,
            vt,
            facecolor=rug_color,
            marker='|',
            s=35,
            label=label)
        xs.append(parametric_distribution.vs[0])
        xs.append(parametric_distribution.vs[-1])
        return xs, ys

    def view_distribution(self, distribution_ids=None, extreme_values=None, extreme_condition='greater than', show_rug=False, show_kde=False, show_histogram=False, show_chi_square=False, show_g_test=False, show_maximum_likelihood=False, show_statistics=False, show_confidence_interval=False, show_filled_tail=False, show_arrow_tail=False, density_id='observed', histogram_id='original', rug_color='k', bar_color='gray', step_color='k', csq_color='darkorange', gt_color='purple', mle_color='steelblue', confidence_color='darkgreen', mu_color='purple', sigma_color='blue', median_color='darkgreen', mode_color='darkred', tail_fill_color='r', tail_arrow_color='k', kde_colors='darkblue', kde_style='curve', sharex=False, sharey=False, collapse_x=False, collapse_y=False, figsize=None, save=False, layout='single'):
        if layout is None:
            permutable_layouts = ['single']
            if 2 <= self.n < 4:
                permutable_layouts.append('horizontal')
            elif (self.n > 2) and (self.n % 2 == 0):
                permutable_layouts.append('square')
            if self.n >= 2:
                permutable_layouts.append('vertical')
            self.view_layout_permutations(
                f=self.view_distribution,
                layouts=permutable_layouts,
                distribution_ids=distribution_ids,
                extreme_values=extreme_values,
                extreme_condition=extreme_condition,
                show_rug=show_rug,
                show_kde=show_kde,
                show_histogram=show_histogram,
                show_chi_square=show_chi_square,
                show_g_test=show_g_test,
                show_maximum_likelihood=show_maximum_likelihood,
                show_statistics=show_statistics,
                show_confidence_interval=show_confidence_interval,
                show_filled_tail=show_filled_tail,
                show_arrow_tail=show_arrow_tail,
                density_id=density_id,
                histogram_id=histogram_id,
                rug_color=rug_color,
                bar_color=bar_color,
                step_color=step_color,
                csq_color=csq_color,
                gt_color=gt_color,
                mle_color=mle_color,
                confidence_color=confidence_color,
                mu_color=mu_color,
                sigma_color=sigma_color,
                median_color=median_color,
                mode_color=mode_color,
                tail_fill_color=tail_fill_color,
                tail_arrow_color=tail_arrow_color,
                kde_colors=kde_colors,
                kde_style=kde_style,
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
                visualizer.view_distribution(
                    distribution_ids=distribution_ids,
                    extreme_values=extreme_values,
                    extreme_condition=extreme_condition,
                    show_rug=show_rug,
                    show_kde=show_kde,
                    show_histogram=show_histogram,
                    show_chi_square=show_chi_square,
                    show_g_test=show_g_test,
                    show_maximum_likelihood=show_maximum_likelihood,
                    show_statistics=show_statistics,
                    show_confidence_interval=show_confidence_interval,
                    show_filled_tail=show_filled_tail,
                    show_arrow_tail=show_arrow_tail,
                    density_id=density_id,
                    histogram_id=histogram_id,
                    rug_color=rug_color,
                    bar_color=bar_color,
                    step_color=step_color,
                    csq_color=csq_color,
                    gt_color=gt_color,
                    mle_color=mle_color,
                    confidence_color=confidence_color,
                    mu_color=mu_color,
                    sigma_color=sigma_color,
                    median_color=median_color,
                    mode_color=mode_color,
                    tail_fill_color=tail_fill_color,
                    tail_arrow_color=tail_arrow_color,
                    kde_colors=kde_colors,
                    kde_style=kde_style,
                    sharex=sharex,
                    sharey=sharey,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y,
                    figsize=figsize,
                    save=save,
                    layout='overlay')
        else:
            ## verify user input
            if not isinstance(extreme_values, (tuple, list, np.ndarray)):
                extreme_values = [extreme_values]
            if not any([show_kde, show_histogram, show_chi_square, show_g_test, show_maximum_likelihood, show_statistics, show_confidence_interval, show_filled_tail, show_arrow_tail]):
                raise ValueError("set at least one of the following inputs to True: 'show_kde', 'show_histogram', 'show_chi_square', 'show_g_test', 'show_maximum_likelihood', 'show_statistics', 'show_confidence_interval', 'show_filled_tail', 'show_arrow_tail'")
            if density_id not in ('observed', 'probability'):
                raise ValueError("invalid density_id: {}".format(density_id))
            if density_id == 'probability':
                optimization_metric = 'probability density'
            else:
                if show_histogram:
                    optimization_metric = 'observed density'
                else:
                    optimization_metric = 'observed frequency'
            if show_kde:
                if not isinstance(kde_colors, (tuple, list, np.ndarray)):
                    kde_colors = [kde_colors]
            if distribution_ids is None:
                distribution_ids = self.get_mutually_common_distribution_ids()
            elif isinstance(distribution_ids, str):
                distribution_ids = [distribution_ids]
            elif not isinstance(distribution_ids, (tuple, list, np.ndarray)):
                raise ValueError("invalid type(distribution_ids): {}".format(type(distribution_ids)))
            for distribution_id in distribution_ids:
                for extreme_value in extreme_values:
                    xs, ys = [0], [0]
                    parameter_labels, alt_parameter_labels, unit_labels = [], [], []
                    handles, labels = [], []
                    ## get figure and axes
                    kws = self.get_number_of_figure_rows_and_columns(self.n, layout)
                    fig, axes = plt.subplots(figsize=figsize, **kws)
                    ## initialize plot
                    if layout == 'overlay':
                        if self.n != 1:
                            raise ValueError("layout='overlay' for this method will only work for one series")
                        axes = np.array([axes])
                        textcolor = 'k'
                    else:
                        textcolor = True
                    for i, (ax, series) in enumerate(zip(axes.ravel(), self.series)):
                        parametric_distribution = series[distribution_id]
                        parameter_label = series['parameter mapping'][parametric_distribution.extreme_parameter]
                        alt_parameter_label = series['generalized parameter mapping'][parametric_distribution.extreme_parameter]
                        unit_label = series['unit mapping'][parametric_distribution.extreme_parameter]
                        if parametric_distribution.is_log_transformed:
                            parameter_label = 'log {}'.format(parameter_label)
                            alt_parameter_label = 'log {}'.format(alt_parameter_label)
                            unit_label = 'log {}'.format(unit_label)
                        parameter_labels.append(parameter_label)
                        alt_parameter_labels.append(alt_parameter_label)
                        unit_labels.append(unit_label)
                        if show_kde:
                            xs, ys = self.subview_kernel_density_estimation(
                                ax=ax,
                                series=series,
                                distribution_id=distribution_id,
                                density_id=density_id,
                                histogram_id=histogram_id,
                                kde_colors=kde_colors,
                                kde_style=kde_style,
                                i=i,
                                xs=xs,
                                ys=ys,
                                swap_xy=False)
                        if show_histogram:
                            xs, ys, _ = self.subview_distribution_histogram(
                                ax=ax,
                                series=series,
                                distribution_id=distribution_id,
                                density_id=density_id,
                                histogram_id=histogram_id,
                                bar_color=bar_color,
                                step_color=step_color,
                                i=i,
                                xs=xs,
                                ys=ys,
                                alpha=0.775 if show_rug else 1,
                                swap_xy=False,
                                separate_label=None)
                        if any([show_chi_square, show_g_test, show_maximum_likelihood]):
                            nfits = np.sum([show_chi_square, show_g_test, show_maximum_likelihood])
                            fits_alpha = 1 / nfits
                            if (fits_alpha < 0.5) or (not np.isfinite(fits_alpha)):
                                fits_alpha = 0.5
                            xs, ys = self.subview_optimized_fit(
                                ax=ax,
                                series=series,
                                distribution_id=distribution_id,
                                optimization_metric=optimization_metric,
                                csq_color=csq_color,
                                gt_color=gt_color,
                                mle_color=mle_color,
                                i=i,
                                xs=xs,
                                ys=ys,
                                show_chi_square=show_chi_square,
                                show_g_test=show_g_test,
                                show_maximum_likelihood=show_maximum_likelihood,
                                alpha=fits_alpha,
                                swap_xy=False,
                                separate_label=None)
                        if show_statistics:
                            nfits = np.sum([show_chi_square, show_g_test, show_maximum_likelihood])
                            if nfits != 1:
                                raise ValueError("this method 'show_statistics' requires that only one of the following be input as True: 'show_chi_square', 'show_g_test', 'show_maximum_likelihood'")
                            self.subview_statistics(
                                ax=ax,
                                series=series,
                                distribution_id=distribution_id,
                                unit_label=unit_label,
                                i=i,
                                mu_color=mu_color,
                                sigma_color=sigma_color,
                                median_color=median_color,
                                mode_color=mode_color,
                                show_chi_square=show_chi_square,
                                show_g_test=show_g_test,
                                show_maximum_likelihood=show_maximum_likelihood,
                                swap_xy=False,
                                separate_label=None)
                        if show_confidence_interval:
                            nfits = np.sum([show_chi_square, show_g_test, show_maximum_likelihood])
                            if nfits != 1:
                                raise ValueError("this method 'show_confidence_interval' requires that only one of the following be input as True: 'show_chi_square', 'show_g_test', 'show_maximum_likelihood'")
                            self.subview_confidence_interval(
                                ax=ax,
                                series=series,
                                distribution_id=distribution_id,
                                optimization_metric=optimization_metric,
                                confidence_color=confidence_color,
                                i=i,
                                show_chi_square=show_chi_square,
                                show_g_test=show_g_test,
                                show_maximum_likelihood=show_maximum_likelihood,
                                swap_xy=False)
                        if show_filled_tail:
                            nfits = np.sum([show_chi_square, show_g_test, show_maximum_likelihood])
                            if nfits != 1:
                                raise ValueError("this method 'show_statistics' requires that only one of the following be input as True: 'show_chi_square', 'show_g_test', 'show_maximum_likelihood'")
                            self.subview_filled_tail(
                                ax=ax,
                                series=series,
                                distribution_id=distribution_id,
                                optimization_metric=optimization_metric,
                                extreme_value=extreme_value,
                                extreme_condition=extreme_condition,
                                tail_fill_color=tail_fill_color,
                                i=i,
                                show_chi_square=show_chi_square,
                                show_g_test=show_g_test,
                                show_maximum_likelihood=show_maximum_likelihood)
                        if show_arrow_tail:
                            nfits = np.sum([show_chi_square, show_g_test, show_maximum_likelihood])
                            if nfits != 1:
                                raise ValueError("this method 'show_statistics' requires that only one of the following be input as True: 'show_chi_square', 'show_g_test', 'show_maximum_likelihood'")
                            self.subview_arrow_to_tail(
                                ax=ax,
                                series=series,
                                distribution_id=distribution_id,
                                optimization_metric=optimization_metric,
                                extreme_value=extreme_value,
                                tail_arrow_color=tail_arrow_color,
                                show_chi_square=show_chi_square,
                                show_g_test=show_g_test,
                                show_maximum_likelihood=show_maximum_likelihood)
                        ax.set_xlabel('{} [{}]'.format(parameter_label, unit_label), fontsize=self.labelsize)
                        ax.set_ylabel(density_id.title(), fontsize=self.labelsize)
                        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
                        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
                        ax.set_title(series['identifiers']['series id'], fontsize=self.titlesize)
                        if (i == 0) and (not show_rug):
                            _handles, _labels = ax.get_legend_handles_labels()
                            handles.extend(_handles)
                            labels.extend(_labels)
                    if show_rug:
                        for i, (ax, series) in enumerate(zip(axes.ravel(), self.series)):
                            xs, ys = self.subview_rug(
                                ax=ax,
                                series=series,
                                distribution_id=distribution_id,
                                rug_color=rug_color,
                                i=i,
                                xs=xs,
                                ys=ys)
                            if i == 0:
                                _handles, _labels = ax.get_legend_handles_labels()
                                handles.extend(_handles)
                                labels.extend(_labels)
                    hspace = 0.425 if 'vertical' in layout else 0.3
                    fig.subplots_adjust(hspace=hspace)
                    ## update axes
                    xfmt = '{x:,.0f}' if np.nanmax(xs) > 1000 else '{x:,.2f}'
                    yfmt = '{x:,.0f}' if np.nanmax(ys) > 1000 else '{x:,.4f}'
                    if self.is_same_elements(elements=parameter_labels, s='', n=self.n):
                        if self.is_same_elements(elements=unit_labels, s='', n=self.n):
                            shared_xlabel = '{} [{}]'.format(parameter_labels[0], unit_labels[0])
                        else:
                            shared_xlabel = '{}'.format(parameter_labels[0])
                    else:
                        if self.is_same_elements(elements=alt_parameter_labels, s='', n=self.n):
                            if self.is_same_elements(elements=unit_labels, s='', n=self.n):
                                shared_xlabel = '{} [{}]'.format(alt_parameter_labels[0], unit_labels[0])
                            else:
                                shared_xlabel = '{}'.format(alt_parameter_labels[0])
                        else:
                            if self.is_same_elements(elements=unit_labels, s='', n=self.n):
                                # shared_xlabel = '{} [{}]'.format(parameter_labels[0], unit_labels[0])
                                shared_xlabel = 'Parameter [{}]'.format(unit_labels[0])
                            else:
                                shared_xlabel = 'Parameter'
                    shared_ylabel = '{} Density'.format(density_id.title())
                    self.share_axes(
                        axes=axes,
                        layout=layout,
                        xs=xs,
                        ys=ys,
                        sharex=sharex,
                        sharey=sharey,
                        xticks=True,
                        yticks=True,
                        xfmt=xfmt,
                        yfmt=yfmt,
                        xlim=True,
                        ylim=True,
                        xlabel=shared_xlabel,
                        ylabel=shared_ylabel,
                        collapse_x=collapse_x,
                        collapse_y=collapse_y)
                    fig.align_ylabels()
                    for ax in axes.ravel():
                        self.apply_grid(ax)
                    ## update legend
                    if layout == 'vertical':
                        fig.subplots_adjust(hspace=0.325)
                    self.subview_legend(
                        fig=fig,
                        ax=axes.ravel()[0],
                        handles=handles,
                        labels=labels,
                        title='{}'.format(distribution_id.title()),
                        bottom=0.2,
                        textcolor=textcolor,
                        facecolor='white',
                        edgecolor='k',
                        titlecolor='k',
                        ncol=None)
                    ## update title
                    # s = ""
                    # fig.suptitle(s, fontsize=self.titlesize)
                    ## show / save
                    if save:
                        savename = 'RawAnalysis_{}_Distribution'.format(
                            distribution_id.replace('_', '-').title().replace(' ', '_'))
                        if show_rug:
                            savename = '{}_RUG'.format(savename)
                        if show_kde:
                            savename = '{}_KDE-{}'.format(savename, kde_style)
                        if show_histogram:
                            savename = '{}_HIST'.format(savename)
                        if show_chi_square:
                            savename = '{}_CSQ'.format(savename)
                        if show_g_test:
                            savename = '{}_GT'.format(savename)
                        if show_maximum_likelihood:
                            savename = '{}_MLE'.format(savename)
                        if show_statistics:
                            savename = '{}_STAT'.format(savename)
                        if show_confidence_interval:
                            savename = '{}_CONF'.format(savename)
                        if show_filled_tail:
                            savename = '{}_TAIL-F'.format(savename)
                        if show_arrow_tail:
                            savename = '{}_TAIL-A'.format(savename)
                        if extreme_value is not None:
                            savename = '{}_EX-{}'.format(savename, extreme_value)
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

    def view_error_space(self, ndim, distribution_ids=None, show_chi_square=False, show_g_test=False, show_maximum_likelihood=False, extremum_color='k', cmap='Oranges', color_spacing=None, levels=None, azim=-60, elev=30, rstride=1, cstride=1, alpha=0.8, show_colorbar=False, show_fills=False, show_lines=False, show_inline_labels=False, sharex=False, sharey=False, collapse_x=False, collapse_y=False, figsize=None, save=False, layout='single'):
        show_values = np.array([show_chi_square, show_g_test, show_maximum_likelihood])
        nshow = np.sum(show_values)
        if nshow < 1:
            raise ValueError("input at least one of the following to use this method: 'show_chi_square', 'show_g_test', 'show_maximum_likelihood'")
        elif nshow > 1:
            container_of_args = [
                show_chi_square,
                show_g_test,
                show_maximum_likelihood]
            container_of_kwargs = [
                dict(show_chi_square=show_chi_square),
                dict(show_g_test=show_g_test),
                dict(show_maximum_likelihood=show_maximum_likelihood)]
            for args, kwargs in zip(container_of_args, container_of_kwargs):
                if args:
                    self.view_error_space(
                        ndim=ndim,
                        distribution_ids=distribution_ids,
                        extremum_color=extremum_color,
                        cmap=cmap,
                        color_spacing=color_spacing,
                        levels=levels,
                        azim=azim,
                        elev=elev,
                        rstride=rstride,
                        cstride=cstride,
                        show_colorbar=show_colorbar,
                        show_fills=show_fills,
                        show_lines=show_lines,
                        show_inline_labels=show_inline_labels,
                        sharex=sharex,
                        sharey=sharey,
                        collapse_x=collapse_x,
                        collapse_y=collapse_y,
                        figsize=figsize,
                        save=save,
                        layout=layout,
                        **kwargs)
        else: # nshow == 1
            if layout is None:
                permutable_layouts = ['single']
                if 2 <= self.n < 4:
                    permutable_layouts.append('horizontal')
                elif (self.n > 2) and (self.n % 2 == 0):
                    permutable_layouts.append('square')
                if self.n >= 2:
                    permutable_layouts.append('vertical')
                self.view_layout_permutations(
                    f=self.view_error_space,
                    layouts=permutable_layouts,
                    ndim=ndim,
                    distribution_ids=distribution_ids,
                    show_chi_square=show_chi_square,
                    show_g_test=show_g_test,
                    show_maximum_likelihood=show_maximum_likelihood,
                    extremum_color=extremum_color,
                    cmap=cmap,
                    color_spacing=color_spacing,
                    levels=levels,
                    azim=azim,
                    elev=elev,
                    show_colorbar=show_colorbar,
                    show_fills=show_fills,
                    show_lines=show_lines,
                    show_inline_labels=show_inline_labels,
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
                    visualizer.view_error_space(
                        ndim=ndim,
                        distribution_ids=distribution_ids,
                        show_chi_square=show_chi_square,
                        show_g_test=show_g_test,
                        show_maximum_likelihood=show_maximum_likelihood,
                        extremum_color=extremum_color,
                        cmap=cmap,
                        color_spacing=color_spacing,
                        levels=levels,
                        azim=azim,
                        elev=elev,
                        show_colorbar=show_colorbar,
                        show_fills=show_fills,
                        show_lines=show_lines,
                        show_inline_labels=show_inline_labels,
                        sharex=sharex,
                        sharey=sharey,
                        collapse_x=collapse_x,
                        collapse_y=collapse_y,
                        figsize=figsize,
                        save=save,
                        layout='overlay')
            else:
                ## verify inputs
                if layout == 'overlay':
                    textcolor = 'k'
                    if self.n != 1:
                        raise ValueError("layout='overlay' for this method will only work for one series")
                else:
                    textcolor = True
                if ndim not in (2, 3):
                    raise ValueError("invalid ndim: {}".format(ndim))
                if distribution_ids is None:
                    distribution_ids = self.get_mutually_common_distribution_ids()
                elif isinstance(distribution_ids, str):
                    distribution_ids = [distribution_ids]
                elif not isinstance(distribution_ids, (tuple, list, np.ndarray)):
                    raise ValueError("invalid type(distribution_ids): {}".format(type(distribution_ids)))
                for distribution_id in distribution_ids:
                    _optimizer_id = self.get_exclusive_optimizer_id(
                        show_chi_square=show_chi_square,
                        show_g_test=show_g_test,
                        show_maximum_likelihood=show_maximum_likelihood)
                    optimizer_id = _optimizer_id.replace(' ', '_').replace('-', '_')
                    parameter_labels, alt_parameter_labels, unit_labels = [], [], []
                    event_types = []
                    ## get z-colorbar data
                    zs = []
                    for series in self.series:
                        event_type = series['identifiers']['event type']
                        event_types.append(event_type)
                        parametric_distribution = series[distribution_id]
                        parameter_label = series['parameter mapping'][parametric_distribution.extreme_parameter]
                        alt_parameter_label = series['generalized parameter mapping'][parametric_distribution.extreme_parameter]
                        unit_label = series['unit mapping'][parametric_distribution.extreme_parameter]
                        if parametric_distribution.is_log_transformed:
                            parameter_label = 'log {}'.format(parameter_label)
                            alt_parameter_label = 'log {}'.format(alt_parameter_label)
                            unit_label = 'log {}'.format(unit_label)
                        parameter_labels.append(parameter_label)
                        alt_parameter_labels.append(alt_parameter_label)
                        unit_labels.append(unit_label)
                        optimization_result = getattr(parametric_distribution, optimizer_id)
                        optimizer = optimization_result['optimizer']
                        zs.append(np.nanmin(optimizer.Ze))
                        zs.append(np.nanmax(optimizer.Ze))
                    zs = np.array(zs)
                    if self.is_same_elements(elements=parameter_labels, s='', n=self.n):
                        _extreme_parameter = parameter_labels[0]
                    else:
                        if self.is_same_elements(elements=alt_parameter_labels, s='', n=self.n):
                            _extreme_parameter = alt_parameter_labels[0]
                        else:
                            _extreme_parameter = 'Parameter'
                    if self.is_same_elements(elements=event_types, s='', n=self.n):
                        _event_type = event_types[0]
                    else:
                        _event_type = 'Parameter'
                    color_spacing, error_levels, color_bar_levels, zfmt = self.autocorrect_error_space_configuration(
                        extreme_parameter=_extreme_parameter,
                        event_type=_event_type,
                        optimizer_id=_optimizer_id,
                        levels=levels,
                        color_spacing=color_spacing)
                    norm, zfmt, vmin, vmax = self.get_colormap_configuration(
                        z=zs,
                        color_spacing=color_spacing,
                        levels=error_levels,
                        fmt=zfmt)
                    ## initialize bounds per ax (sharex, sharey)
                    xs, ys = [], []
                    handles, labels = [], []
                    ## get figure and axes
                    kws = self.get_number_of_figure_rows_and_columns(self.n, layout)
                    if ndim == 2:
                        fig, axes = plt.subplots(figsize=figsize, **kws)
                        if not isinstance(axes, np.ndarray):
                            axes = np.array([axes])
                    else:
                        row_loc = np.arange(kws['nrows']).astype(int) + 1
                        col_loc = np.arange(kws['ncols']).astype(int) + 1
                        nth_subplot = 0
                        axes = []
                        fig = plt.figure(figsize=figsize)
                        for i in row_loc:
                            for j in col_loc:
                                nth_subplot += 1
                                ax = fig.add_subplot(row_loc[-1], col_loc[-1], nth_subplot, projection='3d')
                                axes.append(ax)
                        axes = np.array(axes).reshape((kws['nrows'], kws['ncols']))
                    for i, (ax, series) in enumerate(zip(axes.ravel(), self.series)):
                        parametric_distribution = series[distribution_id]
                        optimization_result = getattr(parametric_distribution, optimizer_id)
                        optimizer = optimization_result['optimizer']
                        optimization_label = self.get_optimization_label(
                            distribution_id=distribution_id,
                            optimization_result=optimization_result,
                            unit_label=unit_label)
                        if self.n > 1:
                            optimization_label = '{}\n{}'.format(series['identifiers']['series id'], optimization_label)
                        inline_fmt = deepcopy(zfmt)
                        ## initialize plot
                        if ndim == 2:
                            scatter_args = tuple(optimization_result['calculation parameters'])
                            scatter_kwargs = {
                                'color' : extremum_color,
                                'marker' : '*',
                                'linewidth' : 0,
                                's' : 50,
                                'label' : optimization_label}
                            im_handle = self.subview_contour_space(
                                ax=ax,
                                X=optimizer.Xe,
                                Y=optimizer.Ye,
                                Z=optimizer.Ze,
                                norm=norm,
                                levels=error_levels,
                                cmap=cmap,
                                extremum_color=extremum_color,
                                show_fills=show_fills,
                                show_lines=show_lines,
                                show_inline_labels=show_inline_labels,
                                inline_fmt=inline_fmt,
                                scatter_args=scatter_args,
                                **scatter_kwargs)
                        else:
                            scatter_args = tuple([optimization_result['calculation parameters'][0], optimization_result['calculation parameters'][1], optimization_result['fun']])
                            scatter_kwargs = {
                                'color' : extremum_color,
                                'marker' : '*',
                                'linewidth' : 0,
                                's' : 50,
                                'label' : optimization_label}
                            im_handle = self.subview_surface_space(
                                ax=ax,
                                X=optimizer.Xe,
                                Y=optimizer.Ye,
                                Z=optimizer.Ze,
                                norm=norm,
                                levels=error_levels,
                                cmap=cmap,
                                extremum_color=extremum_color,
                                show_lines=show_lines,
                                rstride=rstride,
                                cstride=cstride,
                                alpha=alpha,
                                azim=azim,
                                elev=elev,
                                scatter_args=scatter_args,
                                **scatter_kwargs)
                        xs.append(np.nanmin(optimizer.Xe))
                        xs.append(np.nanmax(optimizer.Xe))
                        ys.append(np.nanmin(optimizer.Ye))
                        ys.append(np.nanmax(optimizer.Ye))
                        _handles, _labels = ax.get_legend_handles_labels()
                        handles.extend(_handles)
                        labels.extend(_labels)
                        ax.set_title(series['identifiers']['series id'], fontsize=self.titlesize)
                    if show_colorbar:
                        cax = fig.add_axes([0.925, 0.225, 0.025, 0.625]) # (x0, y0, dx, dy)
                        orientation = 'vertical'
                        # if self.n == 1:
                        #     cax = fig.add_axes([0.925, 0.15, 0.025, 0.7])
                        #     orientation = 'vertical'
                        # else:
                        #     cax = None
                        #     orientation = 'horizontal'
                        self.subview_color_bar(
                            fig=fig,
                            ax=axes.ravel().tolist(),
                            cax=cax,
                            handle=im_handle,
                            title=self.error_space_mapping[_optimizer_id],
                            # title='{}'.format(_optimizer_id.title()),
                            levels=color_bar_levels,
                            norm=norm,
                            extend='max' if ndim == 3 else None,
                            orientation=orientation,
                            pad=0.2)
                        fig.subplots_adjust(left=0.1)
                    hspace = 0.425 if 'vertical' in layout else 0.3
                    fig.subplots_adjust(hspace=hspace)
                    ## update axes
                    shared_xlabel = '$\mu_{normal}$'
                    shared_ylabel = '$\sigma_{normal}$'
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
                        for ax in axes.ravel():
                            self.apply_grid(ax)
                    else:
                        self.share_dim3_axes(
                            axes=axes,
                            xs=xs,
                            ys=ys,
                            zs=zs,
                            xlim=True,
                            ylim=True,
                            zlim=True,
                            xticks=True,
                            yticks=True,
                            zticks=True,
                            xfmt=None,
                            yfmt=None,
                            zfmt=zfmt,
                            xlabel=shared_xlabel,
                            ylabel=shared_ylabel,
                            zlabel=self.error_space_mapping[_optimizer_id],
                            # zlabel='{}'.format(_optimizer_id.title()),
                            basex=None,
                            basey=None,
                            basez=None)
                    try:
                        fig.align_ylabels()
                    except:
                        pass
                    self.subview_legend(
                        fig=fig,
                        ax=axes.ravel()[0],
                        handles=handles,
                        labels=labels,
                        title=distribution_id.title(),
                        bottom=0.3,
                        textcolor=textcolor,
                        facecolor='lightgray',
                        edgecolor='k',
                        titlecolor='k',
                        ncol=None if self.n == 1 else self.n)
                    ## update title
                    s = '{} Error-Space'.format(_optimizer_id.title())
                    fig.suptitle(s, fontsize=self.titlesize)
                    ## show / save
                    if save:
                        savename = 'RawAnalysis_{}_Distribution'.format(
                            distribution_id.replace('_', '-').title().replace(' ', '_'))
                        savename = '{}_{}-D_error-space_{}-levels'.format(savename, ndim, color_spacing)
                        if show_chi_square:
                            savename = '{}_CSQ'.format(savename)
                        if show_g_test:
                            savename = '{}_GT'.format(savename)
                        if show_maximum_likelihood:
                            savename = '{}_MLE'.format(savename)
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

    def view_distribution_tail(self, extreme_values, extreme_condition='greater than', distribution_ids=None, density_id='observed', histogram_id='original', basex=None, basey=None, extreme_color='darkred', bar_color='gray', step_color='k', sharex=False, sharey=False, collapse_x=False, collapse_y=False, figsize=None, save=False, layout='single'):
        if layout is None:
            permutable_layouts = ['single']
            if 2 <= self.n < 4:
                permutable_layouts.append('horizontal')
            elif (self.n > 2) and (self.n % 2 == 0):
                permutable_layouts.append('square')
            if self.n >= 2:
                permutable_layouts.append('vertical')
            self.view_layout_permutations(
                f=self.view_distribution_tail,
                layouts=permutable_layouts,
                extreme_values=extreme_values,
                extreme_condition=extreme_condition,
                distribution_ids=distribution_ids,
                density_id=density_id,
                histogram_id=histogram_id,
                basex=basex,
                basey=basey,
                extreme_color=extreme_color,
                bar_color=bar_color,
                step_color=step_color,
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
                visualizer.view_distribution_tail(
                    extreme_values=extreme_values,
                    extreme_condition=extreme_condition,
                    distribution_ids=distribution_ids,
                    density_id=density_id,
                    histogram_id=histogram_id,
                    basex=basex,
                    basey=basey,
                    extreme_color=extreme_color,
                    bar_color=bar_color,
                    step_color=step_color,
                    sharex=sharex,
                    sharey=sharey,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y,
                    figsize=figsize,
                    save=save,
                    layout='overlay')
        else:
            ## verify user input
            if not isinstance(extreme_values, (tuple, list, np.ndarray)):
                extreme_values = [extreme_values]
            if density_id == 'probability':
                optimization_metric = 'probability density'
            elif density_id == 'observed':
                optimization_metric = 'observed density'
            else:
                raise ValueError("invalid density_id: {}".format(density_id))
            if distribution_ids is None:
                distribution_ids = self.get_mutually_common_distribution_ids()
            elif isinstance(distribution_ids, str):
                distribution_ids = [distribution_ids]
            elif not isinstance(distribution_ids, (tuple, list, np.ndarray)):
                raise ValueError("invalid type(distribution_ids): {}".format(type(distribution_ids)))
            for distribution_id in distribution_ids:
                for extreme_value in extreme_values:
                    if not isinstance(extreme_value, (int, float)):
                        raise ValueError("invalid type(extreme_value): {}".format(type(extreme_value)))
                    handles, labels = [], []
                    parameter_labels, alt_parameter_labels, unit_labels = [], [], []
                    xs, ys = [], []
                    if basey is None:
                        ys = [0]
                    else:
                        ys = [1 / basey]
                    ## get figure and axes
                    kws = self.get_number_of_figure_rows_and_columns(self.n, layout)
                    fig, axes = plt.subplots(figsize=figsize, **kws)
                    ## initialize plot
                    if layout == 'overlay':
                        if self.n != 1:
                            raise ValueError("layout='overlay' for this method will only work for one series")
                        axes = np.array([axes])
                        textcolor = 'k'
                    else:
                        textcolor = True
                    for i, (ax, series) in enumerate(zip(axes.ravel(), self.series)):
                        parametric_distribution = series[distribution_id]
                        parameter_label = series['parameter mapping'][parametric_distribution.extreme_parameter]
                        alt_parameter_label = series['generalized parameter mapping'][parametric_distribution.extreme_parameter]
                        unit_label = series['unit mapping'][parametric_distribution.extreme_parameter]
                        if parametric_distribution.is_log_transformed:
                            parameter_label = 'log {}'.format(parameter_label)
                            alt_parameter_label = 'log {}'.format(alt_parameter_label)
                            unit_label = 'log {}'.format(unit_label)
                        parameter_labels.append(parameter_label)
                        alt_parameter_labels.append(alt_parameter_label)
                        unit_labels.append(unit_label)
                        if parametric_distribution.is_log_transformed:
                            ev = np.log(extreme_value)
                        else:
                            ev = extreme_value
                        _, _ys, histogram = self.subview_distribution_histogram(
                            ax=ax,
                            series=series,
                            distribution_id=distribution_id,
                            histogram_id=histogram_id,
                            density_id=density_id,
                            bar_color=bar_color,
                            step_color=step_color,
                            i=i,
                            xs=[],
                            ys=[])
                        if basex is None:
                            xs.append(ev * 0.875)
                            xs.append(histogram.edges[-1] * 1.125)
                        else:
                            _, lower_x = self.get_logarithmic_bounds(
                                num=ev + 0.5,
                                base=basex,
                                bound_id='lower')
                            _, upper_x = self.get_logarithmic_bounds(
                                num=histogram.edges[-1] + 0.5,
                                base=basex,
                                bound_id='upper')
                            xs.append(lower_x)
                            xs.append(upper_x)
                        extreme_loc = np.where(ev <= histogram.edges[:-1])[0]
                        if basey is None:
                            upper_y = np.nanmax(histogram.counts[extreme_loc]) * 1.125
                        else:
                            _, upper_y = self.get_logarithmic_bounds(
                                num=np.nanmax(histogram.counts[extreme_loc]) * 1.25,
                                base=basey,
                                bound_id='upper')
                        ys.append(upper_y)
                        event_searcher = EventSearcher({parametric_distribution.extreme_parameter : parametric_distribution.vs})
                        extreme_events, _ = event_searcher.search_events(
                            parameters=parametric_distribution.extreme_parameter,
                            conditions=extreme_condition,
                            values=ev)
                        ax.axvline(
                            x=ev,
                            color=extreme_color,
                            linestyle=':',
                            label='Extreme Value Threshold' if i == 0 else None)
                        text_box = ax.text(
                            0.95,
                            0.95,
                            '${:,}$ Extreme Events'.format(extreme_events[parametric_distribution.extreme_parameter].size),
                            fontsize=self.textsize,
                            horizontalalignment='right',
                            verticalalignment='top',
                            transform=ax.transAxes)
                        text_box.set_bbox(dict(facecolor='gray', alpha=0.25, edgecolor='k'))
                        ax.set_xlabel('{} [{}]'.format(parameter_label, unit_label), fontsize=self.labelsize)
                        ax.set_ylabel(density_id.title(), fontsize=self.labelsize)
                        ax.set_title(series['identifiers']['series id'], fontsize=self.titlesize)
                        if i == 0:
                            _handles, _labels = ax.get_legend_handles_labels()
                            handles.extend(_handles)
                            labels.extend(_labels)
                    hspace = 0.425 if 'vertical' in layout else 0.3
                    fig.subplots_adjust(hspace=hspace)
                    ## update axes
                    if self.is_same_elements(elements=parameter_labels, s='', n=self.n):
                        if self.is_same_elements(elements=unit_labels, s='', n=self.n):
                            shared_xlabel = '{} [{}]'.format(parameter_labels[0], unit_labels[0])
                        else:
                            shared_xlabel = '{}'.format(parameter_labels[0])
                    else:
                        if self.is_same_elements(elements=alt_parameter_labels, s='', n=self.n):
                            if self.is_same_elements(elements=unit_labels, s='', n=self.n):
                                shared_xlabel = '{} [{}]'.format(alt_parameter_labels[0], unit_labels[0])
                            else:
                                shared_xlabel = '{}'.format(alt_parameter_labels[0])
                        else:
                            if self.is_same_elements(elements=unit_labels, s='', n=self.n):
                                # shared_xlabel = '{} [{}]'.format(parameter_labels[0], unit_labels[0])
                                shared_xlabel = 'Parameter [{}]'.format(unit_labels[0])
                            else:
                                shared_xlabel = 'Parameter'
                    shared_ylabel = '{} Density'.format(density_id.title())
                    xfmt = ticker.StrMethodFormatter('{x:,.0f}') if np.nanmax(xs) > 100 else ticker.StrMethodFormatter('{x:,.2f}')
                    yfmt = ticker.StrMethodFormatter('{x:,.0f}') if np.nanmax(ys) > 100 else ticker.StrMethodFormatter('{x:,.2f}')
                    self.share_axes(
                        axes=axes,
                        layout=layout,
                        xs=xs,
                        ys=ys,
                        sharex=sharex,
                        sharey=sharey,
                        xticks=True,
                        yticks=True,
                        xfmt=xfmt,
                        yfmt=yfmt,
                        xlim=True,
                        ylim=True,
                        basex=basex,
                        basey=basey,
                        xlabel=shared_xlabel,
                        ylabel=shared_ylabel,
                        collapse_x=collapse_x,
                        collapse_y=collapse_y)
                    for ax in axes.ravel():
                        self.apply_grid(ax)
                    fig.align_ylabels()
                    self.subview_legend(
                        fig=fig,
                        ax=axes.ravel()[0],
                        handles=handles,
                        labels=labels,
                        title=distribution_id.title(),
                        bottom=0.2,
                        textcolor=textcolor,
                        facecolor='white',
                        edgecolor='k',
                        titlecolor='k',
                        ncol=None)
                    ## update title
                    ...
                    ## show / save
                    if save:
                        savename = 'RawAnalysis_{}_DistributionTail_{}'.format(
                            distribution_id.replace('_', '-').title().replace(' ', '_'),
                            density_id.title().replace(' ', '-'))
                        if basex is not None:
                            savename = '{}-xLOG_{}'.format(savename, basex)
                        if basey is not None:
                            savename = '{}-yLOG_{}'.format(savename, basey)
                        if extreme_value is not None:
                            savename = '{}_EX-{}'.format(savename, extreme_value)
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

    def subview_connection_between_normal_and_lognormal(self, ax, series, normal_ys, connector_color, mu_color, sigma_color, show_chi_square, show_g_test, show_maximum_likelihood, show_statistics):
        lognormal_distribution = series['lognormal distribution']
        ## show log/exp curve
        ax.scatter(
            lognormal_distribution.vs,
            lognormal_distribution.normal_distribution.vs,
            color=connector_color,
            label=r'$X = e^{Y}$',
            marker='.',
            s=2)
        ## get optimization results
        _optimizer_id = self.get_exclusive_optimizer_id(
            show_chi_square=show_chi_square,
            show_g_test=show_g_test,
            show_maximum_likelihood=show_maximum_likelihood)
        optimizer_id = _optimizer_id.replace(' ', '_').replace('-', '_')
        normal_optimization_result = getattr(lognormal_distribution.normal_distribution, optimizer_id)
        lognormal_optimization_result = getattr(lognormal_distribution, optimizer_id)
        normal_prms = normal_optimization_result['true parameters']
        lognormal_prms = lognormal_optimization_result['true parameters']
        if show_statistics:
            leg_labels = (None, None)
        else:
            leg_labels = (r'$\mu_{opt}$', r'$\sigma_{opt}$')
        ## show lines for mean and standard deviation
        for j, facecolor, label in zip((0, 1), (mu_color, sigma_color), (leg_labels)):
            ax.hlines(
                y=normal_prms[j],
                xmin=0,
                xmax=lognormal_prms[j],
                color=facecolor,
                linestyle=':',
                alpha=0.7,
                label=label)
            ax.vlines(
                x=lognormal_prms[j],
                ymin=normal_prms[j],
                # ymax=np.nanmax(normal_ys),
                ymax=0,
                color=facecolor,
                linestyle=':',
                alpha=0.7)
        return ax

    def view_normal_and_lognormal_relation(self, show_kde=False, show_histogram=False, show_chi_square=False, show_g_test=False, show_maximum_likelihood=False, show_statistics=False, show_confidence_interval=False, density_id='observed', histogram_id='original', bar_color='gray', step_color='k', csq_color='darkorange', gt_color='purple', mle_color='steelblue', confidence_color='darkgreen', mu_color='purple', sigma_color='blue', median_color='darkgreen', mode_color='darkred', kde_colors='darkblue', kde_style='curve', connector_color='k', figsize=None, save=False, layout='single'):
        if layout is None:
            permutable_layouts = ['single']
            self.view_layout_permutations(
                f=self.view_normal_and_lognormal_relation,
                layouts=permutable_layouts,
                show_kde=show_kde,
                show_histogram=show_histogram,
                show_chi_square=show_chi_square,
                show_g_test=show_g_test,
                show_maximum_likelihood=show_maximum_likelihood,
                show_statistics=show_statistics,
                show_confidence_interval=show_confidence_interval,
                density_id=density_id,
                histogram_id=histogram_id,
                bar_color=bar_color,
                step_color=step_color,
                csq_color=csq_color,
                gt_color=gt_color,
                mle_color=mle_color,
                confidence_color=confidence_color,
                mu_color=mu_color,
                sigma_color=sigma_color,
                median_color=median_color,
                mode_color=mode_color,
                kde_colors=kde_colors,
                kde_style=kde_style,
                connector_color=connector_color,
                figsize=figsize,
                save=save)
        elif layout == 'single':
            for series in self.series:
                cls = deepcopy(self.cls)
                visualizer = cls(
                    series=[series],
                    savedir=self.savedir)
                visualizer.view_normal_and_lognormal_relation(
                    show_kde=show_kde,
                    show_histogram=show_histogram,
                    show_chi_square=show_chi_square,
                    show_g_test=show_g_test,
                    show_maximum_likelihood=show_maximum_likelihood,
                    show_statistics=show_statistics,
                    show_confidence_interval=show_confidence_interval,
                    density_id=density_id,
                    histogram_id=histogram_id,
                    bar_color=bar_color,
                    step_color=step_color,
                    csq_color=csq_color,
                    gt_color=gt_color,
                    mle_color=mle_color,
                    confidence_color=confidence_color,
                    mu_color=mu_color,
                    sigma_color=sigma_color,
                    median_color=median_color,
                    mode_color=mode_color,
                    kde_colors=kde_colors,
                    kde_style=kde_style,
                    connector_color=connector_color,
                    figsize=figsize,
                    save=save,
                    layout='overlay')
        else:
            ## verify inputs
            if layout == 'overlay':
                if self.n != 1:
                    raise ValueError("layout='overlay' for this method will only work for one series")
            else:
                raise ValueError("invalid layout for this method: {}".format(layout))
            series = self.series[0]
            if 'lognormal distribution' not in list(series.keys()):
                raise ValueError("lognormal distribution is not initialized")
            lognormal_distribution = series['lognormal distribution']
            if lognormal_distribution.normal_distribution is None:
                raise ValueError("normal distribution is not initialized")
            if not any([show_kde, show_histogram, show_chi_square, show_g_test, show_maximum_likelihood, show_statistics, show_confidence_interval]):
                raise ValueError("set at least one of the following inputs to True: 'show_kde', 'show_histogram', 'show_chi_square', 'show_g_test', 'show_maximum_likelihood', 'show_statistics', 'show_confidence_interval'")
            if density_id not in ('observed', 'probability'):
                raise ValueError("invalid density_id: {}".format(density_id))
            if density_id == 'probability':
                optimization_metric = 'probability density'
            else:
                if show_histogram:
                    optimization_metric = 'observed density'
                else:
                    optimization_metric = 'observed frequency'
            if show_kde:
                if not isinstance(kde_colors, (tuple, list, np.ndarray)):
                    kde_colors = [kde_colors]
            ## initialize plot parameters
            normal_xs, normal_ys = [0], [0]
            lognormal_xs, lognormal_ys = [0], [0]
            lognormal_parameter_label = series['parameter mapping'][lognormal_distribution.extreme_parameter]
            lognormal_unit_label = series['unit mapping'][lognormal_distribution.extreme_parameter]
            normal_parameter_label = 'log {}'.format(series['parameter mapping'][lognormal_distribution.normal_distribution.extreme_parameter])
            normal_unit_label = 'log {}'.format(series['unit mapping'][lognormal_distribution.normal_distribution.extreme_parameter])
            handles, labels = [], []
            event_type = series['identifiers']['event type']
            ## initialize figure and axes
            axes = []
            row_loc = np.arange(2).astype(int) + 1
            col_loc = np.arange(2).astype(int) + 1
            nth_subplot = 0
            fig = plt.figure(figsize=figsize)
            for i in row_loc:
                for j in col_loc:
                    nth_subplot += 1
                    ax = fig.add_subplot(row_loc[-1], col_loc[-1], nth_subplot)
                    axes.append(ax)
            axes = np.array(axes).reshape((2, 2))
            axes[0, 0].axis('off')
            ## initialize plots
            if show_kde:
                lognormal_xs, lognormal_ys = self.subview_kernel_density_estimation(
                    ax=axes[0, 1],
                    series=series,
                    distribution_id='lognormal distribution',
                    density_id=density_id,
                    histogram_id=histogram_id,
                    kde_colors=kde_colors,
                    kde_style=kde_style,
                    i=i,
                    xs=lognormal_xs,
                    ys=lognormal_ys)
                normal_xs, normal_ys = self.subview_kernel_density_estimation(
                    ax=axes[1, 0],
                    series=series,
                    distribution_id='normal distribution',
                    density_id=density_id,
                    histogram_id=histogram_id,
                    kde_colors=kde_colors,
                    kde_style=kde_style,
                    i=i+1,
                    xs=normal_xs,
                    ys=normal_ys,
                    swap_xy=True)
            if show_histogram:
                lognormal_xs, lognormal_ys, _ = self.subview_distribution_histogram(
                    ax=axes[0, 1],
                    series=series,
                    distribution_id='lognormal distribution',
                    histogram_id=histogram_id,
                    density_id=density_id,
                    bar_color=bar_color,
                    step_color=step_color,
                    i=i,
                    xs=lognormal_xs,
                    ys=lognormal_ys,
                    alpha=1)
                normal_xs, normal_ys, _ = self.subview_distribution_histogram(
                    ax=axes[1, 0],
                    series=series,
                    distribution_id='normal distribution',
                    histogram_id=histogram_id,
                    density_id=density_id,
                    bar_color=bar_color,
                    step_color=step_color,
                    i=i+1,
                    xs=normal_xs,
                    ys=normal_ys,
                    alpha=1,
                    swap_xy=True)
            if any([show_chi_square, show_g_test, show_maximum_likelihood]):
                nfits = np.sum([show_chi_square, show_g_test, show_maximum_likelihood])
                fits_alpha = 1 / nfits
                if (fits_alpha < 0.5) or (not np.isfinite(fits_alpha)):
                    fits_alpha = 0.5
                lognormal_xs, lognormal_ys = self.subview_optimized_fit(
                    ax=axes[0, 1],
                    series=series,
                    distribution_id='lognormal distribution',
                    optimization_metric=optimization_metric,
                    csq_color=csq_color,
                    gt_color=gt_color,
                    mle_color=mle_color,
                    i=i,
                    xs=lognormal_xs,
                    ys=lognormal_ys,
                    show_chi_square=show_chi_square,
                    show_g_test=show_g_test,
                    show_maximum_likelihood=show_maximum_likelihood,
                    alpha=fits_alpha)
                normal_xs, normal_ys = self.subview_optimized_fit(
                    ax=axes[1, 0],
                    series=series,
                    distribution_id='normal distribution',
                    optimization_metric=optimization_metric,
                    csq_color=csq_color,
                    gt_color=gt_color,
                    mle_color=mle_color,
                    i=i+1,
                    xs=normal_xs,
                    ys=normal_ys,
                    show_chi_square=show_chi_square,
                    show_g_test=show_g_test,
                    show_maximum_likelihood=show_maximum_likelihood,
                    alpha=fits_alpha,
                    swap_xy=True)
            if show_statistics:
                nfits = np.sum([show_chi_square, show_g_test, show_maximum_likelihood])
                if nfits != 1:
                    raise ValueError("this method 'show_statistics' requires that only one of the following be input as True: 'show_chi_square', 'show_g_test', 'show_maximum_likelihood'")
                self.subview_statistics(
                    ax=axes[0, 1],
                    series=series,
                    distribution_id='lognormal distribution',
                    unit_label=lognormal_unit_label,
                    i=i,
                    mu_color=mu_color,
                    sigma_color=sigma_color,
                    median_color=median_color,
                    mode_color=mode_color,
                    show_chi_square=show_chi_square,
                    show_g_test=show_g_test,
                    show_maximum_likelihood=show_maximum_likelihood)
                self.subview_statistics(
                    ax=axes[1, 0],
                    series=series,
                    distribution_id='normal distribution',
                    unit_label=normal_unit_label,
                    i=i+1,
                    mu_color=mu_color,
                    sigma_color=sigma_color,
                    median_color=median_color,
                    mode_color=mode_color,
                    show_chi_square=show_chi_square,
                    show_g_test=show_g_test,
                    show_maximum_likelihood=show_maximum_likelihood,
                    swap_xy=True)
            if show_confidence_interval:
                nfits = np.sum([show_chi_square, show_g_test, show_maximum_likelihood])
                if nfits != 1:
                    raise ValueError("this method 'show_confidence_interval' requires that only one of the following be input as True: 'show_chi_square', 'show_g_test', 'show_maximum_likelihood'")
                self.subview_confidence_interval(
                    ax=axes[0, 1],
                    series=series,
                    distribution_id='lognormal distribution',
                    optimization_metric=optimization_metric,
                    confidence_color=confidence_color,
                    i=i,
                    show_chi_square=show_chi_square,
                    show_g_test=show_g_test,
                    show_maximum_likelihood=show_maximum_likelihood)
                self.subview_confidence_interval(
                    ax=axes[1, 0],
                    series=series,
                    distribution_id='normal distribution',
                    optimization_metric=optimization_metric,
                    confidence_color=confidence_color,
                    i=i+1,
                    show_chi_square=show_chi_square,
                    show_g_test=show_g_test,
                    show_maximum_likelihood=show_maximum_likelihood,
                    swap_xy=True)
            ## show connector
            self.subview_connection_between_normal_and_lognormal(
                ax=axes[1, 1],
                series=series,
                normal_ys=normal_ys,
                connector_color=connector_color,
                mu_color=mu_color,
                sigma_color=sigma_color,
                show_chi_square=show_chi_square,
                show_g_test=show_g_test,
                show_maximum_likelihood=show_maximum_likelihood,
                show_statistics=show_statistics)
            ## update legend handles and axis ticks
            for i, ax in enumerate(axes.ravel()):
                _handles, _labels = ax.get_legend_handles_labels()
                handles.extend(_handles)
                labels.extend(_labels)
                if i > 0:
                    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
                    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            for i, ax in enumerate(axes[:, -1].ravel()):
                if i == 0:
                    ax.xaxis.tick_top()
                    ax.xaxis.set(label_position='top')
                ax.yaxis.tick_right()
                ax.yaxis.set(label_position='right', offset_position='right')
                ax.yaxis.label.set_rotation(-90)
            # axes[1, 1].tick_params(bottom=True, top=True, left=True, right=True, labelbottom=True, labeltop=False, labelleft=False, labelright=True)
            axes[1, 1].tick_params(bottom=True, top=False, left=False, right=True, labelbottom=True, labeltop=False, labelleft=False, labelright=True)
            ## add text to top-right corner
            lognormal_label = r'Lognormal Distribution: $X$ ~ ln($\mathcal{N}(\mu, {\sigma}^2)$)'
            normal_label = r'Normal Distribution: $Y$ ~ $\mathcal{N}(\mu, {\sigma}^2)$'
            connector_caption = r'$X$ is lognormally distributed $\longrightarrow$ $Y = ln(X)$ is normally distributed'
            connector_formula = r'$X = e^{Y}$'
            connector_label = '{}\n{}'.format(connector_caption, connector_formula)
            text_label = '{}\n\n{}\n{}\n\n{}'.format(series['identifiers']['series id'], lognormal_label, normal_label, connector_label)
            text_box = axes[0, 0].text(
                0.5,
                0.5,
                text_label,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=self.titlesize,
                transform=axes[0, 0].transAxes)
            # text_box.set_bbox(dict(facecolor='gray', alpha=0.25, edgecolor='k'))
            # fig.subplots_adjust(hspace=0.425)
            fig.subplots_adjust(hspace=0.025, wspace=0.025)
            ## get axis labels and fmt
            normal_xfmt = '{x:,.0f}' if np.nanmax(normal_xs) > 1000 else '{x:,.2f}'
            normal_yfmt = '{x:,.0f}' if np.nanmax(normal_ys) > 1000 else '{x:,.2f}'
            lognormal_xfmt = '{x:,.0f}' if np.nanmax(lognormal_xs) > 1000 else '{x:,.4f}'
            lognormal_yfmt = '{x:,.0f}' if np.nanmax(lognormal_ys) > 1000 else '{x:,.4f}'
            normal_shared_xlabel = '{} [{}]'.format(normal_parameter_label, normal_unit_label)
            lognormal_shared_xlabel = '{} [{}]'.format(lognormal_parameter_label, lognormal_unit_label)
            lognormal_shared_ylabel = '{} Density'.format(density_id.title())
            normal_shared_ylabel = '{} Density'.format(density_id.title())
            ## swap xy-axes for normal distribution plot (lower left corner)
            normal_xs, normal_ys = normal_ys, normal_xs
            normal_shared_xlabel, normal_shared_ylabel = normal_shared_ylabel, normal_shared_xlabel
            normal_xfmt, normal_yfmt = normal_yfmt, normal_xfmt
            ## update axes
            self.share_axes(
                axes=np.array([axes[0, 1]]),
                layout='overlay',
                xs=lognormal_xs,
                ys=lognormal_ys,
                sharex=True,
                sharey=True,
                xticks=True,
                yticks=True,
                xfmt=lognormal_xfmt,
                yfmt=lognormal_yfmt,
                xlim=True,
                ylim=True,
                xlabel=lognormal_shared_xlabel,
                ylabel=lognormal_shared_ylabel,
                collapse_x=False,
                collapse_y=False)
            self.share_axes(
                axes=np.array([axes[1, 0]]),
                layout='overlay',
                xs=normal_xs,
                ys=normal_ys,
                sharex=True,
                sharey=True,
                xticks=True,
                yticks=True,
                xfmt=normal_xfmt,
                yfmt=normal_yfmt,
                xlim=True,
                ylim=True,
                xlabel=normal_shared_xlabel,
                ylabel=normal_shared_ylabel,
                collapse_x=False,
                collapse_y=False)
            connector_xlabel = r'$X = e^{Y}$'
            connector_ylabel = r'$Y = ln(X)$'
            self.share_axes(
                axes=np.array([axes[1, 1]]),
                layout='overlay',
                xs=lognormal_xs,
                ys=normal_ys,
                sharex=True,
                sharey=True,
                xticks=True,
                yticks=True,
                xfmt=lognormal_xfmt,
                yfmt=normal_yfmt,
                xlim=True,
                ylim=True,
                xlabel=connector_xlabel,
                ylabel=connector_ylabel,
                collapse_x=False,
                collapse_y=False)
            for ax in (axes[1, :].ravel()):
                ylim = ax.get_ylim()
                ax.set_ylim([ylim[1], ylim[0]])
            for ax in axes.ravel():
                self.apply_grid(ax)
            fig.align_ylabels()
            ## update legend
            self.subview_legend(
                fig=fig,
                ax=axes[0, 1],
                handles=handles,
                labels=labels,
                title=None,
                bottom=0.2,
                textcolor='k',
                facecolor='white',
                edgecolor='k',
                titlecolor='k',
                ncol=None)
            ## update title
            # fig.suptitle(series['identifiers']['series id'], fontsize=self.titlesize)
            ## show / save
            if save:
                savename = 'RawAnalysis_Lognormal-Normal-Connect'
                if show_kde:
                    savename = '{}_KDE-{}'.format(savename, kde_style)
                if show_histogram:
                    savename = '{}_HIST'.format(savename)
                if show_chi_square:
                    savename = '{}_CSQ'.format(savename)
                if show_g_test:
                    savename = '{}_GT'.format(savename)
                if show_maximum_likelihood:
                    savename = '{}_MLE'.format(savename)
                if show_statistics:
                    savename = '{}_STAT'.format(savename)
                if show_confidence_interval:
                    savename = '{}_CONF'.format(savename)
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

    def view_distribution_subseries(self, distribution_ids=None, show_kde=False, show_histogram=False, show_chi_square=False, show_g_test=False, show_maximum_likelihood=False, show_statistics=False, show_confidence_interval=False, density_id='observed', histogram_id='original', bar_color='gray', step_color='k', csq_color='darkorange', gt_color='purple', mle_color='steelblue', confidence_color='darkgreen', mu_color='purple', sigma_color='blue', median_color='darkgreen', mode_color='darkred', kde_colors='darkblue', kde_style='curve', sharex=False, sharey=False, collapse_x=False, collapse_y=False, figsize=None, save=False, layout=None):
        if distribution_ids is None:
            distribution_ids = self.get_mutually_common_distribution_ids()
        elif isinstance(distribution_ids, str):
            distribution_ids = [distribution_ids]
        elif not isinstance(distribution_ids, (tuple, list, np.ndarray)):
            raise ValueError("invalid type(distribution_ids): {}".format(type(distribution_ids)))
        for distribution_id in distribution_ids:
            for series in self.series:
                cls = deepcopy(self.cls)
                visualizer = cls(
                    series=[series],
                    savedir=self.savedir)
                permutable_layouts = []
                if layout is None:
                    parametric_distribution = series[distribution_id]
                    n = len(parametric_distribution.sub_series)
                    if n == 0:
                        raise ValueError("sub-series is not initialized")
                    n += 1
                    if (n > 2) and (n % 2 == 0):
                        permutable_layouts.append('square')
                    if (n >= 2) and (n < 4):
                        permutable_layouts.append('horizontal')
                    permutable_layouts.append('vertical')
                    # if ...:
                    #     permutable_layouts.append('overlay')
                    for _layout in permutable_layouts:
                        visualizer.view_distribution_subseries(
                            distribution_ids=distribution_id,
                            show_kde=show_kde,
                            show_histogram=show_histogram,
                            show_chi_square=show_chi_square,
                            show_g_test=show_g_test,
                            show_maximum_likelihood=show_maximum_likelihood,
                            show_statistics=show_statistics,
                            show_confidence_interval=show_confidence_interval,
                            density_id=density_id,
                            histogram_id=histogram_id,
                            bar_color=bar_color,
                            step_color=step_color,
                            csq_color=csq_color,
                            gt_color=gt_color,
                            mle_color=mle_color,
                            confidence_color=confidence_color,
                            mu_color=mu_color,
                            sigma_color=sigma_color,
                            median_color=median_color,
                            mode_color=mode_color,
                            kde_colors=kde_colors,
                            kde_style=kde_style,
                            sharex=sharex,
                            sharey=sharey,
                            collapse_x=collapse_x,
                            collapse_y=collapse_y,
                            figsize=figsize,
                            save=save,
                            layout=_layout)
                elif isinstance(layout, str):
                    if layout == 'single':
                        raise ValueError("this method produces a new figure for each series by default, so layout='single' is invalid for this method")
                    elif layout == 'overlay':
                        raise ValueError("not yet implemented")
                    else:
                        ## verify user input
                        if not any([show_kde, show_histogram, show_chi_square, show_g_test, show_maximum_likelihood, show_statistics, show_confidence_interval]):
                            raise ValueError("set at least one of the following inputs to True: 'show_kde', 'show_histogram', 'show_chi_square', 'show_g_test', 'show_maximum_likelihood', 'show_statistics', 'show_confidence_interval'")
                        if density_id not in ('observed', 'probability'):
                            raise ValueError("invalid density_id: {}".format(density_id))
                        if density_id == 'probability':
                            optimization_metric = 'probability density'
                        else:
                            if show_histogram:
                                optimization_metric = 'observed density'
                            else:
                                optimization_metric = 'observed frequency'
                        if show_kde:
                            if not isinstance(kde_colors, (tuple, list, np.ndarray)):
                                kde_colors = [kde_colors]
                        parametric_distribution = series[distribution_id]
                        n = len(parametric_distribution.sub_series)
                        if n == 0:
                            raise ValueError("sub-series is not initialized")
                        n += 1
                        xs, ys = [0], [0]
                        parameter_label = series['parameter mapping'][parametric_distribution.extreme_parameter]
                        alt_parameter_label = series['generalized parameter mapping'][parametric_distribution.extreme_parameter]
                        unit_label = series['unit mapping'][parametric_distribution.extreme_parameter]
                        handles, labels = [], []
                        parametric_distribution = series[distribution_id]
                        ## get figure and axes
                        kws = self.get_number_of_figure_rows_and_columns(n, layout)
                        fig, axes = plt.subplots(figsize=figsize, **kws)
                        for i, ax in enumerate(axes.ravel()):
                            if i == 0:
                                if show_kde:
                                    xs, ys = self.subview_kernel_density_estimation(
                                        ax=ax,
                                        series=series,
                                        distribution_id=distribution_id,
                                        density_id=density_id,
                                        histogram_id=histogram_id,
                                        kde_colors=kde_colors,
                                        kde_style=kde_style,
                                        i=i,
                                        xs=xs,
                                        ys=ys,
                                        swap_xy=False)
                                if show_histogram:
                                    xs, ys, _ = self.subview_distribution_histogram(
                                        ax=ax,
                                        series=series,
                                        distribution_id=distribution_id,
                                        density_id=density_id,
                                        histogram_id=histogram_id,
                                        bar_color=bar_color,
                                        step_color=step_color,
                                        i=i,
                                        xs=xs,
                                        ys=ys,
                                        alpha=0.775,
                                        swap_xy=False,
                                        separate_label=None)
                                if any([show_chi_square, show_g_test, show_maximum_likelihood]):
                                    nfits = np.sum([show_chi_square, show_g_test, show_maximum_likelihood])
                                    fits_alpha = 1 / nfits
                                    if (fits_alpha < 0.5) or (not np.isfinite(fits_alpha)):
                                        fits_alpha = 0.5
                                    xs, ys = self.subview_optimized_fit(
                                        ax=ax,
                                        series=series,
                                        distribution_id=distribution_id,
                                        optimization_metric=optimization_metric,
                                        csq_color=csq_color,
                                        gt_color=gt_color,
                                        mle_color=mle_color,
                                        i=i,
                                        xs=xs,
                                        ys=ys,
                                        show_chi_square=show_chi_square,
                                        show_g_test=show_g_test,
                                        show_maximum_likelihood=show_maximum_likelihood,
                                        alpha=fits_alpha,
                                        swap_xy=False,
                                        separate_label=None)
                                if show_statistics:
                                    raise ValueError("not yet implemented")
                                    # nfits = np.sum([show_chi_square, show_g_test, show_maximum_likelihood])
                                    # if nfits != 1:
                                    #     raise ValueError("this method 'show_statistics' requires that only one of the following be input as True: 'show_chi_square', 'show_g_test', 'show_maximum_likelihood'")
                                    # self.subview_statistics(
                                    #     ax=ax,
                                    #     series=series,
                                    #     distribution_id=distribution_id,
                                    #     unit_label=unit_label,
                                    #     i=i,
                                    #     mu_color=mu_color,
                                    #     sigma_color=sigma_color,
                                    #     median_color=median_color,
                                    #     mode_color=mode_color,
                                    #     show_chi_square=show_chi_square,
                                    #     show_g_test=show_g_test,
                                    #     show_maximum_likelihood=show_maximum_likelihood,
                                    #     swap_xy=False,
                                    #     separate_label=None)
                                if show_confidence_interval:
                                    nfits = np.sum([show_chi_square, show_g_test, show_maximum_likelihood])
                                    if nfits != 1:
                                        raise ValueError("this method 'show_confidence_interval' requires that only one of the following be input as True: 'show_chi_square', 'show_g_test', 'show_maximum_likelihood'")
                                    self.subview_confidence_interval(
                                        ax=ax,
                                        series=series,
                                        distribution_id=distribution_id,
                                        optimization_metric=optimization_metric,
                                        confidence_color=confidence_color,
                                        i=i,
                                        show_chi_square=show_chi_square,
                                        show_g_test=show_g_test,
                                        show_maximum_likelihood=show_maximum_likelihood,
                                        swap_xy=False)
                                _handles, _labels = ax.get_legend_handles_labels()
                                handles.extend(_handles)
                                labels.extend(_labels)
                            else:
                                sub_series = parametric_distribution.sub_series[i-1]
                                search_kwargs = parametric_distribution.container_of_search_kwargs[i-1]
                                if show_kde:
                                    xs, ys = self.subview_kernel_density_estimation(
                                        ax=ax,
                                        series=sub_series,
                                        distribution_id=distribution_id,
                                        density_id=density_id,
                                        histogram_id=histogram_id,
                                        kde_colors=kde_colors,
                                        kde_style=kde_style,
                                        i=i,
                                        xs=xs,
                                        ys=ys,
                                        swap_xy=False)
                                if show_histogram:
                                    xs, ys, _ = self.subview_distribution_histogram(
                                        ax=ax,
                                        series=sub_series,
                                        distribution_id=distribution_id,
                                        density_id=density_id,
                                        histogram_id=histogram_id,
                                        bar_color=bar_color,
                                        step_color=step_color,
                                        i=i,
                                        xs=xs,
                                        ys=ys,
                                        alpha=0.775,
                                        swap_xy=False,
                                        separate_label=None)
                                if any([show_chi_square, show_g_test, show_maximum_likelihood]):
                                    nfits = np.sum([show_chi_square, show_g_test, show_maximum_likelihood])
                                    fits_alpha = 1 / nfits
                                    if (fits_alpha < 0.5) or (not np.isfinite(fits_alpha)):
                                        fits_alpha = 0.5
                                    xs, ys = self.subview_optimized_fit(
                                        ax=ax,
                                        series=sub_series,
                                        distribution_id=distribution_id,
                                        optimization_metric=optimization_metric,
                                        csq_color=csq_color,
                                        gt_color=gt_color,
                                        mle_color=mle_color,
                                        i=i,
                                        xs=xs,
                                        ys=ys,
                                        show_chi_square=show_chi_square,
                                        show_g_test=show_g_test,
                                        show_maximum_likelihood=show_maximum_likelihood,
                                        alpha=fits_alpha,
                                        swap_xy=False,
                                        separate_label=None)
                                if show_statistics:
                                    raise ValueError("not yet implemented")
                                    # nfits = np.sum([show_chi_square, show_g_test, show_maximum_likelihood])
                                    # if nfits != 1:
                                    #     raise ValueError("this method 'show_statistics' requires that only one of the following be input as True: 'show_chi_square', 'show_g_test', 'show_maximum_likelihood'")
                                    # self.subview_statistics(
                                    #     ax=ax,
                                    #     series=sub_series,
                                    #     distribution_id=distribution_id,
                                    #     unit_label=unit_label,
                                    #     i=i,
                                    #     mu_color=mu_color,
                                    #     sigma_color=sigma_color,
                                    #     median_color=median_color,
                                    #     mode_color=mode_color,
                                    #     show_chi_square=show_chi_square,
                                    #     show_g_test=show_g_test,
                                    #     show_maximum_likelihood=show_maximum_likelihood,
                                    #     swap_xy=False,
                                    #     separate_label=None)
                                if show_confidence_interval:
                                    nfits = np.sum([show_chi_square, show_g_test, show_maximum_likelihood])
                                    if nfits != 1:
                                        raise ValueError("this method 'show_confidence_interval' requires that only one of the following be input as True: 'show_chi_square', 'show_g_test', 'show_maximum_likelihood'")
                                    self.subview_confidence_interval(
                                        ax=ax,
                                        series=sub_series,
                                        distribution_id=distribution_id,
                                        optimization_metric=optimization_metric,
                                        confidence_color=confidence_color,
                                        i=i,
                                        show_chi_square=show_chi_square,
                                        show_g_test=show_g_test,
                                        show_maximum_likelihood=show_maximum_likelihood,
                                        swap_xy=False)
                                ax.set_title(search_kwargs, fontsize=self.titlesize)
                            # ax.set_xlabel('{} [{}]'.format(parameter_label, unit_label), fontsize=self.labelsize)
                            # ax.set_ylabel(density_id.title(), fontsize=self.labelsize)
                            # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
                            # ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
                            # ax.set_title(series['identifiers']['series id'], fontsize=self.titlesize)
                        hspace = 0.425 if 'vertical' in layout else 0.3
                        fig.subplots_adjust(hspace=hspace)
                        ## update axes
                        xfmt = '{x:,.0f}' if np.nanmax(xs) > 1000 else '{x:,.2f}'
                        yfmt = '{x:,.0f}' if np.nanmax(ys) > 1000 else '{x:,.4f}'
                        self.share_axes(
                            axes=axes,
                            layout=layout,
                            xs=xs,
                            ys=ys,
                            sharex=sharex,
                            sharey=sharey,
                            xticks=True,
                            yticks=True,
                            xfmt=xfmt,
                            yfmt=yfmt,
                            xlim=True,
                            ylim=True,
                            xlabel='{} [{}]'.format(parameter_label, unit_label),
                            ylabel='{} Density'.format(density_id.title()),
                            collapse_x=collapse_x,
                            collapse_y=collapse_y)
                        fig.align_ylabels()
                        for ax in axes.ravel():
                            self.apply_grid(ax)
                        ## update legend
                        if layout == 'vertical':
                            fig.subplots_adjust(hspace=0.325)
                        self.subview_legend(
                            fig=fig,
                            ax=axes.ravel()[0],
                            handles=handles,
                            labels=labels,
                            title='{}'.format(distribution_id.title()),
                            bottom=0.2,
                            textcolor=True,
                            facecolor='white',
                            edgecolor='k',
                            titlecolor='k',
                            ncol=None)
                        ## update title
                        fig.suptitle(series['identifiers']['series id'], fontsize=self.titlesize)
                        ## show / save
                        if save:
                            savename = 'RawAnalysis_{}_DistributionSearchVariants'.format(
                                distribution_id.replace('_', '-').title().replace(' ', '_'))
                            if show_kde:
                                savename = '{}_KDE-{}'.format(savename, kde_style)
                            if show_histogram:
                                savename = '{}_HIST'.format(savename)
                            if show_chi_square:
                                savename = '{}_CSQ'.format(savename)
                            if show_g_test:
                                savename = '{}_GT'.format(savename)
                            if show_maximum_likelihood:
                                savename = '{}_MLE'.format(savename)
                            if show_statistics:
                                savename = '{}_STAT'.format(savename)
                            if show_confidence_interval:
                                savename = '{}_CONF'.format(savename)
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
                else:
                    raise ValueError("invalid type(layout): {}".format(type(layout)))

    def subview_nonperiodic_solar_cycle_separations(self, axes, layout, solar_cycles, separation_color='r', arrow_color='k', linestyle=':'):
        arrowprops = dict(facecolor=arrow_color, arrowstyle="->")
        if solar_cycles.size > 1:
            break_points = []
            dt_cycle_configuration = TemporalConfiguration()
            ## get datetime bounds per cycle number
            for k, (icycle, jcycle) in enumerate(zip(solar_cycles[:-1], solar_cycles[1:])):
                ibounds = dt_cycle_configuration.solar_cycles[icycle]['full']
                jbounds = dt_cycle_configuration.solar_cycles[jcycle]['full']
                if k == 0:
                    break_point = ibounds[0]
                    break_points.append(break_point)
                    break_point = ibounds[-1] + 0.5 * (jbounds[0] - ibounds[-1])
                    break_points.append(break_point)
                elif k == solar_cycles.size - 2:
                    break_point = jbounds[-1]
                    break_points.append(break_point)
                else:
                    break_point = ibounds[-1] + 0.5 * (jbounds[0] - ibounds[-1])
                    break_points.append(break_point)
                for ax in axes.ravel():
                    ## draw vertical line separating consecutive cycles
                    ylim = ax.get_ylim()
                    ax.axvline(
                        ibounds[-1],
                        ymin=0,
                        ymax=ylim[-1],
                        color=separation_color,
                        linestyle=linestyle)
            break_points = np.array(break_points)
            if len(axes.shape) == 1:
                if layout in ('overlay', 'horizontal'):
                    top_axes = axes
                else:
                    # top_axes = np.array(axes[0])
                    top_axes = np.array([axes.ravel()[0]])
            elif len(axes.shape) == 2:
                top_axes = axes[0, :]
            else:
                raise ValueError("invalid axes.shape: {}".format(axes.shape))
            for k, (ipoint, jpoint) in enumerate(zip(break_points[:-1], break_points[1:])):
                icycle = solar_cycles[k]
                jcycle = solar_cycles[k+1]
                delta = 0.2 * (jpoint - ipoint)
                left_edge, right_edge = jpoint - delta, jpoint + delta
                for ax in top_axes.ravel():
                    ylim = ax.get_ylim()
                    yarrow = ylim[1] * 0.8
                    ytext = ylim[1] * 0.85
                    ax.annotate(
                        ' ',
                        xy=(left_edge, yarrow),
                        xytext=(jpoint, yarrow),
                        horizontalalignment='center',
                        verticalalignment='center',
                        arrowprops=arrowprops)
                    ax.text(
                        left_edge,
                        ytext,
                        'SC ${}$'.format(icycle),
                        horizontalalignment='right',
                        verticalalignment='center',
                        fontsize=self.labelsize)
                    ax.annotate(
                        ' ',
                        xy=(right_edge, yarrow),
                        xytext=(jpoint, yarrow),
                        horizontalalignment='center',
                        verticalalignment='center',
                        arrowprops=arrowprops)
                    ax.text(
                        right_edge,
                        ytext,
                        'SC ${}$'.format(jcycle),
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=self.labelsize)

    def subview_nonperiodic_temporal_frequency_by_solar_cycle(self, fig, axes, layout, solar_cycles, search_labels, facecolors):
        w = 1 / solar_cycles.size
        h = 0.2
        # handles, labels = [], []
        for i, solar_cycle in enumerate(solar_cycles):
            leg_title = 'SC ${}$'.format(solar_cycle)
            handles, labels = [], []
            for j, (series, search_label) in enumerate(zip(self.series, search_labels)):
                if layout == 'overlay':
                    ax = axes.ravel()[0]
                else:
                    ax = axes.ravel()[j]
                if series['identifiers']['event type'] == 'Sunspot':
                    temporal_histogram = series['temporal histogram']
                    dt_bounds = TemporalConfiguration().solar_cycles[solar_cycle]['full']
                    loc = np.where((temporal_histogram.edges >= dt_bounds[0]) & (temporal_histogram.edges <= dt_bounds[1]))[0]
                    nevents = np.sum(temporal_histogram.counts[loc[:-1]])
                else:
                    event_searcher = EventSearcher(series['data'])
                    try:
                        data, _ = event_searcher.search_events(
                            parameters='solar cycle',
                            conditions='equal',
                            values=solar_cycle)
                        key = list(series['data'].keys())[0]
                        nevents = data[key].size
                    except ValueError:
                        nevents = 0
                if search_label in ('', None):
                    _label = '${:,}$ {}'.format(nevents, self.make_plural(series['identifiers']['event type']))
                else:
                    event_label = '${:,}$ {}'.format(nevents, self.make_plural(series['identifiers']['event type']))
                    # _label = r'%s$_{%s}$' % (event_label, search_label.replace('$', ''))
                    _label = r'{} ({})'.format(event_label, search_label)
                labels.append(_label)
                if layout != 'overlay':
                    _handles, _ = ax.get_legend_handles_labels()
                    handles.extend(_handles)
            if layout == 'overlay':
                handles, _ = axes.ravel()[0].get_legend_handles_labels()
            ## select corners of legend box
            if i < solar_cycles.size//2:
                loc = 'lower left'
            elif i > solar_cycles.size//2:
                loc = 'lower right'
            else:
                loc = 'lower center'
            ## initialize legend
            self.subview_legend(
                fig=fig,
                ax=axes.ravel()[0],
                handles=handles,
                labels=labels,
                title=leg_title,
                bottom=0.2,
                textcolor='k',
                facecolor='silver',
                edgecolor='k',
                titlecolor='k',
                ncol=None if len(handles) == 1 else 2,
                bbox_to_anchor=(i*w, 0, w, h))

    def subview_periodic_frequency_by_series(self, fig, axes, handles, labels, counts, search_labels):
        ## get specs of legend boxes
        w = 1 / self.n
        h = 0.275
        fig.subplots_adjust(bottom=0.4)
        for i, (series, search_label) in enumerate(zip(self.series, search_labels)):
            curr_handles = handles[i]
            curr_labels = labels[i]
            curr_counts = counts[i]
            # # leg_title = '{}: ${:,}$ Events'.format(series['identifiers']['series id'], np.sum(curr_counts))
            # partial_label = r'%s$_{%s}$' % (self.make_plural(series['event type']), search_label.replace('$', ''))
            # leg_title = '{:,} {}'.format(np.sum(curr_counts), partial_label)
            if search_label in ('', None):
                leg_title = r'{:,} {}'.format(np.sum(curr_counts), self.make_plural(series['identifiers']['event type']))
            else:
                leg_title = r'{:,} {} ({})'.format(np.sum(curr_counts), self.make_plural(series['identifiers']['event type']), search_label)
            ## select bottom corner of legend box
            if i < self.n//2:
                loc = 'lower left'
            elif i > self.n//2:
                loc = 'lower right'
            else:
                loc = 'lower center'
            ## initialize legend
            if self.n <= 4:
                bbox_coordinates = (i*w, 0, w, h)
            else:
                if self.n > 8:
                    raise ValueError("not yet implemented")
                if i < 4:
                    bbox_coordinates = (i*w, h, w, h)
                else:
                    bbox_coordinates = (i*w, 0, w, h)
            self.subview_legend(
                fig=fig,
                ax=axes.ravel()[0],
                handles=curr_handles,
                labels=curr_labels,
                title=leg_title,
                bottom=None,
                textcolor=True,
                facecolor='silver',
                edgecolor='k',
                titlecolor='k',
                ncol=None if len(curr_handles) == 1 else 2,
                bbox_to_anchor=bbox_coordinates)

    def view_temporal_frequency_nonperiodically(self, fig, axes, layout, tfmt, facecolors, background_color=None, separation_color='r', arrow_color='k', show_cycle_separations=False, show_frequency_by_cycle=False, sharex=False, sharey=False, collapse_x=False, collapse_y=False, save=False):
        ## verify inputs
        if not isinstance(facecolors, (tuple, list, np.ndarray)):
            facecolors = [facecolors]
        nc = len(facecolors)
        if nc < self.n:
            raise ValueError("{} facecolors for {} series".format(nc, self.n))
        solar_cycles = []
        for series in self.series:
            solar_cycles.append(
                np.unique(series['data']['solar cycle']))
        solar_cycles = np.unique(np.concatenate(solar_cycles, axis=0))
        xs, ys = [], [0]
        handles, labels = [], []
        ylabels = []
        search_labels = []
        shared_xlabel = 'Date'
        ## initialize plots
        for i, (series, facecolor) in enumerate(zip(self.series, facecolors)):
            if layout == 'overlay':
                ax = axes.ravel()[0]
            else:
                ax = axes.ravel()[i]
            alpha = 1 / self.n if layout == 'overlay' else 0.8
            search_kwargs = series['identifiers']['search kwargs']
            if search_kwargs is None:
                search_label = ''
            else:
                search_args = self.autocorrect_search_inputs(search_kwargs['parameters'], search_kwargs['conditions'], search_kwargs['values'])
                search_parameters, search_conditions, search_values, search_modifiers = search_args
                if len(search_parameters) == len(search_conditions) == len(search_values) == 1:
                    unit_label = series['unit mapping'][search_parameters[0]]
                    search_label = '{} {} {:,}'.format(
                        series['parameter mapping'][search_parameters[0]],
                        self.relational_mapping[search_conditions[0]],
                        search_values[0])
                    if unit_label is not None:
                        search_label = '{} {}'.format(search_label, unit_label)
                else:
                    search_label = self.get_search_label(**search_kwargs)
            search_labels.append(search_label)
            if series['identifiers']['event type'] == 'Sunspot':
                temporal_histogram = series['temporal histogram']
                nevents = np.sum(temporal_histogram.counts)
            else:
                temporal_histogram = series['temporal frequency'].temporal_histogram
                nevents = int(np.sum(series['data']['is event']))
            leg_label = '${:,}$ {}'.format(nevents, self.make_plural(series['identifiers']['event type']))
            xs.append(temporal_histogram.edges[0])
            xs.append(temporal_histogram.edges[-1])
            ys.append(np.nanmax(temporal_histogram.counts) * 1.125)
            if (layout == 'overlay') and (self.n > 1):
                ylabel = 'Event Frequency'
            else:
                if search_label in ('', None):
                    ylabel = '{} Frequency'.format(series['identifiers']['event type'])
                else:
                    ylabel = '{} Frequency\n{}'.format(series['identifiers']['event type'], search_label)
            ylabels.append(ylabel)
            ax.plot(
                temporal_histogram.midpoints,
                temporal_histogram.counts,
                color=facecolor,
                label=leg_label,
                linestyle='-',
                lw=1,
                alpha=alpha)
            ax.set_title(series['identifiers']['series id'], fontsize=self.titlesize)
            ## collect legend parameters
            if ((layout == 'overlay') and (self.n > 1)):
                if i == self.n - 1:
                    _handles, _labels = ax.get_legend_handles_labels()
                    handles.extend(_handles)
                    labels.extend(_labels)
            else:
                _handles, _labels = ax.get_legend_handles_labels()
                handles.extend(_handles)
                labels.extend(_labels)
            ax = self.subview_datetime_axis(
                ax=ax,
                axis='x',
                major_interval=12,
                minor_interval=1,
                sfmt=tfmt)
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.set_xlabel(shared_xlabel, fontsize=self.labelsize)
            ax.set_ylabel(ylabel, fontsize=self.labelsize)
            ax.tick_params(axis='both', labelsize=self.ticksize)
            self.apply_grid(ax)
        ## update axes
        if self.is_same_elements(elements=ylabels, s='', n=self.n):
            shared_ylabel = '{}'.format(ylabels[0])
        else:
            shared_ylabel = None
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
            ax.tick_params(axis='x', labelrotation=15)
        fig.align_ylabels()
        if background_color is None:
            for ax in axes.ravel():
                self.apply_grid(ax)
        else:
            for ax in axes.ravel():
                ax.set_facecolor(background_color)
                if background_color in ('k', 'black') or ('dark' in background_color):
                    ax.grid(color='white', alpha=0.3, linestyle=':')
        if layout != 'overlay':
            for ax, facecolor in zip(axes.ravel(), facecolors):
                ax.tick_params(
                    axis='x',
                    which='both',
                    labelsize=self.ticksize,
                    colors=facecolor,
                    rotation=15)
                ax.tick_params(
                    axis='y',
                    which='both',
                    labelsize=self.ticksize,
                    colors=facecolor)
                ax.xaxis.label.set_color(facecolor)
                ax.yaxis.label.set_color(facecolor)
        if show_cycle_separations:
            self.subview_nonperiodic_solar_cycle_separations(
                axes=axes,
                layout=layout,
                solar_cycles=solar_cycles,
                separation_color=separation_color,
                arrow_color=arrow_color)
        ## update legend
        if layout in ('vertical', 'square'):
            hspace = 0.425
        else:
            hspace = 0.3
        fig.subplots_adjust(hspace=hspace, bottom=0.2)
        if show_frequency_by_cycle:
            self.subview_nonperiodic_temporal_frequency_by_solar_cycle(
                fig=fig,
                axes=axes,
                layout=layout,
                solar_cycles=solar_cycles,
                search_labels=search_labels,
                facecolors=facecolors)
        else:
            if self.n < 6:
                if self.n == 1:
                    ncol = None
                else:
                    ncol = self.n
            else:
                ncol = self.n // 2
            self.subview_legend(
                fig=fig,
                ax=axes.ravel()[0],
                handles=handles,
                labels=labels,
                title='Frequency of Events',
                bottom=None,
                textcolor='k',
                facecolor='silver',
                edgecolor='k',
                titlecolor='k',
                ncol=ncol)
        ## update title
        s = 'Frequency of Events'
        if ('horizontal' not in layout) and (layout != 'square'):
            axes.ravel()[0].set_title(s, fontsize=self.titlesize)
        else:
            fig.suptitle(s, fontsize=self.titlesize)
        ## save or show
        if save:
            savename = "RawAnalaysis_TempFreq_P0"
            if show_cycle_separations:
                savename = '{}_SEP'.format(savename)
            if show_frequency_by_cycle:
                savename = '{}_CYCLE'.format(savename)
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
            savename = '{}_{}'.format(savename, layout)
            savename = savename.replace(' ', '_')
        else:
            savename = None
        self.display_image(fig, savename=savename)

    def subview_periodic_frequency_by_data(self, ax, series, period, subperiod, event_searcher, period_values, facecolors, i, total_nevents, _handles, _labels, _counts, xs, ys):
        for period_value, facecolor in zip(period_values, facecolors):
            ## get data per visible period
            period_events, period_indices = event_searcher.search_events(
                parameters=period,
                conditions='equal',
                values=period_value)
            nperiod_events = np.sum(period_indices)
            total_nevents[i] += nperiod_events
            ## get counts per subperiod
            xt, yc = [], []
            subperiod_values = np.unique(period_events[subperiod])
            subtot = 0
            for subperiod_value in subperiod_values:
                subsearcher = EventSearcher(period_events)
                subevents, subindices = subsearcher.search_events(
                    parameters=subperiod,
                    conditions='equal',
                    values=subperiod_value)
                _subtot = np.sum(subindices)
                subtot = _subtot + subtot # += is in-place
                if period == 'solar cycle':
                    xt.append(subperiod_value - np.min(subperiod_values))
                else:
                    xt.append(subperiod_value)
                yc.append(_subtot)
            if period == 'solar cycle':
                xs.append(np.nanmax(xt))
            ys.append(np.nanmax(yc) * 1.125)
            ## get legend label
            if period == 'solar cycle':
                label = 'SC ${}$: {:,} Events'.format(period_value, subtot)
            elif period in ('year', 'hour', 'minute', 'second'):
                label = '${}$: {:,} Events'.format(period_value, subtot)
            elif period == 'month':
                label = '${}$: {:,} Events'.format(calendar.month_abbr[period_value], subtot)
            elif period == 'day':
                label = '${}$: {:,} Events'.format(calendar.day_abbr[period_value], subtot)
            else:
                raise ValueError("invalid period: {}".format(period))
            ## do plot
            handle, = ax.plot(
                xt,
                yc,
                color=facecolor,
                label=label,
                lw=1,
                # linestyle='-')
                # marker='.',
                markersize=5,
                marker='o',
                linestyle='--')
            _handles[i].append(handle)
            _labels[i].append(label)
            _counts[i].append(subtot)
        return xs, ys, total_nevents, _handles, _labels, _counts

    def view_temporal_frequency_periodically(self, fig, axes, layout, period, cmaps, background_color, sharex=False, sharey=False, collapse_x=False, collapse_y=False, save=False):
        ## verify inputs
        if not isinstance(cmaps, (tuple, list, np.ndarray)):
            cmaps = [cmaps]
        nc = len(cmaps)
        if nc < self.n:
            if nc == 1:
                cmaps = [cmaps[0] for i in range(self.n)]
            else:
                raise ValueError("{} cmaps for {} series".format(nc, self.n))
        subperiod = self.period_map[period]['subperiod']
        dt_bounds = dict()
        ys = [0]
        if period == 'solar cycle':
            xs = [0]
            shared_xlabel = 'Elapsed Years Since Onset of Solar Cycle'
        else:
            xs = self.period_map[period]['ticks']
            shared_xlabel = '{}'.format(subperiod.title())
        search_labels = []
        shared_ylabels = []
        total_nevents = dict()
        _handles, _labels, _counts = dict(), dict(), dict()
        ## initialize plots
        for i, (series, cmap) in enumerate(zip(self.series, cmaps)):
            if layout == 'overlay':
                ax = axes.ravel()[0]
            else:
                ax = axes.ravel()[i]
            dt_bounds[i] = (np.nanmin(series['data']['datetime']), np.nanmax(series['data']['datetime']))
            total_nevents[i] = 0
            _handles[i] = []
            _labels[i] = []
            _counts[i] = []
            search_kwargs = series['identifiers']['search kwargs']
            if search_kwargs is None:
                search_label = ''
            else:
                search_args = self.autocorrect_search_inputs(search_kwargs['parameters'], search_kwargs['conditions'], search_kwargs['values'])
                search_parameters, search_conditions, search_values, search_modifiers = search_args
                if len(search_parameters) == len(search_conditions) == len(search_values) == 1:
                    unit_label = series['unit mapping'][search_parameters[0]]
                    search_label = '{} {} {:,}'.format(
                        series['parameter mapping'][search_parameters[0]],
                        self.relational_mapping[search_conditions[0]],
                        search_values[0])
                    if unit_label is not None:
                        search_label = '{} {}'.format(search_label, unit_label)
                else:
                    search_label = self.get_search_label(**search_kwargs)
            if (layout == 'overlay') and (self.n > 1):
                ylabel = 'Event Frequency'
            else:
                if search_label in ('', None):
                    ylabel = '{} Frequency'.format(series['identifiers']['event type'])
                else:
                    ylabel = '{} Frequency\n{}'.format(series['identifiers']['event type'], search_label)
            search_labels.append(search_label)
            shared_ylabels.append(ylabel)
            if series['identifiers']['event type'] == 'Sunspot':
                temporal_histogram = series['temporal histogram']
                binned_dts = []
                for midpoint, count in zip(temporal_histogram.midpoints, temporal_histogram.counts):
                    binned_dts.extend([midpoint for jc in range(count)])
                binned_dts = np.array(binned_dts)
                events = temporal_histogram.consolidate_datetime_components(
                    dts=binned_dts,
                    prefix=None)
                cycle_numbers, _ = temporal_histogram.group_cycles_by_datetime(
                    dts=binned_dts,
                    solar_cycles=None,
                    activity_type='full',
                    bias='left',
                    verify_activity_type=False,
                    verify_cycle_numbers=False)
                events['solar cycle'] = cycle_numbers
                event_searcher = EventSearcher(events=events)
            else:
                event_searcher = EventSearcher(events=series['data'])
            period_values = np.unique(event_searcher.events[period])
            norm = Normalize(vmin=period_values[0], vmax=period_values[-1])
            facecolors = self.get_facecolors_from_cmap(
                cmap=cmap,
                norm=norm,
                arr=period_values)
            xs, ys, total_nevents, _handles, _labels, _counts = self.subview_periodic_frequency_by_data(
                ax=ax,
                series=series,
                period=period,
                subperiod=subperiod,
                event_searcher=event_searcher,
                period_values=period_values,
                facecolors=facecolors,
                i=i,
                total_nevents=total_nevents,
                _handles=_handles,
                _labels=_labels,
                _counts=_counts,
                xs=xs,
                ys=ys)
            if (self.n > 1) and (layout != 'overlay'):
                ax.set_title(series['identifiers']['series id'], fontsize=self.titlesize)
            ax.set_xlabel(shared_xlabel, fontsize=self.labelsize)
            ax.set_ylabel(ylabel, fontsize=self.labelsize)
            ax.tick_params(axis='both', labelsize=self.ticksize)
            self.apply_grid(ax)
        ## update axes
        major_xticks = np.copy(self.period_map[period]['ticks'])
        if period == 'solar cycle':
            minor_xticks = StatisticsConfiguration(vs=None).get_midpoints(major_xticks)
            xticks = [major_xticks, minor_xticks]
        else:
            xticks = [major_xticks, []]
        if self.is_same_elements(elements=shared_ylabels, s='', n=self.n):
            shared_ylabel = '{}'.format(shared_ylabels[0])
        else:
            shared_ylabel = None
        self.share_axes(
            axes=axes,
            layout=layout,
            xs=xs,
            ys=ys,
            sharex=sharex,
            sharey=sharey,
            xlim=True,
            ylim=True,
            xticks=xticks,
            yticks=True,
            xfmt='{x:,.0f}',
            yfmt='{x:,.1f}',
            xlabel=shared_xlabel,
            ylabel=shared_ylabel)
        if period in ('year', 'month'):
            for ax in axes.ravel():
                ax.set_xticklabels(self.period_map[period]['ticklabels'], fontsize=self.ticksize)
        self.unshare_axis_parameters(
            axes=axes,
            layout=layout,
            collapse_x=collapse_x,
            collapse_y=collapse_y)
        fig.align_ylabels()
        if background_color is not None:
            for ax in axes.ravel():
                ax.set_facecolor(background_color)
                if background_color in ('k', 'black'):
                    ax.grid(color='white', alpha=0.3, linestyle=':')
        ## update legend
        if self.n == 1:
            handles, labels = [], []
            for ax in axes.ravel():
                _handles, _labels = ax.get_legend_handles_labels()
                handles.extend(_handles)
                labels.extend(_labels)
            first_search_label = search_labels[0]
            if first_search_label in ('', None):
                leg_title = r'{:,} {}'.format(total_nevents[0], self.make_plural(self.series[0]['identifiers']['event type']))
            else:
                leg_title = r'{:,} {} ({})'.format(total_nevents[0], self.make_plural(self.series[0]['identifiers']['event type']), first_search_label)
            if period == 'solar cycle':
                leg_title = '{} from SC ${}$ $-$ ${}$'.format(leg_title, period_values[0], period_values[-1])
            elif period == 'year':
                leg_title = '{} from ${}$ $-$ ${}$'.format(leg_title, period_values[0], period_values[-1])
            else:
                leg_title = '{} from ${}$ $-$ ${}$'.format(leg_title, dt_bounds[0][0], dt_bounds[0][-1])
            self.subview_legend(
                fig=fig,
                ax=ax,
                handles=handles,
                labels=labels,
                title=leg_title,
                bottom=0.275,
                textcolor=True,
                facecolor='silver',
                edgecolor='k',
                titlecolor='k',
                ncol=None if len(handles) == 1 else 5)
        else:
            if layout in ('vertical', 'square'):
                fig.subplots_adjust(hspace=0.325)
            self.subview_periodic_frequency_by_series(
                fig=fig,
                axes=axes,
                handles=_handles,
                labels=_labels,
                counts=_counts,
                search_labels=search_labels)
        ## update title
        if period == 'solar cycle':
            s = 'Frequency of Events by Solar Cycle'
        else:
            s = '{}ly Frequency of Events'.format(period.title())
        if ('horizontal' not in layout) and (layout != 'square'):
            axes.ravel()[0].set_title(s, fontsize=self.titlesize)
        else:
            fig.suptitle(s, fontsize=self.titlesize)
        ## save or show
        if save:
            savename = "RawAnalaysis_TempFreq_P-{}".format(period.title().replace(' ', ''))
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
            savename = '{}_{}'.format(savename, layout)
            savename = savename.replace(' ', '_')
        else:
            savename = None
        self.display_image(fig, savename=savename)

    def view_temporal_frequency(self, show_cycle_separations=False, show_frequency_by_cycle=False, period=None, cmaps=('Oranges', 'Blues', 'Greens', 'Purples'), facecolors=('darkorange', 'darkgreen', 'purple', 'steelblue'), background_color=None, separation_color='r', arrow_color='k', tfmt='%Y-%m', sharex=False, sharey=False, collapse_x=False, collapse_y=False, figsize=None, save=False, layout='single'):
        if layout is None:
            permutable_layouts = ['single']
            if period is None:
                if self.n > 1:
                    permutable_layouts.append('overlay')
                    permutable_layouts.append('vertical')
                if 2 <= self.n < 4:
                    permutable_layouts.append('horizontal')
                elif (self.n > 2) and (self.n % 2 == 0):
                    permutable_layouts.append('square')
            else:
                if (self.n > 2) and (self.n % 2 == 0):
                    permutable_layouts.append('square')
                if 2 <= self.n < 4:
                    permutable_layouts.append('horizontal')
                if self.n > 1:
                    permutable_layouts.append('vertical')
            self.view_layout_permutations(
                f=self.view_temporal_frequency,
                layouts=permutable_layouts,
                show_cycle_separations=show_cycle_separations,
                show_frequency_by_cycle=show_frequency_by_cycle,
                period=period,
                cmaps=cmaps,
                facecolors=facecolors,
                background_color=background_color,
                separation_color=separation_color,
                arrow_color=arrow_color,
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
                visualizer.view_temporal_frequency(
                    show_cycle_separations=show_cycle_separations,
                    show_frequency_by_cycle=show_frequency_by_cycle,
                    period=period,
                    cmaps=cmaps,
                    facecolors=facecolors,
                    background_color=background_color,
                    separation_color=separation_color,
                    arrow_color=arrow_color,
                    tfmt=tfmt,
                    sharex=sharex,
                    sharey=sharey,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y,
                    figsize=figsize,
                    save=save,
                    layout='overlay')
        else:
            ## get figure and axes
            kws = self.get_number_of_figure_rows_and_columns(self.n, layout)
            fig, axes = plt.subplots(figsize=figsize, **kws)
            if layout == 'overlay':
                axes = np.array([axes])
            if period is None:
                self.view_temporal_frequency_nonperiodically(
                    fig=fig,
                    axes=axes,
                    layout=layout,
                    tfmt=tfmt,
                    facecolors=facecolors,
                    background_color=background_color,
                    separation_color=separation_color,
                    arrow_color=arrow_color,
                    show_cycle_separations=show_cycle_separations,
                    show_frequency_by_cycle=show_frequency_by_cycle,
                    sharex=sharex,
                    sharey=sharey,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y,
                    save=save)
            else:
                if period not in ('solar cycle', 'year'):
                    raise ValueError("not yet implemented")
                self.view_temporal_frequency_periodically(
                    fig=fig,
                    axes=axes,
                    layout=layout,
                    period=period,
                    cmaps=cmaps,
                    background_color=background_color,
                    sharex=sharex,
                    sharey=sharey,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y,
                    save=save)

    def subview_heat_map_separations_by_solar_cycle(self, axes, layout, dtmin, dtmax, xparameter, yparameter, sharex, sharey, separation_color='r', linestyle='-'):
        tconfig = TemporalConfiguration()
        discrete_solar_cycles, _ = tconfig.group_cycles_by_datetime(
            dts=np.array([dtmin, dtmax]))
        solar_cycles = np.arange(discrete_solar_cycles[0], discrete_solar_cycles[-1] + 1).astype(int)
        if solar_cycles.size < 1:
            raise ValueError("invalid solar_cycles: {}".format(solar_cycles))
        elif solar_cycles.size > 1:
            for solar_cycle in solar_cycles[:-1]:
                dt_right = tconfig.solar_cycles[solar_cycle]['full'][1]
                for i, (ax, series) in enumerate(zip(axes.ravel(), self.series)):
                    mixed_frequency_result = series['mixed frequency']
                    dt_edges, p_edges = mixed_frequency_result.mixed_histogram['x edges'], mixed_frequency_result.mixed_histogram['y edges']
                    scale_factor = mixed_frequency_result.temporal_histogram.relative_time_scales[mixed_frequency_result.temporal_histogram.time_step]
                    t_right = (dt_right - dtmin).total_seconds() / scale_factor
                    if t_right > 0:
                        if xparameter == 'temporal value':
                            ax.axvline(
                                x=t_right,
                                color=separation_color,
                                linestyle='-')
                            if layout == 'overlay':
                                j = -1
                            else:
                                if len(axes.shape) == 1:
                                    if layout in ('horizontal', 'vertical'):
                                        j = axes.size - 2
                                    else:
                                        j = np.array([axes.ravel()[0]]).size - 1
                                elif len(axes.shape) == 2:
                                    j = axes.size - axes[0, :].size - 1
                                else:
                                    raise ValueError("invalid axes.shape: {}".format(axes.shape))
                            if i > j:
                                ylim = ax.get_ylim()
                                yticks = ax.get_yticks()
                                if self.n > 1:
                                    y_btm = ylim[0] - 0.5 * (yticks[1] - yticks[0])
                                    verticalalignment = 'bottom'
                                else:
                                    y_btm = ylim[0] - 0.15 * (yticks[1] - yticks[0])
                                    verticalalignment = 'top'
                                ax.text(
                                    x=t_right,
                                    y=y_btm,
                                    s='$\longleftarrow$ SC ${}$'.format(solar_cycle) + ' '*5 + 'SC ${}$ $\longrightarrow$'.format(solar_cycle + 1),
                                    color=separation_color,
                                    fontsize=self.textsize,
                                    horizontalalignment='center',
                                    verticalalignment=verticalalignment)
                        else:
                            ax.axhline(
                                x=t_right,
                                color=separation_color,
                                linestyle='-')
                            if len(axes.shape) == 1:
                                if layout in ('overlay', 'vertical'):
                                    j = axes.size
                                else:
                                    j = np.array([axes.ravel()[0]]).size
                            elif len(axes.shape) == 2:
                                j = axes[:, 0].size
                            else:
                                raise ValueError("invalid axes.shape: {}".format(axes.shape))
                            if i < j:
                                xlim = ax.get_xlim()
                                ax.text(
                                    x=xlim[1],
                                    y=t_right,
                                    s='$\longleftarrow$ SC ${}$'.format(solar_cycle) + ' '*5 + 'SC ${}$ $\longrightarrow$'.format(solar_cycle + 1),
                                    rotation=90,
                                    color=separation_color,
                                    fontsize=self.textsize,
                                    horizontalalignment='left',
                                    verticalalignment='center')

    def view_temporal_frequency_heatmap(self, xparameter='temporal value', yparameter='extreme value', show_cycle_separations=False, show_colorbar=False, show_legend=False, cmap='Oranges', invalid_color=None, color_spacing='linear', separation_color='gray', tfmt='%Y-%m', interpolation='nearest', aspect='auto', sharex=False, sharey=False, collapse_x=False, collapse_y=False, figsize=None, save=False, layout='single'):
        if layout is None:
            permutable_layouts = ['single']
            # if 2 <= self.n < 4:
            #     # permutable_layouts.append('horizontal')
            #     permutable_layouts.append('vertical')
            # elif (self.n > 2) and (self.n % 2 == 0):
            #     permutable_layouts.append('square')
            self.view_layout_permutations(
                f=self.view_temporal_frequency_heatmap,
                layouts=permutable_layouts,
                xparameter=xparameter,
                yparameter=yparameter,
                show_cycle_separations=show_cycle_separations,
                show_colorbar=show_colorbar,
                show_legend=show_legend,
                cmap=cmap,
                invalid_color=invalid_color,
                color_spacing=color_spacing,
                separation_color=separation_color,
                tfmt=tfmt,
                interpolation=interpolation,
                aspect=aspect,
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
                visualizer.view_temporal_frequency_heatmap(
                    xparameter=xparameter,
                    yparameter=yparameter,
                    show_cycle_separations=show_cycle_separations,
                    show_colorbar=show_colorbar,
                    show_legend=show_legend,
                    cmap=cmap,
                    invalid_color=invalid_color,
                    color_spacing=color_spacing,
                    separation_color=separation_color,
                    tfmt=tfmt,
                    interpolation=interpolation,
                    aspect=aspect,
                    sharex=sharex,
                    sharey=sharey,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y,
                    figsize=figsize,
                    save=save,
                    layout='overlay')
        else:
            ## verify inputs
            if layout != 'overlay':
                raise ValueError("not yet implemented")
            if 'temporal value' not in (xparameter, yparameter):
                raise ValueError("invalid xparameter={} and/or yparameter={}".format(xparameter, yparameter))
            if 'extreme value' not in (xparameter, yparameter):
                raise ValueError("invalid xparameter={} and/or yparameter={}".format(xparameter, yparameter))
            if xparameter == yparameter:
                raise ValueError("invalid xparameter={} and/or yparameter={}".format(xparameter, yparameter))
            ## collect datetime bounds and largest z-value per series
            zs, dt_bounds = [], dict()
            for i, series in enumerate(self.series):
                mixed_histogram = series['mixed frequency'].mixed_histogram
                xy_counts = mixed_histogram['counts']
                zs.append(np.nanmax(xy_counts))
                dt_bounds[i] = (mixed_histogram['x edges'][0], mixed_histogram['x edges'][-1])
            zmax = np.max(zs)
            dt_bounds_flat = np.array(list(dt_bounds.values()))
            dtmin, dtmax = np.min(dt_bounds_flat), np.max(dt_bounds_flat)
            dt_min = dtmin
            dt_max = dtmax
            offset_edges, offset_edge_labels = dict(), dict()
            ## convert z-values to normalized color-mapping
            if color_spacing == 'linear':
                levels = np.arange(np.max(zs) + 1).astype(int)
                norm = Normalize(vmin=0, vmax=levels[-1])
                facecolors = self.get_facecolors_from_cmap(
                    cmap=cmap,
                    norm=norm,
                    arr=levels)
            elif color_spacing == 'logarithmic':
                left_exponent, _ = self.get_logarithmic_bounds(
                    num=0.1,
                    base=10,
                    bound_id='lower')
                largest_level = self.round_up(
                    num=zmax,
                    divisor=10**(len(str(int(zmax))) - 1))
                right_exponent, _ = self.get_logarithmic_bounds(
                    num=largest_level,
                    base=10,
                    bound_id='upper')
                left_exponent, right_exponent = int(left_exponent), int(right_exponent)
                levels = np.logspace(left_exponent, right_exponent, right_exponent - left_exponent + 1)
                norm = LogNorm(vmin=levels[0], vmax=levels[-1])
                # levels = np.insert(levels, 0, 0)
                levels[0] = 0
            else:
                raise ValueError("invalid color_spacing: {}".format(color_spacing))
            facecolors = self.get_facecolors_from_cmap(
                cmap=cmap,
                norm=norm,
                arr=levels)
            if invalid_color is not None:
                facecolors[0] = to_rgba(invalid_color)
            if zmax < 21:
                _cmap = ListedColormap(facecolors)
            else:
                _cmap = plt.get_cmap(cmap)
            if invalid_color is not None:
                _cmap.set_bad(color=invalid_color)
                _cmap.set_under(color=invalid_color)
            ## initialize axis parameters
            xs, ys = [], []
            shared_xlabels, shared_ylabels = [], []
            ## get figure and axes
            kws = self.get_number_of_figure_rows_and_columns(self.n, layout)
            fig, axes = plt.subplots(figsize=figsize, **kws)
            ## do plot
            if layout == 'overlay':
                if self.n != 1:
                    raise ValueError("layout='overlay' for this method will only work for one series")
                axes = np.array([axes])
            for i, (ax, series) in enumerate(zip(axes.ravel(), self.series)):
                ## convert datetime edges to elapsed edges
                mixed_frequency_result = series['mixed frequency']
                dt_edges, p_edges = mixed_frequency_result.mixed_histogram['x edges'], mixed_frequency_result.mixed_histogram['y edges']
                scale_factor = mixed_frequency_result.temporal_histogram.relative_time_scales[mixed_frequency_result.temporal_histogram.time_step]
                elapsed_edges = np.array([(dt_edge - dt_min).total_seconds() / scale_factor for dt_edge in dt_edges])
                offset_edges[i] = elapsed_edges
                offset_edge_labels[i] = dt_edges
                ## get axis parameters and select orientation
                parameter_label = series['parameter mapping'][mixed_frequency_result.extreme_parameter]
                unit_label = series['unit mapping'][mixed_frequency_result.extreme_parameter]
                p_label = '{} [{}]'.format(parameter_label, unit_label)
                if xparameter == 'extreme value':
                    dt_axis = 'y'
                    xlabel = p_label[:]
                    ylabel = 'Date'
                    xticks = True
                    yticks = False
                    x_edges = np.copy(p_edges)
                    y_edges = np.copy(elapsed_edges)
                    xy_counts = mixed_frequency_result.mixed_histogram['counts']
                    extent = (p_edges[0], p_edges[-1], elapsed_edges[0], elapsed_edges[-1])
                    xs.append(p_edges[0])
                    xs.append(p_edges[-1])
                    ys.append(elapsed_edges[0])
                    ys.append(elapsed_edges[-1])
                else:
                    dt_axis = 'x'
                    xlabel = 'Date'
                    ylabel = p_label[:]
                    xticks = False
                    yticks = True
                    xfmt = None
                    yfmt = '{x:,.0f}'
                    x_edges = np.copy(elapsed_edges)
                    y_edges = np.copy(p_edges)
                    xy_counts = mixed_frequency_result.mixed_histogram['counts'].T
                    extent = (elapsed_edges[0], elapsed_edges[-1], p_edges[0], p_edges[-1])
                    xs.append(elapsed_edges[0])
                    xs.append(elapsed_edges[-1])
                    ys.append(p_edges[0])
                    ys.append(p_edges[-1])
            shared_xlabels.append(xlabel)
            shared_ylabels.append(ylabel)
            ## initialize plot
            im_handle = ax.imshow(
                xy_counts,
                cmap=_cmap,
                norm=norm,
                interpolation=interpolation,
                aspect=aspect,
                origin='lower',
                extent=extent)
            ## update axes
            ax.set_xlabel(xlabel, fontsize=self.labelsize)
            ax.set_ylabel(ylabel, fontsize=self.labelsize)
            ax.set_title(series['identifiers']['series id'], fontsize=self.titlesize)
            ax.tick_params(axis='both', labelsize=self.ticksize)
            if xparameter == 'extreme value':
                xfmt = '{x:,.1f}'
                yfmt = None
                _xlim = True
                _ylim = True
                ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
                ax.xaxis.set_major_formatter(ticker.StrMethodFormatter(xfmt))
            else:
                xfmt = None
                yfmt = '{x:,.1f}'
                _xlim = True
                _ylim = True
                ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
                ax.yaxis.set_major_formatter(ticker.StrMethodFormatter(yfmt))
            if self.is_same_elements(elements=shared_xlabels, s='', n=self.n):
                shared_xlabel = '{}'.format(shared_xlabels[0])
            else:
                shared_xlabel = None
            if self.is_same_elements(elements=shared_ylabels, s='', n=self.n):
                shared_ylabel = '{}'.format(shared_ylabels[0])
            else:
                shared_ylabel = None
            self.share_axes(
                axes=axes,
                layout=layout,
                xs=xs,
                ys=ys,
                sharex=sharex,
                sharey=sharey,
                xlim=_xlim,
                ylim=_ylim,
                xticks=xticks,
                yticks=yticks,
                xfmt=xfmt,
                yfmt=yfmt,
                xlabel=shared_xlabel,
                ylabel=shared_ylabel,
                collapse_x=collapse_x,
                collapse_y=collapse_y)
            if ((xparameter == 'temporal value') and (sharex)) or ((yparameter == 'temporal value') and sharey):
                if xparameter == 'temporal value':
                    for ax in axes.ravel():
                        xlim = ax.get_xlim()
                        ax.set_xticks([xlim[0], xlim[-1]])
                        ax.set_xticklabels([dtmin.strftime(tfmt), dtmax.strftime(tfmt)], fontsize=self.ticksize, rotation=15)
                        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter(yfmt))
                else:
                    for ax in axes.ravel():
                        ylim = ax.get_ylim()
                        ax.set_yticks([ylim[0], ylim[-1]])
                        ax.set_yticklabels([dtmin.strftime(tfmt), dtmax.strftime(tfmt)], fontsize=self.ticksize, rotation=15)
                        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter(xfmt))
            else:
                for i, (ax, series) in enumerate(zip(axes.ravel(), self.series)):
                    edges = offset_edges[i]
                    edge_labels = offset_edge_labels[i]
                    if xparameter == 'temporal value':
                        ax.set_xticks([edges[0], edges[-1]])
                        ax.set_xticklabels([edge_labels[0].strftime(tfmt), edge_labels[-1].strftime(tfmt)], fontsize=self.ticksize, rotation=15)
                        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter(yfmt))
                    else:
                        ax.set_yticks([edges[0], edges[-1]])
                        ax.set_yticklabels([edge_labels[0].strftime(tfmt), edge_labels[-1].strftime(tfmt)], fontsize=self.ticksize, rotation=15)
                        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter(xfmt))
            self.unshare_axis_parameters(
                axes=axes,
                layout=layout,
                collapse_x=collapse_x,
                collapse_y=collapse_y)
            if show_cycle_separations:
                self.subview_heat_map_separations_by_solar_cycle(
                    axes=axes,
                    layout=layout,
                    dtmin=dtmin,
                    dtmax=dtmax,
                    xparameter=xparameter,
                    yparameter=yparameter,
                    sharex=sharex,
                    sharey=sharey,
                    separation_color=separation_color,
                    linestyle='-')
            fig.align_ylabels()
            if invalid_color in ('k', 'black') or ('dark' in str(invalid_color)):
                for ax in axes.ravel():
                    ax.grid(color='white', linestyle=':', alpha=0.3)
            else:
                for ax in axes.ravel():
                    self.apply_grid(ax)
            ## add legend
            if show_legend:
                fig.subplots_adjust(hspace=0.3, bottom=0.25)
                handles, labels = [], []
                for i, (facecolor, level) in enumerate(zip(facecolors, levels)):
                    label = '${}$'.format(int(level))
                    labels.append(label)
                    patch = mpatches.Patch(
                        color=facecolor,
                        label=label)
                    handles.append(patch)
                nlevels = len(handles)
                ncol = nlevels if 1 < nlevels < 9 else None
                self.subview_legend(
                    fig=fig,
                    ax=axes.ravel()[0],
                    handles=handles,
                    labels=labels,
                    title='Number of Events',
                    bottom=None,
                    textcolor='k',
                    facecolor='silver',
                    edgecolor='k',
                    titlecolor='k',
                    ncol=ncol)
            else:
                fig.subplots_adjust(hspace=0.3)
            ## add colorbar
            if show_colorbar:
                cax = fig.add_axes([0.925, 0.225, 0.025, 0.625]) # (x0, y0, dx, dy)
                orientation = 'vertical'
                self.subview_color_bar(
                    fig=fig,
                    ax=axes.ravel().tolist(),
                    handle=im_handle,
                    title='Number\nof Events',
                    levels=levels,
                    norm=norm,
                    extend='max',
                    orientation=orientation,
                    pad=0.2,
                    cax=cax,
                    shrink=None)
            ## update title
            fig.suptitle('Heat-Map of Event Frequency', fontsize=self.titlesize)
            ## save or show
            if save:
                savename = "RawAnalaysis_FreqHeatMap_CS-{}".format(color_spacing)
                if show_cycle_separations:
                    savename = '{}_SEP'.format(savename)
                if show_colorbar:
                    savename = '{}_CBAR'.format(savename)
                if show_legend:
                    savename = '{}_LEG'.format(savename)
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
                savename = '{}_{}'.format(savename, layout)
                savename = savename.replace(' ', '_')
            else:
                savename = None
            self.display_image(fig, savename=savename)


##
