from data_processing import *
from raw_analysis_configuration import *
from regular_analysis_configuration import *

def perform_flare_analysis(flares, raw=False, regular=False, savedir=None):

    if raw:

        ## load raw series
        for energy_type in ('high energy',): # 'low energy'):
            search_kwargs = {
                'parameters' : energy_type,
                'conditions' : 'greater than',
                'values' : 0}
            for solar_cycles in (23, 24):
                flares.load_raw_series(
                    parameter_id=energy_type,
                    datetime_prefix='peak',
                    series_id=None,
                    solar_cycles=solar_cycles,
                    activity_type='full',
                    nan_policy='omit',
                    nan_replacement=-1,
                    squeeze_interval=True,
                    elapsed_unit='minute',
                    search_kwargs=search_kwargs)

        ## initialize regular analysis
        flare_raw_analysis = RawAnalaysis(
            series=flares.raw_series,
            savedir=savedir)

        ## add inter-exceedamces
        inter_exceedance_kwargs = {
            'wbin' : 15,
            'lbin' : 0,
            'include_inverse_transform_sample' : True,
            'squeeze_trails' : True,
            'tol' : 2}
        flare_raw_analysis.add_inter_exceedances(
            extreme_parameter=None,
            extreme_condition='greater than',
            extreme_values=(10,),
            series_indices=None,
            **inter_exceedance_kwargs)

        ## view inter-exceedance times
        flare_raw_analysis.view_histogram_of_inter_exceedance_times(
            extreme_values=None,
            show_inverse_transform=True,
            facecolors=('darkorange', 'green', 'purple', 'steelblue'),
            sample_color='k',
            sharex=True,
            sharey=True,
            collapse_x=True,
            collapse_y=True,
            figsize=(12, 7),
            save=True,
            layout=None)

        ## add temporal frequency
        temporal_frequency_kwargs = {
            'wbin' : 1,
            'time_step' : 'day',
            'time_scale' : 'relative'}
        flare_raw_analysis.add_temporal_frequency(
            extreme_parameter=None,
            series_indices=None,
            **temporal_frequency_kwargs)

        ## add frequency heat-map
        mixed_frequency_kwargs = {
            'parameter_bin_kwargs' : {
                'wbin' : 10,
                'lbin' : 0,
                'rbin' : 1000},
            'temporal_bin_kwargs' : {
                'wbin' : 2,
                'time_step' : 'day',
                'time_scale' : 'relative'}}
        flare_raw_analysis.add_mixed_frequency(
            extreme_parameter=None,
            series_indices=None,
            **mixed_frequency_kwargs)

        ## view temporal frequency
        flare_raw_analysis.view_temporal_frequency(
            show_cycle_separations=True,
            show_frequency_by_cycle=True,
            period=None,
            cmaps=('Oranges', 'Blues', 'Greens', 'Greys'),
            facecolors=('darkorange', 'darkgreen', 'purple', 'steelblue'),
            background_color=None,
            separation_color='r',
            arrow_color='k',
            tfmt='%Y-%m',
            sharex=True,
            sharey=False,
            collapse_x=True,
            collapse_y=True,
            figsize=(12, 7),
            save=True,
            layout=None)
        flare_raw_analysis.view_temporal_frequency(
            show_cycle_separations=True,
            show_frequency_by_cycle=False,
            period=None,
            cmaps=('Oranges', 'Blues', 'Greens', 'Greys'),
            facecolors=('darkorange', 'darkgreen', 'purple', 'steelblue'),
            background_color=None,
            separation_color='r',
            arrow_color='k',
            tfmt='%Y-%m',
            sharex=True,
            sharey=False,
            collapse_x=True,
            collapse_y=True,
            figsize=(12, 7),
            save=True,
            layout=None)
        flare_raw_analysis.view_temporal_frequency(
            show_cycle_separations=False,
            show_frequency_by_cycle=True,
            period=None,
            facecolors=('darkorange', 'darkgreen', 'purple', 'steelblue'),
            background_color=None,
            separation_color='r',
            arrow_color='k',
            tfmt='%Y-%m',
            sharex=True,
            sharey=False,
            collapse_x=True,
            collapse_y=True,
            figsize=(12, 7),
            save=True,
            layout=None)
        flare_raw_analysis.view_temporal_frequency(
            show_cycle_separations=False,
            show_frequency_by_cycle=False,
            period=None,
            facecolors=('darkorange', 'darkgreen', 'purple', 'steelblue'),
            background_color=None,
            separation_color='r',
            arrow_color='k',
            tfmt='%Y-%m',
            sharex=True,
            sharey=False,
            collapse_x=True,
            collapse_y=True,
            figsize=(12, 7),
            save=True,
            layout=None)
        for period in ('solar cycle', 'year'):
            flare_raw_analysis.view_temporal_frequency(
                period=period,
                background_color='k',
                sharex=True,
                sharey=False,
                collapse_x=True,
                collapse_y=True,
                figsize=(12, 7),
                save=True,
                layout=None)

        ## view temporal frequency heatmap
        for color_spacing in ('linear', 'logarithmic'):
            for show_cycle_separations in (True, False):
                flare_raw_analysis.view_temporal_frequency_heatmap(
                    xparameter='temporal value',
                    yparameter='extreme value',
                    color_spacing=color_spacing,
                    show_colorbar=True,
                    show_legend=True,
                    show_cycle_separations=show_cycle_separations,
                    cmap='jet',
                    invalid_color='k',
                    sharex=True,
                    # sharey=True,
                    collapse_x=True,
                    collapse_y=True,
                    figsize=(12, 7),
                    save=True,
                    layout=None)

    if regular:

        ## load regular series
        for energy_type in ('high energy',): # 'low energy'):
            search_kwargs = {
                'parameters' : energy_type,
                'conditions' : 'greater than',
                'values' : 0}
            for solar_cycles in (23, 24):
                flares.load_regular_series(
                    parameter_id='high energy',
                    datetime_prefix='peak',
                    series_id=None,
                    group_by='maximum',
                    solar_cycles=solar_cycles,
                    activity_type='high-activity',
                    nan_policy='omit',
                    nan_replacement=-1,
                    squeeze_interval=False,
                    elapsed_unit='minute',
                    search_kwargs=search_kwargs)

        ## initialize regular analysis
        flare_regular_analysis = RegularAnalaysis(
            series=flares.regular_series,
            savedir=savedir)

        # ## add inter-exceedamces
        # inter_exceedance_kwargs = {
        #     'wbin' : 15,
        #     'lbin' : 0,
        #     'include_inverse_transform_sample' : True,
        #     'squeeze_trails' : True,
        #     'tol' : 2}
        # flare_regular_analysis.add_inter_exceedances(
        #     extreme_parameter=None,
        #     extreme_condition='greater than',
        #     extreme_values=(10, 15),
        #     series_indices=None,
        #     **inter_exceedance_kwargs)
        #
        # ## view inter-exceedance times
        # flare_regular_analysis.view_histogram_of_inter_exceedance_times(
        #     extreme_values=None,
        #     show_inverse_transform=True,
        #     facecolors=('darkorange', 'green', 'purple', 'steelblue'),
        #     sample_color='k',
        #     sharex=True,
        #     sharey=True,
        #     collapse_x=True,
        #     collapse_y=True,
        #     figsize=(12, 7),
        #     save=True,
        #     layout=None)
        #
        # ## add unbiased estimators
        # unbiased_estimators_kwargs = {
        #     'prms_guess' : None,
        #     'nresamples' : 1000,
        #     'nshuffles' : 3,
        #     'with_replacement' : False,
        #     'value_to_replace' : None,
        #     'ddof' : 0,
        #     'scale' : 'local',
        #     'method' : 'Nelder-Mead',
        #     'alpha_histogram_kwargs' : {
        #         'nbins' : 20},
        #     'intercept_histogram_kwargs' : {
        #         'nbins' : 20},
        #     'theta_histogram_kwargs' : {
        #         'edges' : np.arange(0, 1.01, 0.025)}}
        # flare_regular_analysis.add_unbiased_estimators(
        #     # extreme_parameter=None,
        #     extreme_indices=np.arange(4, 13).astype(int),
        #     series_indices=None,
        #     **unbiased_estimators_kwargs)
        #
        # for series in flare_regular_analysis.series:
        #     ub = series['unbiased estimators']
        #     print("\n\n{}\n\n".format(series['identifiers']['series id']))
        #     print(ub)
        #     print(ub.extreme_value_estimates)
        #     print("\n\n" + "="*10 + "\n")
        #
        # ## view unbiased estimators
        # flare_regular_analysis.view_histogram_of_unbiased_estimators_parameter(
        #     parameters=('alpha', 'intercept', 'theta'),
        #     show_mean=True,
        #     show_kurtosis=True,
        #     show_skew=True,
        #     sharex=True,
        #     sharey=True,
        #     collapse_x=True,
        #     collapse_y=True,
        #     figsize=(12, 7),
        #     save=True,
        #     layout=None)
        # flare_regular_analysis.view_max_spectrum(
        #     show_points=True,
        #     show_fit=True,
        #     show_log_scale=True,
        #     sharex=True,
        #     sharey=True,
        #     collapse_x=True,
        #     collapse_y=True,
        #     figsize=(12, 7),
        #     save=True,
        #     layout=None)
        # flare_regular_analysis.view_max_spectrum(
        #     show_points=True,
        #     show_standard_deviation=True,
        #     show_standard_error=True,
        #     show_linear_scale=True,
        #     sharex=True,
        #     sharey=True,
        #     collapse_x=True,
        #     collapse_y=True,
        #     figsize=(12, 7),
        #     save=True,
        #     layout=None)
        # flare_regular_analysis.view_max_spectrum(
        #     show_points=True,
        #     show_fit=True,
        #     show_standard_deviation=True,
        #     show_standard_error=True,
        #     show_linear_scale=True,
        #     show_log_scale=True,
        #     sharex=True,
        #     sharey=True,
        #     collapse_x=True,
        #     collapse_y=True,
        #     figsize=(12, 7),
        #     save=True,
        #     layout=None)
        # flare_regular_analysis.view_point_estimators_of_extremal_index(
        #     sharex=True,
        #     sharey=True,
        #     collapse_x=True,
        #     collapse_y=True,
        #     figsize=(12, 7),
        #     save=True,
        #     layout=None)

        ## add temporal clustering
        temporal_clustering_kwargs = {
            'apply_first_order_bias' : True,
            'apply_threshold_bias' : True,
            'baseline_theta' : 0.5,
            'include_all_clusters' : True,
            'include_lone_clusters' : True,
            'include_non_lone_clusters' : True,
            'intra_cluster_times_histogram_kwargs' : {
                'lbin' : 0,
                'wbin' : 1,
                'tol' : 2},
            'intra_cluster_durations_histogram_kwargs' : {
                'lbin' : 0,
                'nbins' : 15, # 10,
                'tol' : 2},
            'inter_cluster_durations_histogram_kwargs' : {
                'criteria' : 'fd',
                'squeeze_trails' : True,
                'tol' : 2}}
        flare_regular_analysis.add_temporal_clustering(
            extreme_parameter=None,
            extreme_condition='greater than',
            extreme_values=(10, 15),
            series_indices=None,
            **temporal_clustering_kwargs)
        flare_regular_analysis.add_temporal_clustering_parameterization(
            extreme_parameter=None,
            extreme_condition='greater than',
            extreme_values=np.arange(1, 501, 1).astype(int),
            series_indices=None,
            **temporal_clustering_kwargs)

        # ## view temporal clustering
        # flare_regular_analysis.view_cluster_size_statistics(
        #     extreme_values=None,
        #     cluster_ids='non-lone clusters', # ('all clusters', 'lone clusters',
        #     show_first_order_bias=True,
        #     show_threshold_bias=True,
        #     show_baseline=True,
        #     sharex=True,
        #     sharey=True,
        #     collapse_x=True,
        #     collapse_y=True,
        #     figsize=(12, 7),
        #     save=True,
        #     layout=None)
        # flare_regular_analysis.view_histogram_of_intra_cluster_times(
        #     extreme_values=None,
        #     cluster_ids='non-lone clusters', # ('all clusters', 'lone clusters',
        #     show_first_order_bias=True,
        #     show_threshold_bias=True,
        #     show_baseline=True,
        #     sharex=True,
        #     sharey=True,
        #     collapse_x=True,
        #     collapse_y=True,
        #     figsize=(12, 7),
        #     save=True,
        #     layout=None)
        # flare_regular_analysis.view_histogram_of_intra_cluster_durations(
        #     extreme_values=None,
        #     cluster_ids='non-lone clusters', # ('all clusters', 'lone clusters',
        #     show_first_order_bias=True,
        #     show_threshold_bias=True,
        #     show_baseline=True,
        #     sharex=True,
        #     sharey=True,
        #     collapse_x=True,
        #     collapse_y=True,
        #     figsize=(12, 7),
        #     save=True,
        #     layout=None)
        # flare_regular_analysis.view_histogram_of_inter_cluster_durations(
        #     extreme_values=None,
        #     cluster_ids=('all clusters', 'lone clusters', 'non-lone clusters'),
        #     show_first_order_bias=True,
        #     show_threshold_bias=True,
        #     show_baseline=True,
        #     sharex=True,
        #     sharey=True,
        #     collapse_x=True,
        #     collapse_y=True,
        #     figsize=(12, 7),
        #     save=True,
        #     layout=None)
        # flare_regular_analysis.view_alternating_clusters(
        #     parameter='cluster size',
        #     statistic='mean',
        #     extreme_values=None,
        #     cluster_ids='non-lone clusters',
        #     show_first_order_bias=False,
        #     show_threshold_bias=True,
        #     show_baseline=False,
        #     facecolors=('darkorange', 'darkgreen'),
        #     tfmt='%Y-%m',
        #     sharex=True,
        #     sharey=True,
        #     collapse_x=True,
        #     collapse_y=True,
        #     figsize=(12, 7),
        #     save=True,
        #     layout=None)
        #
        # ## view tables
        # flare_regular_analysis.view_cluster_size_table(
        #     extreme_values=None,
        #     cluster_ids='non-lone clusters',
        #     show_first_order_bias=True,
        #     show_threshold_bias=True,
        #     show_baseline=True,
        #     column_color='orange',
        #     facecolors=('peachpuff', 'bisque'),
        #     ddof=0,
        #     figsize=(12, 7),
        #     save=True,
        #     layout='single')
        # flare_regular_analysis.view_analysis_table(
        #     extreme_values=None,
        #     cluster_ids='non-lone clusters',
        #     show_first_order_bias=False,
        #     show_threshold_bias=True,
        #     show_baseline=True,
        #     row_color='orange',
        #     column_color='orange',
        #     facecolors=('peachpuff', 'bisque'),
        #     ddof=0,
        #     figsize=(12, 7),
        #     save=True,
        #     layout=None)

        ## view temporal clustering parameterization
        flare_regular_analysis.view_cluster_parameterization_moment_estimators_and_time_thresholds(
            extreme_values=None,
            cluster_ids='non-lone clusters',
            show_first_order_bias=True,
            show_threshold_bias=True,
            show_baseline=True,
            threshold_color='steelblue',
            first_order_color='darkorange',
            baseline_color='green',
            sharex=True,
            sharey=True,
            collapse_x=True,
            collapse_y=True,
            figsize=(12, 7),
            save=True,
            layout=None)







##
