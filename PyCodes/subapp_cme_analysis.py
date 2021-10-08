from data_processing import *
from raw_analysis_configuration import *
from regular_analysis_configuration import *

def perform_cme_analysis(cmes, raw=False, regular=False, savedir=None):

    if raw:

        ## load raw series
        for speed_type in ('second order initial speed', 'linear speed'):
            search_kwargs = {
                'parameters' : speed_type,
                'conditions' : 'greater than',
                'values' : 0}
            for solar_cycles in (23, 24):
                cmes.load_raw_series(
                    parameter_id=speed_type,
                    series_id=None,
                    solar_cycles=solar_cycles,
                    activity_type='full',
                    nan_policy='omit',
                    nan_replacement=-1,
                    squeeze_interval=True,
                    elapsed_unit='hour',
                    search_kwargs=search_kwargs)

        ## initialize regular analysis
        cme_raw_analysis = RawAnalaysis(
            series=cmes.raw_series,
            savedir=savedir)

        ## add inter-exceedamces
        inter_exceedance_kwargs = {
            'wbin' : 15,
            'lbin' : 0,
            'include_inverse_transform_sample' : True,
            'squeeze_trails' : True,
            'tol' : 2}
        cme_raw_analysis.add_inter_exceedances(
            extreme_parameter=None,
            extreme_condition='greater than',
            extreme_values=(700, 800, 1000),
            series_indices=None,
            **inter_exceedance_kwargs)

        ## view inter-exceedance times
        cme_raw_analysis.view_histogram_of_inter_exceedance_times(
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
        cme_raw_analysis.add_temporal_frequency(
            extreme_parameter=None,
            series_indices=None,
            **temporal_frequency_kwargs)

        ## add frequency heat-map
        mixed_frequency_kwargs = {
            'parameter_bin_kwargs' : {
                'wbin' : 50,
                'lbin' : 0,
                'rbin' : 3000},
            'temporal_bin_kwargs' : {
                'wbin' : 2,
                'time_step' : 'day',
                'time_scale' : 'relative'}}
        cme_raw_analysis.add_mixed_frequency(
            extreme_parameter=None,
            series_indices=None,
            **mixed_frequency_kwargs)

        ## view temporal frequency
        cme_raw_analysis.view_temporal_frequency(
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
        cme_raw_analysis.view_temporal_frequency(
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
        cme_raw_analysis.view_temporal_frequency(
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
        cme_raw_analysis.view_temporal_frequency(
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
            cme_raw_analysis.view_temporal_frequency(
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
                cme_raw_analysis.view_temporal_frequency_heatmap(
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

        ## add lognormal series
        first_search_kwargs = {
            'parameters' : 'acceleration',
            'conditions' : 'greater than',
            'values' : 0}
        second_search_kwargs = {
            'parameters' : 'acceleration',
            'conditions' : 'less than',
            'values' : 0}
        third_search_kwargs = {
            'parameters' : 'halo',
            'conditions' : 'equal',
            'values' : False}
        container_of_search_kwargs = [
            first_search_kwargs,
            second_search_kwargs,
            third_search_kwargs]
        lognormal_kwargs = {
            'kernel_density_estimation_kwargs' : {
                'bandwidths' : [0.01, 0.1, 'scott', 'silverman'],
                'kernels' : 'gaussian'},
            'histogram_kwargs' : {
                'wbin' : 70,
                'lbin' : 0,
                'threshold' : 5},
            'chi_square_kwargs' : {
                'initial_parameter_guess' : None,
                'error_statistic_id' : 'reduced chi square',
                'scale' : 'local',
                'method' : 'Nelder-Mead'},
            'g_test_kwargs' : {
                'initial_parameter_guess' : None,
                'error_statistic_id' : 'reduced g-test',
                'scale' : 'local',
                'method' : 'Nelder-Mead'},
            'maximum_likelihood_kwargs' : {
                'initial_parameter_guess' : None,
                'scale' : 'local',
                'method' : 'Nelder-Mead'},
            'chi_square_error_kwargs' : {
                'x' : np.arange(1, 11.1, 0.1),
                'y' : np.arange(0.25, 2.11, 0.05)},
            'g_test_error_kwargs' : {
                'x' : np.arange(1, 11.1, 0.1),
                'y' : np.arange(0.25, 2.11, 0.05)},
            'maximum_likelihood_error_kwargs' : {
                'x' : np.arange(1, 11.1, 0.1),
                'y' : np.arange(0.25, 2.11, 0.05)},
            'normal_kwargs' : {
                'kernel_density_estimation_kwargs' : {
                    'bandwidths' : [0.01, 0.1, 'scott', 'silverman'],
                    'kernels' : 'gaussian'},
                'histogram_kwargs' : {
                    'nbins' : 20,
                    'lbin' : 0,
                    'threshold' : 5},
                'chi_square_kwargs' : {
                    'initial_parameter_guess' : None,
                    'error_statistic_id' : 'reduced chi square',
                    'scale' : 'local',
                    'method' : 'Nelder-Mead'},
                'g_test_kwargs' : {
                    'initial_parameter_guess' : None,
                    'error_statistic_id' : 'reduced g-test',
                    'scale' : 'local',
                    'method' : 'Nelder-Mead'},
                'maximum_likelihood_kwargs' : {
                    'initial_parameter_guess' : None,
                    'scale' : 'local',
                    'method' : 'Nelder-Mead'},
                'chi_square_error_kwargs' : {
                    'x' : np.arange(1, 11.1, 0.1),
                    'y' : np.arange(0.25, 2.11, 0.05)},
                'g_test_error_kwargs' : {
                    'x' : np.arange(1, 11.1, 0.1),
                    'y' : np.arange(0.25, 2.11, 0.05)},
                'maximum_likelihood_error_kwargs' : {
                    'x' : np.arange(1, 11.1, 0.1),
                    'y' : np.arange(0.25, 2.11, 0.05)}},
            'container_of_search_kwargs' : container_of_search_kwargs}
        cme_raw_analysis.add_lognormal_distribution(
            extreme_parameter=None,
            series_indices=None,
            **lognormal_kwargs)

        ## view random variable
        cme_raw_analysis.view_random_variable(
            distribution_ids=None,
            show_chronological_distribution=True,
            show_empirical_accumulation=True,
            show_empirical_survival=True,
            facecolors=('darkorange', 'steelblue', 'green', 'purple'),
            fmt='%Y-%m',
            sharex=True,
            sharey=True,
            collapse_x=True,
            collapse_y=True,
            figsize=(12, 7),
            save=True,
            layout=None)
        cme_raw_analysis.view_exceedance_probability(
            distribution_ids=None,
            show_chi_square=True,
            show_g_test=True,
            show_maximum_likelihood=True,
            facecolor='b',
            sharex=True,
            sharey=True,
            collapse_x=True,
            collapse_y=True,
            figsize=(12, 7),
            save=True,
            layout=None)
        cme_raw_analysis.view_qq(
            distribution_ids=None,
            show_chi_square=True,
            show_g_test=True,
            show_maximum_likelihood=True,
            quantile_color='b',
            line_color='r',
            sharex=True,
            sharey=True,
            collapse_x=True,
            collapse_y=True,
            figsize=(12, 7),
            save=True,
            layout=None)

        ## view distribution
        for density_id in ('observed', 'probability'):
            cme_raw_analysis.view_distribution(
                density_id=density_id,
                show_kde=True,
                kde_style='fill',
                kde_colors=('red', 'green', 'blue', 'k'),
                sharex=True,
                sharey=True,
                collapse_x=True,
                collapse_y=True,
                figsize=(12, 7),
                save=True,
                layout=None)
            cme_raw_analysis.view_distribution(
                density_id=density_id,
                show_kde=True,
                kde_style='curve',
                kde_colors=('red', 'green', 'blue', 'k'),
                sharex=True,
                sharey=True,
                collapse_x=True,
                collapse_y=True,
                figsize=(12, 7),
                save=True,
                layout=None)
            cme_raw_analysis.view_distribution(
                density_id=density_id,
                show_rug=True,
                show_histogram=True,
                rug_color='darkorange',
                bar_color='gray',
                step_color='k',
                sharex=True,
                sharey=True,
                collapse_x=True,
                collapse_y=True,
                figsize=(12, 7),
                save=True,
                layout=None)
            cme_raw_analysis.view_distribution(
                density_id=density_id,
                show_rug=True,
                show_chi_square=True,
                show_g_test=True,
                show_maximum_likelihood=True,
                sharex=True,
                sharey=True,
                collapse_x=True,
                collapse_y=True,
                figsize=(12, 7),
                save=True,
                layout=None)
            cme_raw_analysis.view_distribution(
                density_id=density_id,
                show_rug=True,
                show_histogram=True,
                step_color='k',
                bar_color=None,
                show_g_test=True,
                sharex=True,
                sharey=True,
                collapse_x=True,
                collapse_y=True,
                figsize=(12, 7),
                save=True,
                layout=None)
            cme_raw_analysis.view_distribution(
                density_id=density_id,
                show_statistics=True,
                show_confidence_interval=True,
                show_g_test=True,
                sharex=True,
                sharey=True,
                collapse_x=True,
                collapse_y=True,
                figsize=(12, 7),
                save=True,
                layout=None)
            cme_raw_analysis.view_distribution(
                density_id=density_id,
                extreme_values=(700, 800, 1000),
                show_filled_tail=True,
                show_arrow_tail=True,
                show_g_test=True,
                sharex=True,
                sharey=True,
                collapse_x=True,
                collapse_y=True,
                figsize=(12,7),
                save=True,
                layout=None)

        ## view error-space
        for ndim in (2, 3):
            show_colorbar = True if ndim == 3 else False
            cme_raw_analysis.view_error_space(
                ndim=ndim,
                show_maximum_likelihood=True,
                extremum_color='k',
                cmap='Oranges',
                show_colorbar=show_colorbar,
                show_fills=True,
                show_lines=True,
                show_inline_labels=True,
                sharex=True,
                sharey=True,
                collapse_x=True,
                collapse_y=True,
                figsize=(12, 7),
                save=True,
                layout=None)
            cme_raw_analysis.view_error_space(
                ndim=ndim,
                show_g_test=True,
                extremum_color='k',
                cmap='Oranges',
                show_colorbar=show_colorbar,
                show_fills=True,
                show_lines=True,
                show_inline_labels=True,
                sharex=True,
                sharey=True,
                collapse_x=True,
                collapse_y=True,
                figsize=(12, 7),
                save=True,
                layout=None)
            cme_raw_analysis.view_error_space(
                ndim=ndim,
                show_chi_square=True,
                extremum_color='k',
                cmap='Oranges',
                show_colorbar=show_colorbar,
                show_fills=True,
                show_lines=True,
                show_inline_labels=True,
                sharex=True,
                sharey=True,
                collapse_x=True,
                collapse_y=True,
                figsize=(12, 7),
                save=True,
                layout=None)

        ## view distribution tail
        for log_base in (None, 5, 10):
            cme_raw_analysis.view_distribution_tail(
                extreme_values=(700, 800, 1000),
                extreme_condition='greater than',
                distribution_ids=None,
                density_id='observed',
                histogram_id='original',
                basex=log_base,
                basey=log_base,
                extreme_color='darkred',
                bar_color='gray',
                step_color='k',
                sharex=True,
                sharey=True,
                collapse_x=True,
                collapse_y=True,
                figsize=(12, 7),
                save=True,
                layout=None)

        ## view lognormal-normal connection
        cme_raw_analysis.view_normal_and_lognormal_relation(
            show_kde=True,
            show_histogram=True,
            show_g_test=True,
            show_statistics=True,
            kde_style='fill',
            kde_colors=('red', 'green', 'blue', 'k'),
            bar_color=None,
            step_color='k',
            gt_color='darkorange',
            figsize=(12, 7),
            save=True,
            layout=None)
        cme_raw_analysis.view_normal_and_lognormal_relation(
            show_g_test=True,
            show_confidence_interval=True,
            show_statistics=True,
            gt_color='darkorange',
            figsize=(12, 7),
            save=True,
            layout=None)

        ## view searched variants of distribution
        cme_raw_analysis.view_distribution_subseries(
            distribution_ids='lognormal distribution', # None, # 'lognormal distribution',
            show_kde=False,
            show_histogram=True,
            show_chi_square=False,
            show_g_test=True,
            show_maximum_likelihood=False,
            show_statistics=False,
            show_confidence_interval=False,
            density_id='observed',
            histogram_id='original',
            bar_color='gray',
            step_color='k',
            csq_color='darkorange',
            gt_color='purple',
            mle_color='steelblue',
            confidence_color='darkgreen',
            mu_color='purple',
            sigma_color='blue',
            median_color='darkgreen',
            mode_color='darkred',
            kde_colors='darkblue',
            kde_style='curve',
            sharex=True,
            sharey=True,
            collapse_x=True,
            collapse_y=True,
            figsize=(12, 7),
            save=True,
            layout=None)
        cme_raw_analysis.view_distribution_subseries(
            distribution_ids='lognormal distribution', # None,
            show_kde=True,
            show_histogram=False,
            show_chi_square=False,
            show_g_test=False,
            show_maximum_likelihood=False,
            show_statistics=False,
            show_confidence_interval=False,
            density_id='observed',
            histogram_id='original',
            bar_color='gray',
            step_color='k',
            csq_color='darkorange',
            gt_color='purple',
            mle_color='steelblue',
            confidence_color='darkgreen',
            mu_color='purple',
            sigma_color='blue',
            median_color='darkgreen',
            mode_color='darkred',
            kde_colors=('red', 'green', 'blue', 'k'),
            kde_style='curve',
            sharex=True,
            sharey=True,
            collapse_x=True,
            collapse_y=True,
            figsize=(12, 7),
            save=True,
            layout=None)
        cme_raw_analysis.view_distribution_subseries(
            distribution_ids='lognormal distribution', # None,
            show_kde=False,
            show_histogram=False,
            show_chi_square=False,
            show_g_test=True,
            show_maximum_likelihood=False,
            show_statistics=False, # True,
            show_confidence_interval=True,
            density_id='observed',
            histogram_id='original',
            bar_color='gray',
            step_color='k',
            csq_color='darkorange',
            gt_color='purple',
            mle_color='steelblue',
            confidence_color='darkgreen',
            mu_color='purple',
            sigma_color='blue',
            median_color='darkgreen',
            mode_color='darkred',
            kde_colors='darkblue',
            kde_style='curve',
            sharex=True,
            sharey=True,
            collapse_x=True,
            collapse_y=True,
            figsize=(12, 7),
            save=True,
            layout=None)

    if regular:

        ## load regular series
        for speed_type in ('second order initial speed', 'linear speed'):
            search_kwargs = {
                'parameters' : speed_type,
                'conditions' : 'greater than',
                'values' : 0}
            for solar_cycles in (23, 24):
                cmes.load_regular_series(
                    parameter_id=speed_type,
                    series_id=None,
                    group_by='maximum',
                    solar_cycles=solar_cycles,
                    activity_type='high-activity',
                    nan_policy='omit',
                    nan_replacement=-1,
                    squeeze_interval=False,
                    elapsed_unit='hour',
                    search_kwargs=search_kwargs,
                    integer_pad=-1)

        ## initialize regular analysis
        cme_regular_analysis = RegularAnalaysis(
            series=cmes.regular_series,
            savedir=savedir)

        # ## add lag-correlations
        # lag_correlation_kwargs = {
        #     'lag_ks' : np.arange(151, dtype=int),
        #     'nresamples' : 10}
        # cme_regular_analysis.add_naive_lag_correlations(
        #     extreme_parameter=None,
        #     extreme_condition='greater than',
        #     extreme_values=(700, 800, 1000),
        #     series_indices=None,
        #     **lag_correlation_kwargs)
        #
        # ## view lag-correlations
        # cme_regular_analysis.view_lag_correlations(
        #     extreme_values=None,
        #     show_original=True,
        #     sharex=True,
        #     sharey=True,
        #     collapse_x=True,
        #     collapse_y=True,
        #     figsize=(12, 7),
        #     save=True,
        #     layout=None)
        # cme_regular_analysis.view_lag_correlations(
        #     extreme_values=None,
        #     show_time_randomized=True,
        #     sharex=True,
        #     sharey=True,
        #     collapse_x=True,
        #     collapse_y=True,
        #     figsize=(12, 7),
        #     save=True,
        #     layout=None)
        # cme_regular_analysis.view_lag_correlations(
        #     extreme_values=None,
        #     show_difference=True,
        #     sharex=True,
        #     sharey=True,
        #     collapse_x=True,
        #     collapse_y=True,
        #     figsize=(12, 7),
        #     save=True,
        #     layout=None)
        # cme_regular_analysis.view_lag_correlations(
        #     extreme_values=None,
        #     show_original=True,
        #     show_time_randomized=True,
        #     sharex=True,
        #     sharey=True,
        #     collapse_x=True,
        #     collapse_y=True,
        #     figsize=(12, 7),
        #     save=True,
        #     layout=None)

        ## add lag-correlations
        lag_correlation_parameterization_kwargs = {
            'lag_ks' : np.arange(151, dtype=int),
            'nresamples' : 10}
        cme_regular_analysis.add_naive_lag_correlations_parameterization(
            extreme_parameter=None,
            extreme_condition='greater than',
            extreme_values=np.arange(500, 2301, 10).astype(int),
            series_indices=None,
            **lag_correlation_parameterization_kwargs)

        ## view lag-correlations parameterization
        for ndim in (2, 3):
            for zparameter in ('original lambda', 'difference lambda'):
                for color_spacing in ('linear', 'log'):
                    for show_lines in (True, False):
                        for show_colorbar in (True, False):
                            cme_regular_analysis.view_lag_correlations_parameterization(
                                ndim=ndim,
                                xparameter='time-lag k',
                                yparameter='extreme value',
                                zparameter=zparameter,
                                cmap='Oranges',
                                extremum_color='k',
                                color_spacing=color_spacing,
                                levels=None,
                                zfmt=None,
                                azim=-60,
                                elev=30,
                                rstride=1,
                                cstride=1,
                                show_lines=show_lines,
                                show_colorbar=show_colorbar,
                                sharex=True,
                                sharey=True,
                                collapse_x=True,
                                collapse_y=True,
                                figsize=(12, 7),
                                save=True,
                                layout='single' if ndim == 3 else None)

        # ## add inter-exceedamces
        # inter_exceedance_kwargs = {
        #     'wbin' : 15,
        #     'lbin' : 0,
        #     'include_inverse_transform_sample' : True,
        #     'squeeze_trails' : True,
        #     'tol' : 2}
        # cme_regular_analysis.add_inter_exceedances(
        #     extreme_parameter=None,
        #     extreme_condition='greater than',
        #     extreme_values=(700, 800, 1000),
        #     series_indices=None,
        #     **inter_exceedance_kwargs)
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
        # cme_regular_analysis.add_unbiased_estimators(
        #     # extreme_parameter=None,
        #     extreme_indices=np.arange(4, 13).astype(int),
        #     series_indices=None,
        #     **unbiased_estimators_kwargs)
        #
        # ## add temporal clustering
        # temporal_clustering_kwargs = {
        #     'apply_first_order_bias' : True,
        #     'apply_threshold_bias' : True,
        #     'baseline_theta' : 0.5,
        #     'include_all_clusters' : True,
        #     'include_lone_clusters' : True,
        #     'include_non_lone_clusters' : True,
        #     'intra_cluster_times_histogram_kwargs' : {
        #         'lbin' : 0,
        #         'wbin' : 1,
        #         'tol' : 2},
        #     'intra_cluster_durations_histogram_kwargs' : {
        #         'lbin' : 0,
        #         'nbins' : 15, # 10,
        #         'tol' : 2},
        #     'inter_cluster_durations_histogram_kwargs' : {
        #         'criteria' : 'fd',
        #         'squeeze_trails' : True,
        #         'tol' : 2}}
        # cme_regular_analysis.add_temporal_clustering(
        #     extreme_parameter=None,
        #     extreme_condition='greater than',
        #     extreme_values=(700, 800, 1000),
        #     series_indices=None,
        #     **temporal_clustering_kwargs)
        # cme_regular_analysis.add_temporal_clustering_parameterization(
        #     extreme_parameter=None,
        #     extreme_condition='greater than',
        #     extreme_values=np.arange(500, 2301, 10).astype(int),
        #     series_indices=None,
        #     **temporal_clustering_kwargs)
        #
        # ## view inter-exceedance times
        # cme_regular_analysis.view_histogram_of_inter_exceedance_times(
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
        # ## view unbiased estimators
        # cme_regular_analysis.view_histogram_of_unbiased_estimators_parameter(
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
        # cme_regular_analysis.view_max_spectrum(
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
        # cme_regular_analysis.view_max_spectrum(
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
        # cme_regular_analysis.view_max_spectrum(
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
        # cme_regular_analysis.view_point_estimators_of_extremal_index(
        #     sharex=True,
        #     sharey=True,
        #     collapse_x=True,
        #     collapse_y=True,
        #     figsize=(12, 7),
        #     save=True,
        #     layout=None)
        #
        # ## view temporal clustering
        # cme_regular_analysis.view_cluster_size_statistics(
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
        # cme_regular_analysis.view_histogram_of_intra_cluster_times(
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
        # cme_regular_analysis.view_histogram_of_intra_cluster_durations(
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
        # cme_regular_analysis.view_histogram_of_inter_cluster_durations(
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
        # cme_regular_analysis.view_alternating_clusters(
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
        # cme_regular_analysis.view_cluster_size_table(
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
        # cme_regular_analysis.view_analysis_table(
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
        #
        # ## view temporal clustering parameterization
        # cme_regular_analysis.view_cluster_parameterization_moment_estimators_and_time_thresholds(
        #     extreme_values=None,
        #     cluster_ids='non-lone clusters',
        #     show_first_order_bias=True,
        #     show_threshold_bias=True,
        #     show_baseline=True,
        #     threshold_color='steelblue',
        #     first_order_color='darkorange',
        #     baseline_color='green',
        #     sharex=True,
        #     sharey=True,
        #     collapse_x=True,
        #     collapse_y=True,
        #     figsize=(12, 7),
        #     save=True,
        #     layout=None)
        # ## debug <>
        # # cme_regular_analysis.view_cluster_parameterization_quantity(
        # #     extreme_values=None,
        # #     quantities=None,
        # #     cluster_ids='non-lone clusters',
        # #     show_first_order_bias=True,
        # #     show_threshold_bias=True,
        # #     show_baseline=True,
        # #     show_min_and_max=True,
        # #     central_statistic='mean',
        # #     error_statistic='standard deviation',
        # #     sharex=True,
        # #     sharey=True,
        # #     collapse_x=True,
        # #     collapse_y=True,
        # #     figsize=(12, 7),
        # #     save=True,
        # #     layout=None)
