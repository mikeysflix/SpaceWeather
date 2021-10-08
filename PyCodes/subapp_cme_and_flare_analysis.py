from data_processing import *
from raw_analysis_configuration import *
from regular_analysis_configuration import *

def perform_cme_and_flare_analysis(cmes, flares, solar_cycles, speed_type='second order initial speed', energy_type='high energy', savedir=None):

    for cycle_nums in solar_cycles:
        cmes.load_raw_series(
            parameter_id=speed_type,
            series_id=None,
            solar_cycles=cycle_nums,
            activity_type='full',
            nan_policy='omit',
            nan_replacement=-1,
            squeeze_interval=True,
            elapsed_unit='hour',
            search_kwargs={
                'parameters' : speed_type,
                'conditions' : 'greater than',
                'values' : 0})

        flares.load_raw_series(
            parameter_id=energy_type,
            datetime_prefix='peak',
            series_id=None,
            solar_cycles=cycle_nums,
            activity_type='full',
            nan_policy='omit',
            nan_replacement=-1,
            squeeze_interval=True,
            elapsed_unit='minute',
            search_kwargs={
                'parameters' : energy_type,
                'conditions' : 'greater than',
                'values' : 0})

    raw_analysis = RawAnalaysis(
        series=(cmes.raw_series + flares.raw_series),
        savedir=savedir)

    ## add temporal frequency
    temporal_frequency_kwargs = {
        'wbin' : 1,
        'time_step' : 'day',
        'time_scale' : 'relative'}
    raw_analysis.add_temporal_frequency(
        extreme_parameter=speed_type,
        series_indices=0,
        **temporal_frequency_kwargs)
    raw_analysis.add_temporal_frequency(
        extreme_parameter=energy_type,
        series_indices=1,
        **temporal_frequency_kwargs)

    ## view temporal frequency
    raw_analysis.view_temporal_frequency(
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
    raw_analysis.view_temporal_frequency(
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
    raw_analysis.view_temporal_frequency(
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
    raw_analysis.view_temporal_frequency(
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
        raw_analysis.view_temporal_frequency(
            period=period,
            background_color='k',
            sharex=True,
            sharey=False,
            collapse_x=True,
            collapse_y=True,
            figsize=(12, 7),
            save=True,
            layout=None)

    ## add frequency heat-map
    cme_mixed_frequency_kwargs = {
        'parameter_bin_kwargs' : {
            'wbin' : 50,
            'lbin' : 0,
            'rbin' : 3000},
        'temporal_bin_kwargs' : {
            'wbin' : 2,
            'time_step' : 'day',
            'time_scale' : 'relative'}}
    raw_analysis.add_mixed_frequency(
        extreme_parameter=speed_type,
        series_indices=0,
        **cme_mixed_frequency_kwargs)
    flare_mixed_frequency_kwargs = {
        'parameter_bin_kwargs' : {
            'wbin' : 10,
            'lbin' : 0,
            'rbin' : 1000},
        'temporal_bin_kwargs' : {
            'wbin' : 2,
            'time_step' : 'day',
            'time_scale' : 'relative'}}
    raw_analysis.add_mixed_frequency(
        extreme_parameter=energy_type,
        series_indices=1,
        **flare_mixed_frequency_kwargs)

    ## view temporal frequency heatmap
    for color_spacing in ('linear', 'logarithmic'):
        for show_cycle_separations in (True, False):
            raw_analysis.view_temporal_frequency_heatmap(
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


##
