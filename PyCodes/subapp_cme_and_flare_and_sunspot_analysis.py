from data_processing import *
from raw_analysis_configuration import *
from regular_analysis_configuration import *

def perform_cme_and_flare_and_sunspot_analysis(cmes, flares, sunspots, solar_cycles, speed_type='second order initial speed', energy_type='high energy', savedir=None):

    ## initialize series
    for cycle_nums in solar_cycles:
        sunspots.load_raw_series(
            series_id=None,
            solar_cycles=cycle_nums,
            activity_type='full',
            squeeze_interval=True,
            nan_policy='omit')
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
                'values' : 800})
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
        series=(sunspots.raw_series + cmes.raw_series + flares.raw_series),
        savedir=savedir)

    ## add temporal frequency
    temporal_frequency_kwargs = {
        'wbin' : 1,
        'time_step' : 'day',
        'time_scale' : 'relative'}
    raw_analysis.add_temporal_frequency(
        extreme_parameter=speed_type,
        series_indices=(1, 2),
        **temporal_frequency_kwargs)
    raw_analysis.add_temporal_frequency(
        extreme_parameter=energy_type,
        series_indices=3,
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

#
