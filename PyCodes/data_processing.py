import os
import re
import requests
from collections import OrderedDict
from astropy.io import fits
from astropy import wcs
import imageio
from pygifsicle import optimize as io_optimize
import moviepy.video.io.ImageSequenceClip
from histogram_configuration import *
from visual_configuration import *

class EventSeries(TemporalConfiguration):

    def __init__(self, event_type, event_source, url=None, date_fmt='%Y/%m/%d', time_fmt='%H:%M:%S', cycle_bias='left'):
        super().__init__()
        self.event_type = event_type
        self.event_source = event_source
        self.url = url
        self.date_fmt = date_fmt
        self.time_fmt = time_fmt
        self.datetime_fmt = '{} {}'.format(date_fmt, time_fmt)
        self.cycle_bias = cycle_bias
        self._access_stamp = None
        self._observed_data = None
        self._raw_series = []
        self._irregular_series = []
        self._regular_series = []
        self._parameter_mapping = dict()
        self._unit_mapping = dict()
        self._generalized_parameter_mapping = dict()

    @property
    def access_stamp(self):
        return self._access_stamp

    @property
    def observed_data(self):
        return self._observed_data

    @property
    def raw_series(self):
        return self._raw_series

    @property
    def irregular_series(self):
        return self._irregular_series

    @property
    def regular_series(self):
        return self._regular_series

    @property
    def parameter_mapping(self):
        return self._parameter_mapping

    @property
    def unit_mapping(self):
        return self._unit_mapping

    @property
    def generalized_parameter_mapping(self):
        return self._generalized_parameter_mapping

    @staticmethod
    def deal_with_nans(data, parameter_id, nan_policy='propagate', nan_replacement=None):
        ## skip checking for NaN
        if parameter_id is None:
            return deepcopy(data)
        else:
            arr = np.copy(data[parameter_id])
            result = dict()
            ## discard values for all parameters at indices where parameter values is NaN
            if nan_policy == 'omit': # discard':
                condition = np.isnan(arr)
                indices = np.where(np.invert(condition))[0]
                result.update({key : np.copy(value[indices]) for key, value in data.items()})
            ## replace parameter values at indices where parameter values is NaN
            elif nan_policy == 'replace':
                condition = np.isnan(arr)
                indices = np.where(condition)[0]
                arr[indices] = nan_replacement
                result.update({key : np.copy(value) for key, value in data.items() if key != parameter_id})
                result[parameter_id] = arr
            ## propagate NaN
            elif nan_policy == 'propagate':
                result = deepcopy(data)
            else:
                raise ValueError("invalid nan_policy: {}".format(nan_policy))
            return result

    @staticmethod
    def squeeze_interval(data):
        is_event = np.where(data['is event'])[0]
        event_indices = np.arange(np.min(is_event), np.max(is_event)+1).astype(int)
        data = {key : np.copy(value[event_indices])
            for key, value in data.items()}
        data['elapsed'] -= data['elapsed'][0]
        return data

    # @staticmethod
    # def collapse_keys(data, new_key='', copy_key='', old_keys=None):
    #     if old_keys is None:
    #         return data
    #     else:
    #         result = dict()
    #         for key, value in data.items():
    #             if key in old_keys:
    #                 if key == copy_key:
    #                     result[new_key] = value
    #             else:
    #                 result[key] = value
    #         return result

    def get_current_datetime_string(self):
        ## date of access date/time -stamp
        return datetime.datetime.now().strftime(self.datetime_fmt)

    def get_last_modified_datetime_string(self, filepath):
        ## datetime when file was most recently modified
        t = os.path.getmtime(filepath)
        dt = datetime.datetime.fromtimestamp(t)
        return datetime.datetime.strftime(dt, self.datetime_fmt)

    def get_raw_series_identifiers(self, data, parameter_id, series_id, elapsed_unit, search_kwargs):
        identifiers = dict()
        identifiers['parameter id'] = parameter_id
        try:
            parameter_label = self.parameter_mapping[parameter_id][:]
        except:
            parameter_label = parameter_id.title()
        identifiers['parameter label'] = parameter_label
        identifiers['search kwargs'] = search_kwargs
        if search_kwargs is None:
            identifiers['search label'] = None
        else:
            identifiers['search label'] = self.get_search_label(**search_kwargs)
        solar_cycle_label = self.get_solar_cycles_label(data)
        identifiers['solar cycle label'] = solar_cycle_label
        if series_id is None:
            series_id = '{} {}s ({})'.format(
                solar_cycle_label,
                self.event_type,
                parameter_label)
        elif not isinstance(series_id, str):
            raise ValueError("invalid type(series_id): {}".format(type(series_id)))
        identifiers['series id'] = series_id
        identifiers['elapsed unit'] = elapsed_unit
        identifiers['event type'] = self.event_type
        identifiers['source'] = self.event_source
        return identifiers

    def get_irregular_series_identifiers(self, data, parameter_id, series_id, elapsed_unit, group_by, search_kwargs):
        identifiers = self.get_raw_series_identifiers(
            data=data,
            parameter_id=parameter_id,
            series_id=series_id,
            elapsed_unit=elapsed_unit,
            search_kwargs=search_kwargs)
        identifiers['group by'] = group_by
        return identifiers

    def get_regular_series_identifiers(self, data, parameter_id, series_id, elapsed_unit, group_by, search_kwargs, string_pad, integer_pad, float_pad):
        identifiers = self.get_irregular_series_identifiers(
            data=data,
            parameter_id=parameter_id,
            series_id=series_id,
            elapsed_unit=elapsed_unit,
            group_by=group_by,
            search_kwargs=search_kwargs)
        identifiers['string pad'] = string_pad
        identifiers['integer pad'] = integer_pad
        identifiers['float pad'] = float_pad
        return identifiers

class CoronalMassEjections(EventSeries):

    def __init__(self, cycle_bias='left'):
        super().__init__(
            event_type='CME',
            event_source='SOHO LASCO',
            url='https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL/text_ver/univ_all.txt',
            date_fmt='%Y/%m/%d',
            time_fmt='%H:%M:%S',
            cycle_bias=cycle_bias)
        self._parameter_mapping = {
            'speed' : r'$V_{CME}$',
            'linear speed' : r'$V_{linear}$',
            'second order initial speed' : r'$V_{20 R_{\odot}, i}$',
            'second order final speed' : r'$V_{20 R_{\odot}, f}$',
            'second order 20R speed' : r'$V_{20 R_{\odot}}$',
            'acceleration' : r'$acc_{CME}$',
            'mass' : 'Mass',
            'kinetic energy' : 'Kinetic Energy',
            'angular width' : 'Angular Width',
            'mean position angle' : 'Mean Position Angle',
            'central position angle' : 'Central Position Angle'}
        self._unit_mapping = {
            'speed' : r'$\frac{km}{s}$',
            'linear speed' : r'$\frac{km}{s}$',
            'second order initial speed' : r'$\frac{km}{s}$',
            'second order final speed' : r'$\frac{km}{s}$',
            'second order 20R speed' : r'$\frac{km}{s}$',
            'acceleration' : r'$\frac{%s}{%s^2}$' % ('$km$', '$s$'),
            'mass' : '$g$',
            'kinetic energy' : '$J$',
            'angular width' : r'$\degree$',
            'mean position angle' : r'$\degree$',
            'central position angle' : r'$\degree$'}
        self._generalized_parameter_mapping = {
            'linear speed' : 'speed',
            'second order initial speed' : 'speed',
            'second order final speed' : 'speed',
            'second order 20R speed' : 'speed'}
        self.speed_types = OrderedDict([
            ('linear speed', r'$V_{linear}$'),
            ('second order initial speed', r'$V_{20 R_{\odot}, i}$'),
            ('second order final speed', r'$V_{20 R_{\odot}, f}$'),
            ('second order 20R speed', r'$V_{20 R_{\odot}}$')])

    def load_observed_data(self, filepath=None):
        ## explicit keys
        keys = ['date', 'time', 'central position angle', 'angular width']
        keys += list(self.speed_types.keys())
        keys += ['acceleration', 'mass', 'kinetic energy', 'mean position angle']
        kws = {
            'usecols' : np.arange(len(keys)).astype(int),
            'skiprows' : 4,
            'dtype' : str}
        ## load observed parameters
        if filepath:
            ## load file from directory
            if isinstance(filepath, str):
                self._access_stamp = self.get_last_modified_datetime_string(filepath)
                data = np.loadtxt(
                    filepath,
                    **kws)
            else:
                raise ValueError("invalid type(filepath): {}".format(filepath))
        else:
            ## load file from url
            self._access_stamp = self.get_current_datetime_string()
            with requests.Session() as s:
                download = s.get(self.url)
                data = np.loadtxt(
                    download.iter_lines(),
                    **kws)
        ## update temporal parameters
        result = self.extract_datetime_components(
            dates=np.copy(data[:, 0]),
            times=np.copy(data[:, 1]),
            date_fmt='%Y/%m/%d',
            time_fmt='%H:%M:%S',
            include_components=True)
        result['solar cycle'], _ = self.group_cycles_by_datetime(
            dts=result['datetime'],
            solar_cycles=None,
            activity_type='full',
            bias=self.cycle_bias)
        ## clean dataset (filter from non-numeric types)
        for col, key in enumerate(keys):
            if key not in ('date', 'time'):
                v0 = data[:, col]
                if key in list(self.speed_types.keys()):
                    v1 = np.core.defchararray.replace(v0, '----', 'nan').astype(float)
                    result[key] = v1
                elif key in ('mass', 'kinetic energy'):
                    v1 = np.core.defchararray.replace(v0, '*'*8, 'nan')
                    v2 = np.core.defchararray.replace(v1, '*', '')
                    v3 = np.core.defchararray.replace(v2, '-------', 'nan').astype(float)
                    result[key] = v3
                elif key == 'acceleration':
                    v1 = np.core.defchararray.replace(v0, '------', 'nan').astype(str)
                    v2 = np.core.defchararray.replace(v1, '*', '').astype(float)
                    result[key] = v2
                elif key in ('angular width', 'mean position angle'):
                    result[key] = np.array(v0).astype(int)
                elif key == 'central position angle':
                    v1 = np.core.defchararray.replace(v0, 'Halo', 'nan').astype(float)
                    result[key] = v1
                else:
                    raise ValueError("invalid key: {}".format(key))
        ## update remaining parameters
        result['halo'] = np.isnan(result['central position angle'])
        result['is event'] = np.ones(result['halo'].size, dtype=bool)
        result['source'] = np.repeat(self.event_source, result['halo'].size)
        result['event'] = np.repeat(self.event_type, result['halo'].size)
        self._observed_data = result

    def load_raw_series(self, parameter_id, series_id=None, solar_cycles=None, activity_type='full', nan_policy='propagate', nan_replacement=None, squeeze_interval=False, elapsed_unit='hour', search_kwargs=None):
        ## deal with nans
        data = self.deal_with_nans(
            data=deepcopy(self.observed_data),
            parameter_id=parameter_id,
            nan_policy=nan_policy,
            nan_replacement=nan_replacement)
        ## group solar cycles
        cycle_nums, loc = self.group_cycles_by_datetime(
            dts=np.copy(data['datetime']),
            solar_cycles=solar_cycles,
            activity_type=activity_type,
            bias=self.cycle_bias)
        data = {key : np.copy(value[loc]) for key, value in data.items()}
        data['solar cycle'] = cycle_nums
        ## update data by search criteria
        if search_kwargs is not None:
            event_searcher = EventSearcher(data)
            data, _ = event_searcher.search_events(**search_kwargs)
        ## get elapsed units of time by step-size
        data['elapsed'] = self.get_elapsed_time(
            dts=data['datetime'],
            time_step=elapsed_unit,
            step_size=1,
            time_scale='relative')
        if squeeze_interval:
            data = self.squeeze_interval(data)
        ## load series
        identifiers = self.get_raw_series_identifiers(
            data=data,
            parameter_id=parameter_id,
            series_id=series_id,
            elapsed_unit=elapsed_unit,
            search_kwargs=search_kwargs)
        result = {
            'data' : data,
            'identifiers' : identifiers,
            'parameter mapping' : self.parameter_mapping,
            'unit mapping' : self.unit_mapping,
            'generalized parameter mapping' : self.generalized_parameter_mapping}
        self._raw_series.append(result)

    def load_irregular_series(self, parameter_id, series_id=None, group_by='maximum', solar_cycles=None, activity_type='full', nan_policy='propagate', nan_replacement=None, squeeze_interval=False, elapsed_unit='hour', search_kwargs=None):
        ## deal with nans
        data = self.deal_with_nans(
            data=deepcopy(self.observed_data),
            parameter_id=parameter_id,
            nan_policy=nan_policy,
            nan_replacement=nan_replacement)
        ## group solar cycles
        cycle_nums, loc = self.group_cycles_by_datetime(
            dts=np.copy(data['datetime']),
            solar_cycles=solar_cycles,
            activity_type=activity_type,
            bias=self.cycle_bias)
        data = {key : np.copy(value[loc]) for key, value in data.items()}
        data['solar cycle'] = cycle_nums
        ## update data by search criteria
        if search_kwargs is not None:
            event_searcher = EventSearcher(data)
            data, _ = event_searcher.search_events(**search_kwargs)
        ## get elapsed units of time by step-size
        data['elapsed'] = self.get_elapsed_time(
            dts=data['datetime'],
            time_step=elapsed_unit,
            step_size=1,
            time_scale='absolute')
        if squeeze_interval:
            data = self.squeeze_interval(data)
        ## group extrema of parameter by elapsed
        elapsed_indices = self.get_windowed_indices_of_extrema(
            data=data,
            extremal_parameter=parameter_id,
            group_parameter='elapsed',
            group_by=group_by)
        data = {key : np.copy(value[elapsed_indices])
            for key, value in data.items()}
        ## load series
        identifiers = self.get_irregular_series_identifiers(
            data=data,
            parameter_id=parameter_id,
            series_id=series_id,
            elapsed_unit=elapsed_unit,
            group_by=group_by,
            search_kwargs=search_kwargs)
        result = {
            'data' : data,
            'identifiers' : identifiers,
            'parameter mapping' : self.parameter_mapping,
            'unit mapping' : self.unit_mapping,
            'generalized parameter mapping' : self.generalized_parameter_mapping}
        self._irregular_series.append(result)

    def load_regular_series(self, parameter_id, series_id=None, group_by='maximum', solar_cycles=None, activity_type='full', nan_policy='propagate', nan_replacement=None, squeeze_interval=False, elapsed_unit='hour', search_kwargs=None, integer_pad=-1, float_pad=0., string_pad='NaN'):
        ## deal with nans
        data = self.deal_with_nans(
            data=deepcopy(self.observed_data),
            parameter_id=parameter_id,
            nan_policy=nan_policy,
            nan_replacement=nan_replacement)
        ## group solar cycles
        cycle_nums, loc = self.group_cycles_by_datetime(
            dts=np.copy(data['datetime']),
            solar_cycles=solar_cycles,
            activity_type=activity_type,
            bias=self.cycle_bias)
        data = {key : np.copy(value[loc]) for key, value in data.items()}
        data['solar cycle'] = cycle_nums
        ## update data by search criteria
        if search_kwargs is not None:
            event_searcher = EventSearcher(data)
            data, _ = event_searcher.search_events(**search_kwargs)
        ## get elapsed units of time by step-size
        data['elapsed'] = self.get_elapsed_time(
            dts=data['datetime'],
            time_step=elapsed_unit,
            step_size=1,
            time_scale='absolute')
        ## group extrema of parameter by elapsed
        elapsed_indices = self.get_windowed_indices_of_extrema(
            data=data,
            extremal_parameter=parameter_id,
            group_parameter='elapsed',
            group_by=group_by)
        data = {key : np.copy(value[elapsed_indices])
            for key, value in data.items()}
        ## pad data
        n = data['elapsed'][-1] + 1
        indices = np.copy(data['elapsed'])
        padded = self.apply_temporal_padding(
            data=data,
            time_step=elapsed_unit,
            step_size=1,
            indices=np.copy(data['elapsed']),
            integer_pad=integer_pad,
            float_pad=float_pad,
            string_pad=string_pad)
        temporal_corrections = self.pad_missing_datetimes(
            dts=padded['datetime'],
            time_step=elapsed_unit,
            step_size=1)
        correction_loc = np.nonzero(np.invert(padded['is event']))[0]
        for key, value in temporal_corrections.items():
            padded[key][correction_loc] = value[correction_loc]
        cycle_nums, loc = self.group_cycles_by_datetime(
            dts=np.copy(padded['datetime']),
            solar_cycles=solar_cycles,
            activity_type=activity_type,
            bias=self.cycle_bias)
        padded = {key : np.copy(value[loc]) for key, value in padded.items()}
        padded['solar cycle'] = cycle_nums
        if squeeze_interval:
            data = self.squeeze_interval(padded)
        ## load series
        identifiers = self.get_regular_series_identifiers(
            data=padded,
            parameter_id=parameter_id,
            series_id=series_id,
            elapsed_unit=elapsed_unit,
            group_by=group_by,
            search_kwargs=search_kwargs,
            string_pad=string_pad,
            integer_pad=integer_pad,
            float_pad=float_pad)
        result = {
            'data' : padded,
            'identifiers' : identifiers,
            'parameter mapping' : self.parameter_mapping,
            'unit mapping' : self.unit_mapping,
            'generalized parameter mapping' : self.generalized_parameter_mapping}
        self._regular_series.append(result)

class SolarFlares(EventSeries):

    def __init__(self, cycle_bias='left'):
        super().__init__(
            event_type='Flare',
            event_source='HESSI',
            url='https://hesperia.gsfc.nasa.gov/hessidata/dbase/hessi_flare_list.txt',
            date_fmt='%Y/%m/%d',
            time_fmt='%H:%M:%S',
            cycle_bias=cycle_bias)
        self._parameter_mapping = {
            'duration' : 'Duration',
            'counts per second' : r'Flux',
            'total counts' : 'Counts',
            'energy' : 'Energy',
            'low energy' : 'Low Energy$',
            'high energy' : 'High Energy',
            'x position' : r'$X_{POS}$',
            'y position' : r'$Y_{POS}$',
            'radial position' : r'$R_{POS}$'}
        self._unit_mapping = {
            'duration' : '$s$',
            'counts per second' : r'$\frac{counts}{s}$',
            'total counts' : None,
            'energy' : '$keV$',
            'low energy' : '$keV$',
            'high energy' : '$keV$',
            'x position' : '$arcsec$',
            'y position' : '$arcsec$',
            'radial position' : '$arcsec$'}
        self._generalized_parameter_mapping = {
            'low energy' : 'energy',
            'high energy' : 'energy'}
        self.datetime_prefixes = ('start', 'peak', 'end')

    def load_observed_data(self, filepath=None):
        ## explicit keys
        keys = ['flare id', 'start date', 'start time', 'peak time', 'end time']
        keys += ['duration', 'counts per second', 'total counts', 'energy']
        keys += ['x position', 'y position', 'radial position', 'active region id']
        usecols = np.arange(len(keys)).astype(int)
        kws = {
            'usecols' : usecols,
            'skip_header' : 7,
            'skip_footer' : 38,
            'dtype' : str}
        ## load observed parameters
        if filepath:
            ## load file from directory
            if isinstance(filepath, str):
                self._access_stamp = self.get_last_modified_datetime_string(filepath)
                data = np.genfromtxt(
                    filepath,
                    **kws)
            else:
                raise ValueError("invalid type(filepath): {}".format(filepath))
        else:
            ## load file from url
            self._access_stamp = self.get_current_datetime_string()
            with requests.Session() as s:
                download = s.get(self.url)
                data = np.genfromtxt(
                    download.iter_lines(),
                    **kws)
        ## update non-temporal parameters
        result = dict()
        for col, key in enumerate(keys):
            if key not in ('start date', 'start time', 'peak time', 'end time'):
                if key == 'energy':
                    energy = np.array([s.split('-') for s in data[:, col]]).astype(int)
                    result['low energy'] = energy[:, 0]
                    result['high energy'] = energy[:, 1]
                # elif key in ('id, active region'):
                #     base_result[key] = data[:, col].astype(str)
                else:
                    result[key] = data[:, col].astype(int)
        ## update temporal parameters
        start_dts = self.extract_datetime_components(
            dates=data[:, 1],
            times=data[:, 2],
            date_fmt='%d-%b-%Y',
            time_fmt='%H:%M:%S',
            include_components=False)
        peak_dts = self.extract_datetime_components(
            dates=data[:, 1],
            times=data[:, 3],
            date_fmt='%d-%b-%Y',
            time_fmt='%H:%M:%S',
            include_components=False)
        end_dts = self.extract_datetime_components(
            dates=data[:, 1],
            times=data[:, 4],
            date_fmt='%d-%b-%Y',
            time_fmt='%H:%M:%S',
            include_components=False)
        peak_condition = (peak_dts < start_dts)
        end_condition = (end_dts < start_dts)
        peak_dts[peak_condition] += relativedelta(days=1)
        end_dts[end_condition] += relativedelta(days=1)
        for dts, prefix in zip((start_dts, peak_dts, end_dts), self.datetime_prefixes):
            series_result = self.consolidate_datetime_components(dts, prefix='{} '.format(prefix))
            solar_cycles, _ = self.group_cycles_by_datetime(
                dts=series_result['{} datetime'.format(prefix)],
                solar_cycles=None,
                activity_type='full',
                bias=self.cycle_bias)
            series_result['{} solar cycle'.format(prefix)] = solar_cycles
            result.update(series_result)
        ## update remaining parameters
        result['is event'] = np.ones(result['flare id'].size, dtype=bool)
        result['source'] = np.repeat(self.event_source, result['flare id'].size)
        result['event'] = np.repeat(self.event_type, result['flare id'].size)
        self._observed_data = result

    def get_temporal_parameters_by_prefix(self, data, datetime_prefix):
        ## verify user input
        if datetime_prefix not in self.datetime_prefixes:
            raise ValueError("invalid datetime_prefix: '{}'".format(datetime_prefix))
        ## separate prefix from datetime components
        inclusive_keys, exclusive_keys = [], []
        for component in self.temporal_components:
            inclusive_keys.append('{} {}'.format(datetime_prefix, component))
            for prefix in self.datetime_prefixes:
                if prefix != datetime_prefix:
                    exclusive_keys.append('{} {}'.format(prefix, component))
        nchars = len('{} '.format(datetime_prefix)) # len(datetime_prefix) + 1
        ## get data
        result = dict()
        for key, value in data.items():
            if key in inclusive_keys:
                mod_key = key[nchars:]
                result[mod_key] = np.copy(value)
            elif key not in exclusive_keys:
                result[key] = np.copy(value)
        return result

    def load_raw_series(self, parameter_id, datetime_prefix, series_id=None, solar_cycles=None, activity_type='full', nan_policy='propagate', nan_replacement=None, squeeze_interval=False, elapsed_unit='minute', search_kwargs=None):
        ## deal with nans
        data = self.deal_with_nans(
            data=deepcopy(self.observed_data),
            parameter_id=parameter_id,
            nan_policy=nan_policy,
            nan_replacement=nan_replacement)
        ## select proper keys by datetime prefix
        dt_key = '{} datetime'.format(datetime_prefix)
        sc_key = '{} solar cycle'.format(datetime_prefix)
        ## select bounds of datetime interval by solar cycle
        cycle_nums, loc = self.group_cycles_by_datetime(
            dts=np.copy(data[dt_key]),
            solar_cycles=solar_cycles,
            activity_type=activity_type,
            bias=self.cycle_bias)
        data = {key : np.copy(value[loc]) for key, value in data.items()}
        data[sc_key] = cycle_nums
        ## group data by datetime prefix
        data = self.get_temporal_parameters_by_prefix(
            data=data,
            datetime_prefix=datetime_prefix)
        ## update data by search criteria
        if search_kwargs is not None:
            event_searcher = EventSearcher(data)
            data, _ = event_searcher.search_events(**search_kwargs)
        ## get elapsed units of time by step-size
        data['elapsed'] = self.get_elapsed_time(
            dts=data['datetime'],
            time_step=elapsed_unit,
            step_size=1,
            time_scale='relative')
        if squeeze_interval:
            data = self.squeeze_interval(data)
        ## rename parameter keys
        # data = self.collapse_keys(
        #     data=data,
        #     new_key=new_key,
        #     copy_key=copy_key,
        #     old_keys=old_keys)
        ## load series
        identifiers = self.get_raw_series_identifiers(
            data=data,
            parameter_id=parameter_id,
            series_id=series_id,
            elapsed_unit=elapsed_unit,
            search_kwargs=search_kwargs)
        result = {
            'data' : data,
            'identifiers' : identifiers,
            'parameter mapping' : self.parameter_mapping,
            'unit mapping' : self.unit_mapping,
            'generalized parameter mapping' : self.generalized_parameter_mapping}
        self._raw_series.append(result)

    def load_irregular_series(self, parameter_id, datetime_prefix, series_id=None, group_by='maximum', solar_cycles=None, activity_type='full', nan_policy='propagate', nan_replacement=None, squeeze_interval=False, elapsed_unit='minute', search_kwargs=None):
        ## deal with nans
        data = self.deal_with_nans(
            data=deepcopy(self.observed_data),
            parameter_id=parameter_id,
            nan_policy=nan_policy,
            nan_replacement=nan_replacement)
        ## select proper keys by datetime prefix
        dt_key = '{} datetime'.format(datetime_prefix)
        sc_key = '{} solar cycle'.format(datetime_prefix)
        ## select bounds of datetime interval by solar cycle
        cycle_nums, loc = self.group_cycles_by_datetime(
            dts=np.copy(data[dt_key]),
            solar_cycles=solar_cycles,
            activity_type=activity_type,
            bias=self.cycle_bias)
        data = {key : np.copy(value[loc]) for key, value in data.items()}
        data[sc_key] = cycle_nums
        ## group data by datetime prefix
        data = self.get_temporal_parameters_by_prefix(
            data=data,
            datetime_prefix=datetime_prefix)
        ## update data by search criteria
        if search_kwargs is not None:
            event_searcher = EventSearcher(data)
            data, _ = event_searcher.search_events(**search_kwargs)
        ## get elapsed units of time by step-size
        data['elapsed'] = self.get_elapsed_time(
            dts=data['datetime'],
            time_step=elapsed_unit,
            step_size=1,
            time_scale='absolute')
        if squeeze_interval:
            data = self.squeeze_interval(data)
        ## group extrema of parameter by elapsed
        elapsed_indices = self.get_windowed_indices_of_extrema(
            data=data,
            extremal_parameter=parameter_id,
            group_parameter='elapsed',
            group_by=group_by)
        data = {key : np.copy(value[elapsed_indices])
            for key, value in data.items()}
        ## load series
        identifiers = self.get_irregular_series_identifiers(
            data=data,
            parameter_id=parameter_id,
            series_id=series_id,
            elapsed_unit=elapsed_unit,
            group_by=group_by,
            search_kwargs=search_kwargs)
        result = {
            'data' : data,
            'identifiers' : identifiers,
            'parameter mapping' : self.parameter_mapping,
            'unit mapping' : self.unit_mapping,
            'generalized parameter mapping' : self.generalized_parameter_mapping}
        self._irregular_series.append(result)

    def load_regular_series(self, parameter_id, datetime_prefix, series_id=None, group_by='maximum', solar_cycles=None, activity_type='full', nan_policy='propagate', nan_replacement=None, squeeze_interval=False, elapsed_unit='minute', search_kwargs=None, integer_pad=-1, float_pad=0., string_pad='NaN'):
        ## deal with nans
        data = self.deal_with_nans(
            data=deepcopy(self.observed_data),
            parameter_id=parameter_id,
            nan_policy=nan_policy,
            nan_replacement=nan_replacement)
        ## select proper keys by datetime prefix
        dt_key = '{} datetime'.format(datetime_prefix)
        sc_key = '{} solar cycle'.format(datetime_prefix)
        ## select bounds of datetime interval by solar cycle
        cycle_nums, loc = self.group_cycles_by_datetime(
            dts=np.copy(data[dt_key]),
            solar_cycles=solar_cycles,
            activity_type=activity_type,
            bias=self.cycle_bias)
        data = {key : np.copy(value[loc]) for key, value in data.items()}
        data[sc_key] = cycle_nums
        ## group data by datetime prefix
        data = self.get_temporal_parameters_by_prefix(
            data=data,
            datetime_prefix=datetime_prefix)
        ## update data by search criteria
        if search_kwargs is not None:
            event_searcher = EventSearcher(data)
            data, _ = event_searcher.search_events(**search_kwargs)
        ## get elapsed units of time by step-size
        data['elapsed'] = self.get_elapsed_time(
            dts=data['datetime'],
            time_step=elapsed_unit,
            step_size=1,
            time_scale='absolute')
        ## group extrema of parameter by elapsed
        elapsed_indices = self.get_windowed_indices_of_extrema(
            data=data,
            extremal_parameter=parameter_id,
            group_parameter='elapsed',
            group_by=group_by)
        data = {key : np.copy(value[elapsed_indices])
            for key, value in data.items()}
        ## pad data
        n = data['elapsed'][-1] + 1
        indices = np.copy(data['elapsed'])
        padded = self.apply_temporal_padding(
            data=data,
            time_step=elapsed_unit,
            step_size=1,
            indices=np.copy(data['elapsed']),
            integer_pad=integer_pad,
            float_pad=float_pad,
            string_pad=string_pad)
        temporal_corrections = self.pad_missing_datetimes(
            dts=padded['datetime'],
            time_step=elapsed_unit,
            step_size=1)
        correction_loc = np.nonzero(np.invert(padded['is event']))[0]
        for key, value in temporal_corrections.items():
            padded[key][correction_loc] = value[correction_loc]
        cycle_nums, loc = self.group_cycles_by_datetime(
            dts=np.copy(padded['datetime']),
            solar_cycles=solar_cycles,
            activity_type=activity_type,
            bias=self.cycle_bias)
        padded = {key : np.copy(value[loc]) for key, value in padded.items()}
        padded['solar cycle'] = cycle_nums
        if squeeze_interval:
            data = self.squeeze_interval(padded)
        ## load series
        identifiers = self.get_regular_series_identifiers(
            data=padded,
            parameter_id=parameter_id,
            series_id=series_id,
            elapsed_unit=elapsed_unit,
            group_by=group_by,
            search_kwargs=search_kwargs,
            string_pad=string_pad,
            integer_pad=integer_pad,
            float_pad=float_pad)
        result = {
            'data' : padded,
            'identifiers' : identifiers,
            'parameter mapping' : self.parameter_mapping,
            'unit mapping' : self.unit_mapping,
            'generalized parameter mapping' : self.generalized_parameter_mapping}
        self._regular_series.append(result)

class Sunspots(EventSeries):

    def __init__(self, cycle_bias='left'):
        super().__init__(
            event_type='Sunspot',
            event_source='SILSO',
            url='http://www.sidc.be/silso/DATA/SN_d_tot_V2.0.txt',
            date_fmt='%Y/%m/%d',
            time_fmt='%H:%M:%S',
            cycle_bias=cycle_bias)
        self._temporal_histogram = None

    @property
    def temporal_histogram(self):
        return self._temporal_histogram

    def add_temporal_histogram(self, result):
        temporal_histogram = TemporalHistogram(
            data=result,
            hbias='left')
        midpoints = np.copy(result['datetime'])
        bin_widths = np.ones(midpoints.size).astype(int)
        edges = temporal_histogram.from_midpoint_to_edges(
            midpoints=midpoints,
            bin_widths=bin_widths,
            time_step='day')
        counts = np.copy(result['ssn'])
        counts[counts < 0] = 0
        temporal_histogram.update_edges(
            edges=edges,
            time_step='day',
            time_scale='relative')
        temporal_histogram.update_counts(
            counts=counts,
            extra_counts=True)
        self._temporal_histogram = temporal_histogram

    def load_observed_data(self, filepath=None):
        ## explicit keys
        keys = ['year', 'month', 'day', 'ssn', 'standard deviation', 'number of sources']
        kws = {
            'usecols' : (0, 1, 2, 4, 5, 6),
            'dtype' : str}
        ## load observed parameters
        if filepath:
            ## load file from directory
            if isinstance(filepath, str):
                self._access_stamp = self.get_last_modified_datetime_string(filepath)
                data = np.loadtxt(
                    filepath,
                    **kws)
            else:
                raise ValueError("invalid type(filepath): {}".format(filepath))
        else:
            ## load file from url
            self._access_stamp = self.get_current_datetime_string()
            with requests.Session() as s:
                download = s.get(self.url)
                data = np.loadtxt(
                    download.iter_lines(),
                    **kws)
        result = self.get_datetimes_from_components(
            years=data[:, 0].astype(int),
            months=data[:, 1].astype(int),
            days=data[:, 2].astype(int),
            hours=None,
            minutes=None,
            seconds=None)
        result['solar cycle'], _ = self.group_cycles_by_datetime(
            dts=result['datetime'],
            solar_cycles=None,
            activity_type='full',
            bias=self.cycle_bias)
        ## update non-temporal parameters
        for col, key in enumerate(keys):
            if key not in list(result.keys()):
                value = data[:, col]
                if key in ('ssn', 'number of sources'):
                    result[key] = value.astype(int)
                else:
                    result[key] = value.astype(float)
        ## update remaining parameters
        is_event = (result['ssn'] != -1)
        result['is event'] = is_event
        result['source'] = np.repeat(self.event_source, result['datetime'].size)
        result['event'] = np.repeat(self.event_type, result['datetime'].size)
        ## update counts by temporal parameters
        self._observed_data = deepcopy(result)
        self.add_temporal_histogram(result)

    def load_raw_series(self, series_id=None, solar_cycles=None, activity_type='full', nan_policy='propagate', nan_replacement=None, squeeze_interval=False):
        ## explicit inputs
        parameter_id = 'ssn'
        elapsed_unit = 'day'
        search_kwargs = None
        ## deal with nans
        data = self.deal_with_nans(
            data=deepcopy(self.observed_data),
            parameter_id=parameter_id,
            nan_policy=nan_policy,
            nan_replacement=nan_replacement)
        ## group solar cycles
        cycle_nums, loc = self.group_cycles_by_datetime(
            dts=np.copy(data['datetime']),
            solar_cycles=solar_cycles,
            activity_type=activity_type,
            bias=self.cycle_bias)
        data = {key : np.copy(value[loc]) for key, value in data.items()}
        data['solar cycle'] = cycle_nums
        ## update data by search criteria
        if search_kwargs is not None:
            event_searcher = EventSearcher(data)
            data, _ = event_searcher.search_events(**search_kwargs)
        ## get elapsed units of time by step-size
        data['elapsed'] = self.get_elapsed_time(
            dts=data['datetime'],
            time_step=elapsed_unit,
            step_size=1,
            time_scale='relative')
        if squeeze_interval:
            data = self.squeeze_interval(data)
        ## get temporal histogram
        lbound, rbound = np.min(data['datetime']), np.max(data['datetime'])
        loc = np.where((self.temporal_histogram.edges >= lbound) & (self.temporal_histogram.edges <= rbound))[0]
        temporal_histogram = TemporalHistogram(data)
        temporal_histogram.update_edges(edges=self.temporal_histogram.edges[loc])
        temporal_histogram.update_counts(counts=self.temporal_histogram.counts[loc[:-1]])
        ## load series
        identifiers = self.get_raw_series_identifiers(
            data=data,
            parameter_id=parameter_id,
            series_id=series_id,
            elapsed_unit=elapsed_unit,
            search_kwargs=search_kwargs)
        result = {
            'data' : data,
            'identifiers' : identifiers,
            'temporal histogram' : temporal_histogram,
            'parameter mapping' : None,
            'unit mapping' : None,
            'generalized parameter mapping' : None}
        self._raw_series.append(result)

class ActiveRegions(EventSeries):

    def __init__(self, cycle_bias='left'):
        super().__init__(
            event_type='AR',
            event_source='MDI',
            url='http://soi.stanford.edu/data/full_farside/link_info.html',
            date_fmt='%Y.%m.%d',
            time_fmt='%H:%M:%S',
            cycle_bias=cycle_bias)
        self._raw_series = dict()
        self._irregular_series = None
        self._regular_series = None

    @staticmethod
    def configure_carrington_map(fig, ax, vmin, vmax, rotation_number, start_dt, end_dt, visualizer):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmin, xmax, xn, xscale = 60, 360, 60, 360.
        xticklabels = np.arange(xmin, xmax+1, xn)
        xticks = xticklabels / xscale * xlim[1] + xlim[0]
        mirror_yticklabels = np.array([90, 60, 40, 20, 0, -20, -40, -60, -90]) ## angles (not evenly spaced; haversine distance??)
        yn = mirror_yticklabels.size # 9 # 5
        yticks = np.linspace(min(ylim), max(ylim), yn)
        yticklabels = np.linspace(-1, 1, yticks.size)
        degrees_to_sin_radians = lambda angle : np.sin(angle * np.pi / 180)
        yslope_offset = (ylim[-1] - ylim[0]) / (yticklabels[-1] - yticklabels[0])
        # mirror_yticks = degrees_to_sin_radians()
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=visualizer.ticksize)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=visualizer.ticksize)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis='both', labelsize=visualizer.ticksize)
        ax.set_xlabel('Carrington Longitude', fontsize=visualizer.labelsize)
        ax.set_ylabel('Sine Latitude', fontsize=visualizer.labelsize)
        ax.set_title('Carrington Rotation {} ({} - {})'.format(rotation_number, start_dt, end_dt), fontsize=visualizer.titlesize)
        # ax_mirror = ax.twinx()
        # ax_mirror.set_xlim(xlim)
        # ax_mirror.set_ylim(ylim)
        # # ax_mirror.set_yticks(mirror_yticks)
        # ax_mirror.set_yticklabels(mirror_yticklabels, fontsize=visualizer.ticksize)
        # ax_mirror.set_ylabel('Latitude', fontsize=visualizer.labelsize)
        return fig, ax

    @staticmethod
    def animate_timelapse(img_dir, fps=30, show_gray_scale=False, show_false_color=False, img_extension='.png', mov_extension='.mkv', codec=None, save=False):
        if not save:
            raise ValueError("not yet implemented")
        filepaths = []
        savename = 'MDI_SynopticMap_CarringtonRotations_fps{}'.format(fps)
        if show_gray_scale:
            for filepath in os.listdir(img_dir):
                if 'GRAY-SCALE' in filepath:
                    if filepath.endswith(img_extension):
                        filepaths.append('{}{}'.format(img_dir, filepath))
            savename = '{}_GRAY-SCALE'.format(savename)
        if show_false_color:
            for filepath in os.listdir(img_dir):
                if 'FALSE-COLOR' in filepath:
                    if filepath.endswith(img_extension):
                        filepaths.append('{}{}'.format(img_dir, filepath))
            savename = '{}_FALSE-COLOR'.format(savename)
        if len(filepaths) == 0:
            raise ValueError("no filepaths were found; convert fits --> image or input a different directory")
        savename = savename.replace(' ', '_')
        savepath = r'{}{}{}'.format(img_dir, savename, mov_extension)
        ## ADD CARRINGTON ROTATION NUMBER TO SAVENAME
        filepaths = sorted(filepaths)
        try:
            if mov_extension == '.gif':
                with imageio.get_writer(savepath, mode='I') as writer:
                    for filepath in filepaths:
                        writer.append_data(imageio.imread(filepath))
                io_optimize(savepath)
            else:
                writer_kwargs = dict()
                if codec is not None:
                    writer_kwargs['codec'] = codec
                writer_kwargs['mode'] = 'I'
                writer_kwargs['fps'] = fps
                with imageio.get_writer(savepath, **writer_kwargs) as writer:
                    for filepath in filepaths:
                        image = imageio.imread(filepath)
                        writer.append_data(image)
        except OSError:
            if mov_extension == '.gif':
                ## https://github.com/python-pillow/Pillow/issues/4544
                raise ValueError("not yet implemented")
            else:
                clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(filepaths, fps=fps)
                clip.write_videofile(savepath)

    def get_rotation_number_and_datetime_bounds(self, hdu):
        rotation_number = int(str(hdu[0].header['CAR_ROT']).replace(' ', '').split('/')[0])
        _start_dt = str(hdu[0].header['T_START']).replace('/ Time of Rotation Center', '').replace('_TAI', '').replace(' ', '').replace('_', ' ')
        _end_dt = str(hdu[0].header['T_STOP']).replace('/ Stop Time of Rotation', '').replace('_TAI', '').replace(' ', '').replace('_', ' ')
        start_dt = datetime.datetime.strptime(_start_dt, '{}'.format(self.datetime_fmt))
        end_dt = datetime.datetime.strptime(_end_dt, '{}'.format(self.datetime_fmt))
        return rotation_number, start_dt, end_dt

    def convert_fits_to_images(self, fitsdir=None, savedir=None, convert_to_gray_scale=False, convert_to_false_color=False, vmin=-60, vmax=60, img_extension='.png', save=False, **kwargs):
        if not any([convert_to_gray_scale, convert_to_false_color]):
            raise ValueError("at least one of the following inputs must be True: 'convert_to_gray_scale', 'convert_to_false_color'")
        if all([convert_to_gray_scale, convert_to_false_color]):
            for do_gray, do_color in zip([True, False], [False, True]):
                self.convert_fits_to_images(
                    fitsdir=fitsdir,
                    savedir=savedir,
                    convert_to_gray_scale=do_gray,
                    convert_to_false_color=do_color,
                    vmin=vmin,
                    vmax=vmax,
                    img_extension=img_extension,
                    save=save,
                    **kwargs)
        else:
            if fitsdir is None:
                raise ValueError("robots.txt explicitly disallows web-scraping")
            filepaths = sorted(['{}{}'.format(fitsdir, filename) for filename in os.listdir(fitsdir) if filename.endswith('.fits')])
            visualizer = VisualConfiguration(savedir=savedir)
            for filepath in filepaths:
                hdu = fits.open(filepath)
                rotation_number, start_dt, end_dt = self.get_rotation_number_and_datetime_bounds(hdu)
                hdu.close()
                img_data = fits.getdata(filepath, ext=0)
                fig, ax = plt.subplots(**kwargs)
                if (vmin is None) or (vmax is None):
                    ## by visual inspection, estimate vmin/vmax ~ ±60 < v < ±100
                    histogram = Histogram(
                        vs=img_data.reshape(-1))
                    histogram.update_edges(
                        criteria='auto')
                    histogram.update_counts(
                        extra_counts=True,
                        tol=10,
                        squeeze_leads=True,
                        squeeze_trails=True)
                    if vmin is None:
                        vmin = np.min(histogram.edges)
                    if vmax is None:
                        vmax = np.max(histogram.edges)
                savename = 'MDI_SynopticMap_{}'.format(filepath.rsplit('/', 1)[-1])
                if convert_to_gray_scale:
                    ax.imshow(
                        img_data,
                        cmap='gray',
                        origin='lower',
                        vmin=vmin,
                        vmax=vmax)
                    savename = '{}_GRAY-SCALE'.format(savename) if save else None
                else: # elif convert_to_false_color:
                    ax.imshow(
                        img_data,
                        cmap='hot',
                        origin='lower',
                        vmin=vmin,
                        vmax=vmax)
                    savename = '{}_FALSE-COLOR'.format(savename) if save else None
                fig, ax = self.configure_carrington_map(
                    fig=fig,
                    ax=ax,
                    vmin=vmin,
                    vmax=vmax,
                    rotation_number=rotation_number,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    visualizer=visualizer)
                if not save:
                    savename = None
                visualizer.display_fits_image(
                    fig=fig,
                    savename=savename,
                    extension=img_extension)

    def view_pixel_data(self, parameters, savedir=None, **kwargs):
        if isinstance(parameters, str):
            parameters = [parameters]
        elif not isinstance(parameters, (tuple, list, np.ndarray)):
            raise ValueError("invalid type(parameters): {}".format(type(parameters)))
        for parameter in parameters:
            if parameter not in ('pixel peak', 'pixel trough', 'pixel difference', 'pixel total', 'pixel mean', 'pixel median', 'pixel standard deviation'):
                raise ValueError("invalid parameter: {}".format(parameter))
        parameter_to_facecolor_mapping = {
            'pixel peak' : 'red',
            'pixel trough' : 'blue',
            'pixel difference' : 'purple',
            'pixel total' : 'k',
            'pixel mean' : 'darkgreen',
            'pixel median' : 'darkorange',
            'pixel standard deviation' : 'gray'}
        x_midpoints = self.raw_series['datetime']
        visualizer = VisualConfiguration(savedir=savedir)
        fig, ax = plt.subplots(**kwargs)
        save = True if savedir is not None else None
        if save:
            rot_numbers = np.sort(list(self.observed_data.keys()))
            savename = 'ActiveRegions_CR_{}-{}_'.format(rot_numbers[0], rot_numbers[-1])
        else:
            savename = None
        for parameter in parameters:
            if save:
                savename += '{}_'.format(parameter.title().replace(' ', ''))
            facecolor = parameter_to_facecolor_mapping[parameter]
            if parameter == 'standard deviation':
                ax.errorbar(
                    x_midpoints,
                    self.raw_series['pixel mean'],
                    yerr=self.raw_series['standard deviation'],
                    capsize=5,
                    ecolor=facecolor,
                    fmt='none',
                    label=parameter.title(),
                    alpha=1/len(parameters))
            else:
                y_counts = self.raw_series[parameter]
                ax.scatter(
                    x_midpoints,
                    y_counts,
                    color=facecolor,
                    label=parameter.title(),
                    alpha=1/len(parameters),
                    marker='.',
                    s=5)
        visualizer.subview_datetime_axis(
            ax=ax,
            axis='x',
            major_interval=12,
            minor_interval=1,
            sfmt='%Y-%m',
            locator='month')
        visualizer.apply_grid(ax)
        rot_numbers = list(self.observed_data.keys())
        first_dt = self.observed_data[rot_numbers[0]]['start datetime']
        last_dt = self.observed_data[rot_numbers[-1]]['end datetime']
        ax.set_xlabel('DateTime', fontsize=visualizer.labelsize)
        ax.set_ylabel('Pixel Data', fontsize=visualizer.labelsize)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis='both', labelsize=visualizer.ticksize)
        ax.set_title('Active Regions during SC $23$ (Carrington Rotations ${} - {}$)'.format(rot_numbers[0], rot_numbers[-1]), fontsize=visualizer.titlesize)
        handles, labels = ax.get_legend_handles_labels()
        dt_fmt = "%Y-%m-%d %H:%M:%S"
        visualizer.subview_legend(
            fig=fig,
            ax=ax,
            handles=handles,
            labels=labels,
            title='Analysis of .fits via Michelson Doppler Imager ({} - {})'.format(first_dt.strftime(dt_fmt), last_dt.strftime(dt_fmt)),
            bottom=0.2,
            textcolor=True)
        if save:
            savename = savename[:-1]
        visualizer.display_image(fig=fig, savename=savename)

    def load_observed_data(self, fitsdir=None):
        if fitsdir is None:
            raise ValueError("robots.txt explicitly disallows web-scraping")
        filepaths = sorted(['{}{}'.format(fitsdir, filename) for filename in os.listdir(fitsdir) if filename.endswith('.fits')])
        if len(filepaths) == 0:
            raise ValueError("no paths were found; invalid fitsdir: {}".format(fitsdir))
        data = dict()
        for i, filepath in enumerate(filepaths):
            if i == 0:
                self._access_stamp = self.get_last_modified_datetime_string(filepath)
            hdu = fits.open(filepath)
            rotation_number, start_dt, end_dt = self.get_rotation_number_and_datetime_bounds(hdu)
            img_data = fits.getdata(filepath, ext=0)
            data[rotation_number] = {
                'center carrington time' : hdu[0].header['CARRTIME'],
                'rms' : float(str(hdu[0].header['DATA_RMS']).replace(' ', '')),
                'start carrington time' : str(hdu[0].header['LON_FRST']).replace('/ First Carrington Time - Last Column', '').replace(' ', ''),
                'end carrington time' : str(hdu[0].header['LON_LAST']).replace('/ Last Carrington Time - First Column', '').replace(' ', ''),
                'start datetime' : start_dt,
                'center datetime' : str(hdu[0].header['T_OBS']).replace('/ Time of Rotation Center', '').replace(' ', ''),
                'end datetime' : end_dt,
                'unit' : str(hdu[0].header['BUNIT']).replace("'", "").replace(' ', ''),
                'scale' : float(str(hdu[0].header['BSCALE']).replace(' ', '')),
                'pixels' : img_data} # np.nan_to_num
            hdu.close()
        self._observed_data = data

    def load_raw_series(self):
        ## explicit keys
        keys = ('datetime', 'rotation radians', 'rotation degrees', 'pixel peak', 'pixel trough', 'pixel difference', 'pixel total', 'pixel mean', 'pixel median', 'pixel standard deviation')
        all_pixels, dts, is_event, rot_nums, rot_rads, rot_degs, pix_peak, pix_trg, pix_df, pix_tot, pix_avg, pix_med, pix_std = [], [], [], [], [], [], [], [], [], [], [], [], []
        for rotation_number, data in self.observed_data.items():
            _is_event = []
            pixels = data['pixels']
            all_pixels.append(pixels[:, ::-1])
            nrows, ncols = pixels.shape
            rot_nums.extend([rotation_number for _ in range(ncols)])
            degrees_per_pixel = 360 / ncols
            radians_per_pixel = degrees_per_pixel * np.pi / 180
            dt_start, dt_end = data['start datetime'], data['end datetime']
            delta_dt = dt_end - dt_start
            elapsed_per_pixel = delta_dt.total_seconds() / (ncols + 1)
            _peak = np.nanmax(pixels, axis=0)[::-1]
            _trough = np.nanmin(pixels, axis=0)[::-1]
            pix_peak.extend(
                _peak.tolist())
            pix_trg.extend(
                _trough.tolist())
            pix_df.extend(
                (_peak - _trough).tolist())
            pix_tot.extend(
                np.nansum(pixels, axis=0)[::-1].tolist())
            pix_avg.extend(
                np.nanmean(pixels, axis=0)[::-1].tolist())
            pix_med.extend(
                np.nanmedian(pixels, axis=0)[::-1].tolist())
            pix_std.extend(
                np.nanstd(pixels, axis=0)[::-1].tolist())
            rot_rads.extend(
                np.arange(0, 2 * np.pi + radians_per_pixel, radians_per_pixel)[::-1].tolist())
            rot_degs.extend(
                np.arange(0, 360 + degrees_per_pixel, degrees_per_pixel)[::-1].tolist())
            for i in range(ncols):
                elapsed_seconds = elapsed_per_pixel / 2 + i * elapsed_per_pixel
                dt_midpoint = dt_start + relativedelta(seconds=elapsed_seconds)
                dts.append(dt_midpoint)
                if np.all(np.isnan(pixels[:, i])):
                    _is_event.append(False)
                else:
                    _is_event.append(True)
            is_event.extend(_is_event[::-1])
        result = {
            'pixel data' : all_pixels,
            'datetime' : dts,
            'is event' : is_event,
            'rotation number' : rot_nums,
            'rotation radians' : rot_rads,
            'rotation degrees' : rot_degs,
            'pixel peak' : pix_peak,
            'pixel trough' : pix_trg,
            'pixel difference' : pix_df,
            'pixel total' : pix_tot,
            'pixel mean' : pix_avg,
            'pixel median' : pix_med,
            'pixel standard deviation' : pix_std}
        for key, value in result.items():
            _value = np.array(value)
            if key in ('rotation radians', 'rotation degrees'):
                result[key] = StatisticsConfiguration([]).get_midpoints(_value)
            else:
                result[key] = _value
        result['solar cycle'], _ = self.group_cycles_by_datetime(
            dts=result['datetime'],
            solar_cycles=None,
            activity_type='full',
            bias=self.cycle_bias)
        result['source'] = np.repeat(self.event_source, result['datetime'].size)
        result['event'] = np.repeat(self.event_type, result['datetime'].size)
        self._raw_series = result












##



##
