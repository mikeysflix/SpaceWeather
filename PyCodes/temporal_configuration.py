import calendar
import datetime
from dateutil.relativedelta import *
import numpy_indexed as npi
from search_methods import *

class DateTimeConfiguration(ConditionMapping):

    def __init__(self):
        super().__init__()
        self.temporal_components = (
            'datetime',
            'year',
            'month',
            'day',
            'hour',
            'minute',
            'second',
            'solar cycle')
        self.relative_time_scales = {
            'second' : 1,
            'minute' : 60, ## 60 * 1
            'hour' : 3600, ## 60 * 60
            'day' : 86400, ## 60 * 60 * 24
            'week' : 604800} ## 60 * 60 * 24 * 7

    @staticmethod
    def pad_missing_datetimes(dts, time_step, step_size=1):
        ## get time-step of step-size
        if step_size <= 0:
            raise ValueError("step_size must be greater than zero")
        if time_step in ('year', 'month', 'day', 'hour', 'minute', 'second'):
            kws = {'{}s'.format(time_step) : step_size}
            step = relativedelta(**kws)
            # step = datetime.timedelta(**kws)
        else:
            raise ValueError("invalid time_step for padding: {}".format(time_step))
        ## get bounds of datetime range
        bounds = np.copy([dts[0], dts[-1]])
        lbound, ubound = bounds
        ## get datetime components per step in bounded range
        _dts, years, months, days, hours, minutes, seconds = [], [], [], [], [], [], []
        while lbound < ubound:
            _dts.append(lbound)
            years.append(lbound.year)
            months.append(lbound.month)
            days.append(lbound.day)
            hours.append(lbound.hour)
            minutes.append(lbound.minute)
            seconds.append(lbound.second)
            lbound += step
        ## get result
        result = {
            'datetime' : np.array(_dts),
            'year' : np.array(years),
            'month' : np.array(months),
            'day' : np.array(days),
            'hour' : np.array(hours),
            'minute' : np.array(minutes),
            'second' : np.array(seconds)}
        return result

    @staticmethod
    def get_datetimes_from_components(years, months, days, hours=None, minutes=None, seconds=None):
        ## initialize result
        if years.size == months.size == days.size:
            result = {
                'year' : years,
                'month' : months,
                'day' : days}
        else:
            raise ValueError("number of years/months/days should match")
        ## format hours / minutes / seconds
        for key, component in zip(('hour', 'minute', 'second'), (hours, minutes, seconds)):
            if component is None:
                component = np.zeros(years.size).astype(int)
            elif isinstance(component, int):
                component = np.repeat(component, years.size).astype(int)
            elif not isinstance(component, (tuple, list, np.ndarray)):
                raise ValueError("invalid type({}s): {}".format(key, type(component)))
            result[key] = component
        ## get datetimes
        dts = []
        for year, month, day, hour, minute, second in zip(result['year'], result['month'], result['day'], result['hour'], result['minute'], result['second']):
            ymd_hms = np.array([year, month, day, hour, minute, second], dtype=int)
            dt = datetime.datetime(*ymd_hms)
            dts.append(dt)
        result['datetime'] = np.array(dts)
        return result

    @staticmethod
    def consolidate_datetime_components(dts, prefix=None):
        years, months, days = [], [], []
        hours, minutes, seconds = [], [], []
        for dt in dts:
            years.append(dt.year)
            months.append(dt.month)
            days.append(dt.day)
            hours.append(dt.hour)
            minutes.append(dt.minute)
            seconds.append(dt.second)
        if prefix is None:
            prefix = ''
        elif not isinstance(prefix, str):
            raise ValueError("invalid type(prefix): {}".format(type(prefix)))
        result = {
            '{}datetime'.format(prefix) : np.array(dts),
            '{}year'.format(prefix) : np.array(years),
            '{}month'.format(prefix) : np.array(months),
            '{}day'.format(prefix) : np.array(days),
            '{}hour'.format(prefix) : np.array(hours),
            '{}minute'.format(prefix) : np.array(minutes),
            '{}second'.format(prefix) : np.array(seconds)}
        return result

    @staticmethod
    def extract_datetimes_with_components(dates, times=None, date_fmt='%Y/%m/%d', time_fmt='%H:%M:%S'):
        dts = []
        years, months, days = [], [], []
        ## exclude hour, minute, and second for each observation
        if times is None:
            hms = '00:00:00'
            for ymd in dates:
                dt = datetime.datetime.combine(
                    datetime.datetime.strptime(ymd, date_fmt).date(),
                    datetime.datetime.strptime(hms, time_fmt).time())
                dts.append(dt)
                years.append(dt.year)
                months.append(dt.month)
                days.append(dt.day)
            n = len(dts)
            hours = np.zeros(n, dtype=int)
            minutes = np.zeros(n, dtype=int)
            seconds = np.zeros(n, dtype=int)
        ## include hour, minute, and second for each observation
        else:
            hours, minutes, seconds = [], [], []
            for ymd, hms in zip(dates, times):
                dt = datetime.datetime.combine(
                    datetime.datetime.strptime(ymd, date_fmt).date(),
                    datetime.datetime.strptime(hms, time_fmt).time())
                dts.append(dt)
                years.append(dt.year)
                months.append(dt.month)
                days.append(dt.day)
                hours.append(dt.hour)
                minutes.append(dt.minute)
                seconds.append(dt.second)
        ## get result
        result = {
            'datetime' : np.array(dts),
            'year' : np.array(years),
            'month' : np.array(months),
            'day' : np.array(days),
            'hour' : np.array(hours),
            'minute' : np.array(minutes),
            'second' : np.array(seconds)}
        return result

    @staticmethod
    def extract_datetimes_without_components(dates, times=None, date_fmt='%Y/%m/%d', time_fmt='%H:%M:%S'):
        dts = []
        ## exclude hour, minute, and second for each observation
        if times is None:
            hms = '00:00:00'
            for ymd in dates:
                dt = datetime.datetime.combine(
                    datetime.datetime.strptime(ymd, date_fmt).date(),
                    datetime.datetime.strptime(hms, time_fmt).time())
                dts.append(dt)
        ## include hour, minute, and second for each observation
        else:
            for ymd, hms in zip(dates, times):
                dt = datetime.datetime.combine(
                    datetime.datetime.strptime(ymd, date_fmt).date(),
                    datetime.datetime.strptime(hms, time_fmt).time())
                dts.append(dt)
        return np.array(dts)

    def extract_datetime_components(self, dates, times=None, date_fmt='%Y/%m/%d', time_fmt='%H:%M:%S', include_components=False):
        if include_components:
            result = self.extract_datetimes_with_components(
                dates=dates,
                times=times,
                date_fmt=date_fmt,
                time_fmt=time_fmt)
        else:
            result = self.extract_datetimes_without_components(
                dates=dates,
                times=times,
                date_fmt=date_fmt,
                time_fmt=time_fmt)
        return result

    def get_elapsed_time(self, dts, time_step, step_size=1, time_scale='relative'):
        if time_step not in list(self.relative_time_scales.keys()):
            raise ValueError("invalid time_step: {}".format(time_step))
        if step_size <= 0:
            raise ValueError("step_size must be greater than zero")
        t = self.relative_time_scales[time_step] * step_size
        elapsed = []
        for dt in dts:
            delta = dt - dts[0]
            elapsed.append(delta.total_seconds())
        result = np.array(elapsed) / t
        if time_scale == 'relative':
            return result
        elif time_scale == 'absolute':
            # return np.floor(result, dtype=int)
            return np.floor(result).astype(int)
            # # return result.astype(int)
        else:
            raise ValueError("invalid time_scale: {}".format(time_scale))

class TemporalConfiguration(DateTimeConfiguration):

    def __init__(self):
        super().__init__()
        self.activity_types = ('full', 'high-activity', 'early low-activity', 'late low-activity')
        self._solar_cycles = dict()
        self.initialize_solar_cycles()

    @property
    def solar_cycles(self):
        return self._solar_cycles

    @staticmethod
    def get_windowed_indices_of_extrema(data, extremal_parameter, group_parameter='elapsed', group_by='maximum'):
        x, y = np.copy(data[group_parameter]), np.copy(data[extremal_parameter])
        if group_by == 'maximum':
            indices = npi.group_by(x).argmax(y)[1]
        elif group_by == 'minimum':
            indices = npi.group_by(x).argmin(y)[1]
        else:
            raise ValueError("invalid group_by: {}".format(group_by))
        return indices

    @staticmethod
    def get_solar_cycles_label(data):
        cycle_nums = np.unique(data['solar cycle'])
        if cycle_nums.size == 1:
            sc_label = r'Solar Cycle ${}$'.format(cycle_nums[0])
        else:
            if np.all(np.diff(cycle_nums) == 1):
                sc_label = r'Solar Cycles ${} - {}$'.format(cycle_nums[0], cycle_nums[-1])
            else:
                sc_label = r'Solar Cycles ${}$'.format(','.join(cycle_nums.astype(str)))
        return sc_label

    def initialize_solar_cycles(self):
        ## initialize bounds of solar cycles
        bounds = np.array([
            datetime.datetime(1755, 2, 1, 0, 0, 0), ## sc 1
            datetime.datetime(1766, 6, 15, 0, 0, 0), ## sc 2
            datetime.datetime(1775, 6, 15, 0, 0, 0), ## sc 3
            datetime.datetime(1784, 9, 15, 0, 0, 0),
            datetime.datetime(1798, 4, 15, 0, 0, 0),
            datetime.datetime(1810, 8, 15, 0, 0, 0),
            datetime.datetime(1823, 5, 15, 0, 0, 0),
            datetime.datetime(1833, 11, 15, 0, 0, 0),
            datetime.datetime(1843, 7, 15, 0, 0, 0),
            datetime.datetime(1855, 12, 15, 0, 0, 0),
            datetime.datetime(1867, 3, 15, 0, 0, 0),
            datetime.datetime(1878, 12, 15, 0, 0, 0),
            datetime.datetime(1890, 3, 15, 0, 0, 0),
            datetime.datetime(1902, 1, 15, 0, 0, 0),
            datetime.datetime(1913, 7, 15, 0, 0, 0),
            datetime.datetime(1923, 8, 15, 0, 0, 0),
            datetime.datetime(1933, 9, 15, 0, 0, 0),
            datetime.datetime(1944, 2, 15, 0, 0, 0),
            datetime.datetime(1954, 4, 15, 0, 0, 0),
            datetime.datetime(1964, 10, 15, 0, 0, 0),
            datetime.datetime(1976, 3, 15, 0, 0, 0),
            datetime.datetime(1986, 9, 15, 0, 0, 0),
            datetime.datetime(1996, 8, 15, 0, 0, 0), ## sc 23
            datetime.datetime(2008, 12, 15, 0, 0, 0), ## sc 24
            datetime.datetime(2020, 1, 15, 0, 0, 0), ## sc 25
            datetime.datetime.now()])
        ## update solar cycle number by datetime bounds
        solar_cycles = dict()
        for i, (lbound, ubound) in enumerate(zip(bounds[:-1], bounds[1:])):
            j = i + 1
            result = dict()
            result['full'] = (lbound, ubound)
            if j == 23:
                result['high-activity'] = (datetime.datetime(1999, 2, 5, 0, 0, 0), datetime.datetime(2006, 12, 31, 23, 59, 59))
                result['early low-activity'] = (datetime.datetime(1996, 8, 15, 0, 0, 0), datetime.datetime(1998, 12, 31, 23, 59, 59))
                result['late low-activity'] = (datetime.datetime(2007, 1, 1, 0, 0, 0), datetime.datetime(2008, 12, 15, 0, 0, 0))
            elif j == 24:
                result['high-activity'] = (datetime.datetime(2010, 1, 1, 0, 0, 0), datetime.datetime(2016, 12, 31, 23, 59, 59))
                result['early low-activity'] = (datetime.datetime(2008, 12, 15, 0, 0, 0), datetime.datetime(2009, 12, 31, 23, 59, 59))
                result['late low-activity'] = (datetime.datetime(2017, 1, 1, 0, 0, 0), datetime.datetime(2020, 1, 15, 0, 0, 0))
            solar_cycles[j] = result
        self._solar_cycles.update(solar_cycles)

    def group_cycles_by_datetime(self, dts, solar_cycles=None, activity_type='full', bias='left', verify_activity_type=False, verify_cycle_numbers=False):
        ## verify user input
        if activity_type not in self.activity_types:
            raise ValueError("invalid activity_type: {}".format(activity_type))
        if solar_cycles is None:
            solar_cycles = list(self.solar_cycles.keys())
        elif isinstance(solar_cycles, int):
            solar_cycles = [solar_cycles]
        elif not isinstance(solar_cycles, (tuple, list, np.ndarray)):
            raise ValueError("invalid type(solar_cycles): {}".format(type(solar_cycles)))
        ## initialize cycle numbers
        cycle_numbers = np.zeros(dts.size, dtype=int)
        ## initialize event searcher
        event_searcher = EventSearcher({'datetime' : dts})
        ## group cycles by datetime
        parameters = ('datetime', 'datetime')
        if bias == 'left':
            conditions = ('greater than or equal', 'less than')
        elif bias == 'right':
            conditions = ('greater than', 'less than or equal')
        else:
            raise ValueError("invalid bias: {}".format(bias))
        loc = np.zeros(dts.size, dtype=bool)
        for cycle_num in solar_cycles:
            activity_dts = self.solar_cycles[cycle_num]
            if activity_type in list(activity_dts.keys()):
                values = np.copy(activity_dts[activity_type])
            else:
                if verify_activity_type:
                    raise ValueError("invalid activity_type: {} for solar cycle {}; instead, try one the following: '{}'".format(activity_type, cycle_num, list(activity_dts.keys())))
                else:
                    values = np.copy(activity_dts['full'])
            try:
                _, indices = event_searcher.search_events(parameters, conditions, values)
                cycle_numbers[indices] = cycle_num
                loc[np.where(indices)[0]] = True
            except:
                pass
        if np.any(cycle_numbers == 0):
            if verify_cycle_numbers:
                raise ValueError("invalid datetime; cycle number cannot be zero")
            else:
                cycle_numbers = np.copy(cycle_numbers[loc])
        return cycle_numbers, np.where(loc)[0]

    def apply_temporal_padding(self, data, time_step, step_size, indices, integer_pad, float_pad, string_pad):
        ## initialize padding configuration
        n = data['elapsed'][-1] + 1
        padded = dict()
        padded['elapsed'] = np.arange(n).astype(int)
        indices = np.copy(data['elapsed'])
        ## update non-date/time parameters
        for key, value in data.items():
            if not ((key in self.temporal_components) or (key == 'elapsed')):
                if value.dtype == bool:
                    arr = np.zeros(n, dtype=bool)
                elif value.dtype in (int, np.int64):
                    arr = np.repeat(integer_pad, n).astype(int)
                elif value.dtype in (float, np.float64):
                    try:
                        arr = np.ones(n, dtype=float) * float_pad
                    except:
                        print("\n type(n) = {}, n = {}".format(type(n), n))
                        print("\n type(float_pad) = {}, float_pad = {}".format(type(float_pad), float_pad))
                        arr = np.ones(n, dtype=float) * float_pad
                elif str(value.dtype)[:2] == '<U':
                    arr = np.array([string_pad] * n, dtype=value.dtype)
                else:
                    raise ValueError("unrecognized dtype '{}' for value corresponding to key '{}'".format(value.dtype, key))
                arr[indices] = np.copy(value)
                padded[key] = arr
        ## update date/time parameters
        first_dt = data['datetime'][0]
        dts = []
        for i in range(n):
            kws = {'{}s'.format(time_step) : step_size}
            dt = first_dt + relativedelta(**kws)
            dts.append(dt)
        dts = np.array(dts)
        dts[indices] = np.copy(data['datetime'])
        temporal_pads = self.consolidate_datetime_components(
            dts=dts,
            prefix=None)
        padded.update(temporal_pads)
        period_map = {
            'solar cycle' : ('hour', 'minute', 'second'),
            'year' : ('hour', 'minute', 'second'),
            'month' : ('hour', 'minute', 'second'),
            'day' :  ('hour', 'minute', 'second'),
            'hour' : ('minute', 'second'),
            'minute' : ('second',),
            'second' : None}
        subperiod_ids = period_map[time_step]
        if subperiod_ids is not None:
            temporal_subs = dict()
            for key in subperiod_ids:
                arr = np.zeros(n).astype(int)
                arr[indices] = np.copy(data[key])
                temporal_subs[key] = arr
            inv_loc = np.setdiff1d(padded['elapsed'], indices)
            inv_dts = padded['datetime'][inv_loc]
            padded['datetime'][inv_loc] = np.array([datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
                for year, month, day, hour, minute, second in zip(
                    padded['year'][inv_loc], padded['month'][inv_loc], padded['day'][inv_loc], padded['hour'][inv_loc], padded['minute'][inv_loc], padded['second'][inv_loc])])
        return padded








##
