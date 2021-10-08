import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.dates import DateFormatter, MonthLocator, date2num
from matplotlib.colors import Normalize, LogNorm, to_rgba, to_rgb, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
from matplotlib.legend_handler import HandlerTuple, HandlerRegularPolyCollection
import colorsys
from pandas.plotting import register_matplotlib_converters
from search_methods import *

register_matplotlib_converters()


class LabelConfiguration(ConditionMapping):

    def __init__(self):
        super().__init__()
        self.error_space_mapping = {
            'chi square' : r'$\chi^{2}_{\nu}$',
            'g-test' : r'$G_{\nu}$',
            'maximum likelihood' : r'$log(L_{max})$'}

    @staticmethod
    def make_plural(s, overwrite=False):
        if isinstance(s, str):
            if s == '':
                return ''
            else:
                if s[-1] == 's':
                    if overwrite:
                        return '{}es'.format(s)
                    else:
                        return '{}'.format(s)
                else:
                    return '{}s'.format(s)
        else:
            raise ValueError("invalid type(s): {}".format(type(s)))

    @staticmethod
    def is_same_elements(elements, s=None, n=None):
        if not isinstance(elements, list):
            raise ValueError("invalid type(elements): {}".format(type(elements)))
        nelems = len(elements)
        if elements.count(elements[0]) == nelems:
            result = True
            if (n is not None) and (nelems != n):
                result = False
            if (s is not None) and (s != ''):
            # if (s is not None) and (s != units[0]): # (s != '')
                result = False
        else:
            result = False
        return result

    @staticmethod
    def get_scientific_notation_string(values):
        f1 = ticker.ScalarFormatter(useOffset=False, useMathText=True)
        f2 = lambda x,pos : "${}$".format(f1._formatSciNotation('%1.10e' % x))
        fmt = ticker.FuncFormatter(f2)
        return fmt(values)

    def get_extreme_label(self, extreme_parameter, extreme_condition, extreme_value, parameter_mapping, unit_mapping):
        try:
            parameter_label = parameter_mapping[extreme_parameter]
        except:
            parameter_label = "{}".format(
                extreme_parameter.title())
        try:
            unit_label = unit_mapping[extreme_parameter]
        except:
            unit_label = None
        if unit_label is None:
            result = "{}".format(
                parameter_label)
        else:
            result = "{} {} {:,} {}".format(
                parameter_label,
                self.relational_mapping[extreme_condition],
                extreme_value,
                unit_label)
        result = "Extreme-Value Threshold: {}".format(result)
        return result

    def get_generalized_extreme_label(self, extreme_parameter, extreme_condition, extreme_value, parameter_mapping, unit_mapping, generalized_parameter_mapping):
        try:
            parameter_label = parameter_mapping[generalized_parameter_mapping[extreme_parameter]]
        except:
            parameter_label = "{}".format(
                generalized_parameter_mapping[extreme_parameter].title())
        try:
            unit_label = unit_mapping[extreme_parameter]
        except:
            unit_label = None
        if unit_label is None:
            result = "{}".format(
                parameter_label)
        else:
            result = "{} {} {:,} {}".format(
                parameter_label,
                self.relational_mapping[extreme_condition],
                extreme_value,
                unit_label)
        result = "Extreme-Value Threshold: {}".format(result)
        return result


class RoundingConfiguration(LabelConfiguration):

    def __init__(self):
        super().__init__()

    @staticmethod
    def round_down(num, divisor):
        return num - (num%divisor)

    @staticmethod
    def round_up(num, divisor):
        if not isinstance(divisor, int):
            raise ValueError("invalid type(divisor): {}".format(type(divisor)))
        return int(np.ceil(num / float(divisor))) * divisor

    @staticmethod
    def get_logarithmic_bounds(num, base, bound_id):
        exponent = np.log(num) / np.log(base)
        if bound_id == 'upper':
            exponent = np.ceil(exponent)
        elif bound_id == 'lower':
            exponent = np.floor(exponent)
        else:
            raise ValueError("invalid bound_id: {}".format(bound_id))
        value = base ** exponent
        return exponent, value

class AxesConfiguration(RoundingConfiguration):

    def __init__(self, ticksize=7, labelsize=8):
        super().__init__()
        self.ticksize = ticksize
        self.labelsize = labelsize

    @staticmethod
    def apply_grid(ax, color='gray', linestyle=':', alpha=0.3, **kwargs):
        ax.grid(color=color, linestyle=linestyle, alpha=alpha, **kwargs)
        return ax

    @staticmethod
    def get_mirror_ax(ax, frameon=False):
        mirror_ax = ax.figure.add_subplot(ax.get_subplotspec(), frameon=False)
        return mirror_ax

    @staticmethod
    def subview_datetime_axis(ax, axis, major_interval=12, minor_interval=1, sfmt='%Y-%m', locator='month'):
        fmt = DateFormatter(sfmt) # DateFormatter('%Y-%m' or '%b %d %Y')
        fmap = {
            'month' : MonthLocator,
            }
        f = fmap[locator]
        if axis == 'x':
            ax.xaxis.set_major_locator(f(interval=major_interval))
            ax.xaxis.set_minor_locator(f(interval=minor_interval))
            ax.xaxis.set_major_formatter(fmt)
        elif axis == 'y':
            ax.yaxis.set_major_locator(f(interval=major_interval))
            ax.yaxis.set_minor_locator(f(interval=minor_interval))
            ax.yaxis.set_major_formatter(fmt)
        else:
            raise ValueError("invalid axis: {}".format(axis))
        return ax

    def subview_monthly_axis(self, ax, axis, months=None):
        if months is None:
            major_ticks = np.arange(1, 13).astype(int)
        else:
            major_ticks = np.unique(months)
            if np.any(major_ticks < 0) or np.any(major_ticks > 12):
                raise ValueError("invalid months; 1 ≤ months ≤ 12")
        if axis == 'x':
            ax.set_xticks(major_ticks)
            tick_labels = [calendar.month_name[tck] for tck in major_ticks]
            ax.set_xticklabels(tick_labels, fontsize=self.ticksize)
        elif axis == 'y':
            ax.set_yticks(major_ticks)
            tick_labels = [calendar.month_name[tck] for tck in major_ticks]
            ax.set_yticklabels(tick_labels, fontsize=self.ticksize)
        else:
            raise ValueError("invalid axis: {}".format(axis))
        return ax

    def share_axis_parameters(self, axis, axes, vs, label=None, ticks=False, limits=False, fmt=None, log_base=None):
        ## verify user input
        if axis not in ('x', 'y', 'z'):
            raise ValueError("invalid axis: {}".format(axis))
        ## update axis label
        if label is not None:
            if not isinstance(label, str):
                raise ValueError("invalid type(label): {}".format(type(label)))
            if axis == 'x':
                for ax in axes.ravel():
                    ax.set_xlabel(label, fontsize=self.labelsize)
            elif axis == 'y':
                for ax in axes.ravel():
                    ax.set_ylabel(label, fontsize=self.labelsize)
            else:
                for ax in axes.ravel():
                    ax.set_zlabel(label, fontsize=self.labelsize)
        ## update axis ticks
        if isinstance(ticks, bool):
            if ticks:
                if axis == 'x':
                    if log_base is None:
                        for ax in axes.ravel():
                            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
                            ax.tick_params(axis='x', labelsize=self.ticksize)
                    else:
                        for ax in axes.ravel():
                            ax.set_xscale('log', basex=log_base)
                            ax.xaxis.set_major_locator(ticker.LogLocator(base=log_base))
                            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
                            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
                            ax.tick_params(axis='x', which='both', labelsize=self.ticksize)
                elif axis == 'y':
                    if log_base is None:
                        for ax in axes.ravel():
                            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
                            ax.tick_params(axis='y', labelsize=self.ticksize)
                    else:
                        for ax in axes.ravel():
                            ax.set_yscale('log', basey=log_base)
                            ax.yaxis.set_major_locator(ticker.LogLocator(base=log_base))
                            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
                            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
                            ax.tick_params(axis='y', which='both', labelsize=self.ticksize)
                elif axis == 'z':
                    if log_base is None:
                        for ax in axes.ravel():
                            ax.zaxis.set_minor_locator(ticker.AutoMinorLocator())
                            ax.tick_params(axis='z', labelsize=self.ticksize)
                    else:
                        for ax in axes.ravel():
                            ax.set_yscale('log', basez=log_base)
                            ax.zaxis.set_major_locator(ticker.LogLocator(base=log_base))
                            ax.zaxis.set_minor_formatter(ticker.NullFormatter())
                            ax.zaxis.set_major_formatter(ticker.ScalarFormatter())
                            ax.tick_params(axis='z', which='both', labelsize=self.ticksize)
        elif isinstance(ticks, (tuple, list, np.ndarray)):
            if len(ticks) == 2:
                major_ticks, minor_ticks = ticks
                if axis == 'x':
                    for ax in axes.ravel():
                        ax.set_xticks(major_ticks)
                        ax.set_xticks(minor_ticks, minor=True)
                        ax.tick_params(axis='x', labelsize=self.ticksize)
                elif axis == 'y':
                    for ax in axes.ravel():
                        ax.set_yticks(major_ticks)
                        ax.set_yticks(minor_ticks, minor=True)
                        ax.tick_params(axis='y', labelsize=self.ticksize)
                else:
                    for ax in axes.ravel():
                        ax.set_zticks(major_ticks)
                        ax.set_zticks(minor_ticks, minor=True)
                        ax.tick_params(axis='z', labelsize=self.ticksize)
            else:
                if axis == 'x':
                    for ax in axes.ravel():
                        ax.set_xticks(ticks)
                        ax.tick_params(axis='x', labelsize=self.ticksize)
                elif axis == 'y':
                    for ax in axes.ravel():
                        ax.set_yticks(ticks)
                        ax.tick_params(axis='y', labelsize=self.ticksize)
                else:
                    for ax in axes.ravel():
                        ax.set_zticks(ticks)
                        ax.tick_params(axis='z', labelsize=self.ticksize)
        else:
            raise ValueError("invalid type(ticks): {}".format(type(ticks)))
        ## update axis major-tick formatting
        if fmt is not None:
            try:
                if not isinstance(fmt, str):
                    raise ValueError("invalid type(fmt): {}".format(type(fmt)))
                if axis == 'x':
                    for ax in axes.ravel():
                        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter(fmt))
                elif axis == 'y':
                    for ax in axes.ravel():
                        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter(fmt))
                else:
                    for ax in axes.ravel():
                        ax.zaxis.set_major_formatter(ticker.StrMethodFormatter(fmt))
            except:
                if axis == 'x':
                    for ax in axes.ravel():
                        ax.xaxis.set_major_formatter(fmt)
                elif axis == 'y':
                    for ax in axes.ravel():
                        ax.yaxis.set_major_formatter(fmt)
                else:
                    for ax in axes.ravel():
                        ax.zaxis.set_major_formatter(fmt)
        ## update grid
        for ax in axes.ravel():
            ax.grid(color='k', linestyle=':', alpha=0.3)
        ## update axis limits
        if isinstance(limits, bool):
            if limits:
                try:
                    if log_base is None:
                        _vs = np.copy(vs)
                        _vs = _vs[np.isfinite(_vs)]
                    else:
                        vs = np.array(vs)
                        vs = vs[np.isfinite(vs)]
                        vloc = (vs > 0)
                        if len(vloc) == vs.size:
                            _vs = np.copy(vs)
                        else:
                            _vs = vs[vloc].tolist()
                            _vs.append(0.1)
                except: ## TypeError
                    if not isinstance(_vs[0], datetime.datetime):
                        raise ValueError("invalid vs: {}".format(vs))
                vmin, vmax = np.nanmin(_vs), np.nanmax(_vs)
                if axis == 'x':
                    for ax in axes.ravel():
                        ax.set_xlim([vmin, vmax])
                elif axis == 'y':
                    for ax in axes.ravel():
                        ax.set_ylim([vmin, vmax])
                else:
                    for ax in axes.ravel():
                        ax.set_zlim([vmin, vmax])
        elif isinstance(limits, (tuple, list, np.ndarray)):
            nlim = len(limits)
            if nlim != 2:
                raise ValueError("this function accepts a container of 2 limits (upper and lower) but received {} limits".format(nlim))
            if axis == 'x':
                for ax in axes.ravel():
                    ax.set_xlim(limits)
            elif axis == 'y':
                for ax in axes.ravel():
                    ax.set_ylim(limits)
            else:
                for ax in axes.ravel():
                    ax.set_zlim(limits)
        else:
            raise ValueError("invalid type(limits): {}".format(type(limits)))

    def unshare_axis_parameters(self, axes, layout, collapse_x=False, collapse_y=False):
        if any([collapse_x, collapse_y]):
            if layout != 'overlay':
                ## 1-D axes
                if layout in ('horizontal', 'vertical'):
                    if layout == 'horizontal':
                        nrows, ncols = 0, axes.size
                    else:
                        nrows, ncols = axes.size, 0
                    if collapse_x:
                        if (ncols == 0) and (nrows > 1):
                            for ax in axes[:-1].ravel():
                                ax.set_xticklabels([], fontsize=self.ticksize)
                                ax.set_xlabel('', fontsize=self.labelsize)
                    if collapse_y:
                        if (ncols > 1) and (nrows == 0):
                            for ax in axes[1:].ravel():
                                ax.set_yticklabels([], fontsize=self.ticksize)
                                ax.set_ylabel('', fontsize=self.labelsize)
                ## 2-D axes
                else:
                    if len(axes.shape) != 2:
                        raise ValueError("this method only works for 1-D and 2-D axes; this method does not work for axes of shape {}".format(axes.shape))
                    nrows, ncols = axes.shape
                    if collapse_x:
                        for ax in axes[:-1, :].ravel():
                            ax.set_xticklabels([], fontsize=self.ticksize)
                            ax.set_xlabel('', fontsize=self.labelsize)
                    if collapse_y:
                        for ax in axes[:, 1:].ravel():
                            ax.set_yticklabels([], fontsize=self.ticksize)
                            ax.set_ylabel('', fontsize=self.labelsize)

    def share_mirror_axis_parameters(self, axis, mirror_axes, axes, label=None, ticks=False, ticklabels=False, limits=False, fmt=None, facecolor='k'):
        ## verify user input
        if axis not in ('x', 'y'):
            raise ValueError("invalid axis: {}".format(axis))
        ## update axis label
        if label is not None:
            if not isinstance(label, str):
                raise ValueError("invalid type(label): {}".format(type(label)))
            if axis == 'x':
                for mirror_ax in mirror_axes.ravel():
                    mirror_ax.set_xlabel(label, fontsize=self.labelsize)
                    mirror_ax.xaxis.label.set_color(facecolor)
                    mirror_ax.xaxis.set(label_position='top')
            else:
                for mirror_ax in mirror_axes.ravel():
                    mirror_ax.set_ylabel(label, fontsize=self.labelsize)
                    mirror_ax.yaxis.label.set_color(facecolor)
                    mirror_ax.yaxis.set(label_position='right', offset_position='right')
                    mirror_ax.yaxis.label.set_rotation(-90)
            ## update axis ticks
            if isinstance(ticks, bool):
                if ticks:
                    if axis == 'x':
                        for mirror_ax, ax in zip(mirror_axes.ravel(), axes.ravel()):
                            mirror_ax.xaxis.tick_top()
                            mirror_ax.set_xticks(ax.get_xticks())
                            mirror_ax.tick_params(axis='x', colors=facecolor)
                    else:
                        for mirror_ax, ax in zip(mirror_axes.ravel(), axes.ravel()):
                            mirror_ax.set_yticks(ax.get_yticks())
                            mirror_ax.yaxis.tick_right()
                            mirror_ax.tick_params(axis='y', colors=facecolor)
            elif isinstance(ticks, (tuple, list, np.ndarray)):
                if len(ticks) == 2:
                    major, minor = ticks
                    if axis == 'x':
                        for mirror_ax in mirror_axes.ravel():
                            mirror_ax.xaxis.tick_top()
                            mirror_ax.set_xticks(major)
                            mirror_ax.set_xticks(minor, minor=True)
                            mirror_ax.tick_params(axis='x', colors=facecolor)
                    else:
                        for mirror_ax in mirror_axes.ravel():
                            mirror_ax.set_yticks(major)
                            mirror_ax.set_yticks(minor, minor=True)
                            mirror_ax.yaxis.tick_right()
                            mirror_ax.tick_params(axis='y', colors=facecolor)
                else:
                    if axis == 'x':
                        for mirror_ax in mirror_axes.ravel():
                            mirror_ax.xaxis.tick_top()
                            mirror_ax.set_xticks(ticks)
                            mirror_ax.tick_params(axis='x', colors=facecolor)
                    else:
                        for mirror_ax in mirror_axes.ravel():
                            mirror_ax.set_yticks(ticks)
                            mirror_ax.yaxis.tick_right()
                            mirror_ax.tick_params(axis='y', colors=facecolor)
            else:
                raise ValueError("invalid type(ticks): {}".format(type(ticks)))
            ## update axis ticklabels
            if isinstance(ticklabels, bool):
                if ticklabels:
                    for mirror_ax, ax in zip(mirror_axes.ravel(), axes.ravel()):
                        if axis == 'x':
                            ticklabels = ax.get_xticklabels()
                            if fmt is not None:
                                ticklabels = [fmt.format(ticklabel) for ticklabel in ticklabels]
                            mirror_ax.set_xticklabels(ticklabels, fontsize=self.ticksize)
                            mirror_ax.xaxis.tick_top()
                        else:
                            ticklabels = ax.get_yticklabels()
                            if fmt is not None:
                                ticklabels = [fmt.format(ticklabel) for ticklabel in ticklabels]
                            mirror_ax.set_yticklabels(ticklabels, fontsize=self.ticksize)
                            mirror_ax.yaxis.tick_right()
                else:
                    if axis == 'x':
                        for mirror_ax in mirror_axes.ravel():
                            mirror_ax.set_xticklabels([], fontsize=self.ticksize)
                            mirror_ax.xaxis.tick_top()
                    else:
                        for mirror_ax in mirror_axes.ravel():
                            mirror_ax.set_yticklabels([], fontsize=self.ticksize)
                            mirror_ax.yaxis.tick_right()
            elif isinstance(ticklabels, (tuple, list, np.ndarray)):
                if len(ticklabels) == 2:
                    major, minor = ticklabels
                    if axis == 'x':
                        if fmt is not None:
                            major = [fmt.format(_maj) for _maj in major]
                            minor = [fmt.format(_min) for _min in minor]
                        for mirror_ax in mirror_axes.ravel():
                            mirror_ax.set_xticklabels(major, fontsize=self.ticksize)
                            mirror_ax.set_xticklabels(minor, minor=True, fontsize=self.ticksize)
                            mirror_ax.xaxis.tick_top()
                    else:
                        if fmt is not None:
                            major = [fmt.format(_maj) for _maj in major]
                            minor = [fmt.format(_min) for _min in minor]
                        for mirror_ax in mirror_axes.ravel():
                            mirror_ax.set_yticklabels(major, fontsize=self.ticksize)
                            mirror_ax.set_yticklabels(minor, minor=True, fontsize=self.ticksize)
                            mirror_ax.yaxis.tick_right()
                else:
                    if axis == 'x':
                        if fmt is None:
                            for mirror_ax in mirror_axes.ravel():
                                mirror_ax.set_xticklabels(ticklabels, fontsize=self.ticksize)
                                mirror_ax.xaxis.tick_top()
                        else:
                            _ticklabels = [fmt.format(ticklabel) for ticklabel in _ticklabels]
                            for mirror_ax in mirror_axes.ravel():
                                mirror_ax.set_xticklabels(_ticklabels, fontsize=self.ticksize)
                                mirror_ax.xaxis.tick_top()
                    else:
                        if fmt is None:
                            for mirror_ax in mirror_axes.ravel():
                                mirror_ax.set_yticklabels(ticklabels, fontsize=self.ticksize)
                                mirror_ax.yaxis.tick_right()
                        else:
                            _ticklabels = [fmt.format(ticklabel) for ticklabel in _ticklabels]
                            for mirror_ax in mirror_axes.ravel():
                                mirror_ax.set_yticklabels(_ticklabels, fontsize=self.ticksize)
                                mirror_ax.yaxis.tick_right()
            elif hasattr(ticklabels, '__call__'):
                for mirror_ax, ax in zip(mirror_axes.ravel(), axes.ravel()):
                    if axis == 'x':
                        ticks = ax.get_xticks()
                        _ticklabels = ticklabels(ticks)
                        if fmt is not None:
                            _ticklabels = [fmt.format(ticklabel) for ticklabel in _ticklabels]
                        mirror_ax.set_xticklabels(_ticklabels, fontsize=self.ticksize)
                        mirror_ax.xaxis.tick_top()
                    else:
                        ticks = ax.get_yticks()
                        _ticklabels = ticklabels(ticks)
                        if fmt is not None:
                            _ticklabels = [fmt.format(ticklabel) for ticklabel in _ticklabels]
                        mirror_ax.set_yticklabels(_ticklabels, fontsize=self.ticksize)
                        mirror_ax.yaxis.tick_right()
            else:
                raise ValueError("invalid type(ticklabels): {}".format(type(ticklabels)))
            ## update axis limits
            if isinstance(limits, bool):
                for mirror_ax, ax in zip(mirror_axes.ravel(), axes.ravel()):
                    if limits:
                        if axis == 'x':
                            for mirror_ax in mirror_axes.ravel():
                                mirror_ax.set_xlim(ax.get_xlim())
                        else:
                            for mirror_ax in mirror_axes.ravel():
                                mirror_ax.set_ylim(ax.get_ylim())
            elif isinstance(limits, (tuple, list, np.ndarray)):
                nlim = len(limits)
                if nlim != 2:
                    raise ValueError("this function accepts a container of 2 limits (upper and lower) but received {} limits".format(nlim))
                if axis == 'x':
                    for mirror_ax in mirror_axes.ravel():
                        mirror_ax.set_xlim(limits)
                else:
                    for mirror_ax in mirror_axes.ravel():
                        mirror_ax.set_ylim(limits)
            else:
                raise ValueError("invalid type(limits): {}".format(type(limits)))

    def unshare_mirror_axis_parameters(self, mirror_axes, layout, collapse_x=False, collapse_y=False):
        if any([collapse_x, collapse_y]):
            n = len(mirror_axes.shape)
            ## 1-DIM
            if n == 1:
                if layout == 'horizontal':
                    if collapse_y:
                        for ax in mirror_axes[:-1].ravel():
                            ax.set_yticklabels([], fontsize=self.ticksize)
                            ax.set_ylabel('', fontsize=self.labelsize)
                elif layout == 'vertical':
                    if collapse_x:
                        for ax in mirror_axes[1:].ravel():
                            ax.set_xticklabels([], fontsize=self.ticksize)
                            ax.set_xlabel('', fontsize=self.labelsize)
                elif layout == 'overlay':
                    pass
                else:
                    raise ValueError("something went wrong; see layout")
            ## 2-DIM
            elif n == 2:
                if collapse_y:
                    for ax in mirror_axes[:, :-1].ravel():
                        ax.set_yticklabels([], fontsize=self.ticksize)
                        ax.set_ylabel('', fontsize=self.labelsize)
                if collapse_x:
                    for ax in mirror_axes[1:, :].ravel():
                        ax.set_xticklabels([], fontsize=self.ticksize)
                        ax.set_xlabel('', fontsize=self.labelsize)
            else:
                raise ValueError("this method only works for 1-D and 2-D mirror_axes; this method does not work for mirror_axes of shape {}".format(mirror_axes.shape))

    def share_axes(self, axes, layout, xs, ys, sharex=False, sharey=False, xlim=False, ylim=False, xticks=False, yticks=False, xfmt=None, yfmt=None, xlabel=None, ylabel=None, basex=None, basey=None, collapse_x=False, collapse_y=False):
        for axis, share, vs, label, ticks, limits, fmt, log_base in zip(('x', 'y'), (sharex, sharey), (xs, ys), (xlabel, ylabel), (xticks, yticks), (xlim, ylim), (xfmt, yfmt), (basex, basey)):
            if share:
                self.share_axis_parameters(
                    axis=axis,
                    axes=axes,
                    vs=vs,
                    label=label,
                    ticks=ticks,
                    limits=limits,
                    fmt=fmt,
                    log_base=log_base)
        self.unshare_axis_parameters(
            axes=axes,
            layout=layout,
            collapse_x=collapse_x,
            collapse_y=collapse_y)

    def share_mirror_axes(self, axes, layout, sharex=False, sharey=False, xlim=False, ylim=False, xticks=False, yticks=False, xticklabels=False, yticklabels=False, xfmt=None, yfmt=None, xlabel=None, ylabel=None, xcolor='k', ycolor='k', collapse_x=False, collapse_y=False):
        mirror_axes = []
        for ax in axes.ravel():
            mirror_ax = self.get_mirror_ax(ax)
            mirror_axes.append(mirror_ax)
        mirror_axes = np.array(mirror_axes).reshape(axes.shape)
        for i, (axis, share, label, ticks, ticklabels, limits, fmt, facecolor) in enumerate(zip(('x', 'y'), (sharex, sharey), (xlabel, ylabel), (xticks, yticks), (xticklabels, yticklabels), (xlim, ylim), (xfmt, yfmt), (xcolor, ycolor))):
            if share:
                self.share_mirror_axis_parameters(
                    axis=axis,
                    mirror_axes=mirror_axes,
                    axes=axes,
                    label=label,
                    ticks=ticks,
                    ticklabels=ticklabels,
                    limits=limits,
                    fmt=fmt,
                    facecolor=facecolor)
                self.unshare_mirror_axis_parameters(
                    mirror_axes=mirror_axes,
                    layout=layout,
                    collapse_x=collapse_x,
                    collapse_y=collapse_y)

class LegendConfiguration(AxesConfiguration):

    def __init__(self, ticksize=7, labelsize=8):
        super().__init__(ticksize, labelsize)

    @staticmethod
    def get_empty_handle(ax):
        empty_handle = ax.scatter([np.nan], [np.nan], color='none', alpha=0)
        return empty_handle

    @staticmethod
    def get_number_of_legend_columns(labels):
        if isinstance(labels, int):
            n = labels
        else:
            n = len(labels)
        if n > 2:
            if n >= 17:
                possible_values = []
                for v in range(n-1, 17, -1):
                    gcd = np.gcd(n, v)
                    lcm = np.lcm(n, v)
                    possible_value = lcm / gcd
                    if float(possible_value) == int(possible_value):
                        possible_values.append(int(possible_value))
                _ncol = max(possible_values)
                if _ncol in (n, n-1):
                    ncol = _ncol // 2
                else:
                    ncol = _ncol
            else:
                if n % 3 == 0:
                    ncol = 3
                else:
                    ncol = n // 2
        else:
            ncol = n
        return ncol

    @staticmethod
    def get_customized_line_handle(facecolor, label=None, linewidth=1, linestyle='-', alpha=1):
        return Line2D(
            [0],
            [0],
            color=facecolor,
            label=label,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha)

    @staticmethod
    def get_customized_scatter_handle(facecolor, label=None, marker='.', markersize=10, alpha=1):
        return Line2D(
            [0],
            [0],
            color=facecolor,
            label=label,
            marker=marker,
            linestyle=None,
            markersize=markersize,
            alpha=alpha)

    @staticmethod
    def get_customized_bar_handle(facecolor, label=None, alpha=1):
        return mpatches.Patch(
            color=facecolor,
            label=label,
            alpha=alpha)

    def update_legend_design(self, leg, title=None, textcolor=False, facecolor=None, edgecolor=None, titlecolor=None, borderaxespad=None, apply_semibold=False):
        if title:
            title_prop = {'size' : self.labelsize}
            if apply_semibold:
                title_prop['weight'] = 'semibold'
            leg.set_title(title, prop=title_prop)
            if titlecolor:
                leg.get_title().set_color(titlecolor)
            # leg.get_title().set_ha("center")
        leg._legend_box.align = "center"
        frame = leg.get_frame()
        if facecolor:
            frame.set_facecolor(facecolor)
        if edgecolor:
            frame.set_edgecolor(edgecolor)
        if textcolor:
            if isinstance(textcolor, bool):
                try:
                    for line, text in zip(leg.get_lines(), leg.get_texts()):
                        text.set_color(line.get_color())
                except:
                    for handle, text in zip(leg.legendHandles, leg.get_texts()):
                        text.set_color(handle.get_facecolor()[0])
                # text.set_color(to_rgba(tcolor))
            elif isinstance(textcolor, str):
                for text in leg.get_texts():
                    text.set_color(textcolor)
            else:
                raise ValueError("invalid type(textcolor): {}".format(type(textcolor)))
        elif isinstance(textcolor, (tuple, list, np.ndarray)):
            for handle, text, tcolor in zip(leg.legendHandles, leg.get_texts(), textcolor):
                text.set_color(to_rgba(tcolor))
        else:
            raise ValueError("invalid type(textcolor): {}".format(type(textcolor)))
        return leg

    def subview_legend(self, fig, ax, handles, labels, title=None, bottom=None, textcolor='darkorange', facecolor='white', edgecolor='k', titlecolor='k', ncol=None, bbox_to_anchor=None, borderaxespad=0.1, **kwargs):
        nlabels = len(labels)
        if ncol:
            if (nlabels == 1) and (ncol != 3):
                raise ValueError("ncol should be 3 to place the only legend label in the middle")
        else:
            if nlabels == 1:
                ncol = 3
                empty_handle = self.get_empty_handle(ax)
                handles = [empty_handle, handles[0], empty_handle]
                labels = ['  ', labels[0], '  ']
            else:
                ncol = self.get_number_of_legend_columns(labels)
        if bottom is not None:
            fig.subplots_adjust(bottom=bottom)
        if bbox_to_anchor is None:
            leg = fig.legend(handles=handles, labels=labels, ncol=ncol, loc='lower center', mode='expand', borderaxespad=borderaxespad, fontsize=self.labelsize, **kwargs)
        else:
            leg = fig.legend(handles=handles, labels=labels, ncol=ncol, loc='lower center', mode='expand', borderaxespad=borderaxespad, fontsize=self.labelsize, bbox_to_anchor=bbox_to_anchor, **kwargs)
        leg = self.update_legend_design(leg, title=title, textcolor=textcolor, facecolor=facecolor, edgecolor=edgecolor, titlecolor=titlecolor)
        return leg

class TabularConfiguration(LegendConfiguration):

    def __init__(self, ticksize=7, labelsize=8, textsize=5, titlesize=9, headersize=10, cellsize=15):
        super().__init__(ticksize, labelsize)
        self.textsize = textsize
        self.titlesize = titlesize
        self.headersize = headersize
        self.cellsize = cellsize

    @staticmethod
    def get_diagonal_table_colors(facecolors, nrows, ncols):
        if isinstance(facecolors, str):
            facecolors = [facecolors]
        ncolors = len(facecolors)
        if ncolors != 2:
            raise ValueError("2 facecolors required but only {} provided".format(ncolors))
        cell_colors = []
        for r in range(nrows):
            for c in range(ncols):
                if r % 2 == 0:
                    if c % 2 == 0:
                        cell_colors.append(facecolors[0])
                    else:
                        cell_colors.append(facecolors[1])
                else:
                    if c % 2 == 0:
                        cell_colors.append(facecolors[1])
                    else:
                        cell_colors.append(facecolors[0])
        return np.array(cell_colors).reshape((nrows, ncols))

    def autoformat_table(self, ax, table):
        ## update table scale
        ax.axis('off')
        ax.axis('tight')
        table.auto_set_font_size(False)
        table.set_fontsize(self.cellsize)
        for key, cell in table.get_celld().items():
            row, col = key
            if row == 0:
                cell.set_text_props(fontproperties=FontProperties(variant='small-caps', size=self.ticksize)) # weight='semibold'
        # xscale, yscale = (ncols, nrows)
        # table.scale(xscale, yscale)

class ColorConfiguration(TabularConfiguration):

    def __init__(self, ticksize=7, labelsize=8, textsize=5, titlesize=9, headersize=10, cellsize=6):
        super().__init__(ticksize, labelsize, textsize, titlesize, headersize, cellsize)

    @staticmethod
    def get_facecolors_from_cmap(cmap, norm, arr):
        f = plt.get_cmap(cmap)
        return f(norm(arr))

    @staticmethod
    def get_colormap_configuration(z, color_spacing, levels=None, fmt=None):
        valid_loc = np.isfinite(z)
        _z = np.copy(z[valid_loc])
        zmin, zmax = np.min(_z), np.max(_z)
        if color_spacing not in ('linear', 'log'):
            raise ValueError("invalid color_spacing: {}".format(color_spacing))
        if levels is None:
            vmin, vmax = zmin, zmax
        elif isinstance(levels, bool):
            if levels:
                vmin = 0
                vmax = np.max(_z) * 10
            else:
                vmin, vmax = zmin, zmax
        elif isinstance(levels, (tuple, list, np.ndarray)):
            vmin = levels[0]
            vmax = levels[-1]
        else:
            raise ValueError("invalid type(levels): {}".format(type(levels)))
        if color_spacing == 'linear':
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            if vmin == 0:
                if zmin > 0.1:
                    vmin = 0.1
                else:
                    vmin = 1e-7
            if vmax < vmin:
                vmin, vmax = vmax, vmin
            norm = LogNorm(vmin=vmin, vmax=vmax)
        if fmt is None:
            if zmin < 0.1:
                if zmax < 0.1:
                    fmt = ticker.StrMethodFormatter('{x:,.7f}')
                elif 0.1 <= zmax < 1:
                    fmt = ticker.StrMethodFormatter('{x:,.3f}')
                else:
                    fmt = ticker.StrMethodFormatter('{x:,.0f}')
            elif 0.1 <= zmin < 1:
                fmt = ticker.StrMethodFormatter('{x:,.2f}')
            else:
                fmt = ticker.StrMethodFormatter('{x:,.0f}')
        return norm, fmt, vmin, vmax

    @staticmethod
    def adjust_lightness(facecolor, amount=0.5):
        try:
            c = mc.cnames[facecolor]
        except:
            c = facecolor
        # c = colorsys.rgb_to_hls(*mpl_colors.to_rgb(c))
        c = colorsys.rgb_to_hls(*to_rgb(c))
        return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

    def subview_color_bar(self, fig, ax, handle, title=None, levels=None, norm=None, extend=None, orientation='vertical', pad=0.1, cax=None, shrink=None):
        kwargs = dict()
        if levels is not None:
            kwargs['ticks'] = levels
            if np.nanmax(levels) > 1000:
                fmt = ticker.FormatStrFormatter('%.2e')
                kwargs['format'] = fmt
        if norm is not None:
            kwargs['norm'] = norm
        if extend is not None:
            kwargs['extend'] = extend
        if shrink is not None:
            kwargs['shrink'] = shrink
        if cax is None:
            cbar = fig.colorbar(
                handle,
                ax=ax,
                orientation=orientation,
                pad=pad,
                **kwargs)
        else:
            cbar = fig.colorbar(
                handle,
                cax=cax,
                orientation=orientation,
                pad=pad,
                **kwargs)
        cbar.ax.tick_params(labelsize=self.ticksize)
        # if levels is not None:
        #     if len(levels) > 10:
        #         if orientation == 'vertical':
        #             ticklabels = cbar.ax.get_yticklabels()
        #         else:
        #             ticklabels = cbar.ax.get_xticklabels()
        #         cbar.set_ticklabels([ticklabel if i % 2 == 0 else '' for i, ticklabel in enumerate(ticklabels)])
        if title is not None:
            cbar.ax.set_title(title, fontsize=self.labelsize)
        return cbar

class ThreeDimensionalConfiguration(ColorConfiguration):

    def __init__(self, ticksize=7, labelsize=8, textsize=5, titlesize=9, headersize=10, cellsize=6):
        super().__init__(ticksize, labelsize, textsize, titlesize, headersize, cellsize)

    @staticmethod
    def get_dim3_figure_axes(nrows, ncols, figsize=None):
        row_loc = np.arange(nrows).astype(int) + 1
        col_loc = np.arange(ncols).astype(int) + 1
        nth_subplot = 0
        axes = []
        fig = plt.figure(figsize=figsize)
        for i in row_loc:
            for j in col_loc:
                nth_subplot += 1
                ax = fig.add_subplot(row_loc[-1], col_loc[-1], nth_subplot, projection='3d')
                axes.append(ax)
        axes = np.array(axes).reshape((nrows, ncols))
        return fig, axes

    def subview_contour_space(self, ax, X, Y, Z, norm, levels=None, cmap='Oranges', extremum_color='k', show_fills=False, show_lines=False, show_inline_labels=False, inline_fmt=None, scatter_args=None, **scatter_kwargs):
        if not any([show_fills, show_lines]):
            raise ValueError("cannot show contour-space without contour-fills or contour-lines")
        if (show_inline_labels and not show_lines):
            raise ValueError("cannot show inline-labels without showing lines")
        fill_kwargs = {
            'cmap' : cmap,
            'norm' : norm}
        line_kwargs = {
            'colors' : extremum_color,
            'linewidths' : 0.5}
        if levels is not None:
            fill_kwargs['levels'] = levels
            line_kwargs['levels'] = levels
        cbar_handles = []
        if show_fills:
            fill_handle = ax.contourf(
                X,
                Y,
                Z,
                **fill_kwargs)
            cbar_handles.append(fill_handle)
        if show_lines:
            line_handle = ax.contour(
                X,
                Y,
                Z,
                **line_kwargs)
            if show_inline_labels:
                if show_inline_labels:
                    inline_kwargs = {
                        'inline' : True,
                        'fontsize' : self.ticksize}
                    if inline_fmt is not None:
                        inline_kwargs['fmt'] = inline_fmt
                    ax.clabel(
                        line_handle,
                        **inline_kwargs)
            cbar_handles.append(line_handle)
        if scatter_args is not None:
            ax.scatter(
                *scatter_args,
                **scatter_kwargs)
        return cbar_handles[0]

    def subview_surface_space(self, ax, X, Y, Z, norm, levels=None, cmap='Oranges', extremum_color='k', show_lines=False, rstride=1, cstride=1, alpha=0.8, azim=-60, elev=30, scatter_args=None, **scatter_kwargs):
        surf_handle = ax.plot_surface(
            X,
            Y,
            Z,
            cmap=cmap,
            norm=norm,
            rstride=rstride,
            cstride=cstride,
            lw=0.5,
            alpha=alpha)
        # surf_handle._facecolors2d = surf_handle._facecolors3d
        # surf_handle._edgecolors2d = surf_handle._edgecolors3d
        if show_lines:
            fill_kwargs = {
                'linewidths' : 1,
                'linestyles' : 'solid',
                'cmap' : cmap,
                'norm' : norm,
                'offset' : -1}
            line_kwargs = {
                'linewidths' : 1,
                'linestyles' : 'solid',
                'colors' : extremum_color}
            if levels is not None:
                fill_kwargs['levels'] = levels
                line_kwargs['levels'] = levels
            ax.contour(
                X,
                Y,
                Z,
                **fill_kwargs)
            ax.contour(
                X,
                Y,
                Z,
                **line_kwargs)
        if scatter_args is not None:
            ax.scatter(
                *scatter_args,
                **scatter_kwargs)
        ax.elev = elev
        ax.azim = azim
        return surf_handle

    def share_dim3_axes(self, axes, xs, ys, zs, xlim=False, ylim=False, zlim=False, xticks=False, yticks=False, zticks=False, xfmt=None, yfmt=None, zfmt=None, xlabel=None, ylabel=None, zlabel=None, basex=None, basey=None, basez=None):
        for ax in axes.ravel():
            self.share_axes(
                axes=np.array([ax]),
                layout='overlay',
                xs=xs,
                ys=ys,
                sharex=True,
                sharey=True,
                xticks=xticks,
                yticks=yticks,
                xlim=xlim,
                ylim=ylim,
                xlabel=xlabel,
                ylabel=ylabel,
                xfmt=xfmt,
                yfmt=yfmt,
                basex=basex,
                basey=basey,
                collapse_x=False,
                collapse_y=False)
            self.share_axis_parameters(
                axis='z',
                axes=np.array([ax]),
                vs=zs,
                ticks=zticks,
                limits=zlim,
                label=zlabel,
                fmt=zfmt,
                log_base=basez)
        # for ax in axes.ravel():
        #     zticks = ax.get_zticks()
        #     if np.max(zticks) > 1000:
        #         zfmt = ticker.FormatStrFormatter('%.2e')
        #     else:
        #         zfmt = ticker.StrMethodFormatter('{x:,.3f}')
        #     ax.zaxis.set_minor_locator(ticker.AutoMinorLocator())
        #     ax.zaxis.set_major_formatter(zfmt)
        #     ax.tick_params(axis='z', labelsize=self.ticksize)

class VisualConfiguration(ThreeDimensionalConfiguration):

    def __init__(self, savedir=None, ticksize=7, labelsize=8, textsize=5, titlesize=9, headersize=10, cellsize=6):
        super().__init__(ticksize, labelsize, textsize, titlesize, headersize, cellsize)
        self.savedir = savedir

    @staticmethod
    def view_layout_permutations(f, layouts, *args, **kwargs):
        if isinstance(layouts, str):
            layouts = [layouts]
        for layout in layouts:
            kwargs['layout'] = layout
            f(*args, **kwargs)

    def get_number_of_figure_rows_and_columns(self, nseries, layout='overlay'):
        if (nseries == 1) and (layout != 'overlay'):
            raise ValueError("nseries=1 is only compatible with layout='overlay'")
        if (nseries % 2 != 0) and (layout == 'square'):
            raise ValueError("layout='square' is only compatible with nseries if nseries is odd")
        available_layouts = {
            'overlay' : dict(nrows=1, ncols=1),
            'horizontal' : dict(nrows=1, ncols=nseries),
            'vertical' : dict(nrows=nseries, ncols=1),
            'square' : dict(nrows=nseries//2, ncols=nseries//2)}
        result = available_layouts[layout]
        n = np.prod(list(result.values()))
        if (n != nseries) and (layout != 'overlay'):
            raise ValueError("{} series cannot be represented in {} axes".format(nseries, n))
        return available_layouts[layout]

    def display_image(self, fig, savename=None, dpi=800, bbox_inches='tight', pad_inches=0.1, extension='.png', **kwargs):
        if savename is None:
            plt.show()
        elif isinstance(savename, str):
            if self.savedir is None:
                raise ValueError("cannot save plot; self.savedir is None")
            savepath = '{}{}{}'.format(self.savedir, savename, extension)
            fig.savefig(savepath, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches, **kwargs)
        else:
            raise ValueError("invalid type(savename): {}".format(type(savename)))
        plt.close(fig)

    def display_fits_image(self, fig, savename=None, dpi=800, extension='.png', **kwargs):
        if savename is None:
            plt.show()
        elif isinstance(savename, str):
            if self.savedir is None:
                raise ValueError("cannot save plot; self.savedir is None")
            savepath = '{}{}{}'.format(self.savedir, savename, extension)
            fig.savefig(savepath, dpi=dpi, format=None, transparent=False, **kwargs)
        else:
            raise ValueError("invalid type(savename): {}".format(type(savename)))
        plt.close() # plt.close(fig)


##
