from resampling_configuration import *
from optimization_methods import *

class MaxSpectrum(ResamplingConfiguration):

    def __init__(self, vs, extreme_indices=None):
        super().__init__()
        self.vs = vs
        self.jmax = int(np.floor(np.log2(self.vs.size)))
        self.js = np.linspace(1, self.jmax, self.jmax, dtype=int)
        if extreme_indices is None:
            extreme_indices = np.arange(self.js.size).astype(int)
            self.xj = np.copy(self.js)
        else:
            self.xj = np.copy(self.js[extreme_indices])
        self.extreme_indices = extreme_indices
        self.ts = 2 ** self.js
        self.tj = 2 ** self.xj
        self._max_spectra = []
        self._alphas = []
        self._intercepts = []

    @property
    def max_spectra(self):
        return self._max_spectra

    @property
    def alphas(self):
        return self._alphas

    @property
    def intercepts(self):
        return self._intercepts

    @staticmethod
    def get_shapeshifted_data(vs, jprime):
        desired_size_factor = np.prod([n for n in jprime if n != -1])
        if -1 in jprime:
            desired_size = vs.size // desired_size_factor * desired_size_factor
        else:
            desired_size = desired_size_factor
        return np.copy(vs.flat[:desired_size]).reshape(jprime)

    @staticmethod
    def get_data_maxima(vs, jprime, value_to_replace=None):
        res = np.nanmax(vs, axis=1)
        if value_to_replace is not None:
            condition = (res != value_to_replace)
            res = np.copy(res[condition])
        return np.copy(res[res > 0])

    @staticmethod
    def get_parameter_guess(x, y):
        dy = y[-1] - y[0]
        dx = x[-1] - x[0]
        m = dy / dx
        b = np.mean([y[idx] - m * x[idx] for idx in (0, -1)])
        return (m, b)

    def get_full_spectrum(self, vs, value_to_replace=None, ddof=0):
        keys = ('dlog', 'npad', 'standard deviation', 'standard error', 'yj initial')
        result = {key : [] for key in keys}
        for j in self.js:
            jprime = (-1, 2**j)
            v = self.get_shapeshifted_data(
                vs=vs,
                jprime=jprime)
            v = self.get_data_maxima(
                vs=v,
                jprime=jprime,
                value_to_replace=value_to_replace)
            dlog = np.log2(v)
            st_dev = np.std(dlog, ddof=ddof)
            st_err = sem(dlog, ddof=ddof)
            yj_init = np.sum(dlog) / dlog.size # np.mean(dlog) but dlog is ragged
            for key, args in zip(keys, (dlog, dlog.size, st_dev, st_err, yj_init)):
                result[key].append(args)
        return {key : np.array(args) for key, args in result.items()}

    def logarithmic_power_law_equation(self, prms):
        return prms[0] * self.xj + prms[1] ## line on logarithmic axes ==> power-law

    def get_optimized_spectrum(self, max_spectrum, prms_guess=None, scale='local', **optimization_kwargs):
        yj = np.copy(max_spectrum['yj initial'][self.extreme_indices])
        wts = np.copy(max_spectrum['npad'][self.extreme_indices])
        # wts = 1 / np.copy(max_spectrum['npad'][self.extreme_indices])
        GLS = GeneralizedLeastSquaresEstimator(
            x=self.xj,
            y=yj,
            f=self.logarithmic_power_law_equation,
            wts=wts)
        if prms_guess is None:
            prms_guess = self.get_parameter_guess(
                x=GLS.x,
                y=GLS.y)
        GLS.fit(
            x0=prms_guess,
            scale=scale,
            **optimization_kwargs)
        opt_prms = GLS.extrema[scale]['x']
        inv_slope = 1 / opt_prms[0]
        intercept = opt_prms[1]
        yj_fit = GLS.f(opt_prms)
        return (inv_slope, intercept), yj_fit

    def update_max_spectra(self, vs, prms_guess=None, value_to_replace=None, ddof=0, scale='local', **optimization_kwargs):
        og_spectrum = self.get_full_spectrum(
            vs=vs,
            value_to_replace=value_to_replace,
            ddof=ddof)
        (alpha, intercept), yj_fit = self.get_optimized_spectrum(
            max_spectrum=og_spectrum,
            prms_guess=prms_guess,
            scale=scale,
            **optimization_kwargs)
        og_spectrum['yj fit'] = yj_fit
        self._max_spectra.append(og_spectrum)
        self._alphas.append(alpha)
        self._intercepts.append(intercept)

class ExtremalIndex(MaxSpectrum):

    def __init__(self, vs, extreme_indices=None):
        super().__init__(
            vs=vs,
            extreme_indices=extreme_indices)
        self._thetas = None
        self._point_estimators = dict()
        self._histograms = dict()
        self._extreme_value_estimates = dict()

    @property
    def thetas(self):
        return self._thetas

    @property
    def point_estimators(self):
        return self._point_estimators

    @property
    def histograms(self):
        return self._histograms

    @property
    def extreme_value_estimates(self):
        return self._extreme_value_estimates

    def update_spectra_by_resampling(self, prms_guess=None, nresamples=1000, nshuffles=3, with_replacement=False, value_to_replace=None, ddof=0, scale='local', **optimization_kwargs):
        if len(self.max_spectra) == 0:
            raise ValueError("one should only run this method after running the max spectrum routine withOUT resampling")
        self.update_resampling_criteria(
            nresamples=nresamples,
            nshuffles=nshuffles,
            with_replacement=with_replacement)
        self.update_resampling_indices(
            n=self.vs.size)
        for i in range(self.nresamples):
            self.resample_indices()
            self.update_max_spectra(
                vs=np.copy(self.vs[self.resampled_indices]),
                prms_guess=prms_guess,
                value_to_replace=value_to_replace,
                ddof=ddof,
                scale=scale,
                **optimization_kwargs)
        self._alphas = np.array(self._alphas)
        self._intercepts = np.array(self._intercepts)

    def update_thetas(self):
        if len(self.max_spectra) <= 1:
            raise ValueError("one should only run this method after running the max spectrum routine with and without resampling")
        exponents = []
        for max_spectrum, alpha in zip(self.max_spectra[1:], self.alphas[1:]):
            dy = self.max_spectra[0]['yj fit'] - max_spectrum['yj fit']
            exponents.append(dy.T * alpha)
        self._thetas = 2 ** np.array(exponents)

    def update_point_estimators(self):
        for f, key in zip((np.mean, np.median, np.min, np.max), ('mean', 'median', 'minimum', 'maximum')):
            self._point_estimators[key] = f(self.thetas, axis=0)

    def get_histogram(self, parameter, edges=None, nbins=None, wbin=None, midpoints=None, bin_widths=None, criteria=None, lbin=None, rbin=None, squeeze_trails=False, squeeze_leads=False, tol=0, hbias='left', bias=False, fisher=False, ddof=0, nan_policy='propagate'):
        if parameter not in ('alpha', 'intercept', 'theta'):
            raise ValueError("invalid parameter: {}".format(parameter))
        vs = np.copy(getattr(self, '{}s'.format(parameter)))
        if parameter == 'theta':
            vs = vs.reshape(-1)
        histogram = Histogram(vs, hbias=hbias)
        histogram.update_statistics(
            bias=bias,
            fisher=fisher,
            ddof=ddof,
            nan_policy=nan_policy)
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
            squeeze_trails=squeeze_trails,
            squeeze_leads=squeeze_leads,
            tol=tol)
        return histogram

    def update_alpha_histogram(self, edges=None, nbins=None, wbin=None, midpoints=None, bin_widths=None, criteria=None, lbin=None, rbin=None, squeeze_trails=False, squeeze_leads=False, tol=0, hbias='left', bias=False, fisher=False, ddof=0, nan_policy='propagate'):
        histogram = self.get_histogram(
            parameter='alpha',
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
            nan_policy=nan_policy)
        self._histograms['alpha'] = histogram

    def update_intercept_histogram(self, edges=None, nbins=None, wbin=None, midpoints=None, bin_widths=None, criteria=None, lbin=None, rbin=None, squeeze_trails=False, squeeze_leads=False, tol=0, hbias='left', bias=False, fisher=False, ddof=0, nan_policy='propagate'):
        histogram = self.get_histogram(
            parameter='intercept',
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
            nan_policy=nan_policy)
        self._histograms['intercept'] = histogram

    def update_theta_histogram(self, edges=None, nbins=None, wbin=None, midpoints=None, bin_widths=None, criteria=None, lbin=None, rbin=None, squeeze_trails=False, squeeze_leads=False, tol=0, hbias='left', bias=False, fisher=False, ddof=0, nan_policy='propagate'):
        histogram = self.get_histogram(
            parameter='theta',
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
            nan_policy=nan_policy)
        self._histograms['theta'] = histogram

    def update_extreme_value_estimates(self):
        yj = np.array([max_spectrum['yj fit'] for max_spectrum in self.max_spectra])
        for f, key in zip((np.mean, np.median), ('mean', 'median')):
            values = f(yj, axis=0)
            self._extreme_value_estimates[key] = int(np.min(2**values))

class UnbiasedEstimators(ExtremalIndex):

    def __init__(self, vs, extreme_parameter, extreme_indices=None):
        super().__init__(
            vs=vs,
            extreme_indices=extreme_indices)
        self.extreme_parameter = extreme_parameter

    def __repr__(self):
        return 'UnbiasedEstimators(%r, %r)' % (self.vs, self.extreme_indices)

    def __str__(self):
        return '\n ** Unbiased Estimators ** \n results via %s resamples; with_replacement=%s \n .. mean(alpha) = %s \n .. mean(intercept) = %s \n .. mean(theta) = %s \n' % (
            self.nresamples,
            self.with_replacement,
            self.histograms['alpha'].mean,
            self.histograms['intercept'].mean,
            self.histograms['theta'].mean)

    def __call__(self, prms_guess=None, nresamples=1000, nshuffles=3, with_replacement=False, value_to_replace=None, ddof=0, alpha_histogram_kwargs=None, intercept_histogram_kwargs=None, theta_histogram_kwargs=None, scale='local', **optimization_kwargs):
        self.update_max_spectra(
            vs=self.vs,
            prms_guess=prms_guess,
            value_to_replace=value_to_replace,
            ddof=ddof,
            scale=scale,
            **optimization_kwargs)
        self.update_spectra_by_resampling(
            prms_guess=prms_guess,
            nresamples=nresamples,
            nshuffles=nshuffles,
            with_replacement=with_replacement,
            value_to_replace=value_to_replace,
            ddof=ddof,
            scale=scale,
            **optimization_kwargs)
        self.update_thetas()
        self.update_point_estimators()
        if alpha_histogram_kwargs is not None:
            self.update_alpha_histogram(
                **alpha_histogram_kwargs)
        if intercept_histogram_kwargs is not None:
            self.update_intercept_histogram(
                **intercept_histogram_kwargs)
        if theta_histogram_kwargs is not None:
            self.update_theta_histogram(
                **theta_histogram_kwargs)
        self.update_extreme_value_estimates()






##
