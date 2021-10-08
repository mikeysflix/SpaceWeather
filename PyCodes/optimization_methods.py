from scipy.integrate import quad
from scipy.optimize import minimize, basinhopping, OptimizeResult
from histogram_configuration import *

class Errorspace():

    def __init__(self):
        super().__init__()
        self._xe = None
        self._ye = None
        self._ze = None
        self._Xe = None
        self._Ye = None
        self._Ze = None

    @property
    def xe(self):
        return self._xe

    @property
    def ye(self):
        return self._ye

    @property
    def ze(self):
        return self._ze

    @property
    def Xe(self):
        return self._Xe

    @property
    def Ye(self):
        return self._Ye

    @property
    def Ze(self):
        return self._Ze

    @staticmethod
    def get_parameter_space(prm, frac=1, n=None, w=None, apply_non_negative=False):
        if (n is None) and (w is None):
            raise ValueError("input either 'n' or 'w'")
        if (n is not None) and (w is not None):
            raise ValueError("input either 'n' or 'w'")
        delta = prm * frac
        pmin, pmax = prm - delta, prm + delta
        if apply_non_negative:
            if pmin < 0:
                pmin = 0
        if n is not None:
            pspace = np.linspace(pmin, pmax, n)
        elif w is not None:
            pspace = np.arange(pmin, pmax + w, w)
        return pspace

    def get_zspace(self, zfunc):
        z = np.array([
            zfunc([xi, yi]) for xi, yi in zip(self.Xe.reshape(-1), self.Ye.reshape(-1))])
        Z = z.reshape(self.Xe.shape)
        return z, Z

    def update_error_space(self, prms, zfunc, xfrac=1, yfrac=1, xn=None, yn=None, xw=None, yw=None, xneg=False, yneg=False, x=None, y=None):
        xprm, yprm = prms
        if x is None:
            x = self.get_parameter_space(
                prm=xprm,
                frac=xfrac,
                n=xn,
                w=xw,
                apply_non_negative=xneg)
        self._xe = x
        if y is None:
            y = self.get_parameter_space(
                prm=yprm,
                frac=yfrac,
                n=yn,
                w=yw,
                apply_non_negative=yneg)
        self._ye = y
        X, Y = np.meshgrid(x, y)
        self._Xe = X
        self._Ye = Y
        z, Z = self.get_zspace(zfunc)
        self._ze = z
        self._Ze = Z

class ExtremizerConfiguration(Errorspace):

    def __init__(self):
        super().__init__()
        self._extrema = dict()
        self._error_func = None
        self._wts = None

    @property
    def extrema(self):
        return self._extrema

    @property
    def error_func(self):
        return self._error_func

    @property
    def wts(self):
        return self._wts

    def search_for_local_extremum(self, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
        result = minimize(
            self.error_func,
            x0=x0,
            args=args,
            method=method,
            jac=jac,
            hess=hess,
            hessp=hessp,
            bounds=bounds,
            constraints=constraints,
            tol=tol,
            callback=callback,
            options=options)
        return result

    def search_for_global_extremum(self, x0, niter=100, T=1.0, stepsize=0.5, minimizer_kwargs=None, take_step=None, accept_test=None, callback=None, interval=50, disp=False, niter_success=None, seed=None):
        result = basinhopping(
            self.error_func,
            x0=x0,
            niter=niter,
            T=T,
            stepsize=stepsize,
            minimizer_kwargs=minimizer_kwargs,
            take_step=take_step,
            accept_test=accept_test,
            callback=callback,
            interval=interval,
            disp=disp,
            niter_success=niter_success,
            seed=seed)
        return result.lowest_optimization_result

    def fit(self, x0, scale='local', **kwargs):
        if scale in list(self.extrema.keys()):
            raise ValueError("minimization corresponding to scale='{}' has already been initialized".format(scale))
        if self.error_func is None:
            raise ValueError("error_func is not initialized")
        if scale == 'local':
            result = self.search_for_local_extremum(
                x0=x0,
                **kwargs)
        elif scale == 'global':
            result = self.search_for_global_extremum(
                x0=x0,
                **kwargs)
        else:
            raise ValueError("invalid scale: {}".format(scale))
        if not result.success:
            raise ValueError("optimization was NOT successful; see output below\n{}".format(result))
        self._extrema[scale] = result

class GeneralizedLeastSquaresEstimator(ExtremizerConfiguration):

    def __init__(self, x, y, f, wts=False, tol=1e-9):
        super().__init__()
        self.x = x
        self.y = y
        self.f = f
        self._wts = wts
        self.initialize_error_func(wts, tol)

    @staticmethod
    def get_weights_from_residuals(residuals, tol=1e-9):
        total_error = np.sum(residuals)
        if total_error < tol:
            return np.ones(residuals.size)
        else:
            return 1 / residuals

    def get_unweighted_residuals(self, prms):
        return np.square(self.y - self.f(prms))

    @staticmethod
    def get_weights_from_residuals(residuals, tol=1e-9):
        total_error = np.sum(residuals)
        if total_error < tol:
            return np.ones(residuals.size)
        else:
            return 1 / residuals

    def initialize_error_func(self, wts=False, tol=1e-9):
        if isinstance(wts, bool):
            if wts:
                f1 = lambda prms : self.get_unweighted_residuals(prms)
                f2 = lambda prms : self.get_weights_from_residuals(f1(prms), tol)
                error_func = lambda prms : np.sum(f1(prms) / f2(prms))
            else:
                error_func = lambda prms : np.sum(self.get_unweighted_residuals(prms))
        elif isinstance(wts, np.ndarray):
            error_func = lambda prms : np.sum(self.get_unweighted_residuals(prms) * wts)
        else:
            raise ValueError("invalid type(wts): {}".format(type(wts)))
        self._error_func = error_func

class MaximumLikelihoodEstimator(ExtremizerConfiguration):

    def __init__(self, vs, pdf, wts=None):
        super().__init__()
        self.vs = vs
        self.pdf = pdf
        self._wts = wts
        self.initialize_error_func(wts)

    def __repr__(self):
        return 'MaximumLikelihoodEstimator(%r, %r, %r)' % (
            self.vs,
            self.pdf,
            self.wts)

    def __str__(self):
        s = '\n ** Maximum Likelihood Estimator ** \n'
        for scale in ('local', 'global'):
            try:
                optimization_result = self.extrema[scale]
                s += '\n results via %s optimizer \n\n .. prms = (%s) \n .. function extremum = %s \n' % (
                    scale,
                    optimization_result.x,
                    optimization_result.fun)
            except:
                pass
        return s

    @property
    def wts(self):
        return self._wts

    def get_log_likelihood(self, prms):
        return np.log(self.pdf(prms))

    def initialize_error_func(self, wts=None):
        if wts is None:
            error_func = lambda prms : -1 * np.sum(self.get_log_likelihood(prms))
        else:
            if not isinstance(wts, np.ndarray):
                raise ValueError("invalid type(wts): {}".format(type(wts)))
            if wts.size != self.vs.size:
                raise ValueError("weights of shape={} are incompatible with values of shape={}".format(wts.shape, self.vs.shape))
            error_func = lambda prms : -1 * np.sum(self.get_log_likelihood(prms) * wts)
        self._error_func = error_func

class BinStatisticEstimator(ExtremizerConfiguration):

    def __init__(self, histogram, pdf, nparameters):
        super().__init__()
        self.histogram = histogram
        self.pdf = pdf
        self.integrable_pdf = lambda x, prms : pdf(prms, x)
        nconstraints = nparameters + 1
        self.dof = histogram.vs.size - nconstraints

    def get_pvalue_from_statistic(self, reduced_statistic):
        return SPstats.distributions.chi2.sf(reduced_statistic, self.dof)

    def get_expected_counts(self, prms):
        expected_counts = []
        for lbound, ubound in zip(self.histogram.edges[:-1], self.histogram.edges[1:]):
            ec = quad(self.integrable_pdf, lbound, ubound, prms)[0] * self.histogram.vs.size
            expected_counts.append(ec)
        return np.array(expected_counts)

class ChiSquareEstimator(BinStatisticEstimator):

    def __init__(self, histogram, pdf, nparameters, error_statistic_id='chi square'):
        super().__init__(
            histogram=histogram,
            pdf=pdf,
            nparameters=nparameters)
        self.error_statistic_id = error_statistic_id
        self.initialize_error_func(
            error_statistic_id=error_statistic_id)

    def __repr__(self):
        return 'ChiSquareEstimator(%r, %r, %r, %r)' % (
            self.histogram,
            self.pdf,
            self.nparameters,
            self.error_statistic_id)

    def __str__(self):
        s = '\n ** Chi-Square Estimator ** \n'
        for scale in ('local', 'global'):
            try:
                optimization_result = self.extrema[scale]
                s += '\n results via %s optimizer \n\n .. prms = (%s) \n .. function extremum = %s \n' % (
                    scale,
                    optimization_result.x,
                    optimization_result.fun)
            except:
                pass
        return s

    def get_bin_statistic(self, prms):
        expected_counts = self.get_expected_counts(prms)
        csq = np.sum(np.square(self.histogram.counts - expected_counts) / expected_counts)
        return csq

    def get_reduced_statistic(self, prms):
        csq = self.get_bin_statistic(prms)
        return csq / self.dof

    def get_pvalue(self, prms):
        reduced_statistic = self.get_reduced_statistic(prms)
        return self.get_pvalue_from_statistic(reduced_statistic)

    def initialize_error_func(self, error_statistic_id):
        if error_statistic_id == 'chi square':
            self._error_func = lambda prms : self.get_bin_statistic(prms)
        elif error_statistic_id == 'reduced chi square':
            self._error_func = lambda prms : self.get_reduced_statistic(prms)
        elif error_statistic_id == 'p-value':
            self._error_func = lambda prms : -1 * self.get_pvalue(prms)
        else:
            raise ValueError("invalid error_statistic_id: {}".format(error_statistic_id))

class GTestEstimator(BinStatisticEstimator):

    def __init__(self, histogram, pdf, nparameters, error_statistic_id='g-test'):
        super().__init__(
            histogram=histogram,
            pdf=pdf,
            nparameters=nparameters)
        self.error_statistic_id = error_statistic_id
        self.initialize_error_func(
            error_statistic_id=error_statistic_id)

    def __repr__(self):
        return 'GTestEstimator(%r, %r, %r, %r)' % (
            self.histogram,
            self.pdf,
            self.nparameters,
            self.error_statistic_id)

    def __str__(self):
        s = '\n ** G-Test Estimator ** \n'
        for scale in ('local', 'global'):
            try:
                optimization_result = self.extrema[scale]
                s += '\n results via %s optimizer \n\n .. prms = (%s) \n .. function extremum = %s \n' % (
                    scale,
                    optimization_result.x,
                    optimization_result.fun)
            except:
                pass
        return s

    def get_bin_statistic(self, prms):
        expected_counts = self.get_expected_counts(prms)
        gte = 2 * np.sum(self.histogram.counts * np.log(self.histogram.counts / expected_counts))
        return gte

    def get_reduced_statistic(self, prms):
        gte = self.get_bin_statistic(prms)
        return gte / self.dof

    def get_pvalue(self, prms):
        reduced_statistic = self.get_reduced_statistic(prms)
        return self.get_pvalue_from_statistic(reduced_statistic)

    def initialize_error_func(self, error_statistic_id):
        if error_statistic_id == 'g-test':
            self._error_func = lambda prms : self.get_bin_statistic(prms)
        elif error_statistic_id == 'reduced g-test':
            self._error_func = lambda prms : self.get_reduced_statistic(prms)
        elif error_statistic_id == 'p-value':
            self._error_func = lambda prms : -1 * self.get_pvalue(prms)
        else:
            raise ValueError("invalid error_statistic_id: {}".format(error_statistic_id))















##
