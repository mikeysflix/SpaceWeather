from scipy.special import gamma as f_gamma, gammainc as f_gamma_lower_incomplete
import statsmodels.api as sm
from optimization_methods import *

class NonParametricDistributionConfiguration():

    def __init__(self, data, extreme_parameter):
        super().__init__()
        self.data = data
        self.extreme_parameter = extreme_parameter
        self.us = data[extreme_parameter]
        self.vs = np.sort(self.us)
        self._original_histogram = None
        self._threshold_histogram = None
        self._kernel_density_estimation = None

    @property
    def original_histogram(self):
        return self._original_histogram

    @property
    def threshold_histogram(self):
        return self._threshold_histogram

    @property
    def kernel_density_estimation(self):
        return self._kernel_density_estimation

    def update_histogram(self, edges=None, nbins=None, wbin=None, midpoints=None, bin_widths=None, criteria=None, lbin=None, rbin=None, threshold=None, merge_condition='less than', squeeze_trails=False, squeeze_leads=False, tol=0, hbias='left', bias=False, fisher=False, ddof=0, nan_policy='propagate'):
        original_histogram = Histogram(
            vs=self.vs,
            hbias=hbias)
        original_histogram.update_edges(
            edges=edges,
            nbins=nbins,
            wbin=wbin,
            midpoints=midpoints,
            bin_widths=bin_widths,
            criteria=criteria,
            lbin=lbin,
            rbin=rbin)
        original_histogram.update_statistics(
            bias=bias,
            fisher=fisher,
            ddof=ddof,
            nan_policy=nan_policy)
        original_histogram.update_counts(
            threshold=None,
            merge_condition=merge_condition,
            squeeze_trails=squeeze_trails,
            squeeze_leads=squeeze_leads,
            tol=tol,
            extra_counts=True)
        threshold_histogram = deepcopy(original_histogram)
        if threshold is not None:
            threshold_histogram.apply_threshold_to_bin_frequency(
                threshold=threshold,
                merge_condition=merge_condition,
                extra_counts=True)
        self._original_histogram = original_histogram
        self._threshold_histogram = threshold_histogram

    def update_kernel_density_estimation(self, bandwidths, kernels='gaussian'):
        if not isinstance(bandwidths, (tuple, list, np.ndarray)):
            bandwidths = [bandwidths]
        if not isinstance(kernels, (tuple, list, np.ndarray)):
            kernels = [kernels]
        x = np.linspace(self.vs[0], self.vs[-1], 100)
        for kernel in kernels:
            if kernel == 'gaussian':
                for bandwidth in bandwidths:
                    kde = SPstats.gaussian_kde(
                        self.vs,
                        bw_method=bandwidth)
                    y = kde(x)
                    result = {
                        'bandwidth' : str(bandwidth),
                        'kernel' : kernel,
                        'kde' : kde,
                        'x' : x,
                        'y' : y}
                    if self.kernel_density_estimation is None:
                        self._kernel_density_estimation = [result]
                    else:
                        self._kernel_density_estimation.append(result)
            else:
                raise ValueError("invalid kernel: {}; not yet implemented".format(kernel))
        self._kernel_density_estimation = sorted(self._kernel_density_estimation, key=lambda d : d['bandwidth'])

class ParametricDistributionConfiguration(NonParametricDistributionConfiguration):

    def __init__(self, data, extreme_parameter, distribution_id, nparameters, pdf=None, cdf=None):
        super().__init__(
            data=data,
            extreme_parameter=extreme_parameter)
        self.distribution_id = distribution_id
        self.nparameters = nparameters
        self.pdf = pdf
        self.cdf = cdf
        self._chi_square = None
        self._g_test = None
        self._maximum_likelihood = None
        self._normality_assessment = None
        self._kolmogorov_smirnov_assessment = None
        self._optimizer_ids = []
        self._sub_series = []
        self._container_of_search_kwargs = []

    @property
    def chi_square(self):
        return self._chi_square

    @property
    def g_test(self):
        return self._g_test

    @property
    def maximum_likelihood(self):
        return self._maximum_likelihood

    @property
    def normality_assessment(self):
        return self._normality_assessment

    @property
    def kolmogorov_smirnov_assessment(self):
        return self._kolmogorov_smirnov_assessment

    @property
    def optimizer_ids(self):
        return self._optimizer_ids

    @property
    def sub_series(self):
        return self._sub_series

    @property
    def container_of_search_kwargs(self):
        return self._container_of_search_kwargs

    @staticmethod
    def convert_from_lognormal_to_normal_parameters(prms):
        ## lognormal --> normal
        m, s = prms
        variance = s**2
        tmp = variance / m**2 + 1
        mu = np.log(m / np.sqrt(tmp))
        sigma = np.sqrt(np.log(tmp))
        return (mu, sigma)

    @staticmethod
    def convert_from_normal_to_lognormal_parameters(prms):
        ## normal --> lognormal
        mu, sigma = prms
        m = np.exp(mu + sigma**2 / 2)
        sq_sigma = sigma **2
        s = np.sqrt(np.exp(2 * mu + sq_sigma) * (np.exp(sq_sigma) - 1))
        return (m, s)

    def get_optimization_parameter_guess(self, initial_parameter_guess=None, ddof=0):
        if initial_parameter_guess is None:
            if self.distribution_id == 'normal':
                a = np.nanmean(self.vs)
                b = np.nanstd(self.vs, ddof=ddof)
                return [a, b]
            elif self.distribution_id == 'lognormal':
                a = np.nanmean(self.vs)
                b = np.nanstd(self.vs, ddof=ddof)
                lognormal_prms = [a, b]
                return self.convert_from_lognormal_to_normal_parameters(lognormal_prms)
            else:
                raise ValueError("invalid distribution_id: {}".format(self.distribution_id))
        else:
            if len(initial_parameter_guess) != self.nparameters:
                raise ValueError("invalid initial_parameter_guess: {}".format(initial_parameter_guess))
            return initial_parameter_guess

    def get_preliminary_optimization_result(self, optimizer, scale):
        calculation_prms = np.copy(optimizer.extrema[scale]['x'])
        if self.distribution_id == 'normal':
            true_prms = np.copy(calculation_prms)
            additional_prms = None
            label_mapping = {
                'mu' : r'$\mu_{opt}$ $=$ ' + r'${:,.2f}$'.format(true_prms[0]),
                'sigma' : r'$\sigma_{opt}$ $=$ ' + r'${:,.2f}$'.format(true_prms[1])}
        elif self.distribution_id == 'lognormal':
            true_prms = self.convert_from_normal_to_lognormal_parameters(calculation_prms)
            (true_mu, true_sigma) = true_prms
            # true_median = np.sqrt((np.exp(np.square(true_sigma) - 1)) * (np.exp(2 * true_mu + np.square(true_sigma))))
            # true_mode = np.exp(true_mu - np.square(true_sigma))
            (calculation_mu, calculation_sigma) = calculation_prms
            true_median = np.sqrt((np.exp(np.square(calculation_sigma) - 1)) * (np.exp(2 * calculation_mu + np.square(calculation_sigma))))
            true_mode = np.exp(calculation_mu - np.square(calculation_sigma))
            additional_prms = {
                'median' : true_median,
                'mode' : true_mode}
            label_mapping = {
                'mu' : r'$\mu_{opt}$ $=$ ' + r'${:,.2f}$'.format(true_prms[0]),
                'sigma' : r'$\sigma_{opt}$ $=$ ' + r'${:,.2f}$'.format(true_prms[1]),
                'median' : r'$median_{opt}$ $=$ ' + r'${:,.2f}$'.format(true_median),
                'mode' : r'$mode_{opt}$ $=$ ' + r'${:,.2f}$'.format(true_mode)}
        else:
            raise ValueError("invalid distribution_id: {}".format(self.distribution_id))
        probability_density = self.pdf(calculation_prms)
        observed_frequency = probability_density * self.vs.size
        if self.threshold_histogram is None:
            observed_density = None
        else:
            observed_density = probability_density * self.threshold_histogram.normalization_constant
        result = {
            'error statistic id' : 'negative log-likelihood',
            'probability density' : probability_density,
            'observed frequency' : observed_frequency,
            'observed density' : observed_density,
            'true parameters' : true_prms,
            'calculation parameters' : calculation_prms,
            'additional parameters' : additional_prms,
            'optimizer' : optimizer}
        return result, label_mapping

    def update_chi_square(self, initial_parameter_guess=None, error_statistic_id='reduced chi square', scale='local', **kwargs):
        if self.threshold_histogram is None:
            raise ValueError("this method requires the initialization of a histogram")
        x0 = self.get_optimization_parameter_guess(
            initial_parameter_guess=initial_parameter_guess)
        optimizer = ChiSquareEstimator(
            histogram=self.threshold_histogram,
            pdf=self.pdf,
            nparameters=self.nparameters,
            error_statistic_id=error_statistic_id)
        optimizer.fit(
            x0=x0,
            scale=scale,
            **kwargs)
        result, label_mapping = self.get_preliminary_optimization_result(
            optimizer=optimizer,
            scale=scale)
        if error_statistic_id == 'p-value':
            pvalue = optimizer.extrema[scale]['fun']
            reduced_statistic = optimizer.get_reduced_statistic(result['calculation parameters'])
        elif error_statistic_id == 'reduced chi square':
            reduced_statistic = optimizer.extrema[scale]['fun']
            pvalue = optimizer.get_pvalue_from_statistic(reduced_statistic)
        else:
            reduced_statistic = optimizer.get_reduced_statistic(result['calculation parameters'])
            pvalue = optimizer.get_pvalue_from_statistic(reduced_statistic)
        statistic_label = r"$\chi^{%s}_{\nu=%s}$ $=$ ${%s}$" % (2, '{:,}'.format(optimizer.dof), '{:,.2f}'.format(reduced_statistic))
        pvalue_label = r'p-value $=$ ${:.3}$'.format(pvalue)
        label_mapping['fun'] = '{}\n{}'.format(statistic_label, pvalue_label)
        result['labels'] = label_mapping
        result['fun'] = reduced_statistic
        self._chi_square = result
        self._optimizer_ids.append('chi square')

    def update_g_test(self, initial_parameter_guess=None, error_statistic_id='reduced g-test', scale='local', **kwargs):
        if self.threshold_histogram is None:
            raise ValueError("this method requires the initialization of a histogram")
        x0 = self.get_optimization_parameter_guess(
            initial_parameter_guess=initial_parameter_guess)
        optimizer = GTestEstimator(
            histogram=self.threshold_histogram,
            pdf=self.pdf,
            nparameters=self.nparameters,
            error_statistic_id=error_statistic_id)
        optimizer.fit(
            x0=x0,
            scale=scale,
            **kwargs)
        result, label_mapping = self.get_preliminary_optimization_result(
            optimizer=optimizer,
            scale=scale)
        if error_statistic_id == 'p-value':
            pvalue = optimizer.extrema[scale]['fun']
            reduced_statistic = optimizer.get_reduced_statistic(result['calculation parameters'])
        elif error_statistic_id == 'reduced chi square':
            reduced_statistic = optimizer.extrema[scale]['fun']
            pvalue = optimizer.get_pvalue_from_statistic(reduced_statistic)
        else:
            reduced_statistic = optimizer.get_reduced_statistic(result['calculation parameters'])
            pvalue = optimizer.get_pvalue_from_statistic(reduced_statistic)
        statistic_label = r"$G_{\nu=%s}$ $=$ ${%s}$" % ('{:,}'.format(optimizer.dof), '{:,.2f}'.format(reduced_statistic))
        pvalue_label = r'p-value $=$ ${:.3}$'.format(pvalue)
        label_mapping['fun'] = '{}\n{}'.format(statistic_label, pvalue_label)
        result['labels'] = label_mapping
        result['fun'] = reduced_statistic
        self._g_test = result
        self._optimizer_ids.append('g-test')

    def update_maximum_likelihood(self, initial_parameter_guess=None, scale='local', **kwargs):
        x0 = self.get_optimization_parameter_guess(
            initial_parameter_guess=initial_parameter_guess)
        optimizer = MaximumLikelihoodEstimator(
            vs=self.vs,
            pdf=self.pdf)
        optimizer.fit(
            x0=x0,
            scale=scale,
            **kwargs)
        result, label_mapping = self.get_preliminary_optimization_result(
            optimizer=optimizer,
            scale=scale)
        label_mapping['fun'] = r'$log(L)_{max}$ $=$ ' + r'${:,.2f}$'.format(optimizer.extrema[scale]['fun'])
        result['labels'] = label_mapping
        result['fun'] = optimizer.extrema[scale]['fun']
        self._maximum_likelihood = result
        self._optimizer_ids.append('maximum likelihood')

    def update_chi_square_error_space(self, xfrac=1, yfrac=1, xn=None, yn=None, xw=None, yw=None, xneg=False, yneg=False, x=None, y=None):
        optimizer = self.chi_square['optimizer']
        prms = self.chi_square['calculation parameters']
        optimizer.update_error_space(
            prms=prms,
            zfunc=optimizer.error_func,
            xfrac=xfrac,
            yfrac=yfrac,
            xn=xn,
            yn=yn,
            xw=xw,
            yw=yw,
            xneg=xneg,
            yneg=yneg,
            x=x,
            y=y)
        self._chi_square['optimizer'] = optimizer

    def update_g_test_error_space(self, xfrac=1, yfrac=1, xn=None, yn=None, xw=None, yw=None, xneg=False, yneg=False, x=None, y=None):
        optimizer = self.g_test['optimizer']
        prms = self.g_test['calculation parameters']
        optimizer.update_error_space(
            prms=prms,
            zfunc=optimizer.error_func,
            xfrac=xfrac,
            yfrac=yfrac,
            xn=xn,
            yn=yn,
            xw=xw,
            yw=yw,
            xneg=xneg,
            yneg=yneg,
            x=x,
            y=y)
        self._g_test['optimizer'] = optimizer

    def update_maximum_likelihood_error_space(self, xfrac=1, yfrac=1, xn=None, yn=None, xw=None, yw=None, xneg=False, yneg=False, x=None, y=None):
        optimizer = self.maximum_likelihood['optimizer']
        prms = self.maximum_likelihood['calculation parameters']
        optimizer.update_error_space(
            prms=prms,
            zfunc=optimizer.error_func,
            xfrac=xfrac,
            yfrac=yfrac,
            xn=xn,
            yn=yn,
            xw=xw,
            yw=yw,
            xneg=xneg,
            yneg=yneg,
            x=x,
            y=y)
        self._maximum_likelihood['optimizer'] = optimizer

    def update_kolmogorov_smirnov_assessment(self):
        if len(self.optimizer_ids) < 1:
            raise ValueError("optimizer_ids is not initialized")
        result = dict()
        for optimizer_id in self.optimizer_ids:
            _optimizer_id = optimizer_id.replace(' ', '_').replace('-', '_')
            optimization_result = getattr(self, _optimizer_id)
            (mu, sigma) = optimization_result['calculation parameters']
            # (mu, sigma) = optimization_result['true parameters']
            if self.distribution_id == 'normal':
                data = SPstats.kstest(
                    self.vs,
                    cdf='norm',
                    args=(mu, sigma),
                    alternative='two-sided',
                    mode='asymp')
            elif self.distribution_id == 'lognormal':
                data = SPstats.kstest(
                    self.vs,
                    cdf='lognorm',
                    args=(sigma, 0, np.exp(mu)),
                    alternative='two-sided',
                    mode='asymp')
            else:
                raise ValueError("invalid distribution_id: {}".format(self.distribution_id))
            result[optimizer_id] = data
        self._kolmogorov_smirnov_assessment = result

    def update_normality_assessment(self):
        shapiro_wilk_result = SPstats.shapiro(
            self.vs)
        anderson_darling_result = SPstats.anderson(
            self.vs,
            dist='norm')
        dagostino_pearson_result = SPstats.normaltest(
            self.vs,
            nan_policy='omit')
        normality_assessment = {
            "Shapiro-Wilk Test" : shapiro_wilk_result,
            "Anderson-Darling Test" : anderson_darling_result,
            "D'Agostino-Pearson Test" : dagostino_pearson_result}
        self._normality_assessment = normality_assessment

    def update_base_analysis(self, kernel_density_estimation_kwargs=None, histogram_kwargs=None, chi_square_kwargs=None, chi_square_error_kwargs=None, g_test_kwargs=None, g_test_error_kwargs=None, maximum_likelihood_kwargs=None, maximum_likelihood_error_kwargs=None, assess_normality=False, assess_kolmogorov_smirnov=False):
        if kernel_density_estimation_kwargs is not None:
            self.update_kernel_density_estimation(
                **kernel_density_estimation_kwargs)
        if histogram_kwargs is not None:
            self.update_histogram(
                **histogram_kwargs)
        if chi_square_kwargs is not None:
            self.update_chi_square(
                **chi_square_kwargs)
        if chi_square_error_kwargs is not None:
            self.update_chi_square_error_space(
                **chi_square_error_kwargs)
        if g_test_kwargs is not None:
            self.update_g_test(
                **g_test_kwargs)
        if g_test_error_kwargs is not None:
            self.update_g_test_error_space(
                **g_test_error_kwargs)
        if maximum_likelihood_kwargs is not None:
            self.update_maximum_likelihood(
                **maximum_likelihood_kwargs)
        if maximum_likelihood_error_kwargs is not None:
            self.update_maximum_likelihood_error_space(
                **maximum_likelihood_error_kwargs)
        if assess_normality:
            self.update_normality_assessment()
        if assess_kolmogorov_smirnov:
            self.update_kolmogorov_smirnov_assessment()

class NormalDistributionConfiguration(ParametricDistributionConfiguration):

    def __init__(self, data, extreme_parameter, is_log_transformed=False):
        x = np.sort(data[extreme_parameter])
        pdf = lambda prms, x=x : np.exp(- np.square((x - prms[0])/prms[1]) / 2) / (prms[1] * np.sqrt(2 * np.pi))
        cdf = lambda prms, x=x : None
        super().__init__(
            data=data,
            extreme_parameter=extreme_parameter,
            distribution_id='normal',
            nparameters=2,
            pdf=pdf,
            cdf=cdf)
        self.is_log_transformed = is_log_transformed

    def __call__(self, kernel_density_estimation_kwargs=None, histogram_kwargs=None, chi_square_kwargs=None, chi_square_error_kwargs=None, g_test_kwargs=None, g_test_error_kwargs=None, maximum_likelihood_kwargs=None, maximum_likelihood_error_kwargs=None, assess_normality=False, assess_kolmogorov_smirnov=False, container_of_search_kwargs=None):
        self.update_base_analysis(
            kernel_density_estimation_kwargs=kernel_density_estimation_kwargs,
            histogram_kwargs=histogram_kwargs,
            chi_square_kwargs=chi_square_kwargs,
            chi_square_error_kwargs=chi_square_error_kwargs,
            g_test_kwargs=g_test_kwargs,
            g_test_error_kwargs=g_test_error_kwargs,
            maximum_likelihood_kwargs=maximum_likelihood_kwargs,
            maximum_likelihood_error_kwargs=maximum_likelihood_error_kwargs,
            assess_normality=assess_normality,
            assess_kolmogorov_smirnov=assess_kolmogorov_smirnov)
        if container_of_search_kwargs is not None:
            for search_kwargs in container_of_search_kwargs:
                searcher = EventSearcher(self.data)
                data, _ = searcher.search_events(**search_kwargs)
                normal_distribution = NormalDistributionConfiguration(
                    data=data,
                    extreme_parameter=self.extreme_parameter)
                normal_distribution.update_base_analysis(
                    kernel_density_estimation_kwargs=kernel_density_estimation_kwargs,
                    histogram_kwargs=histogram_kwargs,
                    chi_square_kwargs=chi_square_kwargs,
                    chi_square_error_kwargs=chi_square_error_kwargs,
                    g_test_kwargs=g_test_kwargs,
                    g_test_error_kwargs=g_test_error_kwargs,
                    maximum_likelihood_kwargs=maximum_likelihood_kwargs,
                    maximum_likelihood_error_kwargs=maximum_likelihood_error_kwargs,
                    assess_normality=assess_normality,
                    assess_kolmogorov_smirnov=assess_kolmogorov_smirnov)
                self._sub_series.append(
                    {'normal distribution' : normal_distribution})
                self._container_of_search_kwargs.append(search_kwargs)

class LogNormalDistributionConfiguration(ParametricDistributionConfiguration):

    def __init__(self, data, extreme_parameter):
        x = np.sort(data[extreme_parameter])
        pdf = lambda prms, x=x : np.exp(- np.square((np.log(x) - prms[0]) / prms[1]) / 2) / (x * prms[1] * np.sqrt(2 * np.pi))
        cdf = lambda prms, x=x : None
        super().__init__(
            data=data,
            extreme_parameter=extreme_parameter,
            distribution_id='lognormal',
            nparameters=2,
            pdf=pdf,
            cdf=cdf)
        self._normal_distribution = None
        self.is_log_transformed = False

    @property
    def normal_distribution(self):
        return self._normal_distribution

    def __call__(self, kernel_density_estimation_kwargs=None, histogram_kwargs=None, chi_square_kwargs=None, chi_square_error_kwargs=None, g_test_kwargs=None, g_test_error_kwargs=None, maximum_likelihood_kwargs=None, maximum_likelihood_error_kwargs=None, assess_normality=False, assess_kolmogorov_smirnov=False, normal_kwargs=None, container_of_search_kwargs=None):
        self.update_base_analysis(
            kernel_density_estimation_kwargs=kernel_density_estimation_kwargs,
            histogram_kwargs=histogram_kwargs,
            chi_square_kwargs=chi_square_kwargs,
            chi_square_error_kwargs=chi_square_error_kwargs,
            g_test_kwargs=g_test_kwargs,
            g_test_error_kwargs=g_test_error_kwargs,
            maximum_likelihood_kwargs=maximum_likelihood_kwargs,
            maximum_likelihood_error_kwargs=maximum_likelihood_error_kwargs,
            assess_normality=assess_normality,
            assess_kolmogorov_smirnov=assess_kolmogorov_smirnov)
        if normal_kwargs is not None:
            normal_data = {key : value for key, value in self.data.items() if key != self.extreme_parameter}
            normal_data[self.extreme_parameter] = np.log(self.data[self.extreme_parameter])
            normal_distribution = NormalDistributionConfiguration(
                data=normal_data,
                extreme_parameter=self.extreme_parameter,
                is_log_transformed=True)
            normal_distribution(**normal_kwargs)
            self._normal_distribution = normal_distribution
        if container_of_search_kwargs is not None:
            for search_kwargs in container_of_search_kwargs:
                searcher = EventSearcher(self.data)
                data, _ = searcher.search_events(**search_kwargs)
                lognormal_distribution = LogNormalDistributionConfiguration(
                    data=data,
                    extreme_parameter=self.extreme_parameter)
                lognormal_distribution.update_base_analysis(
                    kernel_density_estimation_kwargs=kernel_density_estimation_kwargs,
                    histogram_kwargs=histogram_kwargs,
                    chi_square_kwargs=chi_square_kwargs,
                    chi_square_error_kwargs=chi_square_error_kwargs,
                    g_test_kwargs=g_test_kwargs,
                    g_test_error_kwargs=g_test_error_kwargs,
                    maximum_likelihood_kwargs=maximum_likelihood_kwargs,
                    maximum_likelihood_error_kwargs=maximum_likelihood_error_kwargs,
                    assess_normality=assess_normality,
                    assess_kolmogorov_smirnov=assess_kolmogorov_smirnov)
                if normal_kwargs is not None:
                    # _normal_data = EventSearcher(lognormal_distribution.data).search_events(**search_kwargs)
                    _normal_data = {key : value for key, value in lognormal_distribution.data.items() if key != lognormal_distribution.extreme_parameter}
                    _normal_data[lognormal_distribution.extreme_parameter] = np.log(lognormal_distribution.data[lognormal_distribution.extreme_parameter])
                    _normal_distribution = NormalDistributionConfiguration(
                        data=_normal_data,
                        extreme_parameter=self.extreme_parameter,
                        is_log_transformed=True)
                    _normal_distribution(
                        **normal_kwargs)
                    lognormal_distribution._normal_distribution = _normal_distribution
                self._sub_series.append(
                    {'lognormal distribution' : lognormal_distribution})
                self._container_of_search_kwargs.append(search_kwargs)










##
