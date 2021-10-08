from search_methods import *

class ResamplingConfiguration(ConditionMapping):

    def __init__(self):
        super().__init__()
        self._nresamples = None
        self._nshuffles = None
        self._resampled_indices = None
        self._unresampled_indices = None
        self._f_resample = None
        self._with_replacement = None

    @property
    def nresamples(self):
        return self._nresamples

    @property
    def nshuffles(self):
        return self._nshuffles

    @property
    def resampled_indices(self):
        return self._resampled_indices

    @property
    def unresampled_indices(self):
        return self._unresampled_indices

    @property
    def f_resample(self):
        return self._f_resample

    @property
    def with_replacement(self):
        return self._with_replacement

    def resample_indices_without_replacement(self):
        for n in range(self.nshuffles):
            np.random.shuffle(self._resampled_indices)
        np.random.shuffle(self._resampled_indices)

    def resample_indices_with_replacement(self, **kwargs):
        for n in range(self.nshuffles):
            np.random.shuffle(self._resampled_indices)
        self._resampled_indices = np.random.choice(
            self.resampled_indices,
            size=self.resampled_indices.size,
            replace=True,
            **kwargs)

    def resample_indices(self, **kwargs):
        self.f_resample(**kwargs)

    def update_resampling_criteria(self, nresamples, nshuffles, with_replacement):
        if not isinstance(nresamples, int):
            raise ValueError("invalid type(nresamples): {}".format(type(nresamples)))
        if nresamples < 0:
            raise ValueError("nresamples should be greater than zero")
        if not isinstance(nshuffles, int):
            raise ValueError("invalid type(nshuffles): {}".format(type(nshuffles)))
        if nshuffles < 0:
            raise ValueError("nshuffles should be greater than zero")
        self._nresamples = nresamples
        self._nshuffles = nshuffles
        if with_replacement: ## independent data
            self._f_resample = self.resample_indices_with_replacement
        else: ## dependent data
            self._f_resample = self.resample_indices_without_replacement
        self._with_replacement = with_replacement

    def update_resampling_indices(self, n):
        self._unresampled_indices = np.arange(n).astype(int)
        self._resampled_indices = np.copy(self.unresampled_indices)
