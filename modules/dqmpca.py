norm_cut = 10000
from modules.histCollection import HistCollection
from modules.histCollection import HistCleaner
class DQMPCA(object):
    """Class to perform PCA specifically on HistCollection objects"""
    import numpy as np
    def __init__(self, use_standard_scaler=False, norm_cut=norm_cut, sse_ncomps=None):
        """Initialize the DQMPCA

        -use_standard_scalar determines whether to use standard scaling
          (zero mean, unit stdev) before feeding into a PCA. This helps
          for some histograms, but hurts for others
        """
        if use_standard_scaler:
            self.pca = Pipeline(
                ("scaler", StandardScaler()),
                ("pca", PCA())
                )
        else:
            self.pca = PCA()

        self.use_standard_scaler = use_standard_scaler
        self.norm_cut = norm_cut
        self.sse_ncomps = sse_ncomps

        self.__is_fit = False

    @property
    def sse_ncomps(self):
        return self.__sse_ncomps

    @sse_ncomps.setter
    def sse_ncomps(self, sse):
        if sse is not None and not isinstance(sse, tuple) and not isinstance(sse, list):
            raise Exception("illigal sse_ncomps value. Should be None or a list/tuple of ints")
        self.__sse_ncomps = sse

    def _check_fit(self):
        if not self.__is_fit:
            raise Exception("Must fit the DQMPCA before calling transform")

    def fit(self, hdata):
        if isinstance(hdata, HistCollection):
            self._hist_cleaner = hdata.hist_cleaner
            cleaned = hdata.hdata
            norms = hdata.norms
            
        else:
            self._hist_cleaner = HistCleaner()
            self._hist_cleaner.fit(hdata)
            cleaned = self._hist_cleaner.transform(hdata)
            norms = np.sum(cleaned, axis=1)

        cleaned = cleaned[norms>self.norm_cut, :]
        self.pca.fit(cleaned)        
        self.__is_fit = True

        if self.sse_ncomps is not None:
            self.sse_cuts = {}
            for ncomp in self.sse_ncomps:
                self.sse_cuts[ncomp] = []
                sses = self.sse(cleaned, ncomp)
                for pct in np.arange(1,101):
                    self.sse_cuts[ncomp].append(np.percentile(sses, pct))
    
    def transform(self, hdata):
        from modules.histCollection import HistCollection
        from modules.histCollection import HistCleaner
        """Transform a set of histograms with the trained PCA"""
        self._check_fit()
        if isinstance(hdata, HistCollection):
            cleaned = hdata.hdata
        else:
            cleaned = self._hist_cleaner.transform(hdata)        
        return self.pca.transform(cleaned)
        
    def inverse_transform(self, xf, n_components=3, restore_bad_bins=False):
        import numpy as np
        self._check_fit()
        xf = np.array(xf)
        trunc = np.zeros((xf.shape[0], self._hist_cleaner.n_good_bins))
        trunc[:,:n_components] = xf[:,:n_components]
        ixf = self.pca.inverse_transform(trunc)
        if not restore_bad_bins:
            return ixf
        else:
            return self._hist_cleaner.restore_bad_bins(ixf)

    def sse(self, hdata, n_components=3):
        if isinstance(hdata, HistCollection):
            cleaned = hdata.hdata
        else:
            cleaned = self._hist_cleaner.transform(hdata)        
        xf = self.transform(cleaned)
        ixf = self.inverse_transform(xf, n_components=n_components)
        return np.sqrt(np.sum((ixf-cleaned)**2, axis=1))
        
    def score(self, hdata, n_components=3):
        if not hasattr(self, "sse_cuts") or n_components not in self.sse_cuts:
            raise Exception("must fit first with {0} in sse_ncomps".format(n_components))
        sse = self.sse(hdata, n_components)
        return np.interp(sse, self.sse_cuts[n_components], np.arange(1,101))

    @property
    def explained_variance_ratio(self):
        if self.use_standard_scaler:
            return self.pca.named_steps["pca"].explained_variance_ratio_
        else:
            return self.pca.explained_variance_ratio_

    @property
    def mean(self):
        if self.use_standard_scaler:
            return self.pca.named_steps["scaler"].inverse_transform(self.pca.named_steps["pca"].mean_)
        else:
            return self.pca.mean_
