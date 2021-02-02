class HistCollection(object):
    """Store a collection of cleaned histograms for use in ML algorithms."""
    def __init__(self, hdata, normalize=True, remove_identical_bins=True, extra_info=None, 
                 hist_cleaner=None):
        """
        Initialize the HistCollection.
        
        hdata is a 2D array of histogram data
          Each row is a histogram and each column a bin
        normalize: whether or not to scale histograms to unit area
        remove_identical_bins: remove bins that are the same in every histogram in the collection
        extra_info: dict containing any auxiliary info you want to be stored
          (e.g. extra_info["runs"] could be a list of runs corresponding to each histogram)

        The histograms will be "cleaned" using the HistCleaner class
        """

        self.hdata = np.array(hdata, dtype=float)
        self.__nhists = self.hdata.shape[0]
        self.__nbins = self.hdata.shape[1]
        self.norms = np.sum(hdata, axis=1)

        if hist_cleaner is not None:
            self.__hist_cleaner = hist_cleaner
        else:
            self.__hist_cleaner = HistCleaner(normalize, remove_identical_bins)
        self.__hist_cleaner.fit(self.hdata)
        self.hdata = self.__hist_cleaner.transform(self.hdata)
        

        self.shape = self.hdata.shape
        self.extra_info = extra_info
        
    
    @property
    def nhists(self):
        return self.__nhists

    @property
    def nbins(self):
        return self.__nbins

    @property
    def hist_cleaner(self):
        return self.__hist_cleaner

    @staticmethod
    def draw(h, ax=None, text=None, **kwargs):
        """
        Plot a single histogram with matplotlib.
          - ax: the matplotlib axis to use. Defaults to plt.gca()
          - text: string to write on the plot
          - kwargs: keywork args to pass to pyplot.hist
        """

        if not ax:
            ax = plt.gca()

        if "histtype" not in kwargs:
            kwargs["histtype"] = 'stepfilled'
        if "color" not in kwargs:
            kwargs["color"] = 'k'
        if "linewidth" not in kwargs and "lw" not in kwargs:
            kwargs["lw"] = 1
        if "facecolor" not in kwargs and "fc" not in kwargs:
            kwargs["fc" ] = "lightskyblue"
        if "linestyle" not in kwargs and "ls" not in kwargs:
            kwargs["linestyle"] = '-'

        nbins = h.size
        ax.hist(np.arange(nbins)+0.5, weights=h, bins=np.arange(nbins+1),
                **kwargs)
        ax.set_ylim(0, np.amax(h)*1.5)
        if np.amax(h) > 10000/1.5:
            ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
        if text:
            ax.text(0.05, 0.9, text, transform=ax.transAxes)
        
    def draw_single(self, idx, restore_bad_bins=True, use_normed=False, draw_title=True, **kwargs):
        """
        Plot the histogram at index idx with matplotlib.
          - ax: the matplotlib axis to use. Defaults to plt.gca()
          - restore_bad_bins: use the HistCleaner to restore bins that were removed for plotting
          - use_normed: whether to draw normalized histograms
          - draw_title: whether to draw the title on the plot (extra_info["title"] must exist)
          - kwargs: arguments to pass to the draw function above
        """

        h = self.hdata[idx, :]
        if restore_bad_bins:
            h = self.__hist_cleaner.restore_bad_bins(h)
        if not use_normed:
            h = h*self.norms[idx]

        if draw_title and "title" in self.extra_info:
            kwargs["text"] = self.extra_info["title"]

        HistCollection.draw(h, **kwargs)
        
class HistCleaner(object):
    #import numpy as np
    """ 
    sklearn-style preprocessing class to perform necessary "cleaning" of histogram collections for use in ML algorithms
    
    Can perform two separate operations, controlled by boolean flags:
     - normalize: whether or not to scale histograms to unit area
     - remove_identical_bins: remove bins that are the same in every histogram in the collection
    """
    def __init__(self, normalize=True, remove_identical_bins=True):
        self.__normalize = normalize
        self.__remove_identical_bins = remove_identical_bins

        # internal use
        self.__is_fit = False

    
    @property
    def normalize(self):
        return self.__normalize

    @normalize.setter
    def normalize(self, norm):
        if not isinstance(norm, bool):
            raise Exception("normalize must be set to a boolean value")
        self.__normalize = norm

    @property
    def remove_identical_bins(self):
        return self.__remove_identical_bins

    @remove_identical_bins.setter
    def remove_identical_bins(self, rib):
        if not isinstance(rib, bool):
            raise Exception("remove_identical_bins must be set to a boolean value")
        self.__remove_identical_bins = rib

    def fit(self, hd):
        self.nbins = hd.shape[1]
        # find the "good" bin indices (those that aren't the same in every histogram)
        #np.tile transform and repeat a given array
        bad_bins = np.all(hd==np.tile(hd[0,:],hd.shape[0]).reshape(hd.shape), axis=0)
        
        good_bins = np.logical_not(bad_bins)
        self.bad_bins = np.arange(self.nbins)[bad_bins]
        self.good_bins = np.arange(self.nbins)[good_bins]
        self.n_good_bins = self.good_bins.size
        self.bad_bin_contents = hd[0,self.bad_bins]

        self.__is_fit = True

    def _check_fit(self):
        if not self.__is_fit:
            raise Exception("Must fit the HistCleaner before calling transform")

    def restore_bad_bins(self, hd):
        self._check_fit()
        init_shape = hd.shape
        if len(init_shape) == 1:
            hd = hd.reshape(1,-1)
        if hd.shape[1] != self.n_good_bins:
            raise Exception("Invalid number of columns")

        ret = np.zeros((hd.shape[0], self.nbins))
        ret[:,self.good_bins] = hd
        ret[:,self.bad_bins] = np.tile(self.bad_bin_contents, hd.shape[0]).reshape(hd.shape[0], self.bad_bins.size)

        if len(init_shape) == 1:
            ret = ret.reshape(ret.size,)
        return ret

    def remove_bad_bins(self, hd):
        self._check_fit() 
        init_shape = hd.shape
        if len(init_shape) == 1:
            hd = hd.reshape(1,-1)
        if hd.shape[1] != self.nbins:
            raise Exception("Invalid number of columns")
        
        ret = hd[:,self.good_bins]
        if len(init_shape) == 1:
            ret = ret.reshape(ret.size,)
        return ret

    def transform(self, hd):
        self._check_fit()
        init_shape = hd.shape
        if len(init_shape)==1:
            hd = hd.reshape(1,-1)
        is_cleaned = False
        if hd.shape[1] != self.nbins:
            if hd.shape[1] == self.n_good_bins:
                is_cleaned = True
            else:
                raise Exception("Invalid shape! Expected {0} or {1} columns, got {2}".format(self.nbins,self.n_good_bins, hd.shape[1]))

        # remove bad bins
        if not is_cleaned and self.remove_identical_bins:
            hd = self.remove_bad_bins(hd)
        import numpy as np
        # normalize each row
        if self.normalize:
            norms = np.sum(hd, axis=1)
            tile = np.tile(norms, self.n_good_bins).reshape(self.n_good_bins, -1).T
            hd = np.divide(hd, tile, out=np.zeros_like(hd), where=tile!=0)

        if len(init_shape) == 1:
            hd = hd.reshape(hd.size,)
        return hd

    def fit_transform(self, hd):
        self.fit(hd)
        return self.transform(hd)
