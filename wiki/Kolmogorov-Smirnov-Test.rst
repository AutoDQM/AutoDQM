
The Kolmogorov-Smirnov (KS) is a comparator plugin used for comparing 1D histograms. The `2-sample KS test <https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Two-sample_Kolmogorov%E2%80%93Smirnov_test>`_ is computed between the data and reference histograms, and the result is used to determine whether the data histogram is anomalous.

Options
-------


* **ks_cut** -- default 0.09: If the KS statistic is greater than this value, the plot is marked as anomalous
* **min_entries** -- default 100,000: The data histogram must have ``entries >= min_entries`` to be displayed as anomalous. This is meant to reduce false-positives in low-statistics runs

Source
------

Under `plugins`: [ks.py](https://github.com/jkguiang/AutoDQM/blob/release-v2.1.0/plugins/ks.py)
