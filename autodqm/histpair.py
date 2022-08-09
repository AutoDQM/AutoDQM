#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json


class HistPair(object):
    """Data class for storing data and ref histograms to be compared by AutoDQM, as well as any relevant configuration parameters."""

    def __init__(self, dqmSource, config,
                 data_series, data_sample, data_run, data_name, data_hist,
                 ref_series, ref_sample, ref_runs, ref_name, ref_hists,
                 data_concat = None, ref_concat = None):

        self.dqmSource = dqmSource
        self.config = config

        self.data_series = data_series
        self.data_sample = data_sample
        self.data_run = data_run
        self.data_name = data_name
        self.data_hist = data_hist
        self.data_concat = data_concat

        self.ref_series = ref_series
        self.ref_sample = ref_sample
        self.ref_runs = ref_runs
        self.ref_name = ref_name
        self.ref_hists = ref_hists
        self.ref_concat = ref_concat

        if self.dqmSource == 'Offline':
            self.comparators = ['pull_values', 'ks_test', 'autodqm_ml_pca', 'beta_binomial']
        else:
            ## Currently ML PCA only trained with Offline data - AWB 2022.06.20
            ## If trained on Online in the future, need to update
            ## plugins/autodqm_ml_pca.py and models/autodqm_ml_pca/
            self.comparators = ['pull_values', 'ks_test', 'beta_binomial']

        if not config['comparators'] is None:
            self.comparators = config['comparators']


    def __eq__(self, other):
        return (isinstance(other, type(self))
                and self.dqmSource == other.dqmSource
                and self.query == other.config
                and self.config == other.config
                and self.data_name == other.data_name
                and self.ref_name == other.ref_name
                and self.comparators == other.comparators)

    def __neq__(self, other):
        return not self == other

    def __hash__(self):
        return hash(
            self.dqmSource + json.dumps(self.config, sort_keys=True) +
            self.data_series + self.data_sample + self.data_run + self.data_name +
            self.ref_series + self.ref_sample + '_'.join(self.ref_runs) + self.ref_name)
