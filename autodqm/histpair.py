#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json


class HistPair(object):
    """Data class for storing data and ref histograms to be compared by AutoDQM, as well as any relevant configuration parameters."""

    def __init__(self, config,
                 data_series, data_sample, data_run, data_name, data_hist,
                 ref_series, ref_sample, ref_run, ref_name, ref_hist):

        self.data_series = data_series
        self.data_sample = data_sample
        self.data_run = data_run
        self.data_name = data_name
        self.data_hist = data_hist

        self.ref_series = ref_series
        self.ref_sample = ref_sample
        self.ref_run = ref_run
        self.ref_name = ref_name
        self.ref_hist = ref_hist

        self.config = config
        self.comparators = config.get(
            'comparators', ('pull_values', 'ks_test'))

    def __eq__(self, other):
        return (isinstance(other, type(self))
                and self.data_name == other.data_name
                and self.ref_name == other.ref_name
                and self.query == other.config
                and self.config == other.config
                and self.comparators == other.comparators)

    def __neq__(self, other):
        return not self == other

    def __hash__(self):
        return hash(
            self.data_series + self.data_sample + self.data_run + self.data_name +
            self.ref_series + self.ref_sample + self.ref_run + self.ref_name +
            json.dumps(self.config, sort_keys=True))
