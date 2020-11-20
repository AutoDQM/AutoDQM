#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os

def identifier(hp, comparator_name):
    """
    Return a `hashed` identifier for the histpair
    """
    data_id = "DATA-{}-{}-{}".format(hp.data_series,
                                     hp.data_sample, hp.data_run)
    ref_id = "REF-{}-{}-{}".format(hp.ref_series, hp.ref_sample, hp.ref_run)
    if hp.data_name == hp.ref_name:
        name_id = hp.data_name
    else:
	name_id = "DATANAME-{}_REFNAME-{}".format(hp.data_name, hp.ref_name)
    comp_id = "COMP-{}".format(comparator_name)

    hash_snippet = str(hash(hp))[-5:]

    idname = "{}_{}_{}_{}_{}".format(
        data_id, ref_id, name_id, comp_id, hash_snippet)
    return idname


class error(Exception):
    pass
