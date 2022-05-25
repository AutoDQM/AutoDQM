#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,'..')


import autodqm.cfg
import os
import argparse
from glob import glob
from tqdm import tqdm
from autodqm.dqm import DQMSession
from autodqm.compare_hists import process



def autodqm_offline(dqmSource, subsystem,
                    data_run, data_sample, data_series,
                    ref_run, ref_sample, ref_series,
                    cfg_dir, output_dir, plugin_dir,
                    sslcert, sslkey, db):

    if not ref_sample:
        ref_sample = data_sample
    if not ref_series:
        ref_series = data_series

    print("Using cert/key pair:")
    print("\tCertificate: {}".format(sslcert))
    print("\tKey: {}".format(sslkey))
    cert = make_cert(sslcert, sslkey)

    # Get root files
    with DQMSession(cert, db) as dqm:
        print('')
        print("Getting data root file...")
        data_path = get_run(dqm, dqmSource, data_series, data_sample, data_run)

        print('')
        print("Getting reference root file...")
        ref_path = get_run(dqm, dqmSource, ref_series, ref_sample, ref_run)

    print('')
    print("Processing results...")
    results = process(0, 9999, cfg_dir, dqmSource, subsystem,
                      data_series, data_sample, data_run, data_path,
                      ref_series, ref_sample, ref_run, ref_path,
                      output_dir=output_dir, plugin_dir=plugin_dir)

    print('')
    print("Results available in {}".format(output_dir))
    return results


def get_run(dqm, dqmSource, series, sample, run):
    stream = dqm.stream_run(dqmSource, series, sample, run)
    first = next(stream)
    path = first.path
    if first.cur == first.total:
        print("Run cached at {}".format(path))
    else:
        with tqdm(total=first.total,
                  unit='B', unit_scale=True, unit_divisor=1024) as t:
            prev = 0
            for p in stream:
                t.update(p.cur - prev)
                prev = p.cur
    return path


def make_cert(sslcert, sslkey):
    return (sslcert, sslkey)


def find_file(pattern):
    '''Find the first file that matches the given pattern.'''
    pattern = os.path.expandvars(pattern)
    pattern = os.path.expanduser(pattern)
    return next((f for f in glob(pattern)), None)


if __name__ == '__main__':

    # Collect command line arguments
    parser = argparse.ArgumentParser(description='Run AutoDQM offline.')
    parser.add_argument('dqmSource', type=str,
                        help="dqmSource configuration to use. Online or Offline")
    parser.add_argument('subsystem', type=str,
                        help="subsystem configuration to use. Examples: CSC, EMTF")

    parser.add_argument('data_series', type=str,
                        help="data series to look for samples in. Examples: Run2017, Commissioning2018")
    parser.add_argument('data_sample', type=str,
                        help="data sample to look for runs in. Examples: ZeroBias, SingleMuon, Cosmics")
    parser.add_argument('data_run', type=str, help="data run number")
    parser.add_argument('ref_run', type=str, help="data run number")

    parser.add_argument('--ref_series', type=str, default=None,
                        help="ref series to look for samples in. Defaults to data_series")
    parser.add_argument('--ref_sample', type=str, default=None,
                        help="ref sample to look for runs in. Defaults to data_ref")

    parser.add_argument('-c', '--config', default=os.environ['ADQM_CONFIG'],
                        help="config directory to use")
    parser.add_argument('-o', '--output', default=os.environ['ADQM_OUT'],
                        help="artifact (pdfs, pngs, txts) output directory")
    parser.add_argument('-p', '--plugins', default=os.environ['ADQM_PLUGINS'],
                        help="comparison plugins directory")
    parser.add_argument('-d', '--db', default=os.environ['ADQM_DB'],
                        help="local database for storing runs")

    parser.add_argument('--sslcert', type=str, default=os.environ['ADQM_SSLCERT'],
                        help="path to a CMS VO public certificate")
    parser.add_argument('--sslkey', type=str, default=os.environ['ADQM_SSLKEY'],
                        help="path to a CMS VO private key")

    args = parser.parse_args()

    sslcert = find_file(args.sslcert)
    sslkey = find_file(args.sslkey)

    autodqm_offline(args.dqmSource, args.subsystem,
                    args.data_run, args.data_sample, args.data_series,
                    args.ref_run, args.ref_sample, args.ref_series,
                    cfg_dir=args.config,
                    output_dir=args.output,
                    plugin_dir=args.plugins,
                    sslcert=sslcert, sslkey=sslkey, db=args.db)
