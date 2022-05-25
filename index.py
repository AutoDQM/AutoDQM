#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import autodqm.cfg
import cgi
import json
import os
import traceback
from autodqm import compare_hists
from autodqm.dqm import DQMSession
from autoref import sql
import sys
sys.path.append('/var/www/cgi-bin/')
from modules.histCollection import HistCollection
from modules.histCollection import HistCleaner
from modules.dqmpca import DQMPCA

VARS = {}
SITE_BASENAME = "/dqm/autodqm"


def handle_request(req):
    err = None
    try:
        load_vars()
        if req['type'] == "fetch_run":
            data = fetch_run(req['dqmSource'], req['series'], req['sample'], req['run'])
        elif req['type'] == "process":
            data = process(int(req['chunk_index']),
                           int(req['chunk_size']),
                           req['dqmSource'],
                           req['subsystem'],
                           req['data_series'],
                           req['data_sample'],
                           req['data_run'],
                           req['ref_series'],
                           req['ref_sample'],
                           req['ref_run'])
        elif req['type'] == "get_dqmSources":
            data = get_dqmSources()
        elif req['type'] == "get_subsystems":
            data = get_subsystems()
        elif req['type'] == "get_series":
            data = get_series(req['dqmSource'])
        elif req['type'] == "get_samples":
            data = get_samples(req['dqmSource'], req['series'])
        elif req['type'] == "get_runs":
            data = get_runs(req['dqmSource'], req['series'], req['sample'])
        elif req['type'] == "get_ref":
            data = get_ref(req['dqmSource'],
                           req['subsystem'],
                           req['run'],
                           req['series'],
                           req['sample'])
        else:
            raise error
    except Exception as e:
        err = e
        tb = traceback.format_exc()
    finally:
        res = {}
        if err:
            res['error'] = {
                'message': str(err),
                'traceback': tb
            }
        else:
            res['data'] = data
        return res


def fetch_run(dqmSource, series, sample, run):
    with make_dqm() as dqm:
        dqm.fetch_run(dqmSource, series, sample, run)
    return {}


def process(chunk_index, chunk_size,
            dqmSource, subsystem,
            data_series, data_sample, data_run,
            ref_series, ref_sample, ref_run):

    with make_dqm() as dqm:
        # Get root file paths
        data_path = dqm.fetch_run(dqmSource, data_series, data_sample, data_run)
        ref_path = dqm.fetch_run(dqmSource, ref_series, ref_sample, ref_run)

    # Get config and results/plugins directories
    results_dir = os.path.join(VARS['PUBLIC'], 'results')
    plugin_dir = VARS['PLUGINS']
    config_dir = VARS['CONFIG']


    # Process this query
    results = compare_hists.process(chunk_index, chunk_size, config_dir,
                                    dqmSource, subsystem,
                                    data_series, data_sample,
                                    data_run, data_path,
                                    ref_series, ref_sample,
                                    ref_run, ref_path,
                                    output_dir=results_dir,
                                    plugin_dir=plugin_dir)

    # Relativize the results paths
    def relativize(p): return os.path.join(
        SITE_BASENAME + '/results', os.path.relpath(p, results_dir))
    for r in results:
        r['pdf_path'] = relativize(r['pdf_path'])
        r['json_path'] = relativize(r['json_path'])
        r['png_path'] = relativize(r['png_path'])

    new_chunk_index = chunk_index + chunk_size;
    if(len(results) == 0):
        new_chunk_index = -1;

    return {'items': results, "chunk_index": new_chunk_index}


def get_dqmSources():
    names = ["Online", "Offline"]
    return {'items': [{"name": n} for n in names]}


def get_subsystems():
    names = autodqm.cfg.list_subsystems(VARS['CONFIG'])
    return {'items': [{"name": n} for n in names]}


def get_series(dqmSource):
    with make_dqm() as dqm:
        rows = dqm.fetch_series_list(dqmSource)
    return {'items': [r._asdict() for r in rows]}


def get_samples(dqmSource, series):
    with make_dqm() as dqm:
        rows = dqm.fetch_sample_list(dqmSource, series)
    return {'items': [r._asdict() for r in rows]}


def get_runs(dqmSource, series, sample):
    with make_dqm() as dqm:
        ## TODO: truncate list to reasonable range, e.g. 2018 and later - AWB 2022.05.23
        rows = dqm.fetch_run_list(dqmSource, series, sample)
    return {'items': [r._asdict() for r in rows]}

def get_ref(dqmSource, subsystem, data_run, series, sample):
    config_dir = VARS['CONFIG']
    with make_dqm() as dqm:
        rows = dqm.fetch_run_list(dqmSource, series, sample)
    ref_runs = []
    for row in [r._asdict() for r in rows]:
        ref_runs.append(row['name'])
    refs = sql.fetch_refs(autodqm.cfg.get_subsystem(config_dir, subsystem), data_run, ref_runs)
    return {'items': refs['ref_data'], 'candidates':refs['ref_cands']}


def load_vars():
    try:
        VARS.update({
            'SSLCERT': os.environ['ADQM_SSLCERT'],
            'SSLKEY': os.environ['ADQM_SSLKEY'],
            'DB': os.environ['ADQM_DB'],
            'PUBLIC': os.environ['ADQM_PUBLIC'],
            'CONFIG': os.environ['ADQM_CONFIG'],
            'PLUGINS': os.environ['ADQM_PLUGINS']
        })
        VARS['CERT'] = (VARS['SSLCERT'], VARS['SSLKEY'])
        VARS['CACHE'] = os.path.join(VARS['DB'], 'dqm_offline')
    except Exception as e:
        raise ServerError("Server incorrectly configured: {}".format(e))

def make_dqm():
    return DQMSession(VARS['CERT'], VARS['DB'], cache=VARS['CACHE'])

class error(Exception):
    pass


class ServerError(error):
    pass


if __name__ == "__main__":
    cgi_req = cgi.FieldStorage()

    req = {}
    for k in list(cgi_req.keys()):
        req[str(k)] = str(cgi_req[k].value)

    res = handle_request(req)

    print("Content-type: application/json")
    print("Access-Control-Allow-Origin: *")
    print("")
    print((json.dumps(res)))
