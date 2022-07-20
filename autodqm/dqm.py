#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import errno
import json
import lxml.html
import os
import requests
from autodqm import cfg
from collections import namedtuple
from requests_futures.sessions import FuturesSession

TIMEOUT = 5
VERBOSE = 0
BASE_URL = 'https://cmsweb.cern.ch'
DQM_URL = 'https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/'
CA_URL = 'https://cafiles.cern.ch/cafiles/certificates/CERN%20Root%20Certification%20Authority%202.crt'

# The following are appended to the db dir
CACHE_DIR = 'cache/'
CA_PATH = 'CERN_Root_CA.crt'

StreamProg = namedtuple('StreamProg', ('cur', 'total', 'path'))
DQMRow = namedtuple('DQMRow', ('name', 'full_name', 'url', 'size', 'date'))

main_config = cfg.get_main(os.environ['ADQM_CONFIG'])
OnlineMap = main_config["OnlineDataMap"]


class DQMSession(FuturesSession):
    """Encapsulates an interface to DQM Offline."""

    def __init__(self, cert, db, cache=None, workers=16):
        super(DQMSession, self).__init__(max_workers=workers)

        self.db = db
        if cache:
            self.cache = cache
        else:
            self.cache = os.path.join(self.db, CACHE_DIR)

        self.cert = cert
        self.verify = os.path.join(db, CA_PATH)
        if not os.path.exists(self.verify):
            _get_cern_ca(self.verify)

    def fetch_run(self, dqmSource, subsystem, series, sample, run):
        """Fetch and cache a run data file.
        Returns the path to the downloaded file."""
        if VERBOSE: print('\ndqm.py fetch_run(dqmSource = %s, subsystem = %s, series = %s, sample = %s, run = %s)' %
                          (dqmSource, subsystem, series, sample, run))
        
        dl = self.stream_run(dqmSource, subsystem, series, sample, run)

        # Get the path from the first yield
        path = next(dl).path

        # Finish the download
        for _ in dl:
            pass
        if VERBOSE: print('  * path = %s' % path)
        return path

    def stream_run(self, dqmSource, subsystem, series, sample, run, chunk_size=4096):
        """Stream and cache a run data file.
        Returns a generator that yields StreamProg tuples corresponding to the download progress."""
        if VERBOSE: print('\ndqm.py stream_run(dqmSource = %s, subsystem = %s, series = %s, sample = %s, run = %s, chunk_size = %d)' %
                          (dqmSource, subsystem, series, sample, run, chunk_size))

        run_path = self._run_path(dqmSource, subsystem, series, sample, run)
        run_dir = os.path.dirname(run_path)

        if not os.path.exists(run_path):
            _try_makedirs(run_dir)

            runs = self.fetch_run_list(dqmSource, subsystem, series, sample, run)

            isOnline = (series.startswith('000') and series.endswith('xxxx'))
            if isOnline:  ## Use cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OnlineData/
                if not subsystem in OnlineMap.keys():
                    raise error("dqm.py stream_run: {} not in OnlineMap".format(subsystem))
                DQM_dir = OnlineMap[subsystem]
                run_info = next(r for r in runs if r.name == run and DQM_dir+'_R000' in r.full_name)
            else:                   ## Use cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/
                ## FIXME: If files from multiple processings of the same dataset are available,
                ## just picks whichever one happens to be first in the list! - AWB 2022.06.28
                run_info = next(r for r in runs if r.name == run)

            for prog in self._stream_file(
                    run_info.url, run_path, chunk_size=chunk_size):
                yield prog

        size = os.path.getsize(run_path)
        if VERBOSE: print('  * run_path = %s, size = %d' % (run_path, size))
        yield StreamProg(size, size, run_path)

    def fetch_series_list(self, dqmSource):
        """Return DQMRows corresponding to the series available on DQM Online or Offline."""
        if VERBOSE: print('\ndqm.py fetch_series_list(%s)' % dqmSource)

        if dqmSource == "Online":
            series_list = []
            for series in _resolve(self._fetch_dqm_rows(DQM_URL+'OnlineData/original/')).data:
                if series.name.startswith('000') and series.name.endswith('xxxx'):
                    series_list.append(series)
            return series_list
        elif dqmSource == "Offline":
            return _resolve(self._fetch_dqm_rows(DQM_URL+'OfflineData/')).data
        else:
            raise error("dqm.py fetch_series_list dqmSource = {}, not Onilne or Offline!".format(dqmSource))

    def fetch_sample_list(self, dqmSource, series):
        """Return DQMRows corresponding to the samples available under the given series."""
        if VERBOSE: print('\ndqm.py fetch_sample_list(dqmSource = %s, series = %s)' % (dqmSource, series))

        series_rows = self.fetch_series_list(dqmSource)
        url = next((r.url for r in series_rows if r.name == series))
        return _resolve(self._fetch_dqm_rows(url)).data

    def fetch_run_list(self, dqmSource, subsystem, series, sample, selRun=None):
        """Return DQMRows corresponding to the runs available under the given series and sample."""
        if VERBOSE: print('\ndqm.py fetch_run_list(dqmSource = %s, subsystem = %s, series = %s, sample = %s, selRun = %s)' %
                          (dqmSource, subsystem, series, sample, selRun))
        if selRun and len(str(selRun)) != 6:
            raise error("dqm.py fetch_run_list selRun = {}, not 6 digits!".format(selRun))

        ## Get list of samples within a series
        ## For OfflineData, primary datasets within Run2018, Run2017, etc.
        ## For OnlineData/original, list of run ranges by first 2 digits of run
        sample_rows = self.fetch_sample_list(dqmSource, series)

        isOnline = (series.startswith('000') and series.endswith('xxxx'))
        if isOnline:
            macrorun_rows = sample_rows
        else:  ## Use cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/
            sample_url = next((r.url for r in sample_rows if r.name == sample))
            # Get all run directories for this sample
            macrorun_rows = _resolve(self._fetch_dqm_rows(sample_url)).data

        ## If fetch_run_list is called with a run number, return only selected rows
        macrorun_rows_sel = []
        for mr in macrorun_rows:
            if isOnline and mr.name != sample: continue
            if (not selRun) or (mr.name == '000'+str(selRun)[0:4]+'xx'):
                macrorun_rows_sel.append(mr)

        ## Determine which run directories are cached
        run_rows = []
        to_req = []
        for mr in macrorun_rows_sel:
            rows = self._get_cache(mr)
            ## If selecting a single run, the cache will not contain a full list of runs
            if rows and not selRun:
                run_rows += rows
            else:
                to_req.append(mr)

        ## Select rows from these directories based on run number and sample (for OnlineData)
        futures = [(mr, self._fetch_dqm_rows(mr.url)) for mr in to_req]
        for mr, fut in futures:
            rows = _resolve(fut).data
            for row in rows:
                if selRun and row.name != selRun: continue
                if isOnline:
                    if not subsystem in OnlineMap.keys():
                        raise error("dqm.py fetch_run_list: {} not in OnlineMap".format(subsystem))
                    DQM_dir = OnlineMap[subsystem]
                    if not (DQM_dir+'_R000' in row.full_name): continue
                run_rows.append(row)
            self._write_cache(mr, rows)

        if VERBOSE: print('  * len(run_rows) = %d' % len(run_rows))
        if VERBOSE + (len(run_rows) < 20) >= 2: print(run_rows)
        return run_rows

    def _get_cache(self, parent_row):
        """Return the DQM page corresponding to parent_row as DQMRows if they are
        cached. Otherwise None."""
        cache_file = self._cache_path(parent_row)
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                dat = json.load(f)
                return [DQMRow(*r) for r in dat]
        else:
            return None

    def _write_cache(self, parent_row, dqm_rows):
        """Write a list of DQMRows to the cache under their parent_row."""
        cache_file = self._cache_path(parent_row)
        _try_makedirs(os.path.dirname(cache_file))
        with open(cache_file, 'w') as f:
            json.dump(dqm_rows, f)

    def _cache_path(self, parent_row):
        """Return the path to a cached DQM page specified by its parent DQMRow."""
        return os.path.join(self.cache, str(abs(hash(parent_row))))

    def _fetch_dqm_rows(self, url, timeout=TIMEOUT):
        """Return a future of DQMRows of a DQM page at url.
        Access the array of DQMRows at _resolve(self._fetch_dqm_rows(...)).data"""

        # Callback to process dqm responses
        def cb(sess, resp):
            resp.data = _parse_dqm_page(resp.text)

        return self.get(url, timeout=timeout, background_callback=cb)

    def _stream_file(self, url, dest, chunk_size=4096):
        """Stream a file into a destination path.

        Returns a generator of StreamProg tuples to indicate download progress."""
        res = _resolve(self.get(url, stream=True))
        if not res:
            raise error("Failed to download file: {}".format(url))

        total = int(res.headers.get('content-length'))
        cur = 0
        try:
            with open(dest, 'wb') as f:
                yield StreamProg(cur, total, dest)
                for data in res.iter_content(chunk_size=chunk_size):
                    cur += len(data)
                    f.write(data)
                    yield StreamProg(cur, total, dest)
            if cur != total:
                raise error(
                    "Failed to stream file: Final size {} less than total {}"
                    .format(cur, total))
        except:
            # Remove the file if anything went wrong in the middle
            os.remove(dest)
            raise

    def _run_path(self, dqmSource, subsystem, series, sample, run):
        """Return the path to the specified run data file in the cached db."""
        if dqmSource == "Online":
            return "{}/{}.root".format(os.path.join(self.db, dqmSource, series, sample, OnlineMap[subsystem]), run)
        elif dqmSource == "Offline":
            return "{}/{}.root".format(os.path.join(self.db, dqmSource, series, sample), run)
        else:
            raise error("dqm.py _run_path dqmSource = {}, not Onilne or Offline!".format(dqmSource))
        


def _parse_dqm_page(content):
    """Return the contents of a DQM series, sample, or macrorun page as a list of DQMRows."""
    if VERBOSE >= 2: print('\ndqm.py _parse_dqm_page(content = %s)' % content)

    dqm_rows = []
    tree = lxml.html.fromstring(content)
    tree.make_links_absolute(BASE_URL)

    for tr in tree.xpath('//tr'):
        td_strs = tr.xpath('td//text()')
        td_urls = tr.xpath('td/a/@href')

        full_name = td_strs[0]
        url = td_urls[0]
        size = int(td_strs[1]) if td_strs[1] != '-' else None
        date = td_strs[2]
        name = _parse_run_full_name(full_name) if size else full_name[:-1]

        dqm_rows.append(DQMRow(name, full_name, url, size, date))

    return dqm_rows


def _parse_run_full_name(full_name):
    """Return the simplified form of a full DQM run name."""
    if VERBOSE >= 2: print('\ndqm.py _parse_run_full_name(full_name = %s)' % full_name)

    if full_name.split('_')[2].startswith('R000'):  ## Format for OfflineData
        """example: DQM_V0001_R000316293__ZeroBias__Run2018A-PromptReco-v2__DQMIO.root => 316293"""
        name = full_name.split('_')[2][1:]
        return str(int(name))
    elif full_name.split('_')[3].startswith('R000'):  ## Format for OnlineData
        """example: DQM_V0001_SiStrip_R000351871.root => 351871"""
        name = full_name.split('_')[3][1:].replace('.root','')
        return str(int(name))
    else:
        raise error("dqm.py _parse_run_full_name({}), failed to parse run number!".format(full_name))
        return 'NULL'


def _get_cern_ca(path):
    """Download the CERN ROOT CA to the specified path."""
    _try_makedirs(os.path.dirname(path))
    r_ca = requests.get(CA_URL)
    with open(path, 'wb') as f:
        f.write(b'-----BEGIN CERTIFICATE-----\n')
        f.write(base64.b64encode(r_ca.content))
        f.write(b'\n-----END CERTIFICATE-----\n')


def _try_makedirs(*args, **kwargs):
    """Make a directory if it doesn't exist"""
    try:
        return os.makedirs(*args, **kwargs)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def _resolve(future):
    """Wrapper to resolve a request future while handling exceptions"""
    try:
        return future.result()
    except requests.ConnectionError as e:
        raise error(e)
    except requests.Timeout as e:
        raise error(e)


class error(Exception):
    pass
