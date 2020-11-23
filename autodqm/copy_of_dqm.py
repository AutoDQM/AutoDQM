#!/usr/bin/env python
# -*- coding: utf-8 -*-

import base64
import errno
import json
#import lxml.html
import os
import requests
from collections import namedtuple
#from requests_futures.sessions import FuturesSession

TIMEOUT = 5

BASE_URL = 'https://cmsweb.cern.ch'
DQM_URL = 'https://cmsweb.cern.ch/dqm/offline/data/browse/ROOT/OfflineData/'
CA_URL = 'https://cafiles.cern.ch/cafiles/certificates/CERN%20Root%20Certification%20Authority%202.crt'

# The following are appended to the db dir
CACHE_DIR = 'cache/'
CA_PATH = 'CERN_Root_CA.crt'

StreamProg = namedtuple('StreamProg', ('cur', 'total', 'path'))
DQMRow = namedtuple('DQMRow', ('name', 'full_name', 'url', 'size', 'date'))

def lxml():
    """
    lxml.html imported from within this function, and is called in the code
    """
    import lxml.html
    return lxml.html    

def _parse_dqm_page(content):
    """
    Return the contents of a DQM series, sample, or macrorun page as a list of DQMRows.
    """
    dqm_rows = []
    tree = lxml().fromstring(content)
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
    """
    Return the simplified form of a full DQM run name.

    example:
    DQM_V0001_R000316293__ZeroBias__Run2018A-PromptReco-v2__DQMIO.root
    => 316293
    """
    name = full_name.split('_')[2][1:]
    return str(int(name))


def _get_cern_ca(path):
    """
    Download the CERN ROOT CA to the specified path.
    """
    _try_makedirs(os.path.dirname(path))
    r_ca = requests.get(CA_URL)
    with open(path, 'wb') as f:
        f.write(b'-----BEGIN CERTIFICATE-----\n')
        f.write(base64.b64encode(r_ca.content))
        f.write(b'\n-----END CERTIFICATE-----\n')

def _try_makedirs(*args, **kwargs):
    """
    Make a directory if it doesn't exist
    """
    try:
        return os.makedirs(*args, **kwargs)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def _resolve(future):
    """
    Wrapper to resolve a request future while handling exceptions
    """
    try:
        return future.result()
    except requests.ConnectionError as e:
        raise error(e)
    except requests.Timeout as e:
        raise error(e)

class error(Exception):
    pass
