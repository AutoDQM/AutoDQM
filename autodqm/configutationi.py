#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os

def get_subsystem(cfg_dir, subsystem):
    """Return the dict-based configuration of subsystem from cfg_dir."""
    fname = subsystem + '.json'
    path = os.path.join(cfg_dir, fname)
    if not os.path.exists(path):
        raise error("Subsystem config '{0}' not found.".format(subsystem))

    with open(os.path.join(cfg_dir, fname)) as f:
        return json.load(f)


class error(Exception):
    pass
