#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

MAIN_CFG = 'main.json'


def list_subsystems(cfg_dir):
    """Return a list of names of subsystem configs in cfg_dir."""
    files = os.listdir(cfg_dir)
    out = []
    for fn in files:
        if (not fn[-5:] == '.json' or fn == MAIN_CFG):
            continue
        out.append(fn[:-5])
    return out


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
