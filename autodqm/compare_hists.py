#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import subprocess
import uproot
from autodqm import cfg
from autodqm.histpair import HistPair
import plotly
import numpy as np


def process(chunk_index, chunk_size, config_dir,
            dqmSource, subsystem,
            data_series, data_sample, data_run, data_path,
            ref_series, ref_sample, ref_runs, ref_paths,
            output_dir='./out/', plugin_dir='./plugins/'):

    # Ensure no graphs are drawn to screen and no root messages are sent to
    # terminal
    histpairs = compile_histpairs(chunk_index, chunk_size, config_dir,
                                  dqmSource, subsystem,
                                  data_series, data_sample, data_run, data_path,
                                  ref_series, ref_sample, ref_runs, ref_paths)

    for d in [output_dir + s for s in ['/pdfs', '/jsons', '/pngs']]:
        if not os.path.exists(d):
            os.makedirs(d)

    hist_outputs = []

    comparator_funcs = load_comparators(plugin_dir)

    for hp in histpairs:
        comparators = []
        for c in hp.comparators:
            try:
                comparators.append((c, comparator_funcs[c]))
            except:
                raise error("Comparator {} was not found in {}/{}.".format(c, dqmSource, subsystem))

        for comp_name, comparator in comparators:
            result_id = identifier(hp, comp_name)
            pdf_path = '{}/pdfs/{}.pdf'.format(output_dir, result_id)
            json_path = '{}/jsons/{}.json'.format(output_dir, result_id)
            png_path = '{}/pngs/{}.html'.format(output_dir, result_id)
            pdf_path = png_path

            if not os.path.isfile(json_path):
                results = comparator(hp, **hp.config)

                # Continue if no results
                if not results:
                    continue

                # Make pdf
                #results.canvas.write_image(pdf_path)

                # Make png
                results.canvas.write_html(png_path, include_plotlyjs='cdn', full_html=False)
                
                #subprocess.Popen(
                #    ['convert', '-density', '50', '-trim', '-fuzz', '1%', pdf_path, png_path])

                # Make json
                info = {
                    'id': result_id,
                    'name': hp.data_name,
                    'comparator': comp_name,
                    'display': results.show or hp.config.get('always_show', False),
                    'config': hp.config,
                    'results': results.info,
                    'pdf_path': pdf_path,
                    'json_path': json_path,
                    'png_path': png_path,
                }
                with open(json_path, 'w') as jf:
                    json.dump(info, jf)
            else:
                with open(json_path) as jf:
                    info = json.load(jf)
            
            hist_outputs.append(info)

    return hist_outputs

def compile_histpairs(chunk_index, chunk_size, config_dir,
                      dqmSource, subsystem,
                      data_series, data_sample, data_run, data_path,
                      ref_series, ref_sample, ref_runs, ref_paths):

    config = cfg.get_subsystem(config_dir, subsystem)
    # Histogram details
    conf_list = config["hists"]
    main_gdir = config["main_gdir"]
    def_comparators = config['comparators'] if 'comparators' in config.keys() else None

    # ROOT files
    data_file = uproot.open(data_path)
    ref_files = [uproot.open(ref_path) for ref_path in ref_paths]

    histPairs = []
    
    missing_data_dirs = []
    missing_ref_dirs  = []

    for hconf in conf_list:
        # Set comparators if there are none
        if not 'comparators' in hconf.keys(): hconf['comparators'] = def_comparators
        # Get name of hist in root file
        h = str(hconf["path"].split("/")[-1])
        # Get parent directory of hist
        gdir = str(hconf["path"].split(h)[0])
        
        data_dirname = "{0}{1}".format(main_gdir.format(data_run), gdir)
        ref_dirnames = ["{0}{1}".format(main_gdir.format(ref_run), gdir) for ref_run in ref_runs]
        
        try:
            data_dir = data_file[data_dirname[:-1]]
        except:
            missing_data_dirs.append(data_dirname)
            continue
            # raise error("Subsystem dir {0} not found in data root file".format(data_dirname))

        ref_dirs = []
        for iRef in range(len(ref_runs)):
            try:
                ref_dirs.append( ref_files[iRef][ref_dirnames[iRef][:-1]] )
            except:
                missing_ref_dirs.append(ref_dirnames)
                continue
                # raise error("Subsystem dir {0} not found in ref root file".format(ref_dirnames[iRef]))

        data_keys = data_dir.keys()
        ref_keyss = [ref_dir.keys() for ref_dir in ref_dirs]

        valid_names = []

        # Add existing histograms that match h
        if "*" not in h:
            if h in [str(keys)[0:-2] for keys in data_keys] and all([ (h in [str(keys)[0:-2] for keys in ref_keys]) for ref_keys in ref_keyss ]):
                try:
                    data_hist = data_dir[h]
                    ref_hists = [ref_dir[h] for ref_dir in ref_dirs]
                except Exception as e:
                    continue

                data_hist_conc, ref_hists_conc = None, None
                if 'concatenate' in hconf.keys():
                    try:
                        data_hist_conc = [data_dir[dhc] for dhc in hconf['concatenate']]
                        ref_hists_conc = [[ref_dir[rhc] for rhc in hconf['concatenate']] for ref_dir in ref_dirs]
                    except Exception as e:
                        continue

                hPair = HistPair(dqmSource, hconf,
                                 data_series, data_sample, data_run, str(h), data_hist,
                                 ref_series, ref_sample, ref_runs, str(h), ref_hists,
                                 data_hist_conc, ref_hists_conc)
                histPairs.append(hPair)
        else:
            # Check entire directory for files matching wildcard (Throw out wildcards with / in them as they are not plottable)
            for name in data_keys:
                if h.split("*")[0] in str(name) and all([ name in ref_keys for ref_keys in ref_keyss ]) and not "<" in str(name):
                    if("/" not in name[:-2]):
                        try:
                            data_hist = data_dir[name[:-2]]
                            ref_hists = [ref_dir[name[:-2]] for ref_dir in ref_dirs]
                        except Exception as e:
                            continue
                        hPair = HistPair(dqmSource, hconf,
                                         data_series, data_sample, data_run, str(name[:-2]), data_hist,
                                         ref_series, ref_sample, ref_runs, str(name[:-2]), ref_hists)
                        histPairs.append(hPair)

    ## TODO: "raise warning" is not an actual function, but need some way to alert
    ## TODO: users that histograms in json config file are missing from ROOT files - AWB 2022.06.11
    # if len(missing_data_dirs) > 0:
    #     raise warning("The folloing subsystem dirs not found in data root file")
    #     for missing_data_dir in missing_data_dirs:
    #         raise warning("{0}".format(missing_data_dir))
    # if len(missing_ref_dirs) > 0:
    #     raise warning("The folloing subsystem dirs not found in ref root file")
    #     for missing_ref_dir in missing_ref_dirs:
    #         raise warning("{0}".format(missing_ref_dirs))

    #Return histpairs that match the chunk_index <<CAN BE IMPROVED IN THE FUTURE TO BE MORE EFFICIENT>>
    return histPairs[min(chunk_index, len(histPairs)):min(chunk_index+chunk_size, len(histPairs))] 

def load_comparators(plugin_dir):
    """Load comparators from each python module in ADQM_PLUGINS."""

    sys.path.insert(0, plugin_dir)

    comparators = dict()

    for modname in os.listdir(plugin_dir):
        if modname[0] == '_' or modname[-4:] == '.pyc' or modname[-4:] == '.swp':
            continue
        if modname[-3:] == '.py':
            modname = modname[:-3]
        try:
            mod = __import__("{}".format(modname))
        except:
            raise error("Failed to import {} from {}.".format(modname, plugin_dir))
        try:
            new_comps = mod.comparators()
        except AttributeError:
            raise error("Plugin {} from {} does not have a comparators() function.".format(mod, plugin_dir))
        comparators.update(new_comps)

    return comparators


def identifier(hp, comparator_name):
    """Return a `hashed` identifier for the histpair"""
    data_id = "DATA-{}-{}-{}".format(hp.data_series,
                                     hp.data_sample, hp.data_run)
    ref_id = "REF-{}-{}-{}".format(hp.ref_series, hp.ref_sample, '_'.join([ref_run for ref_run in hp.ref_runs]))
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
