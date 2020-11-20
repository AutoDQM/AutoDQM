
AutoDQM is an open source project. As such, we are constantly ensuring that AutoDQM is configurable for all subsystems that are interested in using it. With Release v2.1.0, it is now possible to add algorithms to AutoDQM's processing step, making it easy for anyone interested in improving the tool to contribute.

#. `The Configuration JSON <https://github.com/jkguiang/AutoDQM/wiki/Configuration#the-configuration-json>`_
#. `Adding Your Subsystem <https://github.com/jkguiang/AutoDQM/wiki/Configuration#adding-your-subsystem>`_
#. `Adding an Algorithm <https://github.com/jkguiang/AutoDQM/wiki/Configuration#adding-an-algorithm>`_

The Configuration JSON
----------------------

If AutoDQM is not already configured for your subsystem, you can easily push configurations for it. AutoDQM has a conveniently named ```configs.json`` <https://github.com/jkguiang/AutoDQM/blob/release-v2.0.0/configs.json>`_\ , where each entry in the JSON contains the appropriate instructions for AutoDQM's processing algorithms. Let's break it down:

Example entry:
~~~~~~~~~~~~~~

.. code-block:: python

   1 "CSC": {
   2     "main_gdir":"DQMData/Run {0}/CSC/Run summary/CSCOfflineMonitor/",
   3     "hists":[
   4       {
   5         "ks_cut":0.0,
   6         "norm_type": "row",
   7         "path":"Occupancy/hORecHits"
   8       }

   ...

   25      {
   26        "ks_cut":0.0,
   27        "path":"recHits/hRHGlobal*"
   28      },

   ...

   40      {
   41        "always_draw":true,
   42        "path":"Digis/hWirenGroupsTotal"
   43      },

`"CSC"`: Each entry in the JSON is indexed by the acronym of the subsystem: in this case, "CSC". Each subsystem is, itself, another python dictionary.

``"main_gdir"``\ : The path to the DQM histograms for any subsystem included in the DQM ``.root`` files. Generally, each run has a different run number in its directory path, so we put a pythonic ``"{0}"`` in its place so that the path string may be used dynamically in the processing script.

``"hists"``\ : This is a list of python dictionaries which each contain the path to specific histograms for AutoDQM to process as well as several tuning parameters.


* ``"path"``\ : This specifies the path to the histogram. If there are many histograms of the same type, you may use an asterisk to tell AutoDQM to find all histograms of a matching path (see line 27 above). ``**\ *This is the ONLY required parameter for every histogram*\ **``
* ``"ks_cut"``\ : This is the cut for the Kolmogorov-Smirnov Test that is run on 1D histograms.
* ``"norm_type"``: Currently AutoDQM only normalizes histograms by two schemes: row-by-row or as a whole. If you would like AutoDQM to normalize this histogram row-by-row, you would include this option, pointing to ``"row"``. Otherwise, you do not need to include this parameter.
* ``"always_draw"``\ : Include this parameter pointing to the boolean value ``true`` to tell AutoDQM to *always* draw this histogram.

Adding Your Subsystem
---------------------

To add your subsystem, simply clone the latest release, make the necessary additions to ``configs.json``\ , then make a pull request.

Adding an Algorithm
-------------------

To add an algorithm to AutoDQM's processing step, first contact us at autodqm@gmail.com so that we may collaborate on designing an appropriate addition. Assuming that the decision has been made to add your algorithm, you must first ensure that it is properly formatted:

The Comparator Object
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def comparators():
       return {
           'new_algo': new_algo
       }

This object associates a string to the algorithm function you've written. This allows AutoDQM to find and use your algorithm.

Algorithm Arguments and the Histpair object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def new_algo(histpair,
                new_cut=500, min_entries=100000, new_option='new_opt',
                **kwargs):

Every algorithm that AutoDQM uses must handle a ```histpair`` <https://github.com/jkguiang/AutoDQM/blob/release-v2.1.0/autodqm/histpair.py>`_\ object. Put simply, each ``histpair`` object contains all of the information passed from the user's input (i.e. the name of the data and reference runs, the series and samples of those runs, etc.). Any other key word arguments should be specified or otherwise passed through ``**kwargs``.

Plugin Results Object
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from autodqm.plugin_results import PluginResults

   def new_algo( ..., **kwargs ):

       """
           New algorithm contents
       """

       return PluginResults(
           c,
           show=is_outlier,
           info=info,
           artifacts=artifacts

In order for AutoDQM to understand your algorithm's output (i.e. drawn histograms, text files, etc.), you must pass them in a ```PluginResults`` <https://github.com/jkguiang/AutoDQM/blob/release-v2.1.0/autodqm/plugin_results.py>`_\ object.

When your algorithm has been properly formatted, you can make a pull request to AutoDQM's `development <https://github.com/jkguiang/AutoDQM/tree/develop-lxplus>`_ branch, making sure to place it in the `plugins <https://github.com/jkguiang/AutoDQM/tree/develop-lxplus/plugins>`_ directory.
