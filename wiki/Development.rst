
We have written instructions on how to set up a development version of `AutoDQM <https://github.com/jkguiang/AutoDQM/wiki/Running-the-Web-GUI>`_. Unless you have local DQM files on hand, we recommend using a CERN OpenStack VM. Instructions for setting one up can be found `here <https://github.com/jkguiang/AutoDQM/wiki/Creating-a-CERN-OpenStack-VM>`_.

Environment Variables
---------------------


* ``ADQM_CONFIG`` location of the configuration file to use
* ``ADQM_DB`` location to store downloaded root files from offline DQM
* ``ADQM_TMP`` location to store generated temporary pdfs, pngs, etc
* ``ADQM_SSLCERT`` location of CMS VO authorized public key certificate to use in querying offline DQM
* ``ADQM_SSLKEY`` location of CMS VO authorized private ky to use in querying offline DQM
* ``ADQM_CACERT`` location of a CERN Grid CA certificate chain, if needed

Join the team!
--------------

Interested in contributing to AutoDQM? Just shoot us an email at autodqm@gmail.com. If you're anxious to get started, just clone the latest release and follow the instructions in the `README <https://github.com/jkguiang/AutoDQM/blob/release-v2.0.0/README.md>`_ to get a working development environment. Note: you will need Docker on whatever machine or server you clone AutoDQM to in order to run the web browser outside of the dedicated CERN VM.

Current Developers
^^^^^^^^^^^^^^^^^^


* `Jonathan Guiang <https://github.com/jkguiang>`_
* `Alex Aubuchon <https://github.com/A-lxe>`_
