AutoDQM
=======

AutoDQM parses DQM histograms and identifies outliers by various
statistical tests for further analysis by the user. Its output can be
easily parsed by eye on an AutoPlotter-based html page which is
automatically generated when you submit a query from the AutoDQM GUI.
Full documentation for AutoDQM can be found on our
`wiki <https://autodqm-official.readthedocs.io/en/latest/index.html>`__.

AutoDQM_ML
========

| The AutoDQM_ML repository is a toolkit for developing machine learning algorithms to detect anomalies in offline DQM histograms. 
| `Github repo <https://github.com/AutoDQM/AutoDQM_ML>`_ 
| `AutoDQM_ML wiki <https://autodqm-ml.readthedocs.io/en/latest/index.html>`_ 
| `AutoDQM_ML tutorial <https://autodqm.github.io/autodqm_ml.github.io/>`_


Index
========
1. `Features <#features>`__
2. `Setting Up AutoDQM for
   Development <#setting-up-autodqm-for-development>`__
3. `Using AutoDQM Offline <#using-autodqm-offline>`__
4. `Environment Variables <#environment-variables>`__
5. `Deleting Images and Root Files to Free Up Space <#deleting-images>`__

Features
--------

AutoDQM.py
          

-  [x] Outputs histograms that clearly highlight outliers
-  [x] Creates a .txt file along with each .pdf with relevant
   information on it
-  [x] Allows user to easily change input
-  [x] Seeks and accurately finds outliers

index.php
         

-  [x] Previews input in a readable way
-  [x] Gives a clear indication of the status of a user's query

plots.php
         

-  [x] Dynamically displays text files below AutoPlotter toolbar
-  [x] Unique url's for sharing plots pages with the data and reference
   data set names

Setting Up AutoDQM for Development
----------------------------------

This shows how to set up AutoDQM to be served from a machine on CERN OpenStack. This was written based on a fresh CC7 VM on
CERN OpenStack.

You'll need a CERN User/Host certificate authorized with the CMS VO. CMS
VO authorization can take ~8 hours so bear that in mind. Certificates
can be aquired either from https://cern.ch/ca or, on a CC7 machine, by
using auto-enrollment https://ca.cern.ch/ca/Help/?kbid=024000.

Install docker according to https://docs.docker.com/install/ and
docker-compose through pip because CC7 has an old versions in it's
repositories. Enable+start the docker service, and be sure to add your
user to the docker group.

.. code:: sh

    sudo yum-config-manager \
        --add-repo \
        https://download.docker.com/linux/centos/docker-ce.repo
    sudo yum install docker-ce -y
    sudo yum install python3-pip -y
    sudo pip3 install --upgrade pip 
    sudo pip3 install docker-compose

.. code:: sh

    sudo gpasswd -a [user] docker
    sudo systemctl enable --now docker

You may need to relog into your account before the group settings take
effect.

Store your CERN certificate into docker secrets. You may need to extract
your cert from PKCS12 format:

.. code:: sh

    openssl pkcs12 -in cern-cert.p12 -out cern-cert.public.pem -clcerts -nokeys
    openssl pkcs12 -in cern-cert.p12 -out cern-cert.private.key -nocerts -nodes

.. code:: sh

    docker swarm init
    docker secret create cmsvo-cert.pem cern-cert.public.pem
    docker secret create cmsvo-cert.key cern-cert.private.key 

Then initialize a docker swarm, build the autodqm image with
docker-compose, and deploy the image as a docker stack

.. code:: sh

    docker-compose build
    docker stack deploy --compose-file=./docker-compose.yml autodqm

To view AutoDQM, first your browser proxy will need to be set to listen to a port. Insturctions to do this can be found `here <https://github.com/AutoDQM/AutoDQM/wiki/Set-up-Firefox-proxy-for-viewing-private-AutoDQM-version>`_. 

After setting the proxy on your browser, using your local terminal (not ssh-ed into anything), forward your lxplus connection: 

.. code:: sh
    
    ssh <cmsusr>@lxplus.cern.ch -ND <port>

Note: Any port number will work so long as you match this forwarded port number to the port number in the browser network settings.


You can now view AutoDQM at ``<VM name>.cern.ch:8083/dqm/autodqm/``. If you would like to
make your instance of AutoDQM public, open port 8083 to http traffic on
your firewall. For example, on CC7:

.. code:: sh

    sudo firewall-cmd --permanent --add-port=8083/tcp
    sudo firewall-cmd --reload

After making changes to configuration or source code, rebuild and
redeploy the newly built image:

.. code:: sh

    docker-compose build
    docker stack rm autodqm
    docker stack deploy --compose-file=./docker-compose.yml autodqm

If you're using a CC7 image, you may want to disable autoupdate:

.. code:: sh

    sudo systemctl stop yum-autoupdate.service
    sudo systemctl disable yum-autoupdate.service

Using AutoDQM Offline
---------------------

The ``runoffline/run-offline.py`` script can retrieve run data files and process
them without needing a web server. Run ``runoffline/run-offline.py --help`` for
all the options.

``run-offline.py`` requires some packages (listed in ``runoffline/environment.yml``) to run. This environment can be created using conda. If you don't already have a conda installation, you can run: 

.. code:: sh

    curl -O -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b

Then to activate conda:

.. code:: sh

   source ~/.bashrc 
   
To create the environment, go into the ``runoffline`` directory, then run:

.. code:: sh

    conda env create -f environment.yml
    
The conda environment can then be activated with 

.. code:: sh
  
    conda activate autodqm 
    

``run-offline.py`` requires some environment variables to be set in order to run. ``setenvvar.sh`` has all the required environment variables for running the script. It assumes that you cloned AutoDQM into your ``/root/`` directory and that your cert and key lives in ``/root/.globus`` directory. If that is not the case, you can edit the ``setenvvar.sh`` file to match your setup. To set the environment variables, run: 

.. code:: sh 
  
    source setenvvar.sh 
    
You don't need to make the directories defined by ``ADQM_OUT``, ``ADQM_TMP``, ``ADQM_DB`` prior to running ``run-offline.py`` as these will be created the first time you run the script if they do not exist.

Now inside ``runoffline`` directory, you can use ``run-offline.py`` to process data with AutoDQM! 

Example command: 

.. code:: sh
    
    ./run-offline.py Offline RPC Run2022 SingleMuon 355443 355135

This analyzes RPC plots, using Run2022 series, SingleMuon sample (both data and reference), comparing run 355443 (data) and run 355135 (reference). 




Descriptions of the Environment Variables
-----------------------------------------

-  ``ADQM_CONFIG`` location of the configuration file to use
-  ``ADQM_DB`` location to store downloaded root files from offline DQM
-  ``ADQM_TMP`` location to store generated temporary pdfs, pngs, etc
-  ``ADQM_OUT`` location to store the result of AutoDQM
-  ``ADQM_PLUGINS`` location of thep plugins folder
-  ``ADQM_SSLCERT`` location of CMS VO authorized public key certificate
   to use in querying offline DQM
-  ``ADQM_SSLKEY`` location of CMS VO authorized private ky to use in
   querying offline DQM
-  ``ADQM_CACERT`` location of a CERN Grid CA certificate chain, if
   needed



Deleting Docker images and root files to free up space
------------------------------------------------------

Docker images and root files used to test AutoDQM take up space and can accumulate overtime. Since we only have 40Gb max on the large flavor virtual machines, it might be necessary to free up some space in order to keep testing new images and runs. 

To delete a specific docker image, you can do the following: 

.. code:: sh 

    docker image ls #lists docker images 
    docker image rm <image ID> 

To free up space for more root files, head over to this directory 

.. code:: sh 

    /var/lib/docker/volumes/autodqm_adqm-db/_data 

You will see directories `dqm_offline`, `Offline`, and `Online`. It is safe to delete the root files (files names are <run number>.root) and text files (file names are string of numbers) in these directories. DO NOT delete CERN_Root_CA.crt. That is the certificate used to access the root files from DQM. 
