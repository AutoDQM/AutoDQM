
This shows how to set up AutoDQM to be served from your local machine or a machine on `CERN OpenStack <https://openstack.cern.ch/project/instances/>`_. This was written based on a fresh CC7 VM on CERN OpenStack. 

You'll need a CERN User/Host certificate authorized with the CMS VO. CMS VO authorization can take ~8 hours so bear that in mind. Certificates can be aquired either from https://cern.ch/ca or, on a CC7 machine, by using auto-enrollment https://ca.cern.ch/ca/Help/?kbid=024000.

Install docker according to https://docs.docker.com/install/ and docker-compose through pip because CC7 has an old versions in it's repositories.
Enable+start the docker service, and be sure to add your user to the docker group.

.. code-block:: sh

   sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
   sudo yum install docker-ce -y
   sudo yum install python3-pip -y
   sudo pip3 install docker-compose

.. code-block:: sh

   sudo gpasswd -a [user] docker
   sudo systemctl enable --now docker

You may need to relog into your account before the group settings take effect.

Store your CERN certificate into docker secrets. You may need to extract your cert from PKCS12 format:

.. code-block:: sh

   openssl pkcs12 -in cern-cert.p12 -out cern-cert.public.pem -clcerts -nokeys
   openssl pkcs12 -in cern-cert.p12 -out cern-cert.private.key -nocerts -nodes

.. code-block:: sh

   docker swarm init
   docker secret create cmsvo-cert.pem cern-cert.public.pem
   docker secret create cmsvo-cert.key cern-cert.private.key

Then initialize a docker swarm, build the autodqm image with docker-compose, and deploy the image as a docker stack

.. code-block:: sh

   docker-compose build
   docker stack deploy --compose-file=./docker-compose.yml autodqm
   docker run -d -p 80:80 autodqm

You can now view AutoDQM at ``http://[image-name].cern.ch``. If you would like to make your instance of AutoDQM public, open port 80 to http traffic on your firewall. For example, on CC7:

.. code-block:: sh

   sudo firewall-cmd --permanent --add-port=80/tcp
   sudo firewall-cmd --reload

Nonetheless, to access the GUI, you will need to configure your browser to access the GUI via a SOCKS proxy (instructions `here <https://github.com/jkguiang/AutoDQM/wiki/Using-AutoDQM>`_\ ).

After making changes to configuration or source code, rebuild and redeploy the newly built image:

.. code-block:: sh

   docker stack rm autodqm
   docker-compose build
   docker stack deploy --compose-file=./docker-compose.yml autodqm
   docker run -d -p 80:80 autodqm
