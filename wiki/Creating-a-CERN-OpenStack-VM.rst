
First, navigate to CERN `OpenStack <https://openstack.cern.ch/project/instances/>`_. After signing in using your CERN credentials, you should see the page below after selecting "Instances" in the left-hand nav menu under "Compute." Then, click "Launch Instance"

[[images/making-vm/step1.png]]

Now we will begin to configure your VM. First, you need to give it a name

[[images/making-vm/step2.png]]

Now, you will need to specify the OS you would like it to run. For AutoDQM, we use CC7

[[images/making-vm/step3.png]]

[[images/making-vm/step4.png]]

After that, you need to specify the quantity of disk space you would like to reserve for it (go big or go home!)

[[images/making-vm/step5.png]]

Finally, you have to generate a key-pair for it

[[images/making-vm/step6.png]]

With that, you are ready to launch your instance

[[images/making-vm/step7.png]]

After it has finished setting up (it will take a bit), ssh into lxplus. From there, you can ssh into your brand-new virtual machine!

.. code-block:: bash

   ssh [name].cern.ch
