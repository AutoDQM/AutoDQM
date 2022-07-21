
Editing the SOCKS Proxy Settings
--------------------------------


#. Exit all running instances of Chrome.
#. In the terminal (unless you're using `Windows <http://github.com/jkguiang/AutoDQM/wiki/Using-AutoDQM:-Chrome#windows>`_\ ), enter the following command to launch a new window of Chrome with the correct SOCKS Proxy settings:
   ###### Mac OS X
   .. code-block::

      >>> /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --proxy-server="socks5://localhost:1080" --host-resolver-rules="MAP * ~NOTFOUND , EXCLUDE localhost"
   ###### Linux
   .. code-block::

      >>> google-chrome --proxy-server="socks5://localhost:1080" --host-resolver-rules="MAP * ~NOTFOUND , EXCLUDE localhost"
   ###### Windows
#. If you don't already have a shortcut to Chrome on your desktop, navigate to ``chrome.exe`` wherever Chrome is located in your computer (likely location: ``C:\Program Files (x86)\Google\Chrome\Application``\ ). Then, right click on the executable and select "Create a Shortcut."
   [[images/using_autodqm/windows_socks_2.png]]
#. Locate the shortcut you just created, then right click on the icon and select "Properties."
   [[images/using_autodqm/windows_socks_3.png]]
#. The menu should start in the "Shortcut" tab. Add the following command into the "Target" text box after the default text:
   ``--proxy-server="socks5://localhost:1080" --host-resolver-rules="MAP * ~NOTFOUND , EXCLUDE localhost"``.
   The full "Target" text should look like this:
   .. code-block::

      "<path_to_chrome>\chrome.exe" --proxy-server="socks5://localhost:1080" --host-resolver-rules="MAP * ~NOTFOUND , EXCLUDE localhost"
   [[images/using_autodqm/windows_socks_4.png]]
