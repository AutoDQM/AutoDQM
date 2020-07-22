## Online
The AutoDQM GUI is currently hosted on a VM at CERN, which can be accessed in a few simple steps. If you're on the CERN network simply navigate to http://autodqm.cern.ch, otherwise you need to set up a proxy into the CERN network:

1. Setup your browser's SOCKS Proxy settings (Instructions: [Firefox](https://github.com/jkguiang/AutoDQM/wiki/SOCKS-Proxy:-Firefox) (recommended), [Chrome](https://github.com/jkguiang/AutoDQM/wiki/SOCKS-Proxy:-Chrome))
2. `ssh` into CERN from your terminal

   `>> ssh <username>@lxplus.cern.ch -ND 1080`
3. Navigate to http://autodqm.cern.ch from the browser you set up.

## Offline
The `./run-offline.py` script can retrieve run data files and process them without needing a web server. Run `./run-offline.py --help` for all the options.

1. Supply certificate files to the environment variables below. Alternatively, the default uses the files produced when running voms-proxy-init, so that may work instead.
2. Use `./run-offline.py` to process data with AutoDQM