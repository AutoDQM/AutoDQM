import os
import sys

if __name__ == "__main__":
    arg = sys.argv[1] # argument from tools.sh
    c_id = open("temp.txt", "r").read().split("\n")[1].split(" ")[0] # container ID
    # API
    if arg == "open":
        print("Opening {0}".format(c_id))
        os.system("docker exec -ti {0} /bin/bash".format(c_id))
    elif arg == "logs":
        print("Fetching {0} logs".format(c_id))
        os.system("docker logs {0}".format(c_id))
    elif arg == "scp":
        targ_file = sys.argv[2]
        targ_dir = sys.argv[3]
        print("Copying {0} to {1}".format(targ_file, targ_dir))
        os.system("docker cp {0} {1}:{2}".format(targ_file, c_id, targ_dir))
    elif arg == "stop":
        print("Stopping: {0}".format(c_id))
        os.system("docker stop {0}".format(c_id))