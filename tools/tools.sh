docker ps > temp.txt
if [[ "${1}" == "open" ]] ; then
    python tools/tools.py open
elif [[ "${1}" == "logs" ]] ; then
    python tools/tools.py logs
elif [[ "${1}" == "scp" ]] ; then
    if [[ "${2}" != "" ]] && [[ "${3}" != "" ]] ; then
        python tools/tools.py scp ${2} ${3}
    else
        echo "Usage: ./tools.sh scp <origin_path> <target_path>"
    fi
elif [[ "${1}" == "updt" ]] ; then
    docker stack rm autodqm
    python tools/tools.py "stop"
    docker-compose build
    sleep 7
    docker stack deploy --compose-file=./docker-compose.yml autodqm
    docker run -d -p 80:80 autodqm
elif [[ "${1}" == "run" ]] ; then
    docker stack rm autodqm
    docker-compose build
    sleep 7
    docker stack deploy --compose-file=./docker-compose.yml autodqm
    docker run -d -p 80:80 autodqm
elif [[ "${1}" == "keys" ]] ; then
    python tools/tools.py scp cern-cert.public.pem /run/secrets/cmsvo-cert.pem
    python tools/tools.py scp cern-cert.private.key /run/secrets/cmsvo-cert.key
elif [[ "${1}" == "stop" ]] ; then
    docker stack rm autodqm
    python tools/tools.py "stop"
else
    echo "Command ${1} not supported."
fi
rm temp.txt
