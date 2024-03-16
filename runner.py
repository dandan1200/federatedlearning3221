import subprocess

num_clients_subsample = 0
method = 0

run_server = f"python COMP3221_FLServer.py 6000 {num_clients_subsample}"

run_clients = [
    f"python COMP3221_FLClient.py client1 6001 {method}",
    f"python COMP3221_FLClient.py client2 6002 {method}",
    f"python COMP3221_FLClient.py client3 6003 {method}",
    f"python COMP3221_FLClient.py client4 6004 {method}",
    f"python COMP3221_FLClient.py client5 6005 {method}",
]


for program in [run_server] + run_clients:
    command = [
        "osascript",
        "-e",
        'tell app "Terminal" to do script '
        f'"cd ~/Downloads/federated-learning && {program}"',
    ]
    with subprocess.Popen(command):
        ...
