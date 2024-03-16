import pickle
import random
import signal
import socket
import sys
import threading
import time
from statistics import mean

import numpy as np
import torch

from constants import BUFFER_SIZE, NUM_GLOBAL_ITERATIONS, REGISTRATION_WINDOW_SECONDS
from model import MCLR
from sending import recv_data_in_chunks, send_data_in_chunks

np.random.seed(0)  # Set the random seed

# Read command-line arguments
port_server = int(sys.argv[1])
sub_client = int(sys.argv[2])

# Initialise global model
global_model = MCLR()


clients = {}

lock = threading.Lock()
first_client_registered = threading.Event()


def accept_clients():
    # Setup server socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFFER_SIZE)
    server_socket.bind(("localhost", port_server))
    server_socket.listen()

    while True:
        client_socket, _ = server_socket.accept()
        data = client_socket.recv(BUFFER_SIZE)
        hand_shaking_message = pickle.loads(data)

        if hand_shaking_message.get("action") != "register":
            continue

        if not first_client_registered.is_set():
            first_client_registered.set()

        with lock:
            clients[hand_shaking_message["client_id"]] = {
                "socket": client_socket,
                "data_size": hand_shaking_message["data_size"],
            }


def broadcast(message):
    for client_data in clients.values():
        client_socket = client_data["socket"]
        # Send the global model to the client
        send_data_in_chunks(client_socket, pickle.dumps(message), BUFFER_SIZE)


def aggregate_local_models(num_clients_in_subset):
    # Clear global model before aggregation
    for param in global_model.parameters():
        param.data = torch.zeros_like(param.data)

    if num_clients_in_subset == 0:
        clients_subset = clients.keys()
    else:  # Randomly select clients according to number that should be subset
        clients_subset = random.sample(list(clients), num_clients_in_subset)

    total_train_samples = sum(clients[c]["data_size"] for c in clients_subset)
    for client in clients_subset:
        local_model = clients[client]["local_model"]
        for global_param, client_param in zip(
            global_model.parameters(), local_model.parameters()
        ):
            global_param.data = (
                global_param.data
                + client_param.data.clone()
                * clients[client]["data_size"]
                / total_train_samples
            )
    return {
        "average_accuracy": mean(clients[c]["accuracy"] for c in clients_subset),
        "average_loss": mean(clients[c]["loss"] for c in clients_subset),
    }


def timeout_handler(signum, frame):
    raise TimeoutError


def receive_message_from_client(client, timeout=1):
    """Try to receive a message from a client, returns False if client has dropped out."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        # Attempt to receive the message up to 3 times before giving up
        for _ in range(3):
            try:
                client_socket = client["socket"]
                data = recv_data_in_chunks(client_socket, BUFFER_SIZE)
                return data
            except (BlockingIOError, ConnectionResetError):
                # Sleep for 1 second and try again
                time.sleep(1)
            except ConnectionRefusedError:
                # Stop trying if the connection is refused
                break

    except TimeoutError:
        print("Timeout reached")
        return False

    finally:
        signal.alarm(0)  # Cancel the alarm

    return False


def main():
    registration_thread = threading.Thread(target=accept_clients)
    registration_thread.start()

    # Wait for the first client to register and for the registration window to expire
    first_client_registered.wait()
    time.sleep(REGISTRATION_WINDOW_SECONDS)

    average_accuracy = []
    average_loss = []

    for run in range(NUM_GLOBAL_ITERATIONS):
        lock.acquire()
        print("Broadcasting new global model")
        broadcast({"global_model": global_model})

        print(f"Global Iteration {run+1}")

        # Wait for all clients to send their local models
        disconnected_clients = []
        for client_id, client_data in clients.items():
            print(f"Getting local model from client {client_id}")
            response = receive_message_from_client(client_data)
            if not response:
                disconnected_clients.append(client_id)
                continue
            data = pickle.loads(response)
            client_data["local_model"] = data.get("local_model")
            client_data["accuracy"] = data.get("accuracy")
            client_data["loss"] = data.get("loss")

        for client_id in disconnected_clients:
            del clients[client_id]

        print("Aggregating new global model")
        global_model_statistics = aggregate_local_models(sub_client)
        average_accuracy.append(global_model_statistics["average_accuracy"])
        average_loss.append(global_model_statistics["average_loss"])

        print(f"Global Iteration {run + 1}: Aggregating new global model")
        lock.release()
        time.sleep(0.2)

    # Broadcast a message to stop the training process to all clients
    broadcast({"action": "stop"})

    # Exit once complete
    sys.exit(0)


if __name__ == "__main__":
    main()
