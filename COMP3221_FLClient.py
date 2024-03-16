import copy
import json
import pickle
import socket
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from constants import (
    BATCH_SIZE,
    BUFFER_SIZE,
    LEARNING_RATE,
    NUM_LOCAL_EPOCHS,
    SERVER_PORT,
)
from logs import log
from sending import recv_data_in_chunks, send_data_in_chunks

# Read command-line arguments
client_id = int(sys.argv[1].replace("client", ""))
port_client = int(sys.argv[2])
opt_method = {"0": "GD", "1": "MBGD"}[sys.argv[3]]


# Load the client's data
data_path = Path("FLdata")

with (data_path / f"train/mnist_train_client{client_id}.json").open() as f:
    train_data = json.load(f)
with (data_path / f"test/mnist_test_client{client_id}.json").open() as f:
    test_data = json.load(f)


# Connect to the server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFFER_SIZE)
server_socket.connect(("localhost", SERVER_PORT))

# Send hand-shaking message
hand_shaking_message = pickle.dumps(
    {
        "action": "register",
        "client_id": client_id,
        "data_size": train_data["num_samples"][0],
    }
)
server_socket.sendall(hand_shaking_message, BUFFER_SIZE)


# Prepare log file
log_file = Path(f"client{client_id}_log.txt")
if log_file.exists():
    log_file.unlink()


class LocalModel:
    """Class that handles local model training"""

    def __init__(self, client_id, model):
        self.get_data(train_data["user_data"], test_data["user_data"])
        self.train_data = list(zip(self.X_train, self.y_train))
        self.test_data = list(zip(self.X_test, self.y_test))

        self.trainloader = DataLoader(
            self.train_data,
            batch_size=len(self.train_data) if opt_method == "GD" else BATCH_SIZE,
            shuffle=True,
        )
        self.testloader = DataLoader(self.test_data, self.test_samples)

        self.loss = nn.NLLLoss()
        self.model = copy.deepcopy(model)
        self.id = client_id
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE)

    def get_data(self, train, test):
        X_train, y_train, X_test, y_test = (
            train["0"]["x"],
            train["0"]["y"],
            test["0"]["x"],
            test["0"]["y"],
        )
        self.X_train = torch.Tensor(X_train).view(-1, 1, 28, 28).type(torch.float32)
        self.y_train = torch.Tensor(y_train).type(torch.int64)
        self.X_test = torch.Tensor(X_test).view(-1, 1, 28, 28).type(torch.float32)
        self.y_test = torch.Tensor(y_test).type(torch.int64)
        self.train_samples = len(y_train)
        self.test_samples = len(y_test)

    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):  # pylint: disable=unused-variable
            self.model.train()
            for X, y in self.trainloader:
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
        return loss.data

    def test(self):
        self.model.eval()
        test_accuracy = 0
        for x, y in self.testloader:
            output = self.model(x)
            test_accuracy += (
                torch.sum(torch.argmax(output, dim=1) == y) / y.shape[0]
            ).item()
        return test_accuracy


def main():
    while True:
        # Receive the global model or stop message
        data = recv_data_in_chunks(server_socket, BUFFER_SIZE)
        message = pickle.loads(data)

        if message.get("action") == "stop":
            break

        if message.get("global_model") is not None:
            log(f"I am client {client_id}\nReceiving new global model", file=log_file)
            global_model = message.get("global_model")
            client = LocalModel(client_id, global_model)
            loss = float(client.train(NUM_LOCAL_EPOCHS))
            accuracy = client.test()

            log_message = f"""
                Training loss: {loss:.2f}
                Testing accuracy: {accuracy:.2%}
            """

            log(log_message, file=log_file)

            # Send the new local model to the server
            local_model_message = pickle.dumps(
                {"local_model": client.model, "accuracy": accuracy, "loss": loss}
            )

            send_data_in_chunks(server_socket, local_model_message, BUFFER_SIZE)


if __name__ == "__main__":
    main()
