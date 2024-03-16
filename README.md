# COMP3221_A2_Group7

## Run Server

```bash
python COMP3221_FLServer.py [server_port] [client_subsample]
```

- `server_port`: the port number for the server
- `client_subsample`: the numer of clients to randomly subsample when aggregating the global model (passing 0 uses all the clients).
  e.g. passing 3 will use a random random subsample of 3 clients for aggregation.

Example:

```bash
python COMP3221_FLServer.py 6000 3
```

## Run Client

```bash
python COMP3221_FLClient.py [client_id] [client_port] [optimisation_method]
```

- `client_id`:
- `client_port`:
- `optimisation_method`: `0` for Gradient Descent, `1` for Mini-Batch Gradient Descent (batch size set to 20 and can be modified in `constants.py`)

Example:

```bash
python COMP3221_FLClient.py client1 6001 0
```

## Features

The program implements all required features described in the specification, including:

- Subsampling
- Logging
- Aggreagation
- Client connections past the registration window.

It also implements a bonus: dealing with client failures.
To simulate a client failure, simply quit a client terminal process while it is running with 'CTRL + C'.
The server will automatically remove the client from the list of clients maintained on the server and will stop sending the global model to the client or aggregating its previous local models.
