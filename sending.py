def send_data_in_chunks(socket, data, buffer_size):
    data_chunks = [data[i : i + buffer_size] for i in range(0, len(data), buffer_size)]
    i = 0
    for chunk in data_chunks:
        socket.sendall(chunk)
        i += 1
    socket.sendall(b"EOF")


def recv_data_in_chunks(socket, buffer_size):
    data_chunks = []
    i = 0
    while True:
        chunk = socket.recv(buffer_size)
        data_chunks.append(chunk)
        if chunk is None:
            return False
        if chunk.endswith(b"EOF"):
            break
        i += 1
    return b"".join(data_chunks).removesuffix(b"EOF")
