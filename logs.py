def log(message, file=None):
    message = "\n".join(l.strip() for l in message.split("\n")).strip(" \n")
    if file is not None:
        with open(file, "a", encoding="UTF-8") as f:
            f.write(message + "\n")
    print(message)
