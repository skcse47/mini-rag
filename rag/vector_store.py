import json


def save_vector_store(vector_store, filename):

    with open(filename, "w") as file:
        json.dump(vector_store, file)


def load_vector_store(filename):

    with open(filename, "r") as file:
        return json.load(file)