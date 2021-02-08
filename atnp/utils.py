import os
import csv
import json
import pickle
import joblib

LINK_PATTERN = r"https?:\/\/(www\.)?([-\w@:%._\+~#=]+\.[\w()]+)\b([-\w()@:%_\+.~#?&\/\/=,]*)"

def create_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_json(base, name, file):
    create_if_not_exists(base)
    open(os.path.join(base, name + ".json"), "w+").write(json.dumps(file))


def save_pickle(base, name, document):
    create_if_not_exists(base)
    with open(os.path.join(base, name), 'wb') as f:
        pickle.dump(document, f, pickle.HIGHEST_PROTOCOL)

def save_joblib(base, name, document):
    create_if_not_exists(base)
    joblib.dump(document, os.path.join(base, name)) 

def get_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_csv(base, name, table):
    create_if_not_exists(base)

    with open(os.path.join(base, name + ".csv"), "w", newline='\n') as file_csv:
        writer = csv.writer(file_csv)
        writer.writerows(table)