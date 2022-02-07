import json
import os

technology_document_data_path = os.getcwd() + '/../data/input/technologie_document_data.json'
military_document_data_path = os.getcwd() + '/../data/input/military_document_data.json'

relevant_technology_data_path = os.getcwd() + '/../data/output/relevant_documents_tech.json'
relevant_military_data_path = os.getcwd() + '/../data/output/relevant_documents_milt.json'
irrelevant_document_data_path = os.getcwd() + '/../data/output/irrelevant_documents.json'

def read_document_data(filepath):

    try:
        with open(filepath, 'r') as f:
            data_dict = json.load(f)
    except Exception as e:
        data_dict = dict()
        print(e)

    return data_dict

def load_document_data(type='technology'):

    if type == 'technology':
        document_data = read_document_data(technology_document_data_path)
        relevant_document_data = read_document_data(relevant_technology_data_path)
        irrelevant_document_data = read_document_data(irrelevant_document_data_path)
    elif type == 'military':
        document_data = read_document_data(military_document_data_path)
        relevant_document_data = read_document_data(relevant_military_data_path)
        irrelevant_document_data = read_document_data(irrelevant_document_data_path)

    return document_data, relevant_document_data, irrelevant_document_data

def write_document_data(data, filepath):

    with open(filepath, 'w') as f:
        json.dump(data, f)

def save_document_data(document_data, relevant_document_data, irrelevant_document_data, type='technology'):

    if type == 'technology':
        write_document_data(document_data, technology_document_data_path)
        write_document_data(relevant_document_data, relevant_technology_data_path)
        write_document_data(irrelevant_document_data, irrelevant_document_data_path)
    elif type == 'military':
        write_document_data(document_data, military_document_data_path)
        write_document_data(relevant_document_data, relevant_military_data_path)
        write_document_data(irrelevant_document_data, irrelevant_document_data_path)