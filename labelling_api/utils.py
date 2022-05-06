import json
import os

basepath = '/usr/src/web_app/data'
# basepath = 'C:\\Users\\sri.sai.praveen.gadi\\Documents\\Projects\\mitera_data_annotator\\data'

technology_document_data_path = basepath + '/input/technologie_document_data.json'
military_document_data_path =   basepath + '/input/military_document_data.json'
augmented_pos_document_data_path =  basepath + '/input/augmented_pos_document_data.json'

relevant_technology_data_path = basepath + '/output/relevant_documents_tech.json'
relevant_military_data_path =  basepath + '/output/relevant_documents_milt.json'
irrelevant_document_data_path = basepath + '/output/irrelevant_documents.json'

# classified_pos_docs_path = basepath + '/input/predicted_unlabeled_docs.json'
classified_pos_docs_path = basepath + '/input/new_labeled_negative_set.json'
third_class_docs_path = basepath + '/output/third_class_data.txt'
aug_docs_path = basepath + '/output/doc_aug_info_data.txt'

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
    elif type == 'military':
        document_data = read_document_data(military_document_data_path)
    elif type == 'aug_pos':
        document_data = read_document_data(augmented_pos_document_data_path)
        return document_data

    tech_relevant_document_data = read_document_data(relevant_technology_data_path)
    milt_relevant_document_data = read_document_data(relevant_military_data_path)
    irrelevant_document_data = read_document_data(irrelevant_document_data_path)

    return document_data, tech_relevant_document_data, milt_relevant_document_data, irrelevant_document_data

def write_document_data(data, filepath):

    with open(filepath, 'w') as f:
        json.dump(data, f)

def save_document_data(document_data, tech_relevant_document_data, milt_relevant_document_data, irrelevant_document_data, type='technology'):

    if type == 'technology':
        write_document_data(document_data, technology_document_data_path)
    elif type == 'military':
        write_document_data(document_data, military_document_data_path)

    write_document_data(tech_relevant_document_data, relevant_technology_data_path)
    write_document_data(milt_relevant_document_data, relevant_military_data_path)
    write_document_data(irrelevant_document_data, irrelevant_document_data_path)

def write_data_to_file(filepath, data):

    with open(filepath, "a") as f:
        f.write(data+'\n')