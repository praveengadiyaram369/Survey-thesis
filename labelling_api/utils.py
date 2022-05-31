import json
import os
import faiss
import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub

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

es_index = 'mitera_scraped_docs'

tf_model = hub.load(basepath+ '/../input/USE_model')
index = faiss.read_index(basepath+"/../input/vector.index")
doc_df = pd.read_pickle(basepath+'/../input/final_dataframe.pkl')

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

def get_keyword_query(query, lang, key='match'):

    root_query = {key: {}}

    if lang == 1:
        root_query[key]['contents.de'] = query
    elif lang == 2:
        root_query[key]['contents.en'] = query
    elif lang == 3:
        root_query[key]['contents.default'] = query

    return root_query


def generate_query(query, lang, phrase_query, fuzzy_query):
        
    if phrase_query:
        return get_keyword_query(query, lang, key='match_phrase')
    elif fuzzy_query:
        return get_keyword_query(query, lang, key='fuzzy')
    else:
        return get_keyword_query(query, lang, key='match')

def handle_search_queries(es, query, lang, phrase_query, fuzzy_query, match_top):

    query_dict = generate_query(query, lang, phrase_query, fuzzy_query)
    # search_query = json.dumps(query_dict)

    results = es.search(index=es_index, query=query_dict, size=match_top)

    total_hits = results['hits']['total']['value']
    results = results['hits']['hits']

    result_list = []
    for doc_data in results:
        doc_dict = dict()

        doc_dict['title'] = doc_data['_source']['title']
        doc_dict['text'] = doc_data['_source']['contents']['default']
        doc_dict['page_url'] = doc_data['_source']['page_url']

        result_list.append(doc_dict)

    return total_hits, result_list

def handle_count_queries(es, query, match_top):

    result_list = []
    total_hits = 0

    for term in concept_mapping_dict[query]:
        count_query = json.dumps(get_keyword_query(term, 3, key='match_phrase'))
        results = es.count(index=index, body=count_query)

        concept_hits = results['count']
        result_list.append({'term': term, 'hits': concept_hits, 'url': f'?name1=value1'})

        total_hits += concept_hits

    return total_hits, result_list


def get_keyword_query(query, lang, key='match'):

    root_query = {'query':{key: {}}}

    if lang == 1:
        root_query['query'][key]['contents.de'] = query
    elif lang == 2:
        root_query['query'][key]['contents.en'] = query
    elif lang == 3:
        root_query['query'][key]['contents.default'] = query

    return root_query


def generate_query(query, lang, phrase_query):

    if not phrase_query:
        return get_keyword_query(query, lang, key='match')
    elif phrase_query:
        return get_keyword_query(query, lang, key='match_phrase')

def handle_search_queries(es, query, lang, phrase_query, match_top):

    search_query = json.dumps(generate_query(query, lang, phrase_query))
    results = es.search(index=index, body=search_query, size=match_top)

    total_hits = results['hits']['total']['value']
    results = results['hits']['hits']

    result_list = []
    for doc_data in results:
        doc_dict = dict()

        doc_dict['title'] = doc_data['_source']['title']
        doc_dict['text'] = doc_data['_source']['contents']['default']
        doc_dict['page_url'] = doc_data['_source']['page_url']

        result_list.append(doc_dict)

    return total_hits, result_list


def get_query_result(es, query, lang, phrase_query, search_concept, match_top):

    if not search_concept:
        return handle_search_queries(es, query, lang, phrase_query, match_top)

def get_query_result_semantic(query, lang, match_top):

    query_embedding = tf_model(query)['outputs'].numpy()[0].reshape(1, -1)
    result = index.search(np.float32(query_embedding), match_top)

    df = doc_df.iloc[result[1][0]]

    if lang == 1:
        language_code = 'de'
    elif lang == 2:
        language_code = 'en'
    elif lang == 3:
        language_code = 'xlm'

    result_list = []
    for idx, doc_data in df.iterrows():
        doc_dict = dict()

        if language_code == doc_data['lang'] or language_code == 'xlm':

            doc_dict['title'] = doc_data['title']
            doc_dict['text'] = doc_data['text']
            doc_dict['page_url'] = doc_data['url']

            result_list.append(doc_dict)

    total_hits = len(result_list)

    return total_hits, result_list