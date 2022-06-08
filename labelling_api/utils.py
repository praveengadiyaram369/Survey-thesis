from distutils.util import check_environ
import json
import os
import faiss
import pandas as pd
import numpy as np
import fasttext
import spellchecker
# from recursive_split import *

import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
from compound_split import char_split
from spellchecker import SpellChecker

basepath = '/usr/src/web_app/data'
# basepath = 'C:\\Users\\sri.sai.praveen.gadi\\Documents\\Projects\\mitera_data_annotator\\data'

technology_document_data_path = basepath + '/data/input/technologie_document_data.json'
military_document_data_path =   basepath + '/data/input/military_document_data.json'
augmented_pos_document_data_path =  basepath + '/data/input/augmented_pos_document_data.json'

relevant_technology_data_path = basepath + '/data/output/relevant_documents_tech.json'
relevant_military_data_path =  basepath + '/data/output/relevant_documents_milt.json'
irrelevant_document_data_path = basepath + '/data/output/irrelevant_documents.json'

# classified_pos_docs_path = basepath + '/data/input/predicted_unlabeled_docs.json'
classified_pos_docs_path = basepath + '/data/input/new_labeled_negative_set.json'
third_class_docs_path = basepath + '/data/output/third_class_data.txt'
aug_docs_path = basepath + '/data/output/doc_aug_info_data.txt'

es_index = 'mitera_scraped_docs'

tf_model = hub.load(basepath+ '/models/USE_model')
fasttext_model = fasttext.load_model(basepath + '/models/lid.176.bin')
fasttext.FastText.eprint = lambda x: None

index = faiss.read_index(basepath+"/vector.index")
doc_df = pd.read_pickle(basepath+'/final_dataframe.pkl')

english_checker = SpellChecker(language='en')
german_checker = SpellChecker(language='de')

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

def handle_count_queries(es, query, lang, phrase_query=True, fuzzy_query=False):

    query_dict = generate_query(query, lang, phrase_query, fuzzy_query)
    results = es.count(index=es_index, query=query_dict)

    return results['count']


def get_query_result(es, query, lang, phrase_query, fuzzy_query, search_concept, match_top):

    if not search_concept:
        return handle_search_queries(es, query, lang, phrase_query, fuzzy_query, match_top)

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

def portion_of_capital_letters(query):
    upper_cases = ''.join([c for c in query if c.isupper()])
    return len(upper_cases)/len(query)

def detect_language(text):
    
    lang_label = fasttext_model.predict(text)[0][0].split('__label__')[1]
    return lang_label

def detect_spelling_mistake(query, lang):

    spell_checker = None
    if lang == 'de' or lang == 1:
        spell_checker = german_checker
    elif lang == 'en' or lang == 2:
        spell_checker = english_checker

    correct_spelling = spell_checker.correction(query)
    return correct_spelling

def detect_german_compoundword(query):

    if len(query) > 14 and char_split.split_compound(query)[0][0] > 0:
        nouns_list = char_split.split_compound(query)
        if len(nouns_list) > 1 and nouns_list[0][0] > 0:
            return True
    return False

def get_optimum_search_strategy(es, query):

    lang = detect_language(query)
    search_type = None
    query_type = None
    updated_query = None
    comments = None
    query_parts = query.split()

    if lang == 'de':
        lang = 1
    elif lang == 'en':
        lang = 2

    if len(query_parts) > 3:
        search_type = 'semantic_search'
        query_type = None
        updated_query = None
        comments = 'Sentence query, token length > 3'
    elif len(query_parts) < 3 and len(query_parts) > 1:
        search_type = 'es_search'
        query_type = 'phrase_query'
        updated_query = None
        comments = 'Phrase match, token length < 3 and >1'
    elif len(query_parts) == 1:

        if portion_of_capital_letters(query) >= 0.75:
            search_type = 'es_search'
            query_type = None
            updated_query = None
            comments = 'Abbreviation detected'

        if detect_german_compoundword(query):
            search_type = 'semantic_search'
            query_type = None
            updated_query = None
            comments = 'German compound word detected'
        
        query_count_bm25 = handle_count_queries(es, query, lang, phrase_query=True, fuzzy_query=False)
        if query_count_bm25 == 0:
            updated_query = detect_spelling_mistake(query, lang)
            search_type = 'es_search'
            query_type = 'fuzzy_query'
            updated_query = updated_query
            comments = 'Fuzzy match, zero BM-25 results and correct spelling'
        else:
            search_type = 'es_search'
            query_type = None
            updated_query = updated_query
            comments = 'Simple query match'

    return lang, search_type, query_type, updated_query, comments

def get_search_type(search_type):

    if search_type == 'es_search':
        return 'BM-25'
    elif search_type == 'semantic_search':
        return 'Semantic search'
    elif search_type == 'optimistic_search':
        return 'Optimistic search'
    elif search_type == 'hybrid_search':
        return 'Hybrid search'

def get_language(lang):

    if lang == 'de' or lang == 1:
        return 'Deutsch'
    elif lang == 'en' or lang == 2:
        return 'English'
    elif lang == 'xlm' or lang == 3:
        return 'Mulit-lingual'