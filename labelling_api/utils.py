from distutils.util import check_environ
import json
from operator import index
import os
from random import shuffle
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
from settings import *
from db_utils import *
from subtopic_utils import *

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
        doc_dict['pub_date'] = doc_data['_source']['published_date']
        doc_dict['id'] = doc_data['_source']['id']

        result_list.append(doc_dict)

    return total_hits, result_list

def handle_count_queries(es, query, lang, phrase_query=True, fuzzy_query=False):

    query_dict = generate_query(query, lang, phrase_query, fuzzy_query)
    results = es.count(index=es_index, query=query_dict)

    return results['count']

def write_query_results(query, result_list, search_type):

    query_updated = query.lower().replace(' ', '_')
    filename = search_results_folder + f'{query_updated}_{search_type}_result.json'

    data_dict = dict()
    for idx, result in enumerate(result_list):
        data_dict[str(idx)] = result['id']

    with open(filename, 'w') as f:
        json.dump(data_dict, f)

def get_query_result(es, query, lang, phrase_query, fuzzy_query, search_concept, match_top):

    if not search_concept:
        total_hits, result_list = handle_search_queries(es, query, lang, phrase_query, fuzzy_query, match_top)
        write_query_results(query, result_list, 'bm25')
        return total_hits, result_list

def get_max_diff_index(doc_similarities):

    diff_list = []
    sim_list = list(doc_similarities.values())
    for idx in range(1, len(sim_list)):
        diff_list.append(sim_list[idx]-sim_list[idx-1])
    
    max_diff = max(diff_list)
    max_diff_index =  diff_list.index(max_diff)

    return sim_list[max_diff_index]

def get_query_result_semantic(query, lang, match_top, cut_off = 0.64):

    match_top_org = match_top
    match_top += 10

    if lang == 1:
        index = de_index
        doc_df = de_df
    elif lang == 2:
        index = en_index
        doc_df = en_df
    elif lang == 3:
        index = xlm_index
        doc_df = xlm_df

    query_embedding = tf_model(query)['outputs'].numpy()[0]
    result = index.search(np.float32(query_embedding.reshape(1, -1)), match_top)

    df = doc_df.iloc[result[1][0]]

    doc_similarities = {}
    for idx, doc_data in df.iterrows():
        sim = cosine_similarity(get_modified_vectors(query_embedding), doc_data['nc_vec'])[0][0]
        doc_similarities[doc_data['id']] = sim

    doc_similarities = dict(sorted(doc_similarities.items(), key=lambda item: item[1], reverse=True))
    
    doc_similarity_list = list(doc_similarities.values())
    max_sim = max(doc_similarity_list)
    # min_sim = min(doc_similarity_list)
    max_diff_sim = get_max_diff_index(doc_similarities)
    cut_off_sim = min(MIN_THRESHOLD_SEMANTIC, (cut_off*max_sim), max_diff_sim)
    # cut_off_sim = (cut_off * max_sim)

    logging.info(f'\nQuery: {query}')
    # logging.info(f'Cut-off: {cut_off}')
    # logging.info(f'Max similarity: {max_sim}')
    # logging.info(f'Min similarity: {min_sim}')

    # # logging.info(f'Max*0.68 similarity: {0*max_sim}')
    # logging.info(f'Max diff similarity: {max_diff_sim}')
    # logging.info(f'Cut-off similarity: {cut_off_sim}\n')

    result_list = []
    for idx, doc_data in df.iterrows():
        doc_dict = dict()

        sim = doc_similarities[doc_data['id']]
        if sim > cut_off_sim:
            doc_dict['id'] = doc_data['id']
            doc_dict['title'] = doc_data['title']
            doc_dict['text'] = doc_data['text']
            doc_dict['page_url'] = doc_data['url']
            doc_dict['pub_date'] = doc_data['pubDate']

            result_list.append(doc_dict)

    logging.info(f'Semantic search original length: {len(result_list)}')

    if len(result_list) < 10:
        result_list = []
        index = 0
        for idx, doc_data in df.iterrows():
            index += 1
            doc_dict = dict()

            doc_dict['id'] = doc_data['id']
            doc_dict['title'] = doc_data['title']
            doc_dict['text'] = doc_data['text']
            doc_dict['page_url'] = doc_data['url']
            doc_dict['pub_date'] = doc_data['pubDate']

            result_list.append(doc_dict)

            if index == 10:
                break
    elif len(result_list) > match_top_org:
        result_list = result_list[:match_top_org]

    total_hits = len(result_list)
    write_query_results(query, result_list, 'semantic')

    return total_hits, result_list

def get_query_result_semantic_survey(query, match_top, cut_off = 0.64):

    index = xlm_index
    doc_df = xlm_df

    query_embedding = tf_model(query)['outputs'].numpy()[0]
    result = index.search(np.float32(query_embedding.reshape(1, -1)), match_top)

    df = doc_df.iloc[result[1][0]]

    doc_similarities = {}
    for idx, doc_data in df.iterrows():
        sim = cosine_similarity(get_modified_vectors(query_embedding), doc_data['nc_vec'])[0][0]
        doc_similarities[doc_data['id']] = sim

    doc_similarities = dict(sorted(doc_similarities.items(), key=lambda item: item[1], reverse=True))
    
    doc_similarity_list = list(doc_similarities.values())
    max_sim = max(doc_similarity_list)
    # min_sim = min(doc_similarity_list)
    max_diff_sim = get_max_diff_index(doc_similarities)
    cut_off_sim = min(MIN_THRESHOLD_SEMANTIC, (cut_off*max_sim), max_diff_sim)
    # cut_off_sim = (cut_off * max_sim)

    logging.info(f'\nQuery: {query}')

    result_list = []
    for idx, doc_data in df.iterrows():
        doc_dict = dict()

        sim = doc_similarities[doc_data['id']]
        if sim > cut_off_sim:
            doc_dict['id'] = doc_data['id']
            doc_dict['title'] = doc_data['title']
            doc_dict['text'] = doc_data['text']
            doc_dict['page_url'] = doc_data['url']
            doc_dict['pub_date'] = doc_data['pubDate']

            result_list.append(doc_dict)

    logging.info(f'Semantic search original length: {len(result_list)}')

    total_hits = len(result_list)
    write_query_results(query, result_list, 'semantic')

    return total_hits, result_list

def get_query_result_semantic_analysis(query, lang, match_top, cut_off = 0.64):

    match_top_org = match_top
    # match_top += 10

    if lang == 1:
        index = de_index
        doc_df = de_df
    elif lang == 2:
        index = en_index
        doc_df = en_df
    elif lang == 3:
        index = xlm_index
        doc_df = xlm_df

    query_embedding = tf_model(query)['outputs'].numpy()[0]
    result = index.search(np.float32(query_embedding.reshape(1, -1)), match_top)

    df = doc_df.iloc[result[1][0]]

    doc_similarities = {}
    for idx, doc_data in df.iterrows():
        sim = cosine_similarity(get_modified_vectors(query_embedding), doc_data['nc_vec'])[0][0]
        doc_similarities[doc_data['id']] = sim

    doc_similarities = dict(sorted(doc_similarities.items(), key=lambda item: item[1], reverse=True))
    
    doc_similarity_list = list(doc_similarities.values())
    max_sim = max(doc_similarity_list)
    min_sim = min(doc_similarity_list)
    max_diff_sim = get_max_diff_index(doc_similarities)
    # cut_off_sim = min(0.27, (0.64*max_sim), max_diff_sim)

    cut_off_values = [round(val,2) for val in np.arange(0.5, 1.0, step=0.01)]
    cut_off_dict = {'Query': query}

    for cut_off in cut_off_values:
        cut_off_sim = (cut_off * max_sim)

        result_list = []
        for idx, doc_data in df.iterrows():
            doc_dict = dict()

            sim = doc_similarities[doc_data['id']]
            if sim > cut_off_sim:
                doc_dict['id'] = doc_data['id']
                doc_dict['title'] = doc_data['title']
                doc_dict['text'] = doc_data['text']
                doc_dict['page_url'] = doc_data['url']
                doc_dict['pub_date'] = doc_data['pubDate']

                result_list.append(doc_dict)

        total_hits = len(result_list)
        cut_off_dict[cut_off] = total_hits
    # write_query_results(query, result_list, 'semantic')

    query_updated = query.lower().replace(' ', '_')
    filename = document_count_results_folder + f'{query_updated}_doc_cnt_result.json'
    write_document_data(cut_off_dict, filepath=filename)

    return total_hits, result_list

def portion_of_capital_letters(query):
    upper_cases = ''.join([c for c in query if c.isupper()])
    return float(len(upper_cases)/len(query))

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

    nouns_list = char_split.split_compound(query)
    print(nouns_list)
    if len(query) > 14:

        if len(nouns_list) > 1 and nouns_list[0][0] > -0.8:
            return True
    return False

def get_optimum_search_strategy(es, query):

    lang = detect_language(query)
    search_type = None
    query_type = None
    comments = None
    query_parts = query.split()

    if lang == 'de':
        lang = 1
    elif lang == 'en':
        lang = 2

    if len(query_parts) > 3:
        search_type = 'semantic_search'
        query_type = None
        comments = 'Sentence query, token length > 3'
    elif len(query_parts) < 3 and len(query_parts) > 1:
        search_type = 'es_search'
        query_type = 'phrase_query'
        comments = 'Phrase match, token length < 3 and >1'
    elif len(query_parts) == 1:

        if portion_of_capital_letters(query) >= 0.75:
            search_type = 'es_search'
            query_type = None
            comments = 'Abkürzung entdeckt'
        elif detect_german_compoundword(query):
            search_type = 'semantic_search'
            query_type = None
            comments = 'Deutsches Kompositum gefunden' 
        elif handle_count_queries(es, query, lang, phrase_query=True, fuzzy_query=False) == 0 and detect_spelling_mistake(query, lang) != query.lower():
            search_type = 'es_search'
            query_type = 'fuzzy_query'
            comments = 'Fuzzy match, zero BM-25 results and spelling correction'
        else:
            search_type = 'es_search'
            query_type = None
            comments = 'Simple query match'

    return lang, search_type, query_type, comments

def get_search_type(search_type):

    if search_type == 'es_search':
        return 'BM-25'
    elif search_type == 'semantic_search':
        return 'Semantic search'
    elif search_type == 'optimistic_search':
        return 'Optimistic search'
    elif search_type == 'hybrid_search':
        return 'Hybrid search'
    elif search_type == 'top_candidate_pool':
        return 'Candidate label pool'
    elif search_type == 'sub_topic_search':
        return 'Sub topic search'

def get_language(lang):

    if lang == 'de' or lang == 1:
        return 'deutsch'
    elif lang == 'en' or lang == 2:
        return 'englisch'
    elif lang == 'xlm' or lang == 3:
        return 'mulitlingual'

def get_merged_results(results_semantic, results_es):

    final_results = []
    page_id_tracker = []

    for doc in results_semantic:
        if doc['id'] not in page_id_tracker:
            page_id_tracker.append(doc['id'])
            final_results.append(doc)

    for doc in results_es:
        if doc['id'] not in page_id_tracker:
            page_id_tracker.append(doc['id'])
            final_results.append(doc)

    shuffle(final_results)

    return final_results

def filter_results_from_sqlitedb(results, query):

    recorded_inputs = [val[0] for val in get_db_contents_query(query)]

    final_results = []
    for doc in results:
        if doc['id'] not in recorded_inputs:
            final_results.append(doc) 

    return final_results

def get_stopwords():

    stopwords_de = stopwords.words('german')
    stopwords_en = stopwords.words('english')

    stopwords_full = []
    stopwords_full.extend(stopwords_de)
    stopwords_full.extend(stopwords_en)

    stopwords_full = [word.lower() for word in stopwords_full]
    stop_all = set(stopwords_full + list(string.punctuation))

    return stop_all

def get_subtopic(results, query, min_clust_size, min_samples):

    query_vec = tf_model(query)['outputs'].numpy()[0]

    rank_df = pd.DataFrame(results, columns=['id', 'title', 'text', 'page_url', 'pub_date'])
    rank_df = rank_df[['id']]
    
    final_df = pd.concat([rank_df.set_index('id'), final_keywords_dataframe.set_index('id')], axis=1, join='inner').reset_index()

    final_df['keywords_query'] = final_df.apply(lambda x:get_sent_transformers_keywords(x['keywords'], query_vec), axis=1)
    final_df['candidate_pool'] = final_df.apply(lambda x:get_candidate_pool(x['keywords_query'], lower_limit = 0.0, upper_limit = CP_THRESHOLD), axis=1)

    final_candidate_pool = []

    for idx, row in final_df.iterrows():
        final_candidate_pool.extend(row['candidate_pool'])

    final_candidate_pool_vecs = [m.unpackb(rdb.get(nc)) for nc in final_candidate_pool]
    df_data = []
    for word, vec in zip(final_candidate_pool, final_candidate_pool_vecs):
        df_data.append((word, vec))

    cluster_df = pd.DataFrame(df_data, columns= ['candidate_words', 'candidate_vecs'])

    cluster_df = get_clustering_analysis(cluster_df,final_candidate_pool_vecs, min_clust_size, min_samples)
    cluster_data = []
    for cluster_id in set(cluster_df.cluster_id.values):
        
        if cluster_id != -1:
            df = cluster_df[cluster_df['cluster_id'] == cluster_id]
            cluster_data.append((cluster_id, df.candidate_words.values, df.candidate_vecs.values))

    cluster_data_df = pd.DataFrame(cluster_data, columns=['cluster_id', 'candidate_words', 'candidate_vecs'])
    cluster_data_df['mean_vec'] = cluster_data_df.apply(lambda x:get_pool_vec(x['candidate_vecs'], 'mean'), axis=1)
    cluster_data_df['topic'] = cluster_data_df.apply(lambda x:get_nearest_keyword(x['candidate_words'], x['candidate_vecs'], x['mean_vec']), axis=1)
    # cluster_data_df['topic_sim'] = cluster_data_df.apply(lambda x:cosine_similarity(get_modified_vectors(x['mean_vec']), get_modified_vectors(query_vec))[0][0], axis=1)
    cluster_data_df['cluster_size'] = cluster_data_df.apply(lambda x:len(x['candidate_words']), axis=1)

    cluster_data_df['page_id_list'] = cluster_data_df.apply(lambda x:get_topic_documents(x['candidate_words'], final_df), axis=1)

    cluster_data_df['topic_name'] = cluster_data_df.apply(lambda x:x['topic']+' ('+str(len(x['candidate_words']))+' | '+str(len(x['page_id_list']))+')', axis=1)

    cluster_data_df = cluster_data_df.sort_values(by=['cluster_size'], ascending=False)
    cluster_data_df = cluster_data_df.reset_index(drop=True)

    cluster_dict = dict()
    for topic, doc_id_list in zip(cluster_data_df.topic_name.values, cluster_data_df.page_id_list.values):
        cluster_dict[topic] = doc_id_list

    return cluster_dict

def get_topic_documents_clustering(query, doc_id_list):

    df = xlm_df[xlm_df['id'].isin(doc_id_list)]
    query_embedding = tf_model(query)['outputs'].numpy()[0]

    doc_similarities = {}
    for idx, doc_data in df.iterrows():
        sim = cosine_similarity(get_modified_vectors(query_embedding), doc_data['nc_vec'])[0][0]
        doc_similarities[doc_data['id']] = sim

    doc_similarities = dict(sorted(doc_similarities.items(), key=lambda item: item[1], reverse=True))

    result_list = []
    for doc_id, sim in doc_similarities.items():
        doc_dict = dict()

        doc_data = df.loc[df['id'] == doc_id]
        doc_data = doc_data.reset_index().to_dict("records")[0]

        doc_dict['id'] = doc_data['id']
        doc_dict['title'] = doc_data['title']
        doc_dict['text'] = doc_data['text']
        doc_dict['page_url'] = doc_data['url']
        doc_dict['pub_date'] = doc_data['pubDate']

        result_list.append(doc_dict)

        if len(result_list) == 5:
            break

    total_hits = len(result_list)

    return total_hits, result_list

def get_keyword_query_data():

    query_keyword_list = []
    with open(os.getcwd()+'/../query_keywords_example.txt', 'r') as f:
        query_keyword_list = f.readlines()
    
    shuffle(query_keyword_list)

    query_keyword_dict = dict()
    keywords_list = []

    for idx, query_keyword in enumerate(query_keyword_list):
        keywords_list.append({'id': str(idx+1),
        'name': query_keyword})

        query_keyword_dict[str(idx+1)] = query_keyword

    return keywords_list, query_keyword_dict