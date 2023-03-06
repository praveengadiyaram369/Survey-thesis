import logging
import tensorflow_hub as hub
import faiss
import pandas as pd
import numpy as np
import fasttext
import spellchecker
from spellchecker import SpellChecker

import msgpack
import msgpack_numpy as m
m.patch()

import redis

import texthero as hero
from texthero import preprocessing

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

import umap.umap_ as umap

import hdbscan
import string
from nltk.corpus import stopwords

basepath = '/usr/src/web_app/data/'
# basepath = 'C:/Users/sri.sai.praveen.gadi/Music/data_mount/'

technology_document_data_path = basepath + 'data/input/technologie_document_data.json'
military_document_data_path =   basepath + 'data/input/military_document_data.json'
augmented_pos_document_data_path =  basepath + 'data/input/augmented_pos_document_data.json'

relevant_technology_data_path = basepath + 'data/output/relevant_documents_tech.json'
relevant_military_data_path =  basepath + 'data/output/relevant_documents_milt.json'
irrelevant_document_data_path = basepath + 'data/output/irrelevant_documents.json'

# classified_pos_docs_path = basepath + 'data/input/predicted_unlabeled_docs.json'
classified_pos_docs_path = basepath + 'data/input/new_labeled_negative_set.json'
third_class_docs_path = basepath + 'data/output/third_class_data.txt'
aug_docs_path = basepath + 'data/output/doc_aug_info_data.txt'

es_index = 'mitera_scraped_docs'

tf_model = hub.load(basepath+ 'models/USE_model')
fasttext_model = fasttext.load_model(basepath + 'models/lid.176.bin')
fasttext.FastText.eprint = lambda x: None

xlm_index = faiss.read_index(basepath+"xlm_vector.index")
en_index = faiss.read_index(basepath+"en_vector.index")
de_index = faiss.read_index(basepath+"de_vector.index")

xlm_df = pd.read_pickle(basepath+'xlm_dataframe.pkl')
en_df = pd.read_pickle(basepath+'en_dataframe.pkl')
de_df = pd.read_pickle(basepath+'de_dataframe.pkl')

# final_keywords_dataframe = pd.read_pickle(basepath+'final_keywords_dataframe.pkl')
final_keywords_dataframe = pd.read_pickle(basepath+'final_keywords_dataframe_cdd.pkl')

LOG_FILE = basepath + f'mitera_webapp_log.log'
logging.basicConfig(handlers=[logging.FileHandler(filename=LOG_FILE, 
                                                 encoding='utf-8', mode='a+')],
                    level=logging.INFO)

english_checker = SpellChecker(language='en')
german_checker = SpellChecker(language='de')

sqlite_db_path = basepath+'retrieval_test_dataset.db'

search_results_folder = basepath+'search_results_index/'
document_count_results_folder = basepath+'document_count_results_index/'

rdb = redis.StrictRedis(
    host='redis_cache',
    port=6379,
)

MIN_THRESHOLD_SEMANTIC = 0.27
CP_THRESHOLD = 0.85
UMAP_DIM = 5
MIN_CLUSTER_SIZE = 20 
MIN_SAMPLES = 10

MAX_SURVEY_COUNT = 10
survey_sqlite_db_path = basepath+'survey_results_dataset.db'