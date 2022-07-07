import logging
import tensorflow_hub as hub
import faiss
import pandas as pd
import numpy as np
import fasttext
import spellchecker
from spellchecker import SpellChecker

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

LOG_FILE = basepath + f'mitera_webapp_log.log'
logging.basicConfig(handlers=[logging.FileHandler(filename=LOG_FILE, 
                                                 encoding='utf-8', mode='a+')],
                    level=logging.INFO)

english_checker = SpellChecker(language='en')
german_checker = SpellChecker(language='de')