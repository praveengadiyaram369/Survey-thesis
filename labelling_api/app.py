from re import sub
import time
from unittest import result
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from elasticsearch import helpers, Elasticsearch
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from utils import *
from settings import *
from db_utils import *
from survey_db_utils import *
import json
from random import shuffle
import uuid

app = FastAPI(debug=True, root_path="/mda")
templates = Jinja2Templates(directory="templates/")

app.mount('/static',
        StaticFiles(directory='static/'),
        name='static')

username = 'elastic'
password = 'mit22fkie!'

hostname = 'elasticsearch'
port = '9200'

# time.sleep(5)
es = Elasticsearch([f"http://{username}:{password}@{hostname}:{port}"], timeout=300, max_retries=10, retry_on_timeout=True)

if not es.ping():
    print("##################### Connection failed ######################\n")
else:
    print("###################### Connection successful ######################\n")


document_type = None
document_data = None
tech_relevant_document_data = None
milt_relevant_document_data = None
irrelevant_document_data = None
sub_topics_dict = dict()
sub_topic_list = []
topic_dict = dict()

query_keywords_dict = dict()
query_keyword_list = []
session_id = None
current_query = None

@app.get("/")
async def load_homepage(request: Request):
    return templates.TemplateResponse('index.html', context={'request': request})

@app.get("/validate_technology")
async def validate_technology(request: Request):

    global document_data, document_type, tech_relevant_document_data, milt_relevant_document_data, irrelevant_document_data
    document_type = 'technology'
    document_data, tech_relevant_document_data, milt_relevant_document_data, irrelevant_document_data = load_document_data(type=document_type)
    result_list = get_subclass_data(key=None)
    total_doc_cnt = get_total_documentcount()

    return templates.TemplateResponse('document_validation.html', context={'request': request, 'total_doc_cnt': total_doc_cnt,  'result_list': result_list, 'document_type': document_type})

@app.get("/validate_military")
async def validate_military(request: Request):

    global document_data, document_type, tech_relevant_document_data, milt_relevant_document_data, irrelevant_document_data
    document_type = 'military'
    document_data, tech_relevant_document_data, milt_relevant_document_data, irrelevant_document_data  = load_document_data(type=document_type)
    result_list = get_subclass_data(key=None)
    total_doc_cnt = get_total_documentcount()

    return templates.TemplateResponse('document_validation.html', context={'request': request, 'total_doc_cnt': total_doc_cnt,  'result_list': result_list, 'document_type': document_type})

@app.get("/validate_augdata")
async def validate_augdata(request: Request):

    global document_data
    document_type = 'aug_pos'
    document_data = load_document_data(type=document_type)
    result_list = get_subclass_data(key=None)
    total_doc_cnt = get_total_documentcount()
    print(result_list)

    return templates.TemplateResponse('validate_dataaug.html', context={'request': request, 'total_doc_cnt': total_doc_cnt,  'result_list': result_list, 'document_type': document_type})

@app.post("/mark_irrelevant")
async def mark_documents_irrelevant(request: Request, irrelevant_page_id: str=Form(1)):

    print(irrelevant_page_id)
    update_document_data(irrelevant_page_id, None, marked_document_type='irrelevant', type='irrelevant')

@app.post("/mark_relevant_tech")
async def mark_documents_relevant(request: Request, relevant_page_id: str=Form(1), category_list: str=Form(2)):

    print(relevant_page_id)
    print(category_list)
    update_document_data(relevant_page_id, category_list, marked_document_type='technology', type='relevant')

@app.post("/mark_relevant_milt")
async def mark_documents_relevant(request: Request, relevant_page_id: str=Form(1), category_list: str=Form(2)):

    print(relevant_page_id)
    print(category_list)
    update_document_data(relevant_page_id, category_list, marked_document_type='military', type='relevant')

@app.post("/accept_augdata")
async def accept_augdata(request: Request, aug_page_id: str=Form(1)):

    print(aug_page_id)
    write_data_to_file(aug_docs_path, 'accepted:'+aug_page_id)

@app.post("/reject_augdata")
async def reject_data(request: Request, aug_page_id: str=Form(1)):

    print(aug_page_id)
    write_data_to_file(aug_docs_path, 'rejected:'+aug_page_id)

@app.get("/validate_irrelevant")
async def validate_irrelevant(request: Request):
    return templates.TemplateResponse('index.html', context={'request': request})

@app.get("/validate_pos_docs")
async def validate_pos_docs(request: Request):

    data_dict = read_document_data(classified_pos_docs_path)
    result_list = []
    for key, val in data_dict.items():
        result_list.append(val)
    return templates.TemplateResponse('pos_docs.html', context={'request': request, 'result_list': result_list})

@app.post("/add_document_thirdclass")
async def add_document_thirdclass(request: Request, third_page_id: str=Form(1)):

    print(third_page_id)
    write_data_to_file(third_class_docs_path, third_page_id)

@app.get("/search_keyword")
async def load_search_homepage(request: Request):
    return templates.TemplateResponse('search_keyword.html', context={'request': request, 'total_hits': 0, 'result_list': [], 'concept_list': [], 'search_data': dict()})

@app.get("/search_survey")
async def search_survey(request: Request):

    global query_keywords_dict
    global query_keyword_list

    query_keyword_list, query_keywords_dict = get_keyword_query_data()

    global session_id
    session_id = uuid.uuid4() 

    return templates.TemplateResponse('search_survey.html', context={'request': request, 'total_hits': 0, 'result_list': [], 'concept_list': [], 'search_data': dict(), 'sub_topic_list':[], 'query_keyword_list': query_keyword_list})

@app.get("/search_subtopic")
async def search_subtopic(request: Request):
    query = 'Quantentechnologie'
    return templates.TemplateResponse('sub_topic_search.html', context={'request': request, 'total_hits': 0, 'query': query, 'result_list': [], 'concept_list': [], 'search_data': dict(), 'sub_topic_list':[]})

@app.post("/get_sub_topics")
async def get_sub_topics(request: Request, query: str=Form(...), min_clust_size: str=Form(...), min_samples: str=Form(...)):

    global current_query

    query = query.strip()

    if query.isnumeric():
        query = query_keywords_dict[str(query)]

    current_query = query

    logging.info(f'Query selected: {query}')

    min_clust_size = int(min_clust_size)
    min_samples = int(min_samples)

    if min_clust_size == 0:
        min_clust_size = 20
    
    if min_samples == 0:
        min_samples = 10

    lang = 3
    match_top = 55

    is_german_compoundword = False
    for word in query.split():
        if detect_german_compoundword(word):
            is_german_compoundword = True
            break

    search_concept = False
    fuzzy_query = False
    phrase_query = False

    total_hits_semantic, results_semantic = get_query_result_semantic(query, lang, match_top, cut_off=0.75)

    if is_german_compoundword:
        lang = 1
    total_hits_es, results_es = get_query_result(es, query, lang, phrase_query, fuzzy_query, search_concept, match_top)

    results = get_merged_results(results_semantic, results_es)

    global topic_dict
    topic_dict = get_subtopic(results, query, min_clust_size, min_samples)
    sub_topics = list(topic_dict.keys())
    global sub_topics_dict, sub_topic_list
    sub_topics_dict = dict()
    sub_topic_list = []

    for idx, topic in enumerate(sub_topics):
        sub_topics_dict[str(idx+1)] = topic
        sub_topic_list.append({'id':str(idx+1), 'name': topic})

    json_compatible_item_data = jsonable_encoder(json.dumps(sub_topic_list))
    return JSONResponse(content=json_compatible_item_data)

@app.post('/sub_topic_keywords_search')
async def keyword_search(request: Request, query: int=Form(1), sub_topic_selected: int=Form(1), lang: int=Form(1), phrase_query: bool=Form(False), search_concept: bool=Form(False), match_top: str=Form(...), fuzzy_query: str=Form(False), search_type: str=Form(...)):

    query = query_keywords_dict[str(query)]
    sub_topic = sub_topics_dict[str(sub_topic_selected)]
    doc_id_list = topic_dict[sub_topic]

    logging.info(f'Query selected: {query}')
    logging.info(f'sub_topic selected: {sub_topic}')

    search_data = {
        'original_query':f'{query} und {sub_topic}',
        'search_type': 'NA',
        'search_strategy':'Clustering based results',
        'language': 'NA',
        'total_hits': 'NA',
        'comments': 'Sub topic search'
    }

    total_hits, results = get_topic_documents_clustering(doc_id_list)

    return templates.TemplateResponse('sub_topic_search.html', context={'request': request, 'total_hits': total_hits, 'result_list': results, 'concept_list': [], 'query': query, 'search_data':search_data, 'sub_topic_list':sub_topic_list})


@app.post('/sub_topic_search_survey')
async def keyword_search(request: Request, query: str=Form(...), sub_topic_selected: int=Form(1)):

    query = query_keywords_dict[str(query)]
    sub_topic = sub_topics_dict[str(sub_topic_selected)]
    doc_id_list = topic_dict[sub_topic]

    sub_topic = sub_topic.split(' (')[0]
    sub_topic = sub_topic.strip()

    logging.info(f'Query selected: {query}')
    logging.info(f'sub_topic selected: {sub_topic}')

    query_updated = f'Innovation in {query} und {sub_topic}'

    search_data = {
        'original_query':query,
        'sub_topic': sub_topic,
        'search_type': 'NA',
        'search_strategy':'Clustering based results',
        'language': 'NA',
        'total_hits': 'NA',
        'comments': 'Sub topic search'
    }

    total_hits, system_a_results = get_topic_documents_clustering(query_updated, doc_id_list)

    lang = 3
    match_top = len(system_a_results)

    total_hits_semantic, system_b_results = get_query_result_semantic_survey(query_updated, match_top, cut_off=0.75)

    return templates.TemplateResponse('search_survey.html', context={'request': request, 'total_hits': total_hits, 'result_list_1': system_a_results, 'result_list_2': system_b_results, 'concept_list': [], 'query': query, 'search_data':search_data, 'sub_topic_list':sub_topic_list, 'query_keyword_list': query_keyword_list})

@app.post('/get_cdd_pool')
async def get_cdd_pool(request: Request, query: str=Form(...), lang: int=Form(3), phrase_query: bool=Form(False), search_concept: bool=Form(False), match_top: int=Form(...), fuzzy_query: str=Form(False), search_type: str=Form(...)):

    query = query.strip()
    lang = int(lang)
    # match_top = int(match_top)
    match_top = 55

    search_data = {
        'original_query': query,
        'search_type': 'BM-25 und Semantische Suche',
        'search_strategy': 'NA',
        'language': 'multilingual',
        'total_hits': 'NA',
        'comments': 'Candidate label pool'
    }

    search_data['language'] = get_language(lang)
    search_data['search_type'] = get_search_type(search_type)

    is_german_compoundword = False
    for word in query.split():
        if detect_german_compoundword(word):
            is_german_compoundword = True
            break

    if search_type == 'es_search':
        if is_german_compoundword:
            lang = 1
        total_hits_es, results = get_query_result(es, query, lang, phrase_query, fuzzy_query, search_concept, match_top)
        search_data['cdd_result_cnt'] = 0
        search_data['es_result_cnt'] = len(results)
        search_data['ss_result_cnt'] =  0

    elif search_type == 'semantic_search':
        total_hits_semantic, results = get_query_result_semantic(query, lang, match_top)
        search_data['cdd_result_cnt'] = 0
        search_data['es_result_cnt'] = 0
        search_data['ss_result_cnt'] =  len(results)

    elif search_type == 'top_candidate_pool':

        # for cut_off in [0.9, 0.95, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]:
        #     total_hits_semantic, results_semantic = get_query_result_semantic(query, lang, match_top, cut_off)

        # query_list = ['Schutz von unbemannten Systemen', 'Schwachstellenanalyse eigene Waffen-Systeme', 'Waffen Systeme', 'Defense', 'Cyberinformationsraum', 'Multi Range Cyberinformationsraum', 'militärische Entscheidungsfindung', 'unbemannte Wirksysteme', 'Data Centric Warfare', 'Combat Cloud', 'Cyber-Verteidigung', 'Militärische Kommunikation', 'Unbemannte Landsysteme', 'Multifunktionale Radarsysteme', 'Main Ground Combat System (MGCS)', 'Schutz gegen Navigation Warfare', 'GeoInfo spezifisch', 'Soziale Medien', 'DesInformationskampagnen', 'Cyber Attack', 'Künstliche Intelligenz', 'AI', 'Kryptologie', 'Postquantum', 'Quantentechnologie', 'Public Key Infrastructure', 'Predictive Analysis', '5G', 'Zellenbasierter Mobilfunk', 'Optische Kommunikation', 'Wellenformentwicklung', 'Satellitenkommunikation', 'taktisches Routing', 'Resiliente Kommunikationsverbünde', 'Datenlinks', 'Taktische Datenlinks', 'Mensch-Maschine Schnittstellen', 'Big Data, KI für Analyse', 'Multirobotereinsatz', 'IT-Standards', 'Interoperabilität', 'taktische Entnetzung', 'Edge computing', '3D-Modelle', 'Mixed Reality', 'Geoinformationen', 'Befahrbarkeitsmodellierung', 'Wetterdatenfusion', 'Sensordatenfusion', 'Change Detektion', 'Architekturanalyse', 'Datenermittlung und -analyse', 'Cloud Computing', 'Digitalfunk', 'FPGA', 'IT-Bedrohungsanalyse', 'Technologie-Screening', 'IT-Sicherheit bei embedded IT', 'IT-Sicherheitsvorgaben', 'IT-Technologien', 'Kommunikationsnetze', 'Kommunikations-Services', 'Kommunikationstechnologien', 'Kryptographie', 'LTE', 'Methode Architektur', 'Mobile Computing', 'Mobile Kommunikation', 'Prozessunterstützung', 'Radar', 'Robotik', 'Satellitenkommunikation', 'Semantische Technologien', 'Software Defined Networking', 'Softwareentwicklung', 'Taktische Datenlinks', 'Taktisches Routing', 'vernetzte Systeme', 'Virtualisierung', 'Visualisierung', 'Wellenformen und -ausbreitung', 'Zeichenorientierter InfoAustausch']

        # for query in query_list:
        #     total_hits_semantic, results_semantic = get_query_result_semantic_analysis(query, lang, match_top)

        total_hits_semantic, results_semantic = get_query_result_semantic(query, lang, match_top, cut_off=0.75)
        if is_german_compoundword:
            lang = 1
        total_hits_es, results_es = get_query_result(es, query, lang, phrase_query, fuzzy_query, search_concept, match_top)

        results = get_merged_results(results_semantic, results_es)
    
    # results = filter_results_from_sqlitedb(results, query)
        
        search_data['cdd_result_cnt'] = len(results)
        search_data['es_result_cnt'] = len(results_es)
        search_data['ss_result_cnt'] =  total_hits_semantic

    search_data['total_hits'] = len(results)

    return templates.TemplateResponse('search_keyword.html', context={'request': request, 'total_hits': search_data['total_hits'], 'result_list': results, 'concept_list': results, 'query': query, 'search_data':search_data})

@app.post("/save_document_label")
async def save_document_label(request: Request, doc_id: str=Form(1), query: str=Form(1), label: str=Form(1)):

    insert_into_sqlite_db(doc_id, query, label)

@app.post("/submit_survey_question_1")
async def submit_survey_question_1(request: Request, label: str=Form(1)):
    if current_query is not None:
        insert_clustering_output_label(session_id, current_query, label)

@app.post("/submit_survey_question_2")
async def submit_survey_question_2(request: Request, query: str=Form(1), sub_topic: str=Form(1), label: str=Form(1)):

    insert_system_comparision_label(session_id, query, sub_topic, label)


@app.post('/keyword_search')
async def keyword_search(request: Request, query: str=Form(...), lang: int=Form(1), phrase_query: bool=Form(False), search_concept: bool=Form(False), match_top: str=Form(...), fuzzy_query: str=Form(False), search_type: str=Form(...)):

    query = query.strip()

    print(query)
    search_data = {
        'original_query':query,
        'search_type': 'NA',
        'search_strategy':'NA',
        'language': 'NA',
        'total_hits': 'NA',
        'comments': 'NA'
    }

    if search_type != 'optimistic_search':
        search_data['comments'] = 'Nicht optimistisch, Benutzerspezifische Suche'

    match_top = int(match_top)
    semantic_query = False

    if search_type == 'semantic_search':
        semantic_query = True
    elif search_type == 'es_search':
        semantic_query = False
    elif search_type == 'optimistic_search':
        lang, search_type, query_type, comments = get_optimum_search_strategy(es, query)
        search_data['language'] = get_language(lang)
        search_data['search_type'] = get_search_type(search_type)
        search_data['comments'] = comments

        if query_type == 'phrase_query':
            phrase_query = True
            fuzzy_query = False
        elif query_type == 'fuzzy_query':
            phrase_query = False
            fuzzy_query = True
        else:
            phrase_query = False
            fuzzy_query = False   

    search_data['language'] = get_language(lang)
    search_data['search_type'] = get_search_type(search_type)

    if search_type != 'semantic_search':
        if phrase_query:
            search_data['search_strategy'] = 'Phrase match'
        elif fuzzy_query:
            search_data['search_strategy'] = 'Fuzzy match' 
    else:
        semantic_query = True      

    if not semantic_query:
        total_hits, results = get_query_result(es, query, lang, phrase_query, fuzzy_query, search_concept, match_top)
    else:
        total_hits, results = get_query_result_semantic(query, lang, match_top)

    search_data['total_hits'] = total_hits

    if not search_concept:
        return templates.TemplateResponse('search_keyword.html', context={'request': request, 'total_hits': total_hits, 'result_list': results, 'concept_list': [], 'query': query, 'search_data':search_data})
    else:
        return templates.TemplateResponse('search_keyword.html', context={'request': request, 'total_hits': total_hits, 'result_list': [], 'concept_list': results, 'query': query, 'search_data':search_data})

def update_document_data(page_id, category_list, marked_document_type, type):

    global document_data, tech_relevant_document_data, milt_relevant_document_data, irrelevant_document_data

    page_id_data = document_data[page_id]

    if marked_document_type != 'irrelevant':
        page_id_data['category_list'] = category_list

    del document_data[page_id]

    if type == 'irrelevant':
        irrelevant_document_data[page_id] = page_id_data
    elif type == 'relevant' and marked_document_type == 'technology':
        tech_relevant_document_data[page_id] = page_id_data
    elif type == 'relevant' and marked_document_type == 'military':
        milt_relevant_document_data[page_id] = page_id_data

    save_document_data(document_data, tech_relevant_document_data, milt_relevant_document_data, irrelevant_document_data, type=document_type)


def get_subclass_data(key=None):
        
    result_list = []
    for key, val in document_data.items():
        result_list.append(val)

    return result_list

def get_total_documentcount():

    return len(list(document_data.keys()))

@app.post("/load_subclass_details")
async def load_subclass_details(request: Request, subclass: str=Form(1)):

    sub_class, sub_class_list, result_list = get_subclass_data(key=subclass)
    total_doc_cnt = get_total_documentcount()

    return templates.TemplateResponse('document_validation.html', context={'request': request, 'total_doc_cnt': total_doc_cnt, 'sub_class_list': sub_class_list, 'subclass': sub_class, 'result_list': result_list})

if __name__ == '__main__':
    if not os.path.isfile(survey_sqlite_db_path):
        logging.info('DB does not exist, creating db and tables')
        create_table()
    uvicorn.run(app, host='127.0.0.1', port=80)
