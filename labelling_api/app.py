import time
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from elasticsearch import helpers, Elasticsearch
from utils import *

app = FastAPI(debug=True, root_path="/mda")
templates = Jinja2Templates(directory="templates/")

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

@app.get("/mda/search_keyword")
async def load_search_homepage(request: Request):
    return templates.TemplateResponse('search_keyword.html', context={'request': request, 'total_hits': 0, 'result_list': [], 'concept_list': [], 'search_data': dict()})

@app.post('/mda/keyword_search')
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
        search_data['comments'] = 'Not optimistic, User specified search'

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
    uvicorn.run(app, host='127.0.0.1', port=80)
