import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates

from utils import *

app = FastAPI(debug=True)
templates = Jinja2Templates(directory="templates/")

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

    global document_data, document_type, tech_relevant_document_data, milt_relevant_document_data, irrelevant_document_data
    document_type = 'technology'
    document_data, tech_relevant_document_data, milt_relevant_document_data, irrelevant_document_data = load_document_data(type=document_type)
    result_list = get_subclass_data(key=None)
    total_doc_cnt = get_total_documentcount()
    print(result_list[:2])

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
    # update_document_data(irrelevant_page_id, None, marked_document_type='irrelevant', type='irrelevant')

@app.post("/reject_data")
async def reject_data(request: Request, aug_page_id: str=Form(1)):

    print(aug_page_id)
    # update_document_data(irrelevant_page_id, None, marked_document_type='irrelevant', type='irrelevant')

@app.get("/validate_irrelevant")
async def validate_irrelevant(request: Request):
    return templates.TemplateResponse('index.html', context={'request': request})

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