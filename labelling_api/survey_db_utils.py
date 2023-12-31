import sqlite3
from datetime import datetime
from settings import *

def insert_clustering_output_label(session_id, query, survey_label_1):

    logging.info('before inserting into the table')

    insert_table_query = '''INSERT INTO clustering_output_survey_data VALUES (?, ?, ?, ?);'''
    sqlite_common_query_seq(insert_table_query, sql_select=False, sql_insert_params=(session_id, query, int(survey_label_1), datetime.now()))

    select_table_query = """SELECT session_id, query, survey_label_1  FROM clustering_output_survey_data where session_id= ? and query= ?"""
    get_db_contents(select_table_query, [session_id, query])

    logging.info(f'finished inserting into the table: query: {query}, label:{survey_label_1}')

def insert_survey_output_label(session_id, query, survey_label_5):

    logging.info('before inserting into the table')

    insert_table_query = '''INSERT INTO xxx_survey_output VALUES (?, ?, ?, ?);'''
    sqlite_common_query_seq(insert_table_query, sql_select=False, sql_insert_params=(session_id, query, int(survey_label_5), datetime.now()))

    select_table_query = """SELECT session_id, query, survey_label_5  FROM xxx_survey_output where session_id= ? and query= ?"""
    get_db_contents(select_table_query, [session_id, query])

    logging.info(f'finished inserting into the table: query: {query}, label:{survey_label_5}')

def insert_system_comparision_label(session_id, query, sub_topic, survey_label_2, survey_label_3, survey_label_4):

    logging.info('before inserting into the table')

    insert_table_query = '''INSERT INTO system_comparision_survey_data VALUES (?, ?, ?, ?, ?, ?, ?);'''
    sqlite_common_query_seq(insert_table_query, sql_select=False, sql_insert_params=(session_id, query, sub_topic, int(survey_label_2), float(survey_label_3), float(survey_label_4), datetime.now()))
    
    select_table_query = """SELECT session_id, query, sub_topic, survey_label_2, survey_label_3, survey_label_4 FROM system_comparision_survey_data where session_id= ? and query= ? and sub_topic= ?"""
    get_db_contents(select_table_query, [session_id, query, sub_topic])

    logging.info(f'finished inserting into the table: query: {query}, sub_topic: {sub_topic}, label_2:{survey_label_2}, label_3:{survey_label_3}, label_4:{survey_label_4}')


def create_survey_tables():

    logging.info('starting create table ........... ')
    create_table_query_1 = """CREATE TABLE IF NOT EXISTS clustering_output_survey_data (
        session_id text NOT NULL,
        query text NOT NULL,
        survey_label_1 integer NOT NULL, 
        date_modified timestamp,
        PRIMARY KEY (query, session_id)
    );"""
    sqlite_common_query_seq(create_table_query_1)

    create_table_query_2 = """CREATE TABLE IF NOT EXISTS system_comparision_survey_data (
        session_id text NOT NULL,
        query text NOT NULL,
        sub_topic text NOT NULL,
        survey_label_2 integer NOT NULL, 
        survey_label_3 real NOT NULL, 
        survey_label_4 real NOT NULL, 
        date_modified text,
        PRIMARY KEY (query, sub_topic, session_id)
    );"""
    sqlite_common_query_seq(create_table_query_2)

    create_table_query_3 = """CREATE TABLE IF NOT EXISTS xxx_survey_output (
        session_id text NOT NULL,
        query text NOT NULL,
        survey_label_5 integer NOT NULL, 
        date_modified timestamp,
        PRIMARY KEY (query, session_id)
    );"""
    sqlite_common_query_seq(create_table_query_3)
    logging.info('finished create table ........... ')


def get_db_contents(select_table_query, data):

    logging.info(select_table_query)
    query_result = None
    
    try:
        conn = sqlite3.connect(survey_sqlite_db_path)

        cursor = conn.cursor()
        cursor.execute(select_table_query, data)

        query_result = cursor.fetchall()
        conn.close()

        logging.info('finished sqlite_common_query_seq')
    except Exception as e:
        logging.error(e)

    # logging.info(query_result)
    if query_result is not None and len(query_result) > 0:
        logging.info(f'Inserted data in the DB: {query_result}')

    return query_result    

def sqlite_common_query_seq(sql_query, sql_select=False, sql_insert_params=None):

    logging.info('running sqlite_common_query_seq')

    try:
        conn = sqlite3.connect(survey_sqlite_db_path)

        cursor = conn.cursor()

        if sql_insert_params:
            cursor.execute(sql_query, sql_insert_params)
        else:
            cursor.execute(sql_query)

        if sql_select:
            query_result = cursor.fetchall()
            conn.close()

            return query_result

        conn.commit()
        conn.close()

        logging.info('finished sqlite_common_query_seq')
    except Exception as e:
        logging.error(e)