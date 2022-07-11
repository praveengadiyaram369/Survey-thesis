import sqlite3
from settings import *

def insert_into_sqlite_db(doc_id, query, label):

    logging.info('before inserting into the table')

    insert_table_query = '''INSERT INTO retrieval_dataset VALUES (?, ?, ?);'''
    sqlite_common_query_seq(insert_table_query, sql_select=False, sql_insert_params=(query, doc_id, int(label)))
    get_db_contents()

    logging.info(f'finished inserting into the table: query: {query}, label:{label}')


def create_table():

    create_table_query = """CREATE TABLE IF NOT EXISTS retrieval_dataset (
        query text,
        doc_id text,
        label integer,
        PRIMARY KEY (query, doc_id)
    );"""
    sqlite_common_query_seq(create_table_query)
    

def get_db_contents():

    select_table_query = """SELECT * FROM retrieval_dataset"""
    query_result = sqlite_common_query_seq(select_table_query, sql_select=True)   
    # logging.info(query_result)
    logging.info(f'Total records in the DB: {len(query_result)}')
    

def sqlite_common_query_seq(sql_query, sql_select=False, sql_insert_params=None):

    logging.info('running sqlite_common_query_seq')

    try:
        conn = sqlite3.connect(basepath+'retrival_test_dataset.db')

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


create_table()
if __name__ == '__main__':
    get_db_contents()