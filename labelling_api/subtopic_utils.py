import numpy as np
from settings import *
from sklearn.metrics.pairwise import cosine_similarity

def get_modified_vectors(vec_data):
    
    new_data = []
    for val in vec_data:
        new_data.append(val)
    
    new_data = np.array(new_data).reshape(-1, 512)
    return new_data

def get_pool_vec(doc_vec_list, pool):
    
    doc_vec_list = get_modified_vectors(doc_vec_list)
    if pool == 'mean':
        return np.nanmean(doc_vec_list, axis=0)
    elif pool == 'max':
        return np.nanmax(doc_vec_list, axis=0)

def get_document_vec(text):
    
    return tf_model(text)['outputs'].numpy()[0].reshape(1, -1)

def get_representation_vector(document, title):
    
    title_vec = get_document_vec(title)
    
    document_tokens = document.split()
    doc_len = len(document_tokens)
    doc_vecs = []
    
    doc_vecs.append(title_vec)
    
    if doc_len < 550:
        doc_vecs.append(get_document_vec(document))
    else:
        doc_parts = int(doc_len/500)
        for idx in range(doc_parts):
            if (idx+1)*500 >= doc_len:
                doc_temp = ' '.join(document_tokens[idx*500:])
            else:
                doc_temp = ' '.join(document_tokens[idx*500:(idx+1)*500])
                
            doc_vecs.append(get_document_vec(doc_temp))
        
    return get_pool_vec(get_modified_vectors(doc_vecs), pool='mean')

def get_shorter_text(phrase_1, phrase_2):
    
    if len(phrase_1) < len(phrase_2):
        return phrase_1
    else:
        return phrase_2

def get_stopwords():

    stopwords_de = stopwords.words('german')
    stopwords_en = stopwords.words('english')

    stopwords_full = []
    stopwords_full.extend(stopwords_de)
    stopwords_full.extend(stopwords_en)

    stopwords_full = [word.lower() for word in stopwords_full]
    stop_all = set(stopwords_full + list(string.punctuation))

    return stop_all

def remove_stopwords(noun_chunks):
    
    filtered_noun_chunks = []
    stop_all = get_stopwords()
    
    for word_token in noun_chunks:
        if word_token.lower() not in stop_all:
            filtered_noun_chunks.append(word_token)
            
    return filtered_noun_chunks

def get_filtered_nc(noun_chunks):
    
    noun_chunks = list(set(noun_chunks))
    noun_chunks = remove_stopwords(noun_chunks)
    phrases_len = len(noun_chunks)
    remove_phrases = [] 

    for idx_1 in range(phrases_len):

        phrase_1 = noun_chunks[idx_1]
        for idx_2 in range(idx_1 + 1, phrases_len):
            phrase_2 = noun_chunks[idx_2]

            if fuzz.ratio(phrase_1, phrase_2) > 80:
                remove_phrases.append(get_shorter_text(phrase_1, phrase_2))

    final_noun_chunks = list(set(noun_chunks) - set(remove_phrases))
    return final_noun_chunks

def get_sent_transformers_keywords(keywords, query_vec, max_keyword_cnt=30):
    
    candidate_embeddings_keywords = []
    keywords = list(dict(keywords).keys())

    for kw in keywords: 
        candidate_embeddings_keywords.append(m.unpackb(rdb.get(kw)))
        # candidate_embeddings_keywords.append(tf_model(kw)['outputs'].numpy()[0])

                
    query_distances = cosine_similarity([query_vec], candidate_embeddings_keywords)
    subtopic_keywords_dict = dict()
    for index in query_distances.argsort()[0][-max_keyword_cnt:]: 
        subtopic_keywords_dict[keywords[index]] = query_distances[0][index]
    
    subtopic_keywords_dict = sorted(subtopic_keywords_dict.items(), key=lambda x: x[1], reverse=True)

    return subtopic_keywords_dict

def get_candidate_pool(subtopic_keywords_list, cp_threshold = 0.4):
    
    sim_values = []
    for key, value in subtopic_keywords_list:
        sim_values.append(value)
            
    upper_limit = round(np.percentile(sim_values, cp_threshold), 3)
    candidate_pool = []

    for key, value in subtopic_keywords_list:
        
        if value <= upper_limit:
            candidate_pool.append(key)
                
    return candidate_pool

def get_umap_output(vec_array, dim_size=5):
    
    umap_obj = umap.UMAP(n_neighbors=30, 
                        n_components=UMAP_DIM, 
                        min_dist=0.01,
                        metric='cosine',
                        random_state=123).fit(vec_array) 
    
    umap_output = umap_obj.transform(vec_array) 
    return umap_output, umap_obj

def get_hdbscan_output(data_points, min_clust_size, min_samples):
    
    hdbscan_output = hdbscan.HDBSCAN(
                                    min_cluster_size=min_clust_size,
                                    min_samples=min_samples,
                                    metric='euclidean',
                                    cluster_selection_method='eom').fit(data_points)
    return hdbscan_output

def get_clustering_analysis(cluster_df, final_candidate_pool_vecs, min_clust_size, min_samples):
    
    umap_output_5, umap_5 = get_umap_output(final_candidate_pool_vecs)
    hdbscan_output = get_hdbscan_output(umap_output_5, min_clust_size, min_samples)
    
    cluster_df['cluster_id'] = hdbscan_output.labels_
    cluster_df.cluster_id.hist(bins=150)

    return cluster_df

def get_nearest_keyword(keywords, keyword_vecs, mean_vec):
    
    query_distances = cosine_similarity([mean_vec], list(keyword_vecs))
    subtopic_keywords_dict = dict()
    for index in query_distances.argsort()[0]: 
        
        subtopic_keywords_dict[keywords[index]] = query_distances[0][index]
    
    subtopic_keywords_dict = sorted(subtopic_keywords_dict.items(), key=lambda x: x[1], reverse=True)
    return subtopic_keywords_dict[0][0]


def get_topic_documents(topic_words, final_df):

    doc_id_list = []
    for idx, row in final_df.iterrows():

        candidate_pool = row['candidate_pool']
        doc_id = row['id']

        for tw in topic_words:
            if tw in candidate_pool:
                doc_id_list.append(doc_id)
                break

    return list(set(doc_id_list))


