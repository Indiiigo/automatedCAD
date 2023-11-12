# given text, embed it using the specified embedding method
# you can also tokenize the sentences and embed the sentences
# in that case, map from old ids (text level) ---> new ids (sentence level)

from simpletransformers.language_representation import RepresentationModel
from simpletransformers.config.model_args import ModelArgs
from sentence_transformers import SentenceTransformer

import pandas as pd
import numpy as np

import timeit

from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial

import ast

DATAROOT = 'data/'
EMBEDDINGROOT = '../data/embeddings/'

def sent_tokenize(documents, id_field = '_id', text_field = 'text'):
    # input: dataframe with old_id, text
    # output: dataframe with new_id, old_id, text, sent

    import nltk.data
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sents = []

    for n, row in documents.iterrows():
        if type(row[text_field]) == str:
            sentences = tokenizer.tokenize(row[text_field])
        else:
            sentences = [""]
            
        for sentence in sentences:
            sents.append([row[id_field], row[text_field], sentence])

    sent_data = pd.DataFrame({text_field: [i[1] for i in sents],
                              "sent": [i[2] for i in sents],
                              "old_id": [i[0] for i in sents]})
    sent_data['new_id'] = range(len(sent_data))
    sent_data = sent_data.set_index("new_id")
    return sent_data

def embed_one_shot(document, embedding_type = 'sbert'): # todo: maybe give option to tokenize?
    # input: document
    # output: embedding

    # define embedding models      
    if embedding_type == 'roberta':
        model_args = ModelArgs(max_seq_length=156)
        roberta_model = RepresentationModel(
            "roberta",
            "roberta-base",
            args=model_args,
            use_cuda=False
        )

        
        start = timeit.default_timer()
        embeddings = roberta_model.encode_sentences([document], combine_strategy="mean")
        stop = timeit.default_timer()
    
    elif embedding_type == 'sbert':
        sbert_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
        start = timeit.default_timer()
        embeddings = sbert_model.encode([document])
        stop = timeit.default_timer()
    
    print("time taken for %d: %f" %(len([document]), stop-start))
    
    return embeddings[0]

def embed_df(df, filename, embedding_type = 'sbert', save = True, tokenize = False, id_field = '_id', text_field = 'text', sep = '\t'):
    # input: dataframe with two fields, old_id and text
    # output: if not tokenize, old_id, text, embeddings
    # if tokenize, new_id, old_id, sent, embeddings 
    
    # tokenize sentences
    if tokenize:
        sents = sent_tokenize(df, id_field = id_field, text_field = text_field)
        id_field = 'new_id'
        text_field = 'sent'
    else:
        sents = df
    
    # check if any of the texts are null
    sents[text_field] = sents[text_field].fillna('')
        
    
    # define embedding models      
    if embedding_type == 'roberta':
        model_args = ModelArgs(max_seq_length=156)
        roberta_model = RepresentationModel(
            "roberta",
            "roberta-base",
            args=model_args,
            use_cuda=False
        )

        
        start = timeit.default_timer()
        embeddings = roberta_model.encode_sentences(sents[text_field], combine_strategy="mean")
        stop = timeit.default_timer()
    
    elif embedding_type == 'sbert':
        sbert_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
        start = timeit.default_timer()
        embeddings = sbert_model.encode(sents[text_field])
        stop = timeit.default_timer()
    
    print("time taken for %d: %f" %(len(sents), stop-start))
    
    if save:
        with open(EMBEDDINGROOT+"/%s_%s_%s_embeddings.npy" %(filename, tokenize, embedding_type), 'wb') as f:
             np.save(f, embeddings)
    
    sents['embeddings'] = list(embeddings)
    return list(embeddings)



def embed(filename, embedding_type = 'sbert', save = True, tokenize = False, id_field = '_id', text_field = 'text', sep = '\t'):
    # input: dataframe with two fields, old_id and text
    # output: if not tokenize, old_id, text, embeddings
    # if tokenize, new_id, old_id, sent, embeddings 
    
    documents = pd.read_csv(filename, sep = sep)
    
    # tokenize sentences
    if tokenize:
        sents = sent_tokenize(documents, id_field = id_field, text_field = text_field)
        id_field = 'new_id'
        text_field = 'sent'
    else:
        sents = documents
    
    # check if any of the texts are null
    sents[text_field] = sents[text_field].fillna('')
        
    
    # define embedding models      
    if embedding_type == 'roberta':
        model_args = ModelArgs(max_seq_length=156)
        roberta_model = RepresentationModel(
            "roberta",
            "roberta-base",
            args=model_args,
            use_cuda=False
        )

        
        start = timeit.default_timer()
        embeddings = roberta_model.encode_sentences(sents[text_field], combine_strategy="mean")
        stop = timeit.default_timer()
    
    elif embedding_type == 'sbert':
        sbert_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
        start = timeit.default_timer()
        embeddings = sbert_model.encode(sents[text_field])
        stop = timeit.default_timer()
    
    print("time taken for %d: %f" %(len(sents), stop-start))
    
    if save:
        with open(EMBEDDINGROOT+"/%s_%s_%s_embeddings.npy" %(filename, tokenize, embedding_type), 'wb') as f:
             np.save(f, embeddings)
    
    sents['embeddings'] = list(embeddings)
    return list(embeddings)


if __name__ == "__main__":
    filename = ''
    # TODO: Test this
