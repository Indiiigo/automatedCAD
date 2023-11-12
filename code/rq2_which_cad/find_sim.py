# given embedded documents, find the similarity:
# 1. between two docs
# 2. between a reference doc and several other docs

import timeit


from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial

from sentence_transformers import SentenceTransformer, util

def find_similarity(embedded_documents, reference, similarity = 'cosine'):
    start = timeit.default_timer()
    sims = []
    for embed in embedded_documents:
        sims.append(1-spatial.distance.cosine(reference, embed))
    stop = timeit.default_timer()
    print("time take for %d: %f" %(len(sims), (stop-start)))
    return sims


def find_multi_similarity(embeddings1, embeddings2):
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    return cosine_scores


def find_top_k(embedded_documents, reference, similarity = 'cosine'):
    pass