import joblib
import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from cache import SemanticCache

docs = joblib.load("models/documents.pkl")
gmm = joblib.load("models/gmm_model.pkl")
index = faiss.read_index("models/vector_db.index")
encoder = SentenceTransformer('all-MiniLM-L6-v2')

app = FastAPI(title="Semantic Search API")
semantic_cache = SemanticCache() 

class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def root():
    return {"message": "Semantic Search API is running! Add /docs and visit to test the endpoints."}

@app.post("/query")
async def perform_query(request: QueryRequest):
    """
    JUSTIFICATION: The core retrieval endpoint. It uses Sentence-BERT to encode the query, 
    then uses the GMM to predict the cluster. By isolating the cache search to specific 
    clusters, we reduce search time complexity from O(N) to O(N/K).
    """
    query_vec = encoder.encode(request.query).astype('float32')
    
    # JUSTIFICATION: GMM provides soft clustering probabilities. Semantically identical queries 
    # might fall on the boundary of two clusters. Checking the top 2 most probable 
    # clusters resolves cross cluster cache misses.
    probs = gmm.predict_proba([query_vec])[0]
    top_clusters = np.argsort(probs)[-2:][::-1] 
    dominant_cluster = int(top_clusters[0]) 

    cache_hit_data = None
    hit_cluster = None
    
    for cluster_id in top_clusters:
        cache_hit_data = semantic_cache.check(query_vec, int(cluster_id))
        if cache_hit_data:
            hit_cluster = int(cluster_id)
            break 

    if len(top_clusters) == 2:
        if cache_hit_data and hit_cluster == int(top_clusters[1]):
            semantic_cache.misses -= 1
        elif not cache_hit_data:
            semantic_cache.misses -= 1

    if cache_hit_data:
        # Similiarity score is printed in the terminal after extracting
        sim_score = cache_hit_data["similarity_score"]
        
        # Calculate distance back from similarity score: distance = (1 / similarity) - 1
        raw_distance = round((1 / sim_score) - 1, 4)
        
        print(f"CACHE HIT! Query: '{request.query}' | Cluster: {hit_cluster} | Distance: {raw_distance} | Similarity: {sim_score}")
        
        return {
            "query": request.query,
            "cache_hit": True,
            **cache_hit_data,
            "dominant_cluster": hit_cluster
        }

    # JUSTIFICATION: FAISS is utilized for the fallback search because it provides 
    # highly optimized level vector similarity search it is significantly
    # faster than a brute-force similarity scan in Python.
    
    D, I = index.search(np.array([query_vec]), k=1)
    
    # Calculate similarity score for the miss using FAISS distance
    # FAISS IndexFlatL2 returns squared distance, so we take the square root to match cache.py
    faiss_distance = float(np.sqrt(max(0, D[0][0])))
    faiss_sim_score = round(float(1 / (1 + faiss_distance)), 2)
    
    print(f"CACHE MISS! Searching FAISS for: '{request.query}' | Cluster: {dominant_cluster} | Distance: {round(faiss_distance, 4)} | Similarity: {faiss_sim_score}")
    
    search_result_text = docs[I[0][0]]

    semantic_cache.update(dominant_cluster, request.query, query_vec, search_result_text)

    total_cached = sum(len(v) for v in semantic_cache.storage.values())
    print(f"Saved to Cache. Total items now: {total_cached}")

    return {
        "query": request.query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": faiss_sim_score, 
        "result": search_result_text[:500] + "...", 
        "dominant_cluster": dominant_cluster
    }

@app.get("/cache/stats")
async def get_stats():
    """
    JUSTIFICATION: Provides real-time telemetry on cache performance. 
    Hit rate is calculated dynamically to monitor the threshold's effectiveness in production.
    """
    total_requests = semantic_cache.hits + semantic_cache.misses
    hit_rate = semantic_cache.hits / total_requests if total_requests > 0 else 0.0
    
    total_entries = sum(len(cluster_list) for cluster_list in semantic_cache.storage.values())
    
    return {
        "total_entries": total_entries,
        "hit_count": semantic_cache.hits,
        "miss_count": semantic_cache.misses,
        "hit_rate": round(hit_rate, 3)
    }

@app.delete("/cache")
async def flush_cache():
    semantic_cache.clear()
    return {"message": "Cache flushed successfully."}