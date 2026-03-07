import numpy as np

class SemanticCache:
    def __init__(self, threshold=0.9):
        """
        JUSTIFICATION: The cache is implemented as an in-memory dictionary partitioned by 
        GMM cluster IDs. Instead of a flat list, this structural choice reduces search 
        time complexity from O(N) to roughly O(N/K), where K is the number of clusters.
        """
        self.storage = {} 
        self.threshold = threshold
        self.hits = 0
        self.misses = 0

    def check(self, query_vector, cluster_id):
        """
        JUSTIFICATION: Evaluates semantic similarity by calculating the Euclidean (L2) 
        distance between the incoming query vector and historically cached vectors within 
        the target cluster. A tuned threshold of 0.9 prevents false-positive cache hits.
        """
        if cluster_id not in self.storage:
            self.misses += 1
            return None

        best_match = None
        min_dist = float('inf')

        for entry in self.storage[cluster_id]:
            dist = np.linalg.norm(query_vector - entry["vector"])
            if dist < min_dist:
                min_dist = dist
                best_match = entry

        if min_dist <= self.threshold:
            self.hits += 1
            # JUSTIFICATION: Converts the L2 distance into a normalized similarity score 
            # (between 0 and 1) to provide a human-readable confidence metric in the API response.
            similarity_score = round(float(1 / (1 + min_dist)), 2)
            
            return {
                "matched_query": best_match["query"],
                "similarity_score": similarity_score,
                "result": best_match["result"]
            }

        self.misses += 1
        return None

    def update(self, cluster_id, query_text, query_vector, result):
        """
        JUSTIFICATION: Appends the vector and the database result to the specific 
        cluster partition. This allows future queries in the same semantic space 
        to bypass the FAISS vector database.
        """
        if cluster_id not in self.storage:
            self.storage[cluster_id] = []
            
        self.storage[cluster_id].append({
            "query": query_text,
            "vector": query_vector,
            "result": result
        })

    def clear(self):
        """
        JUSTIFICATION: Flushes the in-memory cache and resets telemetry counters.
        """
        self.storage = {}
        self.hits = 0
        self.misses = 0