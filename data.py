import os
import re
import numpy as np
import joblib
import faiss
from sklearn.datasets import load_files
from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture

MODEL_NAME = 'all-MiniLM-L6-v2'
NUM_CLUSTERS = 15
DATA_PATH = "./20_newsgroups" 

def clean_text(text):
    """
    JUSTIFICATION: Raw newsgroup files are formatted like old emails. 
    We must strip headers, quote blocks (lines starting with >), and footers.
    If we leave them in, the embedding model will cluster documents based on 
    who sent them rather than what they mean.
    """
    _, _, body = text.partition('\n\n')
    if not body:
        body = text
        
    lines = body.split('\n')
    good_lines = [line for line in lines if not re.match(r'^[>|:]', line.lstrip())]
    
    for i in range(len(good_lines) - 1, -1, -1):
        if good_lines[i].strip() == '--':
            good_lines = good_lines[:i]
            break
            
    return '\n'.join(good_lines).strip()

def prepare_data():
    dataset = load_files(container_path=DATA_PATH, encoding='latin1', decode_error='replace')
    
    cleaned_docs = [clean_text(doc) for doc in dataset.data]
    valid_documents = [doc for doc in cleaned_docs if len(doc) > 50]
    documents = valid_documents[:5000]

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32') 

    # JUSTIFICATION: GMM allows for soft clustering. A document about medicine laws 
    # will show probabilities for both the 'politics' and 'medicines' clusters.
    gmm = GaussianMixture(n_components=NUM_CLUSTERS, random_state=42, covariance_type='diag')
    gmm.fit(embeddings)
    
    sample_probs = gmm.predict_proba(embeddings[:3])
    for i, prob in enumerate(sample_probs):
        top_2 = np.argsort(prob)[-2:][::-1]
        print(f"Doc {i}: Cluster {top_2[0]} ({prob[top_2[0]]:.2f}), Cluster {top_2[1]} ({prob[top_2[1]]:.2f})")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs("models", exist_ok=True)
    joblib.dump(documents, "models/documents.pkl")
    joblib.dump(gmm, "models/gmm_model.pkl")
    faiss.write_index(index, "models/vector_db.index")

if __name__ == "__main__":
    prepare_data()