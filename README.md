<h1 align="center">Semantic Cache Search System</h1>
<h3 align="center">AI/ML Engineer Assignment | Internship</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-API-green?style=for-the-badge&logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/FAISS-VectorDB-orange?style=for-the-badge" alt="FAISS"/>
  <img src="https://img.shields.io/badge/GMM-Fuzzy%20Clustering-purple?style=for-the-badge" alt="GMM"/>
</p>

<p align="center">
  A lightweight semantic search system implementing <b>vector embeddings, fuzzy clustering, and a custom semantic cache</b> built from first principles.
</p>

<hr>

<h2>📺 Technical Walkthrough</h2>

<p>
  <b>Loom Demo:</b><br>
  <a href="https://www.loom.com/share/eb028f54d2db4bc291ee1815d66d979e">
    <img src="https://img.shields.io/badge/Watch_Demo-Loom-brightgreen?style=for-the-badge&logo=loom" alt="Watch Demo" />
  </a>
</p>

<hr>

<h2>⚡ Quick Start (Ready to Run)</h2>

<p>
  This repository includes the pre-trained <code>models/</code> directory (containing the text data, GMM model, and FAISS index). <b>You do not need the raw 20 Newsgroups dataset to run and test this API.</b>
</p>

<h3>1. Setup Environment</h3>
<pre><code class="language-bash">python -m venv venv
# Activate environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
</code></pre>

<h3>2. Run API Service</h3>
<pre><code class="language-bash">uvicorn main:app --reload</code></pre>

<p>Access the interactive Swagger UI here: <a href="http://localhost:8000/docs">http://localhost:8000/docs</a></p>

<hr>

<details>
  <summary><h2 style="display:inline-block; cursor:pointer;">🛠️ Optional: Train Models from Scratch</h2></summary>
  <br>
  <p>If you do not clone the <code>models/</code> folder, or if you wish to rebuild the vector database and clustering logic from the raw 20 Newsgroups dataset, follow these steps:</p>
  
  <pre><code class="language-bash"># The script will load the dataset, encode the documents, 
# train the Gaussian Mixture Model, and build the FAISS index.
python data.py</code></pre>

  <p>Once complete, the <code>models/</code> directory will be generated, and you can start the uvicorn server as normal.</p>
</details>

<hr>

<h2>🧠 System Architecture</h2>

<pre>
User Query
   │
   ▼
Sentence Embedding (MiniLM)
   │
   ▼
GMM Cluster Probabilities
   │
   ├── Semantic Cache Lookup (Top-2 clusters)
   │         │
   │         ├── Cache Hit → Return cached result
   │         │
   │         └── Cache Miss
   │
   ▼
FAISS Vector Search (Fallback)
   │
   ▼
Cache Update & API Response
</pre>

<hr>

<h2>🌐 API Endpoints & Cache Management</h2>

<p>
  <b>Note on the Loom Demo:</b> To keep the video concise and focused on the core ML architecture (GMM clustering and FAISS fallback), the cache management endpoints below were not explicitly shown. However, you can easily test them yourself by opening the Swagger UI (<code>/docs</code>), clicking <b>Try it out</b>, and hitting <b>Execute</b>!
</p>

<ul>
  <li><b><code>POST /query</code></b>: The core semantic search endpoint. Submits a query, checks the top-2 GMM clusters in the cache, and falls back to FAISS if necessary.</li>
  <li><b><code>GET /cache/stats</code></b>: Returns real-time telemetry, including total cache entries, hit count, miss count, and the overall hit rate.</li>
  <li><b><code>DELETE /cache</code></b>: Flushes the in-memory cache dictionary and resets all telemetry counters back to zero. Perfect for testing cold-start vs. warm-start performance.</li>
</ul>

<hr>

<h2>⚖️ Design Justifications</h2>

<ul>
  <li><b>Fuzzy Clustering (GMM):</b> Chose GMM over K-Means to handle semantic overlap. Documents are assigned a probability distribution, allowing the system to understand "boundary" topics (e.g., a query blending Politics and Firearms).</li>
  <li><b>Semantic Cache:</b> Implemented a custom partitioned cache that checks the <b>Top-2</b> clusters. This optimizes lookup speed to $O(N/K)$ while preventing cache misses on ambiguous queries.</li>
  <li><b>Similarity Metric:</b> A threshold of <b>0.9</b> (Euclidean $L_2$ distance) was selected to maximize cache precision without hallucinating false matches.</li>
</ul>

<hr>

<h2>🔍 API Telemetry (Terminal Output)</h2>
<p>The system provides real-time vector math logs for transparency during use:</p>
<pre><code>✅ CACHE HIT! Query: 'Repairing a punctured tire' | Cluster: 11 | Distance: 0.88 | Similarity: 0.53
❌ CACHE MISS! Searching FAISS for: 'lol' | Cluster: 3 | Distance: 1.17 | Similarity: 0.46</code></pre>

<hr>

<h2>⚖️ Licensing & Open Source Credits</h2>

<p>
  This project leverages several open-source libraries and models. The repository code is released under the <b>MIT License</b>.
</p>

<ul>
  <li><b>FastAPI / Uvicorn:</b> Web Framework (MIT)</li>
  <li><b>Sentence-Transformers:</b> <code>all-MiniLM-L6-v2</code> Model (Apache 2.0)</li>
  <li><b>Scikit-Learn:</b> GMM Clustering (BSD-3)</li>
  <li><b>FAISS:</b> Vector Search Engine (MIT)</li>
  <li><b>Joblib:</b> Object Serialization (BSD-3)</li>
</ul>

<h3>Dataset Attribution</h3>
<p>
  The machine learning models in this repository were trained using the 20 Newsgroups dataset, which is licensed under a <a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International (CC BY 4.0)</a> license.
  <br><br>
  <b>Citation:</b><br>
  <i>Mitchell, T. (1997). Twenty Newsgroups [Dataset]. UCI Machine Learning Repository. <a href="https://doi.org/10.24432/C5C323">https://doi.org/10.24432/C5C323</a></i>
</p>

<hr>

<h2>👤 Author</h2>
<p><b>Ayush</b></p>