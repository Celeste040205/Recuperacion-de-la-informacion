from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datasets import load_dataset
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# Descargar recursos de NLTK de forma segura
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

app = FastAPI(title="NFCorpus IR Service", version="1.0")

# Variables globales para los índices y datos
documents = {}
doc_corpus_texts = []
doc_ids = []
tfidf_vectorizer = None
tfidf_matrix = None
bm25_index = None

def preprocess_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

@app.on_event("startup")
def load_and_index_data():
    global documents, doc_corpus_texts, doc_ids, tfidf_vectorizer, tfidf_matrix, bm25_index
    print("Cargando corpus NFCorpus en el servidor...")
    corpus_ds = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")
    
    for item in corpus_ds:
        doc_id = item["_id"]
        title = item.get("title", "")
        text = item.get("text", "")
        combined_text = f"{title} {text}"
        
        documents[doc_id] = {
            "title": title,
            "text": text,
            "processed": preprocess_text(combined_text)
        }
        doc_corpus_texts.append(documents[doc_id]["processed"])
        doc_ids.append(doc_id)

    # Construir índices TF-IDF y BM25
    tfidf_vectorizer = TfidfVectorizer(max_features=20000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(doc_corpus_texts)
    
    tokenized_corpus = [doc.split() for doc in doc_corpus_texts]
    bm25_index = BM25Okapi(tokenized_corpus)
    print("¡Indexación completada exitosamente en el servidor!")

class SearchRequest(BaseModel):
    query: str
    model: str  # "bm25" o "tfidf"
    top_k: int = 10

# --- NUEVO: Endpoint /health requerido por el examen ---
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "corpus": "nfcorpus",
        "documents_indexed": len(doc_ids)
    }

@app.post("/search")
def search_endpoint(req: SearchRequest):
    model_name = req.model.lower().strip()
    if model_name not in ["tfidf", "bm25"]:
        raise HTTPException(status_code=400, detail="Modelo no válido. Use 'tfidf' o 'bm25'.")
    
    query_processed = preprocess_text(req.query)
    
    if model_name == "tfidf":
        query_vec = tfidf_vectorizer.transform([query_processed])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    else:
        tokenized_query = query_processed.split()
        scores = bm25_index.get_scores(tokenized_query)
        
    top_indices = scores.argsort()[::-1][:req.top_k]
    
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        d_id = doc_ids[idx]
        results.append({
            "rank": rank,
            "document_id": d_id,  # Debe llamarse exactamente 'document_id'
            "score": float(scores[idx]),
            "title": documents[d_id]["title"],
            "snippet": documents[d_id]["text"][:200] + "..."
        })
        
    return {
        "model": model_name,  # Debe incluir el modelo utilizado
        "results": results     # Debe incluir la lista de resultados ordenados
    }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)