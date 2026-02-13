
import http.client
import json
import time
import re
import numpy as np
import faiss
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Configuration
NEW_API_KEY = "YOUR_SERPER_API_KEY"
INPUT_FILE = "./data/mcq_train.jsonl"
OUTPUT_FILE_FULL = "./data/mcq_train_with_full_context.jsonl"
OUTPUT_FILE_TOP5_WEB = "./data/mcq_train_with_top5_web.jsonl"
OUTPUT_FILE_TOP5_HYBRID = "./data/mcq_train_with_top5_hybrid.jsonl"

EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def clean_serper_data(raw_data):
    """
    Convert raw JSON to two formats of plain text context.
    
    Returns:
        tuple: (full_context, filtered_top5_context)
    """
    try:
        data = json.loads(raw_data)
    except:
        return "", ""
        
    full_parts = []
    top5_parts = []
    
    if "organic" in data:
        for item in data["organic"]:
            snippet = item.get("snippet", "")
            if snippet:
                full_parts.append(f"<knowledge>{snippet}")
        for item in data["organic"][:3]:
            snippet = item.get("snippet", "")
            if snippet:
                top5_parts.append(f"<knowledge>{snippet}")

    if "peopleAlsoAsk" in data:
        for item in data["peopleAlsoAsk"]:
            s = item.get("snippet", "")
            if s:
                full_parts.append(f"<knowledge>{s}")
        for item in data["peopleAlsoAsk"][:2]:
            s = item.get("snippet", "")
            if s:
                top5_parts.append(f"<knowledge>{s}")

    return "\n\n".join(full_parts), "\n\n".join(top5_parts)

def get_serper_search(query):
    """Execute search and return cleaned data."""
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({"q": query, "gl": "us", "hl": "en"})
    headers = {
        'X-API-KEY': NEW_API_KEY,
        'Content-Type': 'application/json'
    }
    
    try:
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        raw_data = res.read().decode("utf-8")
        return clean_serper_data(raw_data)
    except Exception as e:
        print(f"\n[Error] API Request Failed: {e}")
        return "", ""
    finally:
        conn.close()

def chunk_text(text, chunk_size=50, overlap=10):
    """Split text into overlapping chunks."""
    sentences = re.split(r'(?<=[。.!?！？])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return [c.strip() for c in chunks if c.strip()]

def hybrid_search(question, chunks, top_k=5):
    """Combine BM25 and semantic search to rank chunks."""
    if not chunks:
        return []
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    question_tokens = question.lower().split()
    bm25_scores = bm25.get_scores(question_tokens)

    question_embedding = EMBED_MODEL.encode(question)
    chunks_embeddings = EMBED_MODEL.encode(chunks)
    chunks_embeddings = np.asarray(chunks_embeddings, dtype='float32')

    temp_index = faiss.IndexFlatL2(chunks_embeddings.shape[1])
    temp_index.add(chunks_embeddings)
    distances, indices = temp_index.search(
        np.array([question_embedding], dtype='float32'),
        min(top_k * 2, len(chunks))
    )

    faiss_scores = np.zeros(len(chunks))
    for idx, (i, dist) in enumerate(zip(indices[0], distances[0])):
        faiss_scores[i] = 1.0 / (1.0 + dist)

    bm25_scores = bm25_scores / (np.max(bm25_scores) + 1e-8)
    combined_scores = 0.5 * bm25_scores + 0.5 * faiss_scores

    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    return [(chunks[i], combined_scores[i]) for i in top_indices]

def process_dataset():
    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_FILE_FULL, "w", encoding="utf-8") as f_full, \
         open(OUTPUT_FILE_TOP5_WEB, "w", encoding="utf-8") as f_top5_web, \
         open(OUTPUT_FILE_TOP5_HYBRID, "w", encoding="utf-8") as f_top5_hybrid:
        
        lines = f_in.readlines()
        print(f"Processing {len(lines)} records")
        print(f"  Full context output: {OUTPUT_FILE_FULL}")
        print(f"  Top5 web output: {OUTPUT_FILE_TOP5_WEB}")
        print(f"  Top5 hybrid output: {OUTPUT_FILE_TOP5_HYBRID}")
        
        for line in tqdm(lines, desc="Processing"):
            data = json.loads(line)
            instruction = data.get("instruction", "")
            raw_input = data.get("input", "")
            output = data.get("output", "")
            question = raw_input.split("<question>")[1].split("?")[0].strip()

            try:
                question = re.sub(r'(\?)\s*.*$', r'\1', question).strip()
            except IndexError:
                print(f"\n[Warning] Failed to extract question")
            print(f"Question: {question}")
            full_context, top5_web_context = get_serper_search(question)
            time.sleep(0.1)

            chunks = chunk_text(full_context, chunk_size=50, overlap=10)
            ranked_chunks = hybrid_search(question, chunks, top_k=5)
            hybrid_context = "\n\n".join([chunk for chunk, score in ranked_chunks])
            data_full = {
                "instruction": instruction,
                "input": f"{full_context}\n\n{raw_input}",
                "output": output
            }
            f_full.write(json.dumps(data_full, ensure_ascii=False) + "\n")

            data_top5_web = {
                "instruction": instruction,
                "input": f"{top5_web_context}\n\n{raw_input}",
                "output": output
            }
            f_top5_web.write(json.dumps(data_top5_web, ensure_ascii=False) + "\n")

            data_top5_hybrid = {
                "instruction": instruction,
                "input": f"{hybrid_context}\n\n{raw_input}",
                "output": output
            }
            f_top5_hybrid.write(json.dumps(data_top5_hybrid, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    process_dataset()