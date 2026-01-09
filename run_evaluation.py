# -*- coding: utf-8 -*-
"""
RAG System Evaluation - Simplified Version
"""
import json
import time
import requests
import chromadb
import os
from quality_metrics import QualityEvaluator

# Config
DEEPSEEK_API_KEY = "sk-f5ad1f0ba096481181d3eb7ecc3b55ee"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
CHROMA_DB_PATH = "."
COLLECTION_NAME = "medical_rag_lite"
EMBEDDING_MODEL_PATH = r"C:\Users\king\.cache\huggingface\hub\models--moka-ai--m3e-base\snapshots\764b537a0e50e5c7d64db883f2d2e051cbe3c64c"
TOP_K = 5

TEST_QUESTIONS = [
    "感冒了怎么办？",
    "邵长荣医生擅长治疗什么疾病？",
    "中医治疗高血压有哪些方法？",
    "气血两虚的症状有哪些？",
    "吴银根医生的临床经验有哪些？",
]

# Force offline mode
os.environ['HF_HUB_OFFLINE'] = '1'

import sentence_transformers
sentence_transformers.util._import_sentence_transformers = lambda *args, **kwargs: None
from sentence_transformers import SentenceTransformer

def check_dependencies():
    print("Checking dependencies...")
    
    # Check data file
    if not os.path.exists("./data/processed_data.json"):
        print("ERROR: Data file not found")
        return False
    print("  OK: Data file exists")
    
    # Check ChromaDB
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        count = collection.count()
        print(f"  OK: ChromaDB has {count} documents")
    except Exception as e:
        print(f"ERROR: ChromaDB - {e}")
        return False
    
    # Check Ollama
    try:
        response = requests.post(OLLAMA_API_URL, json={"model": "qwen3:8b", "prompt": "test", "stream": False}, timeout=5)
        if response.status_code == 200:
            print("  OK: Ollama API")
        else:
            print(f"ERROR: Ollama API - Status {response.status_code}")
            return False
    except Exception as e:
        print(f"ERROR: Ollama API - {e}")
        return False
    
    # Check DeepSeek
    try:
        headers = {'Authorization': f'Bearer {DEEPSEEK_API_KEY}', 'Content-Type': 'application/json'}
        data = {'model': 'deepseek-chat', 'messages': [{'role': 'user', 'content': 'test'}], 'stream': False}
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=5)
        if response.status_code == 200:
            print("  OK: DeepSeek API")
        else:
            print(f"ERROR: DeepSeek API - Status {response.status_code}")
            return False
    except Exception as e:
        print(f"ERROR: DeepSeek API - {e}")
        return False
    
    # Check embedding model
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_PATH)
        test_emb = model.encode(["test"])
        print(f"  OK: Embedding model ({test_emb.shape[1]} dims)")
    except Exception as e:
        print(f"ERROR: Embedding model - {e}")
        return False
    
    print("All checks passed!")
    return True

def call_ollama(query, context):
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    data = {"model": "qwen3:8b", "prompt": prompt, "stream": False}
    try:
        start = time.time()
        response = requests.post(OLLAMA_API_URL, json=data, timeout=120)
        elapsed = time.time() - start
        result = response.json()['response']
        return result, elapsed
    except Exception as e:
        return f"Error: {e}", 0

def call_deepseek(query, context):
    headers = {'Authorization': f'Bearer {DEEPSEEK_API_KEY}', 'Content-Type': 'application/json'}
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    data = {'model': 'deepseek-chat', 'messages': [{'role': 'user', 'content': prompt}], 'stream': False}
    try:
        start = time.time()
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=120)
        elapsed = time.time() - start
        result = response.json()['choices'][0]['message']['content']
        return result, elapsed
    except Exception as e:
        return f"Error: {e}", 0

def search_documents(collection, query, embedding_model):
    query_embedding = embedding_model.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K, include=['metadatas', 'documents', 'distances'])
    
    if not results['ids'][0]:
        return [], []
    
    docs = []
    for metadata, doc in zip(results['metadatas'][0], results['documents'][0]):
        abstract = doc.split("Abstract:", 1)[1].strip() if "Abstract:" in doc else ""
        docs.append({'title': metadata.get('title', ''), 'abstract': abstract, 'content': doc})
    
    return docs, results['distances'][0]

def evaluate():
    print("=" * 80)
    print("RAG System Evaluation")
    print("=" * 80)

    if not check_dependencies():
        print("=" * 80)
        print("Dependencies check failed. Please fix the errors above.")
        print("=" * 80)
        return

    # Load data
    with open("./data/processed_data.json", 'r', encoding='utf-8') as f:
        all_docs = json.load(f)

    # Initialize
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    # Initialize quality evaluator
    quality_evaluator = QualityEvaluator(embedding_model)
    print("\n[OK] Quality evaluator initialized")

    results = {'qwen3_8b': [], 'deepseek': []}
    generation_metrics = {'qwen3_8b': [], 'deepseek': []}
    retrieval_metrics = []
    
    for i, query in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] {query}")

        # Search
        docs, distances = search_documents(collection, query, embedding_model)
        context = "\n\n---\n\n".join([f"Title: {d['title']}\nAbstract: {d['abstract']}" for d in docs])

        # Evaluate retrieval quality
        retrieval_metrics.append(quality_evaluator.evaluate_retrieval_quality(
            query, [d['content'] for d in docs], distances
        ))

        # Qwen3:8b
        print("  Qwen3:8b...")
        answer_qwen, time_qwen = call_ollama(query, context)
        results['qwen3_8b'].append({'query': query, 'answer': answer_qwen, 'response_time': time_qwen, 'context_length': len(context)})
        print(f"    Time: {time_qwen:.2f}s")

        # Evaluate generation quality for qwen3:8b
        gen_metrics_qwen = quality_evaluator.evaluate_generation_quality(query, answer_qwen, context)
        generation_metrics['qwen3_8b'].append(gen_metrics_qwen)

        # DeepSeek
        print("  DeepSeek...")
        answer_ds, time_ds = call_deepseek(query, context)
        results['deepseek'].append({'query': query, 'answer': answer_ds, 'response_time': time_ds, 'context_length': len(context)})
        print(f"    Time: {time_ds:.2f}s")

        # Evaluate generation quality for DeepSeek
        gen_metrics_ds = quality_evaluator.evaluate_generation_quality(query, answer_ds, context)
        generation_metrics['deepseek'].append(gen_metrics_ds)
    
    # Save results for experiment 1
    with open('evaluation_results_exp1.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save generation metrics
    with open('generation_metrics_exp1.json', 'w', encoding='utf-8') as f:
        json.dump(generation_metrics, f, ensure_ascii=False, indent=2)

    # Save retrieval metrics
    with open('retrieval_metrics_exp1.json', 'w', encoding='utf-8') as f:
        json.dump(retrieval_metrics, f, ensure_ascii=False, indent=2)

    # Summary for experiment 1
    print("\n" + "=" * 80)
    print("Experiment 1 Summary: Model Comparison")
    print("=" * 80)
    for model, data in results.items():
        avg_time = sum(d['response_time'] for d in data) / len(data)
        avg_context = sum(d['context_length'] for d in data) / len(data)
        print(f"\n{model}:")
        print(f"  Avg Response Time: {avg_time:.2f}s")
        print(f"  Avg Context Length: {avg_context:.0f} chars")

    # Quality metrics summary
    print("\n" + "=" * 80)
    print("Quality Metrics Summary")
    print("=" * 80)

    # Generation quality
    print("\nGeneration Quality:")
    for model in ['qwen3_8b', 'deepseek']:
        aggregated = quality_evaluator.aggregate_metrics(generation_metrics[model])
        print(f"\n  {model}:")
        for metric_name, values in aggregated.items():
            print(f"    {metric_name}: {values['mean']:.3f} (+/- {values['std']:.3f})")

    # Retrieval quality
    print("\nRetrieval Quality:")
    aggregated_retrieval = quality_evaluator.aggregate_metrics(retrieval_metrics)
    for metric_name, values in aggregated_retrieval.items():
        print(f"  {metric_name}: {values['mean']:.3f} (+/- {values['std']:.3f})")

    print("\nResults saved to evaluation_results_exp1.json")
    print("Quality metrics saved to generation_metrics_exp1.json and retrieval_metrics_exp1.json")
    
    # ==================== Experiment 2: RAG vs Brute Force vs No Context ====================
    print("\n" + "=" * 80)
    print("Experiment 2: RAG vs Brute Force vs No Context")
    print("=" * 80)

    results_exp2 = {'rag': [], 'bruteforce': [], 'no_context': []}
    generation_metrics_exp2 = {'rag': [], 'bruteforce': [], 'no_context': []}
    retrieval_metrics_exp2 = {'rag': [], 'bruteforce': []}
    
    for i, query in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] {query}")

        # Group A: RAG
        print("  [RAG] Retrieving...")
        docs_rag, dist_rag = search_documents(collection, query, embedding_model)
        context_rag = "\n\n---\n\n".join([f"Title: {d['title']}\nAbstract: {d['abstract']}" for d in docs_rag])

        # Evaluate retrieval quality for RAG
        retrieval_metrics_exp2['rag'].append(quality_evaluator.evaluate_retrieval_quality(
            query, [d['content'] for d in docs_rag], dist_rag
        ))

        print("  [RAG] Generating (DeepSeek)...")
        answer_rag, time_rag = call_deepseek(query, context_rag)
        results_exp2['rag'].append({'query': query, 'answer': answer_rag, 'response_time': time_rag, 'context_length': len(context_rag)})
        print(f"    Time: {time_rag:.2f}s")

        # Evaluate generation quality for RAG
        gen_metrics_rag = quality_evaluator.evaluate_generation_quality(query, answer_rag, context_rag)
        generation_metrics_exp2['rag'].append(gen_metrics_rag)

        # Group B: Brute Force
        print("  [Brute Force] Building context...")
        context_bruteforce = "\n\n---\n\n".join([f"Title: {d['title']}\nAbstract: {d['abstract']}" for d in all_docs[:20]])

        # Evaluate retrieval quality for Brute Force (mock - all docs are retrieved)
        retrieval_metrics_exp2['bruteforce'].append(quality_evaluator.evaluate_retrieval_quality(
            query, [d.get('content', '') for d in all_docs[:20]], [0.1] * 20
        ))

        print("  [Brute Force] Generating (DeepSeek)...")
        answer_bf, time_bf = call_deepseek(query, context_bruteforce)
        results_exp2['bruteforce'].append({'query': query, 'answer': answer_bf, 'response_time': time_bf, 'context_length': len(context_bruteforce)})
        print(f"    Time: {time_bf:.2f}s")

        # Evaluate generation quality for Brute Force
        gen_metrics_bf = quality_evaluator.evaluate_generation_quality(query, answer_bf, context_bruteforce)
        generation_metrics_exp2['bruteforce'].append(gen_metrics_bf)

        # Group C: No Context
        print("  [No Context] Generating (DeepSeek)...")
        answer_nc, time_nc = call_deepseek(query, "")
        results_exp2['no_context'].append({'query': query, 'answer': answer_nc, 'response_time': time_nc, 'context_length': 0})
        print(f"    Time: {time_nc:.2f}s")

        # Evaluate generation quality for No Context
        gen_metrics_nc = quality_evaluator.evaluate_generation_quality(query, answer_nc, None)
        generation_metrics_exp2['no_context'].append(gen_metrics_nc)
    
    # Save results for experiment 2
    with open('evaluation_results_exp2.json', 'w', encoding='utf-8') as f:
        json.dump(results_exp2, f, ensure_ascii=False, indent=2)

    # Save generation metrics for experiment 2
    with open('generation_metrics_exp2.json', 'w', encoding='utf-8') as f:
        json.dump(generation_metrics_exp2, f, ensure_ascii=False, indent=2)

    # Save retrieval metrics for experiment 2
    with open('retrieval_metrics_exp2.json', 'w', encoding='utf-8') as f:
        json.dump(retrieval_metrics_exp2, f, ensure_ascii=False, indent=2)

    # Summary for experiment 2
    print("\n" + "=" * 80)
    print("Experiment 2 Summary: RAG vs Brute Force vs No Context")
    print("=" * 80)
    for group, data in results_exp2.items():
        avg_time = sum(d['response_time'] for d in data) / len(data)
        avg_context = sum(d['context_length'] for d in data) / len(data)
        print(f"\n{group}:")
        print(f"  Avg Response Time: {avg_time:.2f}s")
        print(f"  Avg Context Length: {avg_context:.0f} chars")

    # Quality metrics summary for experiment 2
    print("\n" + "=" * 80)
    print("Quality Metrics Summary (Experiment 2)")
    print("=" * 80)

    # Generation quality
    print("\nGeneration Quality:")
    for group in ['rag', 'bruteforce', 'no_context']:
        aggregated = quality_evaluator.aggregate_metrics(generation_metrics_exp2[group])
        print(f"\n  {group}:")
        for metric_name, values in aggregated.items():
            print(f"    {metric_name}: {values['mean']:.3f} (+/- {values['std']:.3f})")

    # Retrieval quality
    print("\nRetrieval Quality:")
    for group in ['rag', 'bruteforce']:
        aggregated = quality_evaluator.aggregate_metrics(retrieval_metrics_exp2[group])
        print(f"\n  {group}:")
        for metric_name, values in aggregated.items():
            print(f"    {metric_name}: {values['mean']:.3f} (+/- {values['std']:.3f})")

    print("\nResults saved to evaluation_results_exp2.json")
    print("Quality metrics saved to generation_metrics_exp2.json and retrieval_metrics_exp2.json")
    print("=" * 80)

if __name__ == "__main__":
    evaluate()