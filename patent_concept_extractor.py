import re
import requests
import json
import argparse
import os
from typing import List, Dict, Tuple
import nltk
from tqdm import tqdm
import networkx as nx
import numpy as np
import pandas as pd
import pickle

# sentence-transformers for embeddings
from sentence_transformers import SentenceTransformer

# scikit-learn
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# NLTK Resource Management
# -------------------------
def download_nltk_resources():
    resources = ["punkt", "punkt_tab"]
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            print(f"NLTK resource '{resource}' not found. Downloading...")
            nltk.download(resource)

# -------------------------
# Utility: Text sectioning
# -------------------------
def split_sections(patent_text: str) -> Dict[str, str]:
    text = patent_text
    abstract = ""
    # Look for abstract in English or Japanese
    m = re.search(r"(?is)(?:abstract|要約)(.*?)(?:\n{2,}|$)", text)
    if m:
        abstract = m.group(1).strip()
    claims = ""
    # Look for claims in English or Japanese
    m = re.search(r"(?is)(?:claims?|請求項)[:：]?\s*.*$", text, re.MULTILINE)
    if m:
        claims = m.group(0).strip()
    desc = text
    if abstract:
        desc = desc.replace(abstract, "")
    if claims:
        desc = desc.replace(claims, "")
    title = ""
    lines = text.strip().splitlines()
    if lines:
        title = lines[0].strip()
    return {"title": title, "abstract": abstract, "description": desc.strip(), "claims": claims}

# -------------------------
# Embedding / clustering / KG (Re-add from previous version)
# -------------------------
def embed_texts(model, texts: List[str], batch_size=32) -> np.ndarray:
    return np.array(model.encode(texts, batch_size=batch_size, show_progress_bar=False))

def cluster_concepts(embeddings: np.ndarray, n_clusters=10) -> Tuple[np.ndarray, object]:
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels, km

def build_concept_kg(nodes: List[Dict], edges: List[Tuple[str,str,str]]) -> nx.DiGraph:
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n['id'], label=n.get('label'), text=n.get('text'))
    for s, r, d in edges:
        G.add_edge(s, d, rel=r)
    return G

INVENTION_PRINCIPLES = {
    "Conversion": "転換型 (Conversion): エネルギー・情報などの形態を変換する (e.g., 光→電気, 圧力→信号)",
    "Regulation": "制御型 (Regulation): 物理量をフィードバック制御する (e.g., PID制御, タイミング補正)",
    "Configuration": "構造型 (Configuration): 形・配置を変えて機能を得る (e.g., 冷却フィンの配置最適化)",
    "Mediation": "媒介型 (Mediation): 異なる系を結合・変換する媒介要素 (e.g., 絶縁層, 触媒, プロトコル)"
}

def classify_invention_principle(concept_texts: List[str], embed_model, principles: Dict[str, str]) -> Dict[str, float]:
    principle_labels = list(principles.keys())
    principle_descs = list(principles.values())
    principle_embs = embed_model.encode(principle_descs, show_progress_bar=False)
    concept_embs = embed_model.encode(concept_texts, show_progress_bar=False)
    sims = cosine_similarity(concept_embs, principle_embs)
    agg_scores = np.max(sims, axis=0)
    norm_scores = agg_scores / (np.sum(agg_scores) + 1e-9)
    return {label: float(score) for label, score in zip(principle_labels, norm_scores)}

# -------------------------
# Ollama-based Knowledge Graph Extraction
# -------------------------
def detect_language_with_ollama(text_snippet: str, ollama_url: str, model: str) -> str:
    """
    Detects if the text is Japanese or English using Ollama.
    """
    prompt = f"""Is the following text primarily Japanese or English?
Answer with only a single word: 'Japanese' or 'English'.

Text:
{text_snippet}
"""
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = requests.post(ollama_url, json=payload, timeout=120)
        response.raise_for_status()
        response_body = response.json()
        lang = response_body.get("response", "").strip().lower()
        if "japanese" in lang:
            return "japanese"
        elif "english" in lang:
            return "english"
        else:
            print(f"[WARN] Language detection returned an unexpected value: '{lang}'. Defaulting to English.")
            return "english"
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"[ERROR] Language detection failed: {e}. Defaulting to English.")
        return "english"


def extract_graph_with_ollama(
    patent_text: str,
    ollama_url: str,
    model: str,
    language: str = "japanese"
) -> Dict:
    """
    Extracts concepts (nodes) and relations (edges) from patent text to form a knowledge graph.
    """
    prompts = {
        "japanese": f"""
あなたは高度な特許分析AIです。以下の特許テキストを分析し、発明の核心的な構造をナレッジグラフとして抽出してください。

抽出する要素：
1.  **Nodes**: 発明の主要な「構成要素」「技術的課題」「解決手段」「作用・効果」を表す概念。
2.  **Edges**: それらの概念間の関係性。「利用する(uses)」「解決する(solves)」「引き起こす(causes)」「有効である(enables)」「である(is_a)」など。

出力は以下のJSON形式のみとし、他のテキストは含めないでください：
{{
  "nodes": [
    {{"id": 0, "label": "概念1", "type": "構成要素"}},
    {{"id": 1, "label": "概念2", "type": "技術的課題"}},
    ...
  ],
  "edges": [
    {{"source": 0, "target": 1, "label": "解決する"}},
    ...
  ]
}}

特許テキスト：
{patent_text[:4000]}
""",
        "english": f"""
You are an advanced patent analysis AI. Analyze the following patent text and extract its core inventive structure as a knowledge graph.

Elements to extract:
1.  **Nodes**: Concepts representing the main "Components", "Technical Problems", "Solutions", and "Effects" of the invention.
2.  **Edges**: The relationships between these concepts, such as "uses", "solves", "causes", "enables", "is_a".

Your output must be only in the following JSON format, with no other text:
{{
  "nodes": [
    {{"id": 0, "label": "Concept 1", "type": "Component"}},
    {{"id": 1, "label": "Concept 2", "type": "Technical Problem"}},
    ...
  ],
  "edges": [
    {{"source": 0, "target": 1, "label": "solves"}},
    ...
  ]
}}

Patent Text:
{patent_text[:4000]}
"""
    }
    
    prompt = prompts.get(language, prompts["english"])
    payload = {"model": model, "prompt": prompt, "stream": False, "format": "json"}

    try:
        response = requests.post(ollama_url, json=payload, timeout=300)
        response.raise_for_status()
        response_body = response.json()
        graph_json_str = response_body.get("response")

        if graph_json_str:
            graph_data = json.loads(graph_json_str)
            # Basic validation
            if "nodes" in graph_data and "edges" in graph_data:
                return graph_data
            else:
                print("[WARN] Ollama response is not a valid graph structure.")
                return None
        return None
    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"[ERROR] Failed to get a valid graph from Ollama: {e}")
        return None

# -------------------------
# Pipeline (Ollama Graph version)
# -------------------------
def pipeline_process_ollama(patent_text: str, embed_model_name="all-mpnet-base-v2"):
    # Load config
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        ollama_url = config.get("ollama_api_url")
        ollama_model = config.get("ollama_model")
        if not ollama_url or not ollama_model:
            raise ValueError("Ollama URL or model not found in config.json")
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"[ERROR] Could not load configuration: {e}")
        # Return a structure indicating failure
        return {"title": "Configuration Error", "error": str(e)}, nx.DiGraph()

    # 1. Detect language
    print("Detecting language...")
    language = detect_language_with_ollama(patent_text[:500], ollama_url=ollama_url, model=ollama_model)
    print(f"Language detected: {language}")

    sections = split_sections(patent_text)
    title = sections['title']

    # 2. Extract graph using Ollama
    print(f"Extracting knowledge graph with Ollama (language: {language})...")
    graph_data = extract_graph_with_ollama(
        patent_text,
        ollama_url=ollama_url,
        model=ollama_model,
        language=language
    )

    if not graph_data:
        print("[WARN] No graph was extracted. Cannot proceed.")
        return {"title": title, "error": "Graph extraction failed."}, nx.DiGraph()

    concept_nodes = graph_data.get("nodes", [])
    concept_texts = [node.get("label", "") for node in concept_nodes]

    if not concept_texts:
        print("[WARN] No concepts found in the extracted graph.")
        return {"title": title, "error": "No concepts extracted."}, nx.DiGraph()

    # 2. Build Knowledge Graph
    G = nx.DiGraph()
    for node in concept_nodes:
        G.add_node(node["id"], label=node["label"], type=node.get("type", "Unknown"))
    for edge in graph_data.get("edges", []):
        G.add_edge(edge["source"], edge["target"], rel=edge["label"])

    # 3. Embeddings and clustering (optional, can be used for node coloring, etc.)
    embed_model = SentenceTransformer(embed_model_name)
    emb = embed_texts(embed_model, concept_texts)

    # 4. Classify invention principle
    invention_classification = classify_invention_principle(concept_texts, embed_model, INVENTION_PRINCIPLES)

    # 5. Output summary
    summary = {
        "title": title,
        "n_concepts": len(concept_texts),
        "invention_principle_classification": invention_classification,
        # Clusters are no longer a primary output, but we can store node types
        "concepts_by_type": {
            node["type"]: [n["label"] for n in concept_nodes if n["type"] == node["type"]]
            for node in concept_nodes
        }
    }
    return summary, G


# -------------------------
# CLI
# -------------------------
def main():
    download_nltk_resources()
    parser = argparse.ArgumentParser(description="Extracts concepts from patent texts using an LLM via Ollama.")
    parser.add_argument("--input", type=str, required=True, help="Path to a patent text file or a directory.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the output files.")
    parser.add_argument("--embed_model", type=str, default="all-mpnet-base-v2")
    parser.add_argument("--n_clusters", type=int, default=5)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    files_to_process = []
    if os.path.isdir(args.input):
        files_to_process = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(".txt")]
    elif os.path.isfile(args.input):
        files_to_process = [args.input]
    else:
        print(f"Error: Input path {args.input} is not a valid file or directory.")
        return

    for filepath in tqdm(files_to_process, desc="Processing patents"):
        print(f"\nProcessing {filepath}...")
        with open(filepath, "r", encoding="utf-8") as f:
            txt = f.read()

        summary, G = pipeline_process_ollama(txt, embed_model_name=args.embed_model)

        base_filename = os.path.splitext(os.path.basename(filepath))[0]
        summary_path = os.path.join(args.output_dir, f"{base_filename}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Summary saved to {summary_path}")

        graph_path = os.path.join(args.output_dir, f"{base_filename}.gpickle")
        with open(graph_path, "wb") as f:
            pickle.dump(G, f)
        print(f"Graph saved to {graph_path}")

if __name__ == "__main__":
    main()
