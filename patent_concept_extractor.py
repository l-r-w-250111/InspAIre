
import argparse
import json
import os
import pickle
import networkx as nx
import requests
from tqdm import tqdm

def extract_knowledge_graph_with_ollama(text, model, api_url, max_retries=3):
    prompt = f"""
    You are an expert patent analyst. Your task is to extract a knowledge graph from the following patent text.
    The graph should represent the core inventive concepts and their relationships.
    Provide the output as a single JSON object containing 'nodes' and 'edges'.
    Nodes must have 'id' (string), 'label' (string), and 'type' (e.g., "Component", "Process", "Material").
    Edges must have 'source' (string, node id), 'target' (string, node id), and 'label' (string, relationship).

    Example:
    {{
      "nodes": [
        {{"id": "node1", "label": "Adjustable Strap", "type": "Component"}},
        {{"id": "node2", "label": "Locking Mechanism", "type": "Component"}}
      ],
      "edges": [
        {{"source": "node1", "target": "node2", "label": "connects to"}}
      ]
    }}

    Now, analyze the following patent text:
    ---
    {text}
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(
                api_url,
                json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
                timeout=120
            )
            response.raise_for_status()
            
            json_string = response.json().get('response', '{}')
            graph_data = json.loads(json_string)

            if 'nodes' in graph_data and 'edges' in graph_data:
                return graph_data

        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"[ERROR] Failed to get a valid graph from Ollama: {e}")
    
    return None

def load_config(config_path="config.json"):
    """Loads the configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Configuration file not found at {config_path}. Please create it.")
        exit(1)
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON in the configuration file: {config_path}.")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Process patent files to extract concepts.")
    parser.add_argument("--input", required=True, help="Input file or directory of patent texts.")
    parser.add_argument("--output", default="output", help="Directory to save the output files.")
    parser.add_argument("--config", default="config.json", help="Path to the configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)
    ollama_model = config.get("ollama_model", "llama3")
    ollama_api_url = config.get("ollama_api_url", "http://localhost:11434/api/generate")

    os.makedirs(args.output, exist_ok=True)

    files_to_process = []
    if os.path.isdir(args.input):
        files_to_process = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(".txt")]
    else:
        files_to_process.append(args.input)

    for filepath in tqdm(files_to_process, desc="Processing patents"):
        print(f"\nProcessing {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        base_name = os.path.splitext(os.path.basename(filepath))[0]
        
        graph_data = extract_knowledge_graph_with_ollama(text, ollama_model, ollama_api_url)
        
        summary = {"title": base_name, "source_file": filepath}
        
        G = nx.DiGraph()
        if graph_data:
            for node in graph_data.get('nodes', []):
                G.add_node(node['id'], label=node.get('label', ''), type=node.get('type', ''))
            for edge in graph_data.get('edges', []):
                G.add_edge(edge['source'], edge['target'], rel=edge.get('label', ''))
            
            summary['concepts_by_type'] = {ntype: [d['label'] for n, d in G.nodes(data=True) if d.get('type') == ntype] for ntype in set(d.get('type') for _, d in G.nodes(data=True))}
        else:
            print("[WARN] No graph was extracted. Cannot proceed.")

        summary_path = os.path.join(args.output, f"{base_name}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        print(f"Summary saved to {summary_path}")

        graph_path = os.path.join(args.output, f"{base_name}.gpickle")
        with open(graph_path, 'wb') as f:
            pickle.dump(G, f)
        print(f"Graph saved to {graph_path}")

if __name__ == "__main__":
    main()
