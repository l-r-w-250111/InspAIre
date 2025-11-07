
import os
import pickle
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def generate_fingerprint(graph, model):
    """Generates a single 'fingerprint' vector for a graph."""
    if not graph.nodes():
        return np.zeros(model.get_sentence_embedding_dimension())
    
    node_labels = [d.get('label', '') for _, d in graph.nodes(data=True)]
    
    # Encode all labels and average them to get a single vector
    embeddings = model.encode(node_labels)
    return np.mean(embeddings, axis=0)

def main():
    output_dir = "output"
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    fingerprints = {}
    
    gpickle_files = [f for f in os.listdir(output_dir) if f.endswith(".gpickle")]
    
    if not gpickle_files:
        print("No .gpickle files found in the output directory. Nothing to do.")
        return

    for filename in tqdm(gpickle_files, desc="Generating fingerprints"):
        base_name = os.path.splitext(filename)[0]
        graph_path = os.path.join(output_dir, filename)
        
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
            
        fingerprints[base_name] = generate_fingerprint(G, model)

    fingerprint_path = os.path.join(output_dir, "fingerprints.pkl")
    with open(fingerprint_path, "wb") as f:
        pickle.dump(fingerprints, f)
        
    print(f"Successfully generated and saved fingerprints for {len(fingerprints)} graphs to {fingerprint_path}")

if __name__ == "__main__":
    main()
