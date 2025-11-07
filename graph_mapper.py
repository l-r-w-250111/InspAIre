
import networkx as nx
from sentence_transformers import SentenceTransformer, util

class GraphAnalogicalMapper:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def find_structural_analogy(self, g1, g2, threshold=0.5):
        if not g1.nodes() or not g2.nodes():
            return {}

        node_embeddings1 = {n: self.model.encode(d.get('label', '')) for n, d in g1.nodes(data=True)}
        node_embeddings2 = {n: self.model.encode(d.get('label', '')) for n, d in g2.nodes(data=True)}

        mapping = {}
        for n1, emb1 in node_embeddings1.items():
            best_match = None
            max_sim = -1
            for n2, emb2 in node_embeddings2.items():
                sim = util.cos_sim(emb1, emb2).item()
                if sim > max_sim:
                    max_sim = sim
                    best_match = n2
            
            if max_sim > threshold:
                mapping[n1] = best_match
        
        return mapping

    def generate_new_concepts_from_analogy(self, g1, g2, mapping):
        """
        Generates new concepts by finding unmapped nodes in g2 that are connected
        to mapped nodes, and transferring them to the context of g1.
        """
        new_concepts = []
        
        # Get the set of nodes in g2 that are part of the mapping
        mapped_g2_nodes = set(mapping.values())
        
        # Invert the mapping to easily find the g1 counterpart
        inverted_mapping = {v: k for k, v in mapping.items()}

        # Find all nodes in g2 that are NOT in the mapping
        unmapped_g2_nodes = set(g2.nodes()) - mapped_g2_nodes

        for unmapped_node in unmapped_g2_nodes:
            # For each unmapped node, find its neighbors
            for neighbor in g2.neighbors(unmapped_node):
                # If a neighbor IS part of the mapping, we've found a transferable concept
                if neighbor in mapped_g2_nodes:
                    
                    # Get the details of the unmapped node and the relationship
                    unmapped_label = g2.nodes[unmapped_node].get('label', '')
                    edge_data = g2.get_edge_data(neighbor, unmapped_node)
                    if not edge_data: # check other direction for undirected graphs
                         edge_data = g2.get_edge_data(unmapped_node, neighbor)
                    
                    relationship = edge_data.get('rel', 'is related to')

                    # Find the corresponding node in g1
                    g1_equivalent_node = inverted_mapping[neighbor]
                    g1_equivalent_label = g1.nodes[g1_equivalent_node].get('label', '')

                    # Form the new concept
                    new_concept = f"Consider '{unmapped_label}' which '{relationship}' '{g1_equivalent_label}'"
                    new_concepts.append(new_concept)
                    
                    # We only need one connection to transfer the concept
                    break 
        
        return list(set(new_concepts)) # Return unique concepts
