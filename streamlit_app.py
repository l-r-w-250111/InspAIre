# streamlit_app.py
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
import os
import pickle
from graph_mapper import GraphAnalogicalMapper
from osborn_classifier import OsbornClassifier
import matplotlib.font_manager as fm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# --- Font Configuration ---

def find_japanese_font():
    """
    Searches for a Japanese font file on the system across common paths.
    """
    font_paths = [
        '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf',  # Linux
        'C:/Windows/Fonts/YuGothB.ttc',  # Windows (Yu Gothic Bold)
        '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',  # macOS (Hiragino Kaku Gothic)
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            return fm.FontProperties(fname=path)
            
    try:
        return fm.FontProperties(family='TakaoPGothic')
    except Exception:
        pass

    return None

FONT_PROP = find_japanese_font()
if FONT_PROP is None:
    st.warning("A suitable Japanese font was not found. Graph labels may not render correctly.")


# --- Functions ---

@st.cache_data
def load_analysis_files(output_dir):
    files = {}
    if os.path.exists(output_dir):
        for filename in sorted(os.listdir(output_dir)):
            if filename.endswith(".json"):
                base_name = os.path.splitext(filename)[0]
                files[base_name] = os.path.join(output_dir, filename)
    return files

def visualize_comparison_graphs(g1, g2, g3, titles):
    fig, axes = plt.subplots(3, 1, figsize=(15, 45), dpi=150) # Vertical layout
    graphs = [g1, g2, g3]
    font_kwargs = {'font_family': FONT_PROP.get_name()} if FONT_PROP else {}
    for i, (graph, title) in enumerate(zip(graphs, titles)):
        ax = axes[i]
        if not graph or not graph.nodes():
            ax.set_title(f"{title}\n(Graph is empty)", size=16)
            ax.axis('off')
            continue
        pos = nx.spring_layout(graph, k=0.8, iterations=50, seed=42)
        labels = {n: graph.nodes[n].get('label', '') for n in graph.nodes()}
        node_colors = ['lightgreen' if graph.nodes[n].get('is_new') else 'lightblue' for n in graph.nodes()]
        nx.draw(graph, pos, ax=ax, labels=labels, node_size=2500, node_color=node_colors, font_size=10, **font_kwargs)
        edge_labels = nx.get_edge_attributes(graph, 'rel')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax, font_color='red', **font_kwargs)
        ax.set_title(title, size=22, fontproperties=FONT_PROP if FONT_PROP else None)
    plt.tight_layout()
    try:
        st.pyplot(fig)
    except Exception as e:
        st.error("An error occurred while rendering the graphs.")
        print(f"Error rendering matplotlib graph: {e}")

# --- Main App ---
st.set_page_config(layout="wide")
st.title("Invention AI: Patent Analysis and Generation")

output_directory = "output"

# --- Sidebar ---
st.sidebar.title("Controls")
if st.sidebar.button("Refresh Analyses"):
    st.cache_data.clear()
    st.rerun()

analysis_files = load_analysis_files(output_directory)
fingerprint_path = os.path.join(output_directory, "fingerprints.pkl")

if not analysis_files:
    st.warning(f"No analysis files found in '{output_directory}'.")
    st.info("Run `patent_concept_extractor.py` and `precompute_embeddings.py`, then click 'Refresh Analyses'.")
else:
    tab1, tab2 = st.tabs(["Single Patent Analysis", "Analogical Generation"])

    # --- Tab 1: Single Patent Analysis ---
    with tab1:
        st.header("Analyze a Single Patent")
        selected_patent_key = st.selectbox(
            "Choose a patent to view:",
            options=list(analysis_files.keys()),
            key="single_patent_selector"
        )
        
        if selected_patent_key:
            json_path = analysis_files[selected_patent_key]
            graph_path = os.path.join(output_directory, f"{selected_patent_key}.gpickle")

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            st.subheader(f"Patent Title: {data.get('title', 'N/A')}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Invention Principle Classification")
                classification = data.get("invention_principle_classification", {})
                if classification:
                    df_class = pd.DataFrame.from_dict(classification, orient='index', columns=['Score'])
                    st.bar_chart(df_class)
                else:
                    st.info("No classification data available.")
                
                st.subheader("Extracted Inventive Concepts by Type")
                concepts_by_type = data.get("concepts_by_type", {})
                if concepts_by_type:
                    for type, concepts in concepts_by_type.items():
                        with st.expander(f"{type} ({len(concepts)})"):
                            for concept in concepts:
                                st.markdown(f"- `{concept}`")
                else:
                    st.info("No typed concepts extracted.")
            
            with col2:
                if os.path.exists(graph_path):
                    with open(graph_path, "rb") as f:
                        G = pickle.load(f)
                    
                    def visualize_single_graph(graph, title):
                        if not graph.nodes():
                            st.info("The concept graph is empty.")
                            return
                        fig, ax = plt.subplots(figsize=(12, 12))
                        pos = nx.spring_layout(graph, k=0.8, iterations=50, seed=42)
                        labels = {n: graph.nodes[n].get('label', '') for n in graph.nodes()}
                        
                        font_kwargs = {'font_family': FONT_PROP.get_name()} if FONT_PROP else {}
                        nx.draw(graph, pos, ax=ax, labels=labels, node_size=3000, node_color='lightblue', font_size=10, **font_kwargs)
                        
                        edge_labels = nx.get_edge_attributes(graph, 'rel')
                        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax, font_color='red', **font_kwargs)
                        ax.set_title(title, size=20, fontproperties=FONT_PROP if FONT_PROP else None)
                        st.pyplot(fig)
                    
                    visualize_single_graph(G, "Patent Knowledge Graph")
                else:
                    st.warning("Graph file not found.")

    # --- Tab 2: One-to-Many Analogy Discovery ---
    with tab2:
        st.header("Discover and Generate Analogies")
        if not os.path.exists(fingerprint_path):
            st.error("Fingerprints not found. Please run `precompute_embeddings.py` first.")
        else:
            with open(fingerprint_path, "rb") as f:
                fingerprints = pickle.load(f)
            if 'target_list' not in st.session_state:
                st.session_state.target_list = None
            if 'selected_target' not in st.session_state:
                st.session_state.selected_target = None
            col1, col2 = st.columns([1, 3])
            with col1:
                st.subheader("1. Select Source Patent")
                source_key = st.selectbox("Source Patent:", list(analysis_files.keys()))
                st.subheader("2. Set Discovery Threshold")
                similarity_threshold = st.slider("Similarity Threshold", 0.1, 1.0, 0.8, 0.05)
                if st.button("1. Discover Similar Patents", use_container_width=True):
                    st.session_state.target_list = []
                    st.session_state.selected_target = None
                    source_fp = fingerprints[source_key]
                    found_targets = []
                    for key, fp in fingerprints.items():
                        if key == source_key: continue
                        sim = cosine_similarity([source_fp], [fp])[0][0]
                        if sim >= similarity_threshold:
                            found_targets.append((key, sim))
                    found_targets.sort(key=lambda x: x[1], reverse=True)
                    st.session_state.target_list = found_targets
                    if not st.session_state.target_list:
                        st.warning("No similar patents found above the threshold.")
                if st.session_state.target_list is not None and st.session_state.target_list:
                    st.subheader("2. Select Target for Analogy")
                    target_options = {f"{key} (Similarity: {sim:.2f})": key for key, sim in st.session_state.target_list}
                    display_options = ["-- Please Select a Target --"] + list(target_options.keys())
                    selected_display = st.selectbox("Target Patent:", options=display_options)
                    if selected_display != "-- Please Select a Target --":
                        st.session_state.selected_target = target_options[selected_display]
                    else:
                        st.session_state.selected_target = None
            with col2:
                if st.session_state.selected_target:
                    target_key = st.session_state.selected_target
                    source_graph_path = os.path.join(output_directory, f"{source_key}.gpickle")
                    target_graph_path = os.path.join(output_directory, f"{target_key}.gpickle")
                    with open(source_graph_path, "rb") as f: g1 = pickle.load(f)
                    with open(target_graph_path, "rb") as f: g2 = pickle.load(f)
                    if not g1.nodes() or not g2.nodes():
                        st.warning("Source or Target graph is empty. Cannot generate analogy.")
                    else:
                        mapper = GraphAnalogicalMapper()
                        mapping = mapper.find_structural_analogy(g1, g2)
                        new_concepts = mapper.generate_new_concepts_from_analogy(g1, g2, mapping)
                        g_augmented = g1.copy()
                        if new_concepts:
                            for i, concept in enumerate(new_concepts):
                                new_node_id = f"new_{i}"
                                g_augmented.add_node(new_node_id, label=concept, is_new=True)
                        st.subheader("Analogical Mapping Visualization")
                        visualize_comparison_graphs(g1, g2, g_augmented, [f"Source: {source_key}", f"Target: {target_key}", "Augmented Idea"])
                        if new_concepts:
                            st.subheader("Generated & Classified Ideas")
                            classifier = OsbornClassifier()
                            classified_ideas = defaultdict(list)
                            for concept in new_concepts:
                                category = classifier.classify_idea(concept)
                                classified_ideas[category].append(concept)
                            for category, ideas in classified_ideas.items():
                                with st.expander(f"**{category}** ({len(ideas)} ideas)"):
                                    for idea in ideas:
                                        st.info(idea)
                        else:
                            st.info("No novel concepts were generated from this pair.")
                else:
                    st.info("After discovering patents, select a target from the list to generate ideas.")
