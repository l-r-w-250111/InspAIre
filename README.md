# InspAIre: Invention AI, Patent Analysis and Generation

This project provides a suite of tools to analyze patent documents and generate novel inventive ideas through graph-based analogical reasoning. It leverages a local Large Language Model (LLM) via Ollama to extract deep, structured knowledge from patent text.

## Features

-   **Configurable LLM**: Easily change the LLM model or connect to a different Ollama server via an external `config.json` file.
-   **LLM-Powered Knowledge Graph Extraction**: Analyzes patent text to build a knowledge graph of key concepts and their relationships.
-   **One-to-Many Analogy Discovery**: Select a single source patent and let the tool automatically discover and generate analogies from all similar patents in your database.
-   **Vertical 3-Graph Comparison**: Visually understand the analogy process with a clear, vertical layout showing the source, target, and augmented graphs.
-   **Osborn's Checklist Classification**: Classifies generated ideas into creative categories (e.g., "Adapt," "Modify," "Combine").
-   **Interactive GUI**: A Streamlit application provides a clear interface for analyzing patents and exploring generated analogies.

## Installation

This project requires Python 3.8+. Using a virtual environment is highly recommended.

### 1. Install and Run Ollama

This tool depends on a locally running LLM.
-   Download and install **Ollama** from [https://ollama.ai/](https://ollama.ai/).
-   Pull a model to be used for the analysis (Llama3 is listed as an example by default):
    ```bash
    ollama pull llama3
    ```
-   Ensure the Ollama server is running in the background. You can start it with `ollama serve`.

### 2. Set Up Python Environment

First, create and activate a virtual environment:

```bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate
```

Then, install the required Python libraries:

```bash
pip install -r requirements.txt
```

## Configuration

Before running the analysis, you can configure the LLM settings in the `config.json` file:

```json
{
  "ollama_model": "llama3",
  "ollama_api_url": "http://localhost:11434/api/generate"
}
```
-   `ollama_model`: The name of the model you have pulled in Ollama (e.g., "llama3", "mistral").
-   `ollama_api_url`: The endpoint for the Ollama generation API. Change this if your server runs on a different address or port.

## Usage Workflow

The process involves two main stages: a one-time data processing step via the command line, followed by interactive exploration in the GUI.

### Step 1: Process Your Patents (CLI)

First, you need to analyze your patent documents and pre-compute their fingerprints.

1.  **Place Your Patents**: Put your patent documents (as `.txt` files) into the `sample_patents` directory.
2.  **Extract Knowledge Graphs**: Run the main extraction script. This will connect to your local Ollama server to analyze each patent based on your `config.json`.
    ```bash
    python patent_concept_extractor.py --input sample_patents
    ```
    This will populate the `output` directory with `.json` summaries and `.gpickle` graph files for each patent.
3.  **Generate Fingerprints**: Run the pre-computation script to enable similarity searches.
    ```bash
    python precompute_embeddings.py
    ```
    This creates a `fingerprints.pkl` file in your `output` directory.

### Step 2: Explore and Generate (GUI)

Once your data is processed, you can launch the interactive application.

```bash
streamlit run streamlit_app.py
```

This will open the application in your web browser.

-   **Single Patent Analysis Tab**: Select and view the detailed analysis and knowledge graph for any single patent you have processed.
-   **Analogical Generation Tab**:
    1.  Select a **Source Patent**.
    2.  Adjust the **Similarity Threshold**.
    3.  Click **"Discover Similar Patents"**. A list of potential target patents will appear below.
    4.  Select a **Target Patent** from the list.
    5.  The application will then display three graphs vertically for clear comparison: the Source, the Target, and the new "Augmented" graph with generated ideas highlighted. The classified ideas will be listed below the graphs.
