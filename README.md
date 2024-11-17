# Graph Transcript Analysis

This project processes a lecture transcript to generate and analyze a directed graph based on shared words between sentences. Each node in the graph represents a sentence, and edges indicate shared words between sentences. The project provides various graph-related analyses and visualizations.

## Features

1. **Graph Construction**:
   - Nodes represent collections of words from sentences in the transcript.
   - Directed edges connect nodes if their sentences share common words.
   - Edges are labeled and weighted by the number of shared words.

2. **Graph Analysis**:
   - Compute and output the number of vertices, edges, and connected components (considered as undirected).
   - Plot the undirected degree distribution.
   - Plot the weighted in-degree and out-degree distributions.

3. **Strongly Connected Components (SCCs)**:
   - Identify SCCs in the graph.
   - Visualize the DAGs of SCCs sorted by the number of vertices.
   - Plot the distribution of SCCs by size (vertices and edges).

4. **Bi-Connected Components (BCCs)**:
   - Convert the graph to an undirected format.
   - Compute BCCs and visualize the BCC forest.

5. **Preprocessing**:
   - Remove punctuation, spaces, line breaks, and stop words during graph construction.

## Requirements
- Python libraries: `pypdf2`, `NLTK`, `spaCy`, `igraph`, `NetworkX`, `matplotlib`, or equivalent.
- A transcript file (e.g., PDF or text format) for processing.
- Optional: Use available preprocessing scripts and visualization tools like `2D Force Graph` or `3D Force-Directed Graph`.

## Inputs
- A transcript in PDF or text format.

