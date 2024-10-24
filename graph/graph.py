import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from typing import Set, List
import os
from collections import defaultdict
import numpy as np

# Stop words definition
STOP_WORDS = set(
    """
a about above across after afterwards again against all almost alone along
already also although always am among amongst amount an and another any anyhow
anyone anything anyway anywhere are around as at

back be became because become becomes becoming been before beforehand behind
being below beside besides between beyond both bottom but by

call can cannot ca could

did do does doing done down due during

each eight either eleven else elsewhere empty enough even ever every
everyone everything everywhere except

few fifteen fifty first five for former formerly forty four from front full
further

get give go

had has have he hence her here hereafter hereby herein hereupon hers herself
him himself his how however hundred

i if in indeed into is it its itself

keep

last latter latterly least less

just

made make many may me meanwhile might mine more moreover most mostly move much
must my myself

name namely neither never nevertheless next nine no nobody none noone nor not
nothing now nowhere

of off often on once one only onto or other others otherwise our ours ourselves
out over own

part per perhaps please put

quite

rather re really regarding

same say see seem seemed seeming seems serious several she should show side
since six sixty so some somehow someone something sometime sometimes somewhere
still such

take ten than that the their them themselves then thence there thereafter
thereby therefore therein thereupon these they third this those though three
through throughout thru thus to together too top toward towards twelve twenty
two

under until up unless upon us used using

various very very via was we well were what whatever when whence whenever where
whereafter whereas whereby wherein whereupon wherever whether which while
whither who whoever whole whom whose why will with within without would

yet you your yours yourself yourselves
""".split()
)

contractions = ["n't", "'d", "'ll", "'m", "'re", "'s", "'ve"]
STOP_WORDS.update(contractions)

for apostrophe in ["'", "'"]:
    for stopword in contractions:
        STOP_WORDS.add(stopword.replace("'", apostrophe))

class GraphComponentsAnalyzer:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.undirected_graph = graph.to_undirected()
        self.sccs = list(nx.strongly_connected_components(self.graph))
        self.bccs = list(nx.biconnected_components(self.undirected_graph))
        
    def get_scc_dag(self):
        """Create DAG of strongly connected components"""
        # Sort SCCs by size in non-decreasing order
        sorted_sccs = sorted(self.sccs, key=len)
        
        # Create mapping of nodes to their SCC index
        node_to_scc = {}
        for i, component in enumerate(sorted_sccs):
            for node in component:
                node_to_scc[node] = i
        
        # Create the DAG
        scc_dag = nx.DiGraph()
        
        # Add nodes with metadata
        for i, component in enumerate(sorted_sccs):
            scc_dag.add_node(i, 
                            size=len(component),
                            nodes=list(component))
        
        # Add edges between components
        for u in self.graph.nodes():
            for v in self.graph.successors(u):
                scc_u = node_to_scc[u]
                scc_v = node_to_scc[v]
                if scc_u != scc_v:
                    scc_dag.add_edge(scc_u, scc_v)
        
        return scc_dag
    
    def plot_scc_dag(self):
        """Create a 3D visualization of the SCC DAG"""
        dag = self.get_scc_dag()
        pos = nx.spring_layout(dag, dim=3)  # Changed to 3D layout
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        edge_x = []
        edge_y = []
        edge_z = []  # Added z coordinates
        for edge in dag.edges():
            x0, y0, z0 = pos[edge[0]]  # Unpack 3D coordinates
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
            
        fig.add_trace(go.Scatter3d(  # Changed to Scatter3d
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'))
        
        # Add nodes
        node_x = []
        node_y = []
        node_z = []  # Added z coordinates
        node_text = []
        node_sizes = []
        
        for node in dag.nodes():
            x, y, z = pos[node]  # Unpack 3D coordinates
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_text.append(f"SCC {node}<br>Size: {dag.nodes[node]['size']}<br>"
                        f"Nodes: {', '.join(map(str, dag.nodes[node]['nodes']))}")
            node_sizes.append(dag.nodes[node]['size'] * 20)
            
        fig.add_trace(go.Scatter3d(  # Changed to Scatter3d
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_sizes,
                color='#1f77b4',
                line_width=2)))
        
        fig.update_layout(
            title='DAG of Strongly Connected Components',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            scene=dict(  # Added 3D scene configuration
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
                aspectmode='cube'
            ),
            paper_bgcolor='white',  # Ensure white background
            plot_bgcolor='white'
        )
        
        return fig

    def plot_bcc_forest(self):
        """Create visualization of the BCC forest with improved node scaling"""
        forest = nx.Graph()
        
        # Add nodes for each BCC
        for i, component in enumerate(self.bccs):
            forest.add_node(i, size=len(component), nodes=list(component))
        
        # Find cut vertices
        cut_vertices = nx.articulation_points(self.undirected_graph)
        
        # Create mapping of nodes to their BCCs
        node_to_bccs = defaultdict(set)
        for i, component in enumerate(self.bccs):
            for node in component:
                node_to_bccs[node].add(i)
        
        # Add edges between BCCs that share cut vertices
        for cut_vertex in cut_vertices:
            connected_bccs = list(node_to_bccs[cut_vertex])
            for i in range(len(connected_bccs)):
                for j in range(i + 1, len(connected_bccs)):
                    forest.add_edge(connected_bccs[i], connected_bccs[j])
        
        # Create visualization with custom layout
        # Use k parameter to increase spacing between nodes
        pos = nx.spring_layout(forest, k=2.0, iterations=50)
        fig = go.Figure()
        
        # Add edges
        edge_x = []
        edge_y = []
        for edge in forest.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'))
        
        # Add nodes with improved sizing
        node_x = []
        node_y = []
        node_text = []
        node_sizes = []
        
        # Get all component sizes for scaling
        sizes = [forest.nodes[node]['size'] for node in forest.nodes()]
        max_size = max(sizes)
        min_size = min(sizes)
        
        # Calculate node sizes using logarithmic scale
        for node in forest.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            size = forest.nodes[node]['size']
            # Use log scale with base 2 and add small constant to avoid log(0)
            log_size = np.log2(size + 1)
            # Scale to reasonable range (10-50)
            scaled_size = 10 + (log_size * 40 / np.log2(max_size + 1))
            node_sizes.append(scaled_size)
            
            # Enhanced hover text with more information
            node_text.append(
                f"BCC {node}<br>" +
                f"Size: {size} nodes<br>" +
                f"Percentage of total: {(size/sum(sizes)*100):.1f}%<br>" +
                f"Nodes: {', '.join(map(str, forest.nodes[node]['nodes']))}"
            )
    
        # Add node trace with improved visual parameters
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_sizes,
                color='#2ca02c',
                line=dict(width=1, color='darkgreen'),
                opacity=0.8,
                symbol='circle',
                sizemode='diameter'
            )))
        
        # Update layout with improved parameters
        fig.update_layout(
            title={
                'text': f'Biconnected Components Forest<br>({len(forest.nodes())} components)',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=60),
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            )
        )
        
        # Add annotation for size information
        size_info = (
            f"Largest component: {max_size} nodes<br>"
            f"Smallest component: {min_size} nodes<br>"
            f"Average size: {sum(sizes)/len(sizes):.1f} nodes"
        )
        fig.add_annotation(
            text=size_info,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
        
        return fig
    
    def plot_component_distributions(self):
        """Plot distributions of SCCs and BCCs by size"""
        # Calculate size distributions
        scc_sizes = [len(component) for component in self.sccs]
        bcc_sizes = [len(component) for component in self.bccs]
        
        # Create subplots
        fig = make_subplots(rows=2, cols=2, 
                           subplot_titles=('SCC Size Distribution (Vertices)',
                                         'BCC Size Distribution (Vertices)',
                                         'SCC Edge Distribution',
                                         'BCC Edge Distribution'))
        
        # Add SCC vertex distribution
        fig.add_trace(
            go.Histogram(x=scc_sizes, name='SCC Vertices',
                        nbinsx=max(10, len(set(scc_sizes))),
                        histnorm='probability'),
            row=1, col=1
        )
        
        # Add BCC vertex distribution
        fig.add_trace(
            go.Histogram(x=bcc_sizes, name='BCC Vertices',
                        nbinsx=max(10, len(set(bcc_sizes))),
                        histnorm='probability'),
            row=1, col=2
        )
        
        # Calculate edge counts for components
        scc_edges = []
        for component in self.sccs:
            subgraph = self.graph.subgraph(component)
            scc_edges.append(subgraph.number_of_edges())
            
        bcc_edges = []
        for component in self.bccs:
            subgraph = self.undirected_graph.subgraph(component)
            bcc_edges.append(subgraph.number_of_edges())
        
        # Add edge distributions
        fig.add_trace(
            go.Histogram(x=scc_edges, name='SCC Edges',
                        nbinsx=max(10, len(set(scc_edges))),
                        histnorm='probability'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=bcc_edges, name='BCC Edges',
                        nbinsx=max(10, len(set(bcc_edges))),
                        histnorm='probability'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False)
        return fig

class SentenceGraph:
    def __init__(self, text: str):
        self.text = text
        self.raw_sentences = self._split_sentences()
        self.sentences = self._extract_sentences()
        self.graph = self._build_graph()
        self.undirected_graph = self.graph.to_undirected()
    
    def _clean_text(self, text: str) -> str:
        """Remove punctuation and convert to lowercase"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text

    def _split_sentences(self) -> List[str]:
        """Split text into raw sentences for display purposes"""
        return [s.strip() for s in re.split('[.\n]', self.text) if s.strip()]

    def _extract_sentences(self) -> List[Set[str]]:
        """Extract and clean sentences from text, removing stop words"""
        cleaned_sentences = []
        
        for sentence in self.raw_sentences:
            cleaned = self._clean_text(sentence)
            words = {word for word in cleaned.split() 
                    if word not in STOP_WORDS and word.strip()}
            if words:
                cleaned_sentences.append(words)
        
        return cleaned_sentences

    def _build_graph(self) -> nx.DiGraph:
        """Build directed graph where edges represent shared words"""
        G = nx.DiGraph()
        
        for i, (words, raw_sentence) in enumerate(zip(self.sentences, self.raw_sentences)):
            G.add_node(i, words=words, raw_sentence=raw_sentence)
        
        for i in range(len(self.sentences)):
            for j in range(i + 1, len(self.sentences)):
                shared_words = self.sentences[i] & self.sentences[j]
                if shared_words:
                    G.add_edge(i, j, 
                            shared_words=shared_words,
                            weight=len(shared_words))
        
        return G

    def get_graph_data(self):
        """Get graph data in format suitable for 3D plotly visualization"""
        # Use 3D spring layout
        pos = nx.spring_layout(self.graph, k=1, iterations=50, dim=3)
        
        edge_x = []
        edge_y = []
        edge_z = []
        edge_text = []
        
        for edge in self.graph.edges(data=True):
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
            edge_text.append(f"Shared words: {', '.join(edge[2]['shared_words'])}<br>Weight: {edge[2]['weight']}")

        node_x = []
        node_y = []
        node_z = []
        node_text = []
        node_sizes = []
        
        for node in self.graph.nodes(data=True):
            x, y, z = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_text.append(
                f"Sentence {node[0]}<br>" +
                f"Original: {node[1]['raw_sentence']}<br>" +
                f"Key words: {', '.join(node[1]['words'])}"
            )
            # Node size based on degree
            node_sizes.append(len(node[1]['words']) * 5)
            
        return node_x, node_y, node_z, edge_x, edge_y, edge_z, node_text, edge_text, node_sizes

    def get_metrics(self):
        """Get basic graph metrics"""
        return {
            'vertices': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'connected_components': nx.number_connected_components(self.undirected_graph),
            'strongly_connected': len(list(nx.strongly_connected_components(self.graph))),
            'biconnected': len(list(nx.biconnected_components(self.undirected_graph))),
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            'density': nx.density(self.graph)
        }

    def get_degree_distributions(self):
        """Get degree distribution data"""
        in_degrees = [sum(d['weight'] for u, v, d in self.graph.in_edges(node, data=True))
                    for node in self.graph.nodes()]
        out_degrees = [sum(d['weight'] for u, v, d in self.graph.out_edges(node, data=True))
                    for node in self.graph.nodes()]
        return in_degrees, out_degrees

def get_token_files():
    """Get list of available token files"""
    token_dir = os.path.join(os.path.expanduser('~/Desktop/graph_transcript/data/token'))
    if not os.path.exists(token_dir):
        return []
    return [f for f in os.listdir(token_dir) if f.endswith('.txt')]

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Sentence Graph Analysis", className="text-center mb-4"),
            dbc.Select(
                id="file-selector",
                options=[
                    {"label": filename, "value": filename}
                    for filename in get_token_files()
                ],
                placeholder="Select a token file...",
                className="mb-3"
            ),
            dbc.Button("Analyze", id="analyze-button", color="primary", className="mb-4"),
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Graph Metrics"),
                dbc.CardBody(id="metrics-output")
            ], className="mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("3D Sentence Graph Visualization"),
                dbc.CardBody([
                    dcc.Graph(
                        id="graph-visualization",
                        style={'height': '800px'}
                    )
                ])
            ], className="mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Strongly Connected Components DAG"),
                dbc.CardBody([
                    dcc.Graph(id="scc-dag", style={'height': '600px'})
                ])
            ], className="mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Component Size Distributions"),
                dbc.CardBody([
                    dcc.Graph(id="component-distributions", style={'height': '800px'})
                ])
            ], className="mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Biconnected Components Forest"),
                dbc.CardBody([
                    dcc.Graph(id="bcc-forest", style={'height': '600px'})
                ])
            ], className="mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Degree Distributions"),
                dbc.CardBody([
                    dcc.Graph(id="degree-distributions")
                ])
            ])
        ])
    ])
], fluid=True)

@app.callback(
    [Output("graph-visualization", "figure"),
     Output("degree-distributions", "figure"),
     Output("metrics-output", "children"),
     Output("scc-dag", "figure"),
     Output("component-distributions", "figure"),
     Output("bcc-forest", "figure")],
    [Input("analyze-button", "n_clicks")],
    [State("file-selector", "value")],
    prevent_initial_call=True
)
def update_output(n_clicks, filename):
    if not filename:
        return dash.no_update
    
    # Read the selected file
    file_path = os.path.join(os.path.expanduser('~/Desktop/graph_transcript/data/token'), filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        return {}, {}, html.P(f"Error reading file: {str(e)}"), {}, {}, {}
    
    # Create graph
    graph = SentenceGraph(text)
    
    # Get graph visualization data
    node_x, node_y, node_z, edge_x, edge_y, edge_z, node_text, edge_text, node_sizes = graph.get_graph_data()
    
    # Create 3D plotly figure for the sentence graph
    fig_graph = go.Figure()

    # Add edges
    fig_graph.add_trace(go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        line=dict(width=0.5, color='gray'),
        text=edge_text,
        hoverinfo='text'
    ))

    # Add nodes
    fig_graph.add_trace(go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode='markers',
        marker=dict(size=5, color='blue', opacity=0.8),
        text=node_text,
        hoverinfo='text'
    ))

    fig_graph.update_layout(
        scene=dict(
            xaxis=dict(
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=''
            ),
            yaxis=dict(
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=''
            ),
            zaxis=dict(
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=''
            ),
            aspectmode='cube'
        ),
        height=800,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='white',
        showlegend=False
    )

    # Get metrics
    metrics = graph.get_metrics()
    metrics_output = html.Ul([
        html.Li(f"Vertices: {metrics['vertices']}"),
        html.Li(f"Edges: {metrics['edges']}"),
        html.Li(f"Connected Components: {metrics['connected_components']}"),
        html.Li(f"Strongly Connected Components: {metrics['strongly_connected']}"),
        html.Li(f"Biconnected Components: {metrics['biconnected']}"),
        html.Li(f"Average Degree: {metrics['avg_degree']:.2f}"),
        html.Li(f"Density: {metrics['density']:.4f}")
    ])

    # Get degree distributions
    in_degrees, out_degrees = graph.get_degree_distributions()
    fig_degrees = go.Figure()
    fig_degrees.add_trace(go.Histogram(
        x=in_degrees,
        name='In-Degree',
        opacity=0.75,
        histnorm='probability'
    ))
    fig_degrees.add_trace(go.Histogram(
        x=out_degrees,
        name='Out-Degree',
        opacity=0.75,
        histnorm='probability'
    ))

    fig_degrees.update_layout(
        title='Degree Distributions',
        xaxis_title='Degree',
        yaxis_title='Probability',
        barmode='overlay'
    )

    # Create component analyzer and get additional visualizations
    component_analyzer = GraphComponentsAnalyzer(graph.graph)
    fig_scc_dag = component_analyzer.plot_scc_dag()
    fig_distributions = component_analyzer.plot_component_distributions()
    fig_bcc_forest = component_analyzer.plot_bcc_forest()

    return (fig_graph, fig_degrees, metrics_output,
            fig_scc_dag, fig_distributions, fig_bcc_forest)

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)