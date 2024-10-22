import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import networkx as nx
import plotly.graph_objects as go
import re
from typing import Set, List
import os

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
    token_dir = os.path.join('.', 'data', 'token')
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
                    dcc.Graph(id="graph-visualization")
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
     Output("metrics-output", "children")],
    [Input("analyze-button", "n_clicks")],
    [State("file-selector", "value")],
    prevent_initial_call=True
)
def update_output(n_clicks, filename):
    if not filename:
        return dash.no_update
    
    # Read the selected file
    file_path = os.path.join('.', 'data', 'token', filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        return {}, {}, html.P(f"Error reading file: {str(e)}")
    
    # Create graph
    graph = SentenceGraph(text)
    
    # Get graph visualization data
    node_x, node_y, node_z, edge_x, edge_y, edge_z,    node_text, edge_text, node_sizes = graph.get_graph_data()
    
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

    fig_graph.update_layout(scene=dict(
        xaxis=dict(title='X Axis'),
        yaxis=dict(title='Y Axis'),
        zaxis=dict(title='Z Axis'),
        aspectmode='cube'
    ))

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

    return fig_graph, fig_degrees, metrics_output

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)

