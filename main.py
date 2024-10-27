import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage

import json
from pathlib import Path

import datetime

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State

import networkx as nx
import plotly.graph_objects as go

import queue
import logging
import sys
import traceback

load_dotenv()
GROQ_MODEL_NAME = "llama3-groq-70b-8192-tool-use-preview"

def setup_logging():
    """Setup logging configuration"""
    logger = logging.getLogger('semantic_mapper')
    if not logger.handlers:  # Only add handlers if they don't exist
        logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_format = logging.Formatter('%(message)s')  # Simplified console format
        console_handler.setFormatter(console_format)
        
        # File handler
        file_handler = logging.FileHandler('semantic_mapper_debug.log')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger

class SemanticNeuralMapper:
    def __init__(self, network_file="neural_network.json"):
        self.logger = setup_logging()
        self.network_file = Path(network_file)
        # Create directory if it doesn't exist
        self.network_file.parent.mkdir(parents=True, exist_ok=True)
        # Create empty network file if it doesn't exist
        if not self.network_file.exists():
            self.create_empty_network_file()
        
        self.graph = nx.Graph()
        self.llm = None
        self.prompt = None
        self.message_queue = queue.Queue()
        self.setup_llm()
        self.load_network()
        self.setup_prompt()

    def create_empty_network_file(self):
        """Create an empty network file with initial structure"""
        initial_data = {
            "nodes": [],
            "connections": []
        }
        try:
            with open(self.network_file, 'w') as f:
                json.dump(initial_data, f, indent=2)
            self.logger.info(f"Created new network file: {self.network_file}")
        except Exception as e:
            self.logger.error(f"Error creating network file: {str(e)}")
            raise
    
    def setup_llm(self):
        """Initialize the Groq LLM"""
        self.logger.info("Setting up LLM...")
        try:
            self.llm = ChatGroq(
                groq_api_key=os.getenv("GROQ_API_KEY"),
                model_name=GROQ_MODEL_NAME,
                temperature=0.7
            )
            self.logger.info("LLM setup complete")
        except Exception as e:
            self.logger.error(f"Error setting up LLM: {str(e)}")
            raise

    def load_network(self):
        """Load existing network from file"""
        self.logger.info(f"Loading network from {self.network_file}")
        if self.network_file.exists():
            try:
                with open(self.network_file, 'r') as f:
                    data = json.load(f)
                    for node in data.get('nodes', []):
                        self.graph.add_node(
                            node['id'],
                            label=node.get('label', node['id']),
                            type=node.get('type', 'concept'),
                            weight=node.get('weight', 1.0)
                        )
                    for conn in data.get('connections', []):
                        self.graph.add_edge(
                            conn['source'],
                            conn['target'],
                            relation=conn.get('relation', 'related'),
                            weight=conn.get('weight', 1.0)
                        )
                self.logger.info(f"Loaded network with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
            except Exception as e:
                self.logger.error(f"Error loading network: {str(e)}")
                self.logger.info("Starting with empty network")
        else:
            self.logger.info("No existing network file found. Starting fresh.")

    def save_network(self):
        """Save current network to file"""
        try:
            data = {
                'nodes': [
                    {
                        'id': node,
                        'label': self.graph.nodes[node].get('label', node),
                        'type': self.graph.nodes[node].get('type', 'concept'),
                        'weight': self.graph.nodes[node].get('weight', 1.0)
                    }
                    for node in self.graph.nodes
                ],
                'connections': [
                    {
                        'source': edge[0],
                        'target': edge[1],
                        'relation': self.graph.edges[edge].get('relation', 'related'),
                        'weight': self.graph.edges[edge].get('weight', 1.0)
                    }
                    for edge in self.graph.edges
                ]
            }
            with open(self.network_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info("Network saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving network: {str(e)}")

    def get_network_context(self):
        """Get current network state as context string"""
        context = "Current network contains:\n"
        for node in self.graph.nodes:
            node_data = self.graph.nodes[node]
            context += f"- Node: {node_data.get('label', node)} (Type: {node_data.get('type', 'concept')})\n"
        return context

    def setup_prompt(self):
        """Initialize the system prompt template"""
        self.logger.info("Setting up prompt template...")
        
        try:
            system_message_content = (
                "You are a semantic neural network analyzer. Analyze the concept and return ONLY a valid JSON "
                "object representing semantic relationships. The response should contain ONLY the JSON, "
                "no additional text or formatting. Format:\n"
                "{\n"
                "    \"nodes\": [\n"
                "        {\n"
                "            \"id\": \"concept1\",\n"
                "            \"label\": \"Concept One\",\n"
                "            \"type\": \"category|concept|property|action\",\n"
                "            \"weight\": 0.8\n"
                "        }\n"
                "    ],\n"
                "    \"connections\": [\n"
                "        {\n"
                "            \"source\": \"concept1\",\n"
                "            \"target\": \"concept2\",\n"
                "            \"relation\": \"relationship_type\",\n"
                "            \"weight\": 0.5\n"
                "        }\n"
                "    ]\n"
                "}\n\n"
                "Current network context: {network_context}"
            )

            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_message_content),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("Analyze this concept and its relationships: {concept}")
            ])
            
            self.prompt = prompt
            self.logger.info("Prompt template setup complete")
            
        except Exception as e:
            self.logger.error(f"Error setting up prompt template: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def analyze_concept(self, concept):
        """Process input text and update the semantic network"""

        self.logger.info("\n" + "="*50)
        self.logger.info("Starting new analysis")
        self.logger.info("="*50)

        try:
            # Input validation
            if not concept or not concept.strip():
                self.logger.error("Empty input received")
                return False

            self.logger.info(f"Analyzing concept: {concept}")
            
            # Get current network context
            network_context = self.get_network_context()
            
            # Create conversation chain
            conversation = LLMChain(
                llm=self.llm,
                prompt=self.prompt,
                verbose=True
            )

            # Get response
            try:
                response = conversation.predict(
                    concept=concept,
                    network_context=network_context,
                    chat_history=[]  # Add empty chat history
                )
                
                self.logger.info("Response received from Groq")
                self.logger.debug(f"Raw response: {response}")
                
            except Exception as e:
                self.logger.error("Error during Groq API call")
                self.logger.error(f"Error: {str(e)}")
                self.logger.error(traceback.format_exc())
                return False

            # Process response
            try:
                # Clean response content
                content = response.strip()
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()
                
                # Parse JSON
                analysis = json.loads(content)
                self.logger.info("JSON parsed successfully")
                self.logger.debug(f"Parsed content: {json.dumps(analysis, indent=2)}")

                # Update network
                self.update_network(analysis)
                return True

            except json.JSONDecodeError as e:
                self.logger.error("JSON parsing error")
                self.logger.error(f"Error: {str(e)}")
                self.logger.error(f"Content that failed to parse: {content}")
                return False
                
            except Exception as e:
                self.logger.error("Unexpected error during response processing")
                self.logger.error(f"Error: {str(e)}")
                self.logger.error(traceback.format_exc())
                return False

        except Exception as e:
            self.logger.error("Critical error in analyze_concept")
            self.logger.error(f"Error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def update_network(self, analysis):
        """Update the network with new nodes and connections"""
        self.logger.info("Updating network with new information...")
        
        # Track changes
        nodes_before = set(self.graph.nodes())
        
        # Create normalized label to node ID mapping
        label_to_id = {
            self.graph.nodes[node].get('label', '').lower().strip(): node 
            for node in self.graph.nodes()
        }
        
        # Process nodes
        for node in analysis['nodes']:
            normalized_label = node['label'].lower().strip()
            
            if normalized_label in label_to_id:
                existing_id = label_to_id[normalized_label]
                node['original_id'] = node['id']
                node['id'] = existing_id
                
                # Update existing node
                self.graph.nodes[existing_id].update({
                    'type': node['type'],
                    'weight': max(node['weight'], 
                                self.graph.nodes[existing_id].get('weight', 0))
                })
                self.logger.debug(f"Updated existing node: {existing_id}")
            else:
                # Add new node
                self.graph.add_node(
                    node['id'],
                    label=node['label'],
                    type=node['type'],
                    weight=node['weight']
                )
                label_to_id[normalized_label] = node['id']
                self.logger.debug(f"Added new node: {node['id']}")
        
        # Process connections
        for conn in analysis['connections']:
            source_id = conn['source']
            target_id = conn['target']
            
            # Remap IDs if needed
            for node in analysis['nodes']:
                if node.get('original_id') == source_id:
                    source_id = node['id']
                if node.get('original_id') == target_id:
                    target_id = node['id']
            
            # Add connection
            if source_id in self.graph.nodes and target_id in self.graph.nodes:
                self.graph.add_edge(
                    source_id,
                    target_id,
                    relation=conn['relation'],
                    weight=conn['weight']
                )
                self.logger.debug(f"Added connection: {source_id} -> {target_id}")
            else:
                self.logger.warning(f"Skipped connection: missing node(s)")
        
        # Save changes
        self.save_network()
        
        # Log summary
        nodes_after = set(self.graph.nodes())
        new_nodes = nodes_after - nodes_before
        self.logger.info(f"Network update complete:")
        self.logger.info(f"- New nodes added: {len(new_nodes)}")
        self.logger.info(f"- Total nodes: {len(self.graph.nodes)}")
        self.logger.info(f"- Total edges: {len(self.graph.edges)}")

    def get_plotly_graph(self):
        """Generate Plotly graph visualization"""
        if not self.graph.nodes:
            return go.Figure()
            
        # Use Kamada-Kawai layout for better node distribution
        pos = nx.kamada_kawai_layout(self.graph)
        
        # Create traces list to store all traces
        traces = []
        
        # Add edges
        edge_x = []
        edge_y = []
        edge_text = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            relation = self.graph.edges[edge].get('relation', '')
            edge_text.extend([relation, relation, None])
        
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines',
            name='Connections'
        )
        traces.append(edge_trace)
        
        # Create separate traces for each node type
        node_types = set(nx.get_node_attributes(self.graph, 'type').values())
        colors = {
            'category': '#FF9999',
            'concept': '#99FF99',
            'property': '#9999FF',
            'action': '#FFBB99'  # Added color for action type
        }
        
        for node_type in node_types:
            node_x = []
            node_y = []
            node_text = []
            node_sizes = []
            
            for node in self.graph.nodes():
                if self.graph.nodes[node].get('type') == node_type:
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    label = self.graph.nodes[node].get('label', node)
                    node_text.append(f"{label} ({node_type})")
                    weight = self.graph.nodes[node].get('weight', 1.0)
                    node_sizes.append(weight * 30)
            
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="bottom center",
                marker=dict(
                    color=colors.get(node_type, '#FFFFFF'),
                    size=node_sizes,
                    line=dict(width=2, color='#FFFFFF')
                ),
                name=node_type.capitalize()
            )
            traces.append(node_trace)
        
        # Create the figure with all traces
        fig = go.Figure(
            data=traces,
            layout=go.Layout(
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='#0d1117',
                paper_bgcolor='#0d1117',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    font=dict(color='white'),
                    bgcolor='rgba(0,0,0,0.5)'
                ),
                font=dict(color='white')
            )
        )
        
        return fig

def create_dash_app(mapper):
    """Create and configure the Dash application"""
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.Div([
            dcc.Textarea(
                id='concept-input',
                placeholder='Enter any text, sentence, or concept...',
                style={
                    'width': '100%',
                    'height': '100px',
                    'margin': '10px',
                    'padding': '10px',
                    'backgroundColor': '#161b22',
                    'color': 'white',
                    'border': '1px solid #30363d',
                    'borderRadius': '6px',
                    'resize': 'vertical'
                }
            ),
            html.Button(
                'Analyze',
                id='analyze-button',
                style={
                    'margin': '10px',
                    'padding': '10px 20px',
                    'backgroundColor': '#238636',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '6px',
                    'cursor': 'pointer'
                }
            ),
            html.Div(id='status-message', style={
                'margin': '10px',
                'color': '#58a6ff'
            })
        ], style={
            'backgroundColor': '#0d1117',
            'padding': '10px',
            'borderBottom': '1px solid #30363d'
        }),
        dcc.Graph(
            id='network-graph',
            style={'height': 'calc(100vh - 180px)'},
            config={'displayModeBar': True}
        ),
        dcc.Store(id='last-update-time'),
        dcc.Interval(
            id='interval-component',
            interval=500,  # Reduced to 500ms for more responsive updates
            n_intervals=0
        )
    ], style={
        'backgroundColor': '#0d1117',
        'color': 'white',
        'minHeight': '100vh'
    })

    @app.callback(
        [Output('network-graph', 'figure'),
         Output('status-message', 'children'),
         Output('last-update-time', 'data')],
        [Input('analyze-button', 'n_clicks'),
         Input('interval-component', 'n_intervals')],
        [State('concept-input', 'value'),
         State('last-update-time', 'data')]
    )
    def update_graph(n_clicks, n_intervals, value, last_update):
        ctx = dash.callback_context
        if not ctx.triggered:
            return mapper.get_plotly_graph(), "", datetime.datetime.now().isoformat()

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        current_time = datetime.datetime.now().isoformat()
        
        if trigger_id == 'analyze-button' and value:
            success = mapper.analyze_concept(value)
            status = "Analysis complete!" if success else "Error during analysis"
            return mapper.get_plotly_graph(), status, current_time
            
        # Check message queue for updates
        try:
            while True:  # Process all queued messages
                mapper.message_queue.get_nowait()
        except queue.Empty:
            pass
            
        return mapper.get_plotly_graph(), "", current_time

    return app

def main():
    """Main application entry point"""
    print("Initializing Semantic Neural Mapper...")
    mapper = SemanticNeuralMapper()
    app = create_dash_app(mapper)
    print("Starting server...")
    app.run_server(debug=True)

if __name__ == "__main__":
    main()