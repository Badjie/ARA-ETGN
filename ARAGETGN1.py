# ============================================================
# Agentic Retrieval-Augmented ETGN for Temporal Fraud Detection
# ============================================================

import pandas as pd
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from transformers import MixtralModel, MixtralConfig

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Agentic Memory
# ============================================================
class AgenticMemory:
    def __init__(self, memory_size=3000):
        self.memory = []
        self.max_size = memory_size

    def add(self, emb):
        if len(self.memory) >= self.max_size:
            self.memory.pop(0)
        self.memory.append(emb.detach().cpu())

    def retrieve(self, query, k=5):
        if len(self.memory) == 0:
            return None

        mem = torch.stack(self.memory).to(query.device)
        query = F.normalize(query, dim=-1)
        mem = F.normalize(mem, dim=-1)
        scores = torch.matmul(mem, query.unsqueeze(-1)).squeeze(-1)
        idx = torch.topk(scores, min(k, len(scores))).indices
        return mem[idx].mean(dim=0)


# ============================================================
# Retrieval Agent
# ============================================================
class RetrievalAgent(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, emb):
        return self.policy(emb).mean()


# ============================================================
# MGGPT Encoder
# ============================================================
class MGGPT(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        
        #GAT Layers
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)
        
        #Projection layer to align dimensions
        self.proj = nn.Linear(hidden_dim * heads, hidden_dim)
        
        #GPT-5 style Encoder by MixtraModel
        cfg = MixtralConfig(
            vocab_size=8,
            hidden_size=hidden_dim,
            num_hidden_layers=4,
            num_attention_heads=heads,
            intermediate_size=hidden_dim * 4,
            num_experts=8,
            num_experts_per_tok=2
        )
        self.gpt = MixtralModel(cfg)

        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)

        # Missing information layers
        self.Wg1 = nn.Linear(hidden_dim, hidden_dim)
        self.Wg2 = nn.Linear(hidden_dim, hidden_dim)
        self.Wb1 = nn.Linear(hidden_dim, hidden_dim)
        self.Wb2 = nn.Linear(hidden_dim, hidden_dim)

    def predict_missing(self, h_v, h_N_v):
        gamma = torch.tanh(self.Wg1(h_v) + self.Wg2(h_N_v))
        beta = torch.tanh(self.Wb1(h_v) + self.Wb2(h_N_v))
        return h_v + (gamma + 1) * beta - h_N_v

    def forward(self, x, edge_index, batch):
        #First GAT Layer
        x = F.relu(self.gat1(x, edge_index))
        
        #projection hidden_dim
        x_proj = self.proj(x)
        
        #Second GAT
        h_N_v = self.gat2(x, edge_index)
        #x = F.relu(self.gat1(x, edge_index))
        #h_N_v = self.gat2(x, edge_index)

        # Missing information correction
        x_corr = self.predict_missing(x_proj, h_N_v)

        # Batch padding for GPT
        num_graphs = batch.max().item() + 1
        max_nodes = torch.bincount(batch).max().item()
        padded = torch.zeros((num_graphs, max_nodes, self.hidden_dim), device=x.device)

        for i in range(num_graphs):
            idx = (batch == i).nonzero(as_tuple=True)[0]
            padded[i, :len(idx)] = x_corr[idx]

        gpt_out = self.gpt(inputs_embeds=padded).last_hidden_state

        flat = torch.cat(
            [gpt_out[i, :torch.sum(batch == i)] for i in range(num_graphs)], dim=0
        )

        fused = self.combine(torch.cat([x_corr, flat], dim=-1))
        return fused


# ============================================================
# Edge Classifier
# ============================================================
class EdgeClassifier:
    def __init__(self):
        self.model = SGDClassifier(
             loss ="log_loss",
             max_iter=2000,
             tol=1e-3,
             random_state=100
             )
        self.is_fitted = False    

    def fit(self, X, y):
        """Initial training"""
        self.model.partial_fit(X, y, classes=np.array([0, 1])) 
        self.is_fitted = True
    
    def partial_fit(self, X, y):
        """ Online update (optional for streaming graphs"""
        if not self.is_fitted:
            self.fit(X, y)
        else:
            self.model.partial_fit(X, y)    
        #self.model.fit(X, y)

    def predict_proba(self, X):
        """Return probability of positive class"""
        probs = self.model.predict_proba(X)
        return probs[:, 1]

# ============================================================
# Agentic Retrieval-Augmented ETGN
# ============================================================
class EnhancedTemporalGraphNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.encoder = MGGPT(input_dim, hidden_dim).to(DEVICE)

        self.memory = AgenticMemory()
        self.agent = RetrievalAgent(hidden_dim * 2).to(DEVICE)
        self.fusion = nn.Linear(hidden_dim * 4, hidden_dim * 2).to(DEVICE)

        self.edge_classifier = EdgeClassifier()
        self.num_layers = num_layers

    def create_edge_embeddings(self, h, edge_index):
        return torch.cat([h[edge_index[0]], h[edge_index[1]]], dim=1)

    def agentic_fusion(self, edge_emb, train=True):
        prob = self.agent(edge_emb)

        if prob > 0.5:
            ctx = self.memory.retrieve(edge_emb)
            if ctx is not None:
                edge_emb = F.relu(self.fusion(torch.cat([edge_emb, ctx], dim=-1)))

        if train:
            self.memory.add(edge_emb)

        return edge_emb

    def forward(self, x, edge_index, batch, train=True):
        h = self.encoder(x, edge_index, batch)
        edge_embeddings = self.create_edge_embeddings(h, edge_index)

        enhanced = []
        for emb in edge_embeddings:
            enhanced.append(self.agentic_fusion(emb, train))

        return torch.stack(enhanced), h


# ============================================================
# ---------------- Graph Construction
# ============================================================

def create_graph(data):
    G = nx.DiGraph()
    nodes = set(data['from_address'].tolist() + data['to_address'].tolist())
    G.add_nodes_from(nodes)
    for _, row in data.iterrows():
        G.add_edge(row['from_address'], row['to_address'], weight=row['timestamp'])
    return G

# ============================================================
# ----------------Fraud and Antifraud Scores Evaluation
# ============================================================
def calculate_fraud_and_antifraud_scores(G):
    # Centrality-based scores
    base_fraud = nx.eigenvector_centrality(G, max_iter=1000)
    base_antifraud = nx.out_degree_centrality(G)

    # Degree-based scores (normalized)
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    max_in = max(in_deg.values()) if in_deg else 1
    max_out = max(out_deg.values()) if out_deg else 1

    in_deg_norm = {n: v / max_in for n, v in in_deg.items()}
    out_deg_norm = {n: v / max_out for n, v in out_deg.items()}

    # Final combined scores
    fraud_scores = {
        n: base_fraud.get(n, 0.0) + out_deg_norm.get(n, 0.0)
        for n in G.nodes()
    }

    antifraud_scores = {
        n: base_antifraud.get(n, 0.0) + in_deg_norm.get(n, 0.0)
        for n in G.nodes()
    }

    return fraud_scores, antifraud_scores


# ============================================================
# ---------------- Labeling Nodes
# ============================================================
def label_nodes(fraud_scores, antifraud_scores, fraud_threshold=0.01, antifraud_threshold=0.01):
    labels = {}
    for node in fraud_scores:
        collection_label = 'collection_irregular' if fraud_scores[node] > fraud_threshold else 'collection_regular'
        pay_label = 'pay_regular' if antifraud_scores[node] > antifraud_threshold else 'pay_irregular'
        labels[node] = (collection_label, pay_label)
    return labels

# ============================================================
# ---------------- Creating Reachability subgraph
# ============================================================
def create_reachability_subgraph(G, node, max_depth=4):
    reachability_subgraph = nx.DiGraph()
    reachability_subgraph.add_node(node)
    current_level = {node}
    for _ in range(max_depth):
        next_level = set()
        for u in current_level:
            for v in G.successors(u):
                if v not in reachability_subgraph:
                    reachability_subgraph.add_edge(u, v, weight=G[u][v]['weight'])
                    next_level.add(v)
        current_level = next_level
    return reachability_subgraph


# ============================================================
# ---------------- Ladeling Edges based on Reachability subgraph
# ============================================================
def label_edges(G, max_depth=4):
    reachability_networks = defaultdict(nx.DiGraph)
    for node in G.nodes:
        reachability_networks[node] = create_reachability_subgraph(G, node, max_depth)
    return reachability_networks

# ============================================================
# ---------------- Counting Edges based on Reachability subgraph
# ============================================================
def count_edges(reach_net, label):
    count = 0
    for _, _, data in reach_net.edges(data=True):
        if label in data:
            count += 1
    return count


# ============================================================
# ---------------- Common Evaluation based on Reachability subgraph
# ============================================================
def common_eval(reachability_networks):
    neighbors = {}
    for node, reach_net in reachability_networks.items():
        neighbors[node] = list(reach_net.neighbors(node))
    return neighbors


# ============================================================
# ---------------- Feature Extaction
# ============================================================
def extract_features(G, node):
    reachability_networks = label_edges(G, max_depth=4)
    neighbors = common_eval(reachability_networks)

    R1 = count_edges(reachability_networks[node], 'collection_regular')
    R2 = count_edges(reachability_networks[node], 'collection_irregular')
    R3 = count_edges(reachability_networks[node], 'payment_regular')
    R4 = count_edges(reachability_networks[node], 'payment_irregular')

    return [
        R1, R2, len(neighbors.get(node, [])),
        R3, R4, len(neighbors.get(node, [])),
        G.in_degree(node), G.out_degree(node)
    ]

# ============================================================
# ---------------- Create Edge Labels
# ============================================================
def create_edge_labels(G, labels, edge_index, node_to_idx):
    y = []
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    for i in range(edge_index.shape[1]):
        u = idx_to_node[edge_index[0, i].item()]
        label = labels[u][0] == 'collection_irregular'
        y.append(int(label))
    return torch.tensor(y, dtype=torch.long)


# ============================================================
# ---------------- Creating Datalist
# ============================================================
def create_data_list(G_list, labels_list):
    data_list = []

    for G, labels in zip(G_list, labels_list):
        node_to_idx = {node: idx for idx, node in enumerate(G.nodes)}

        edge_index = torch.tensor(
            [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges],
            dtype=torch.long
        ).t().contiguous()

        x = torch.tensor(
            [extract_features(G, node) for node in G.nodes],
            dtype=torch.float
        )

        y = create_edge_labels(G, labels, edge_index, node_to_idx)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

    return data_list


# ============================================================
# Training
# ============================================================
def train_model(model, train_loader, epochs=200):
    model.train()
    all_emb, all_labels = [], []

    for epoch in range(epochs):
        for data in train_loader:
            data = data.to(DEVICE)
            emb, _ = model(data.x, data.edge_index, data.batch, train=True)
            all_emb.append(emb.detach().cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

        print(f"Epoch {epoch+1}/{epochs} completed.")

    X = np.concatenate(all_emb)
    y = np.concatenate(all_labels)

    model.edge_classifier.fit(X, y)
    #model.edge_classifier.partial_fit(X, y)
    
    #compute class weights manually
    #pos_weight = len(y) / (2 * np.sum(y) + 1e-6)
    #neg_weight = len(y) / (2 * np.sum(1 - y) + 1e-6)
    
    #sample_weights = np.where(y == 1, pos_weight, neg_weight)
    
    #model.edge_classifier.partial_fit(X, y, sample_weight=sample_weights) 
    print("SGD training completed.")
    return model


# ============================================================
# Evaluation
# ============================================================
def evaluate_model(model, loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            emb, _ = model(batch.x, batch.edge_index, batch.batch, train=False)
            prob = model.edge_classifier.predict_proba(emb.cpu().numpy())
            preds.append(prob)
            labels.append(batch.y.cpu().numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    binary = (preds > 0.5).astype(int)

    auc = roc_auc_score(labels, preds)
    precision = precision_score(labels, binary)
    recall = recall_score(labels, binary)
    f1 = f1_score(labels, binary)

    print(f"AUC={auc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    return auc, precision, recall, f1


# ============================================================
# Main
# ============================================================
def main():
    #Accessing the dataset
    file_path = "soc-sign-bitcoinotc.csv"
    data = pd.read_csv(file_path)

    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    data = data.sort_values(by='timestamp')

    #time-slicing
    data['time_slice'] = (data['timestamp'] - data['timestamp'].min()).dt.days // 31

    threshold = 20
    #time_slice_counts = data['time_slice'].values.counts()
    time_slice_counts = data['time_slice'].value_counts()
    sparse_time_slices = time_slice_counts[time_slice_counts < threshold].index.tolist()

    if sparse_time_slices:
        combined_time_slice = max(data['time_slice']) + 1  
        data.loc[data['time_slice'].isin(sparse_time_slices), 'time_slice'] = combined_time_slice

    data['time_slice'] = data['time_slice'].astype(int)
    new_time_slice_counts = data['time_slice'].value_counts().sort_index()
    print(new_time_slice_counts)
    
    #combined_time_slice = max(data['time_slice']) - 1
    #data.loc[data['time_slice'] >= combined_time_slice, 'time_slice'] = combined_time_slice

    G_list, labels_list = [], []

    for ts in data['time_slice'].unique():
        slice_data = data[data['time_slice'] == ts]
        G = create_graph(slice_data)
        fraud_scores, antifraud_scores = calculate_fraud_and_antifraud_scores(G)
        labels = label_nodes(fraud_scores, antifraud_scores)
        G_list.append(G)
        labels_list.append(labels)

    train_G, test_G, train_labels, test_labels = train_test_split(
        G_list, labels_list, test_size=0.3, shuffle=False
    )

    train_data = create_data_list(train_G, train_labels)
    test_data = create_data_list(test_G, test_labels)

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=8)

    model = EnhancedTemporalGraphNetwork(input_dim=8, hidden_dim=16).to(DEVICE)
    model = train_model(model, train_loader)

    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()
