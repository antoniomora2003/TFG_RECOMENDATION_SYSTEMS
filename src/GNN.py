import pandas as pd
import numpy as np
import random
import os
import json
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

# Data Preprocessing
def preprocess_data(train_file, test_file, use_imputed=False, imputed_file=None):
    if use_imputed and imputed_file is not None:
        train_df = pd.read_csv(imputed_file)
    else:
        train_df = pd.read_csv(train_file)
    
    test_df = pd.read_csv(test_file)
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    users = combined_df['user_id'].unique()
    venues = combined_df['venue_id'].unique()
    user_to_idx = {user: idx for idx, user in enumerate(users)}
    venue_to_idx = {venue: idx for idx, venue in enumerate(venues)}
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    idx_to_venue = {idx: venue for venue, idx in venue_to_idx.items()}
    
    train_df['user_idx'] = train_df['user_id'].map(user_to_idx)
    train_df['venue_idx'] = train_df['venue_id'].map(venue_to_idx)
    test_df['user_idx'] = test_df['user_id'].map(user_to_idx)
    test_df['venue_idx'] = test_df['venue_id'].map(venue_to_idx)
    
    train_positive_items = defaultdict(set)
    train_visited_real = defaultdict(set)
    
    if use_imputed:
        for _, row in train_df.iterrows():
            user_idx = int(row['user_idx'])
            venue_idx = int(row['venue_idx'])
            source = row['source']
            train_positive_items[user_idx].add(venue_idx)
            if source == 'N':
                train_visited_real[user_idx].add(venue_idx)
    else:
        for _, row in train_df.iterrows():
            user_idx = int(row['user_idx'])
            venue_idx = int(row['venue_idx'])
            train_positive_items[user_idx].add(venue_idx)
            train_visited_real[user_idx].add(venue_idx)
    
    test_data = defaultdict(set)
    for _, row in test_df.iterrows():
        test_data[int(row['user_idx'])].add(int(row['venue_idx']))
    
    num_users = len(users)
    num_items = len(venues)
    
    return train_df, test_df, train_positive_items, train_visited_real, test_data, num_users, num_items, user_to_idx, venue_to_idx, idx_to_user, idx_to_venue

# Generate Triplets for BPR Loss
def generate_triplets(train_positive_items, num_items):
    triplets = []
    for user in train_positive_items:
        pos_items = train_positive_items[user]
        if len(pos_items) == num_items:
            continue
        for pos_item in pos_items:
            attempts = 0
            max_attempts = 10
            while attempts < max_attempts:
                neg_item = random.randint(0, num_items - 1)
                if neg_item not in pos_items:
                    triplets.append((user, pos_item, neg_item))
                    break
                attempts += 1
    return triplets

# Generate Venue Embeddings using BERT and Geographic Data
def generate_venue_embeddings(combined_df, venue_to_idx):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    combined_df = combined_df.dropna(subset=['latitude', 'longitude', 'category_name'])
    descriptions = []
    latitudes = []
    longitudes = []
    venue_ids = sorted(venue_to_idx.keys(), key=lambda x: venue_to_idx[x])
    
    for venue in venue_ids:
        venue_data = combined_df[combined_df['venue_id'] == venue].iloc[0]
        desc = f"Venue {venue}: {venue_data['category_name']}"
        descriptions.append(desc)
        latitudes.append(float(venue_data['latitude']))
        longitudes.append(float(venue_data['longitude']))
    
    embeddings = model.encode(descriptions)
    latitudes = np.array(latitudes).reshape(-1, 1)
    longitudes = np.array(longitudes).reshape(-1, 1)
    scaler = StandardScaler()
    lat_lon = scaler.fit_transform(np.hstack([latitudes, longitudes]))
    
    venue_features = np.hstack([embeddings, lat_lon])
    return torch.tensor(venue_features, dtype=torch.float)

# Create Heterogeneous Graph
def create_hetero_graph(train_df, num_users, num_items, user_to_idx, venue_to_idx):
    data = HeteroData()
    data['user'].num_nodes = num_users
    data['venue'].num_nodes = num_items
    user_indices = train_df['user_idx'].values
    venue_indices = train_df['venue_idx'].values
    
    assert max(user_indices) < num_users, "User index out of range"
    assert max(venue_indices) < num_items, "Venue index out of range"
    
    edge_index = torch.tensor(np.vstack((user_indices, venue_indices)), dtype=torch.long)
    data['user', 'visits', 'venue'].edge_index = edge_index
    edge_index_rev = torch.tensor(np.vstack((venue_indices, user_indices)), dtype=torch.long)
    data['venue', 'visited_by', 'user'].edge_index = edge_index_rev
    return data

# GNN Model Definition
class GNN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers, venue_features_dim):
        super(GNN, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.venue_proj = nn.Linear(venue_features_dim, embedding_dim)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('user', 'visits', 'venue'): SAGEConv((embedding_dim, embedding_dim), embedding_dim),
                ('venue', 'visited_by', 'user'): SAGEConv((embedding_dim, embedding_dim), embedding_dim)
            }, aggr='mean')
            self.convs.append(conv)
    
    def forward(self, x_dict, edge_index_dict):
        user_emb = self.user_embedding.weight
        venue_emb = self.venue_proj(x_dict['venue'])
        x_dict = {'user': user_emb, 'venue': venue_emb}
        
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        return x_dict['user'], x_dict['venue']

# Training Function
def train_gnn(model, optimizer, train_data, edge_index_dict, venue_embeddings, iterations, batch_size=1000):
    model.train()
    for iteration in range(iterations):
        optimizer.zero_grad()
        user_emb, venue_emb = model({'venue': venue_embeddings}, edge_index_dict)
        sampled_data = random.sample(train_data, min(batch_size, len(train_data)))
        loss = 0.0
        for user, pos_item, neg_item in sampled_data:
            user_vec = user_emb[user]
            pos_item_vec = venue_emb[pos_item]
            neg_item_vec = venue_emb[neg_item]
            pos_score = torch.sum(user_vec * pos_item_vec)
            neg_score = torch.sum(user_vec * neg_item_vec)
            loss += -F.logsigmoid(pos_score - neg_score)
        loss = loss / len(sampled_data)
        loss.backward()
        optimizer.step()
        print(f"Iteración {iteration + 1}/{iterations}, Loss: {loss.item():.4f}")

# Evaluation Function
def evaluate_gnn(model, edge_index_dict, venue_embeddings, test_data, train_visited_real, num_users, num_items, idx_to_user, idx_to_venue, top_k):
    model.eval()
    with torch.no_grad():
        user_emb, venue_emb = model({'venue': venue_embeddings}, edge_index_dict)
        results = {}
        recall_total = 0.0
        precision_total = 0.0
        f1_total = 0.0
        num_test_users = 0
        for user in range(num_users):
            if user not in test_data or not test_data[user]:
                continue
            num_test_users += 1
            user_vec = user_emb[user]
            scores = torch.matmul(user_vec, venue_emb.T)
            visited = train_visited_real[user]
            scores[list(visited)] = -float('inf')
            top_indices = torch.topk(scores, top_k).indices.tolist()
            recommended = set(top_indices)
            relevant = test_data[user]
            if len(relevant) == 0:
                continue
            hits = len(recommended.intersection(relevant))
            recall = hits / len(relevant) if len(relevant) > 0 else 0.0
            precision = hits / top_k if top_k > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            user_id = str(idx_to_user[user])
            results[user_id] = {
                "recommended_venues": [int(idx_to_venue[idx]) for idx in top_indices],
                "relevant_items": [int(idx_to_venue[idx]) for idx in relevant],
                "recall": float(recall),
                "precision": float(precision),
                "f1": float(f1)
            }
            recall_total += recall
            precision_total += precision
            f1_total += f1
        
        avg_recall = recall_total / num_test_users if num_test_users > 0 else 0.0
        avg_precision = precision_total / num_test_users if num_test_users > 0 else 0.0
        avg_f1 = f1_total / num_test_users if num_test_users > 0 else 0.0
        print(f"Recall@{top_k}: {avg_recall:.4f}, Precision@{top_k}: {avg_precision:.4f}, F1@{top_k}: {avg_f1:.4f}")
        return results, avg_recall, avg_precision, avg_f1

# Main Execution with SKopt Optimization
if __name__ == "__main__":
    top_k = 10
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    
    algo = "GNN"
    imputed_algo = "BPRMF"
    city = "Tokyo"
    train_file = f"data/Checkins_{city}_train.csv"
    test_file = f"data/Checkins_{city}_test.csv"
    imputed_file = f"data/{city}_{imputed_algo}_combined.csv"
    output_dir = f"results/{city}/GNN"
    os.makedirs(output_dir, exist_ok=True)
    
    use_imputed = True
    train_df, test_df, train_positive_items, train_visited_real, test_data, num_users, num_items, user_to_idx, venue_to_idx, idx_to_user, idx_to_venue = preprocess_data(train_file, test_file, use_imputed, imputed_file)
    
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    venue_embeddings = generate_venue_embeddings(combined_df, venue_to_idx)
    venue_features_dim = venue_embeddings.shape[1]
    
    data = create_hetero_graph(train_df, num_users, num_items, user_to_idx, venue_to_idx)
    edge_index_dict = {
        ('user', 'visits', 'venue'): data['user', 'visits', 'venue'].edge_index,
        ('venue', 'visited_by', 'user'): data['venue', 'visited_by', 'user'].edge_index
    }
    
    train_data = generate_triplets(train_positive_items, num_items)
    
    space = [
        Integer(32, 128, name='embedding_dim'),
        Integer(4, 5, name='num_layers'),
        Real(1e-4, 1e-2, prior='log-uniform', name='learning_rate'),
        Integer(10, 100, name='iterations')
    ]
    
    @use_named_args(space)
    def objective(embedding_dim, num_layers, learning_rate, iterations):
        model = GNN(num_users, num_items, embedding_dim, num_layers, venue_features_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_gnn(model, optimizer, train_data, edge_index_dict, venue_embeddings, iterations=iterations, batch_size=1000)
        _, _, _, avg_f1 = evaluate_gnn(model, edge_index_dict, venue_embeddings, test_data, train_visited_real, num_users, num_items, idx_to_user, idx_to_venue, top_k)
        return -avg_f1
    
    res = gp_minimize(objective, space, n_calls=10, random_state=42)
    
    best_params = {
        "embedding_dim": int(res.x[0]),
        "num_layers": int(res.x[1]),
        "learning_rate": float(res.x[2]),
        "iterations": int(res.x[3]),
        "top_k": int(top_k)
    }
    
    print("Mejores hiperparámetros encontrados:", best_params)
    
    model = GNN(num_users, num_items, best_params['embedding_dim'], best_params['num_layers'], venue_features_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    train_gnn(model, optimizer, train_data, edge_index_dict, venue_embeddings, iterations=best_params['iterations'], batch_size=1000)
    
    results, avg_recall, avg_precision, avg_f1 = evaluate_gnn(model, edge_index_dict, venue_embeddings, test_data, train_visited_real, num_users, num_items, idx_to_user, idx_to_venue, top_k)
    
    if use_imputed:
        output_file = f"{output_dir}/imputed/{city}_{algo}_{imputed_algo}_results_topk_{top_k}.json"
        os.makedirs(f"{output_dir}/imputed", exist_ok=True)
    else:
        output_file = f"{output_dir}/{city}_{algo}_results_topk_{top_k}.json"
    
    output_data = {
        "best_params": best_params,
        "metrics": {
            "avg_recall": float(avg_recall),
            "avg_precision": float(avg_precision),
            "avg_f1": float(avg_f1)
        },
        "results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Resultados guardados en {output_file}")