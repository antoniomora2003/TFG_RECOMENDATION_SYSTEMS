import pandas as pd
import numpy as np
import random
import os
import json
from collections import defaultdict
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

class BPRMF:
    def __init__(self, num_users, num_items, factors, reg_user, reg_item, reg_bias, learning_rate, iterations):
        """Inicializa el modelo BPRMF con los parámetros dados."""
        self.num_users = num_users
        self.num_items = num_items
        self.factors = factors
        self.reg_user = reg_user
        self.reg_item = reg_item
        self.reg_bias = reg_bias
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.user_factors = np.random.normal(0, 0.1, (num_users, factors))
        self.item_factors = np.random.normal(0, 0.1, (num_items, factors))
        self.item_bias = np.zeros(num_items)

    def fit(self, train_data):
        """Entrena el modelo BPRMF con muestreo estocástico."""
        for iteration in range(self.iterations):
            random.shuffle(train_data)
            total_loss = 0.0
            num_samples = min(len(train_data), 1000)
            sampled_data = train_data[:num_samples]
            for user, item_pos, item_neg in sampled_data:
                total_loss += self._update(user, item_pos, item_neg)
            avg_loss = total_loss / num_samples if num_samples > 0 else 0
            print(f"Iteración {iteration + 1}/{self.iterations}, Loss: {avg_loss:.4f}")

    def _update(self, user, item_pos, item_neg):
        """Actualiza factores y sesgos para una tripleta usando vectorización."""
        pred_pos = np.dot(self.user_factors[user], self.item_factors[item_pos]) + self.item_bias[item_pos]
        pred_neg = np.dot(self.user_factors[user], self.item_factors[item_neg]) + self.item_bias[item_neg]
        diff = pred_pos - pred_neg
        sigmoid = 1 / (1 + np.exp(-np.clip(diff, -20, 20)))
        loss = -np.log(max(sigmoid, 1e-10))
        user_vec = self.user_factors[user]
        item_pos_vec = self.item_factors[item_pos]
        item_neg_vec = self.item_factors[item_neg]
        bias_pos = self.item_bias[item_pos]
        bias_neg = self.item_bias[item_neg]
        reg_loss = (
            self.reg_user * np.sum(user_vec ** 2) +
            self.reg_item * np.sum(item_pos_vec ** 2) +
            self.reg_item * np.sum(item_neg_vec ** 2) +
            self.reg_bias * (bias_pos ** 2 + bias_neg ** 2)
        )
        grad = 1 - sigmoid
        self.user_factors[user] += self.learning_rate * (
            grad * (item_pos_vec - item_neg_vec) - self.reg_user * user_vec
        )
        self.item_factors[item_pos] += self.learning_rate * (
            grad * user_vec - self.reg_item * item_pos_vec
        )
        self.item_factors[item_neg] += self.learning_rate * (
            -grad * user_vec - self.reg_item * item_neg_vec
        )
        self.item_bias[item_pos] += self.learning_rate * (grad - self.reg_bias * bias_pos)
        self.item_bias[item_neg] += self.learning_rate * (-grad - self.reg_bias * bias_neg)
        return loss + reg_loss

    def predict(self, user, item):
        """Predice el puntaje para un par usuario-ítem."""
        return np.dot(self.user_factors[user], self.item_factors[item]) + self.item_bias[item]

def preprocess_data(train_file, test_file, use_imputed=False, imputed_file=None):
    """Carga y preprocesa los datos usando diccionarios."""
    if use_imputed and imputed_file is not None:
        train_df = pd.read_csv(imputed_file)
    else:
        train_df = pd.read_csv(train_file)
    
    test_df = pd.read_csv(test_file)
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    user_ids = combined_df["user_id"].unique()
    venue_ids = combined_df["venue_id"].unique()
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    venue_to_idx = {vid: idx for idx, vid in enumerate(venue_ids)}
    idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}
    idx_to_venue = {idx: vid for vid, idx in venue_to_idx.items()}
    
    train_df["user_idx"] = train_df["user_id"].map(user_to_idx)
    train_df["venue_idx"] = train_df["venue_id"].map(venue_to_idx)
    test_df["user_idx"] = test_df["user_id"].map(user_to_idx)
    test_df["venue_idx"] = test_df["venue_id"].map(venue_to_idx)
    
    train_positive_items = defaultdict(set)
    train_visited_real = defaultdict(set)
    
    if use_imputed:
        for _, row in train_df.iterrows():
            user_idx = row["user_idx"]
            venue_idx = row["venue_idx"]
            source = row["source"]
            train_positive_items[user_idx].add(venue_idx)
            if source == 'N':
                train_visited_real[user_idx].add(venue_idx)
    else:
        for _, row in train_df.iterrows():
            user_idx = row["user_idx"]
            venue_idx = row["venue_idx"]
            train_positive_items[user_idx].add(venue_idx)
            train_visited_real[user_idx].add(venue_idx)
    
    test_data = defaultdict(set)
    for _, row in test_df.iterrows():
        test_data[row["user_idx"]].add(row["venue_idx"])
    
    return train_df, test_df, train_positive_items, train_visited_real, test_data, len(user_ids), len(venue_ids), user_to_idx, venue_to_idx, idx_to_user, idx_to_venue

def generate_triplets(train_positive_items, num_items):
    """Genera tripletas de entrenamiento desde el diccionario de ítems positivos."""
    train_data = []
    for user, pos_items in train_positive_items.items():
        for item_pos in pos_items:
            neg_candidates = [i for i in range(num_items) if i not in pos_items]
            item_neg = random.choice(neg_candidates) if neg_candidates else random.randint(0, num_items - 1)
            train_data.append((user, item_pos, item_neg))
    return train_data

def evaluate_bprmf(model, test_data, train_visited_real, num_users, num_items, idx_to_user, idx_to_venue, top_k):
    """Evalúa el modelo calculando recall, precision y F1."""
    results = {}
    recall_total = 0
    precision_total = 0
    f1_total = 0
    num_test_users = 0
    for user in range(num_users):
        if user not in test_data or not test_data[user]:
            continue
        num_test_users += 1
        scores = np.array([model.predict(user, item) for item in range(num_items)])
        visited = train_visited_real[user]
        scores[list(visited)] = -np.inf
        top_indices = np.argsort(scores)[-top_k:][::-1]
        recommended = set(top_indices)
        recommended_venues = [str(idx_to_venue[idx]) for idx in top_indices]
        relevant = test_data[user]
        relevant_venues = [str(idx_to_venue[idx]) for idx in relevant]
        hits = len(recommended.intersection(relevant))
        recall = hits / len(relevant) if len(relevant) > 0 else 0
        precision = hits / top_k if top_k > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        user_id = str(idx_to_user[user])
        results[user_id] = {
            "recommended_venues": recommended_venues,
            "relevant_items": relevant_venues,
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1)
        }
        recall_total += recall
        precision_total += precision
        f1_total += f1
    
    avg_recall = recall_total / num_test_users if num_test_users > 0 else 0
    avg_precision = precision_total / num_test_users if num_test_users > 0 else 0
    avg_f1 = f1_total / num_test_users if num_test_users > 0 else 0
    print(f"Recall@{top_k}: {avg_recall:.4f}, Precision@{top_k}: {avg_precision:.4f}, F1@{top_k}: {avg_f1:.4f}")
    return results, avg_recall, avg_precision, avg_f1

space = [
    Integer(50, 250, name='factors'),
    Real(0.001, 0.05, name='reg_user'),
    Real(0.001, 0.05, name='reg_item'),
    Real(0.001, 0.05, name='reg_bias'),
    Real(0.0001, 0.01, name='learning_rate'),
    Integer(20, 100, name='iterations')
]

@use_named_args(space)
def objective(**params):
    """Función objetivo para SKopt."""
    global top_k
    bprmf = BPRMF(
        num_users, num_items, params['factors'], params['reg_user'],
        params['reg_item'], params['reg_bias'], params['learning_rate'], params['iterations']
    )
    bprmf.fit(train_data)
    _, _, _, avg_f1 = evaluate_bprmf(bprmf, test_data, train_visited_real, num_users, num_items, idx_to_user, idx_to_venue, top_k)
    return -avg_f1

if __name__ == "__main__":
    top_k = 10 # Número de recomendaciones a generar, cámbialo según necesites
    np.random.seed(42)
    random.seed(42)
    algo = "BPRMF"  # Algoritmo a usar
    imputed_algo = "BPRMF"  # Algoritmo de imputación, puede ser BPRMF, BPR, etc.
    city = "Tokyo"  # Cambia esto según la ciudad
    train_file = f"data/Checkins_{city}_train.csv"
    test_file = f"data/Checkins_{city}_test.csv"
    imputed_file = f"data/{city}_{imputed_algo}_combined.csv"  # Archivo imputado
    output_dir = f"results/{city}/BPRMF"
    os.makedirs(output_dir, exist_ok=True)
    
    # Cambia use_imputed a True para usar datos imputados
    use_imputed = True
    train_df, test_df, train_positive_items, train_visited_real, test_data, num_users, num_items, user_to_idx, venue_to_idx, idx_to_user, idx_to_venue = preprocess_data(train_file, test_file, use_imputed, imputed_file)
    train_data = generate_triplets(train_positive_items, num_items)
    
    # Optimización con SKopt usando el top_k definido
    res = gp_minimize(objective, space, n_calls=20, random_state=42)
    
    # Mejores hiperparámetros convertidos a tipos serializables
    best_params = {
        "factors": int(res.x[0]),
        "reg_user": float(res.x[1]),
        "reg_item": float(res.x[2]),
        "reg_bias": float(res.x[3]),
        "learning_rate": float(res.x[4]),
        "iterations": int(res.x[5]),
        "top_k": int(top_k)
    }
    print("Mejores hiperparámetros encontrados:")
    print(json.dumps(best_params, indent=4))
    
    # Entrenar modelo final con los mejores hiperparámetros
    bprmf = BPRMF(num_users, num_items, res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], res.x[5])
    bprmf.fit(train_data)
    results, avg_recall, avg_precision, avg_f1 = evaluate_bprmf(bprmf, test_data, train_visited_real, num_users, num_items, idx_to_user, idx_to_venue, top_k)
    
    # Guardar todo en un solo archivo JSON
    if use_imputed:
        output_file = f"{output_dir}/imputed/{city}_{algo}_{imputed_algo}_results_topk_{top_k}_{use_imputed}.json"
    else: 
        output_file = f"{output_dir}/{city}_{algo}_results_topk_{top_k}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "best_params": best_params,
            "metrics": {
                "avg_recall": float(avg_recall),
                "avg_precision": float(avg_precision),
                "avg_f1": float(avg_f1)
            },
            "results": results
        }, f, indent=4)
    print(f"Resultados guardados en {output_file}")