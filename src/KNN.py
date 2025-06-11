# -*- coding: utf-8 -*-
"""
User-based KNN Recommender («Improved») sin normalización ni skopt.
• Vecindad definida por k usuarios más similares (cosine).
• No hay ajustes bayesianos; k y top_k fijos.
• Mantiene el bloque main adaptado para usuario-KNN.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

# ────────────────────────────────────────────────────────────────
# 1. User-KNN Recommender (sin normalización)
# ────────────────────────────────────────────────────────────────
class KNNRecommenderImproved:
    def __init__(self, k: int, num_users: int, num_items: int):
        """
        k         : número de vecinos usuarios
        num_users : total de usuarios
        num_items : total de ítems
        """
        self.k = k
        self.num_users = num_users
        self.num_items = num_items
        self.user_venue_matrix = None  # matriz usuario×ítem

    def fit(self, train_df, user_to_idx, venue_to_idx):
        """Construye la matriz usuario×ítem a partir de train_df."""
        R = np.zeros((self.num_users, self.num_items), dtype=np.float32)
        for _, row in train_df.iterrows():
            u = user_to_idx[row["user_id"]]
            i = venue_to_idx[row["venue_id"]]
            R[u, i] = row.get("score", 1.0)
        self.user_venue_matrix = R

    def predict(self, user_idx: int, n: int, train_visited_real: dict):
        """
        Devuelve top-n ítems no visitados por user_idx,
        ponderados por similitud usuario–usuario.
        """
        if self.user_venue_matrix is None:
            raise RuntimeError("Llama a fit() antes de predict().")

        # 1) similitud coseno entre usuario y todos los usuarios
        sims = cosine_similarity(
            self.user_venue_matrix[user_idx].reshape(1, -1),
            self.user_venue_matrix
        )[0]
        sims[user_idx] = -1  # excluir self

        # 2) k vecinos más similares
        neigh = np.argpartition(sims, -self.k)[-self.k:]

        # 3) score de cada ítem = Σ_{v∈neigh} sim(u,v) * R[v,i]
        scores = np.zeros(self.num_items, dtype=np.float32)
        for v in neigh:
            scores += sims[v] * self.user_venue_matrix[v]

        # 4) penalizar ítems ya visitados
        visited = train_visited_real.get(user_idx, set())
        if visited:
            scores[list(visited)] = -np.inf

        # 5) devolver top-n índices
        top_n = np.argpartition(scores, -n)[-n:]
        return top_n[np.argsort(scores[top_n])[::-1]].tolist()


# ────────────────────────────────────────────────────────────────
# 2. Preprocesado y evaluación
# ────────────────────────────────────────────────────────────────
def preprocess_knn(train_file, test_file, use_imputed=False, imputed_file=None):
    """
    Devuelve:
      train_df, test_df,
      train_positive_items, train_visited_real, test_data,
      num_users, num_items,
      user_to_idx, venue_to_idx, idx_to_user, idx_to_venue
    """
    if use_imputed and imputed_file:
        train_df = pd.read_csv(imputed_file)
    else:
        train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    all_df = pd.concat([train_df, test_df], ignore_index=True)
    user_ids  = all_df["user_id"].unique()
    venue_ids = all_df["venue_id"].unique()

    user_to_idx  = {u: i for i, u in enumerate(user_ids)}
    venue_to_idx = {v: i for i, v in enumerate(venue_ids)}
    idx_to_user  = {i: u for u, i in user_to_idx.items()}
    idx_to_venue = {i: v for v, i in venue_to_idx.items()}

    train_df["user_idx"]  = train_df["user_id"].map(user_to_idx)
    train_df["venue_idx"] = train_df["venue_id"].map(venue_to_idx)
    test_df["user_idx"]   = test_df["user_id"].map(user_to_idx)
    test_df["venue_idx"]  = test_df["venue_id"].map(venue_to_idx)

    train_positive_items = defaultdict(set)
    train_visited_real   = defaultdict(set)
    for _, row in train_df.iterrows():
        u, v = row["user_idx"], row["venue_idx"]
        train_positive_items[u].add(v)
        if not use_imputed or row.get("source", "N") == "N":
            train_visited_real[u].add(v)

    test_data = defaultdict(set)
    for _, row in test_df.iterrows():
        test_data[row["user_idx"]].add(row["venue_idx"])

    return (
        train_df, test_df,
        train_positive_items, train_visited_real, test_data,
        len(user_ids), len(venue_ids),
        user_to_idx, venue_to_idx, idx_to_user, idx_to_venue
    )

def evaluate_knn(model, test_data, train_visited_real,
                 num_users, num_items, idx_to_user, idx_to_venue, top_k=10):
    results = {}
    recall_sum = precision_sum = f1_sum = users_eval = 0

    for u, relevant in test_data.items():
        if not relevant:
            continue
        users_eval += 1

        rec_idx = model.predict(u, top_k, train_visited_real)
        rec_set = set(rec_idx)

        hits = len(rec_set & relevant)
        recall = hits / len(relevant)
        precision = hits / top_k
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        results[idx_to_user[u]] = {
            "recommended_venues": [idx_to_venue[i] for i in rec_idx],
            "relevant_items":     [idx_to_venue[i] for i in relevant],
            "recall": recall, "precision": precision, "f1": f1,
        }

        recall_sum   += recall
        precision_sum+= precision
        f1_sum       += f1

    avg_recall    = recall_sum   / users_eval if users_eval else 0
    avg_precision = precision_sum/ users_eval if users_eval else 0
    avg_f1        = f1_sum       / users_eval if users_eval else 0

    print(f"\nPromedios (k = {model.k}, top-k = {top_k})")
    print(f"Recall@{top_k}:    {avg_recall:.4f}")
    print(f"Precision@{top_k}: {avg_precision:.4f}")
    print(f"F1@{top_k}:         {avg_f1:.4f}")
    return results, avg_recall, avg_precision, avg_f1


# ────────────────────────────────────────────────────────────────
# 3. MAIN — sin skopt, sin normalización
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    city = "Tokyo"
    train_file   = f"data/Checkins_{city}_train.csv"
    test_file    = f"data/Checkins_{city}_test.csv"
    imputed_algo = "BPRMF"
    imputed_file = f"data/{city}_{imputed_algo}_combined.csv"
    use_imputed  = True

    (train_df, test_df,
     train_positive_items, train_visited_real, test_data,
     num_users, num_items,
     user_to_idx, venue_to_idx,
     idx_to_user, idx_to_venue) = preprocess_knn(
        train_file, test_file, use_imputed, imputed_file
    )

    k     = 10   # nº de vecinos
    top_k =  5  # recomendaciones a devolver

    model = KNNRecommenderImproved(k, num_users, num_items)
    model.fit(train_df, user_to_idx, venue_to_idx)

    results, avg_recall, avg_precision, avg_f1 = evaluate_knn(
        model, test_data, train_visited_real,
        num_users, num_items, idx_to_user, idx_to_venue, top_k
    )

    result_dir = f"results/{city}/KNN"
    os.makedirs(result_dir, exist_ok=True)

    if use_imputed:
        out_file = f"{result_dir}/imputed/{city}_KNN_{imputed_algo}_results_topk_{top_k}.json"
        os.makedirs(f"{result_dir}/imputed", exist_ok=True)
    else:
        out_file = f"{result_dir}/{city}_KNN_results_topk_{top_k}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "params": {"k": k, "top_k": top_k},
            "metrics": {
                "avg_recall":    avg_recall,
                "avg_precision": avg_precision,
                "avg_f1":        avg_f1,
            },
            "results": {
                str(uid): {
                    "recommended_venues": [int(v) for v in res["recommended_venues"]],
                    "relevant_items":     [int(v) for v in res["relevant_items"]],
                    "recall":             float(res["recall"]),
                    "precision":          float(res["precision"]),
                    "f1":                 float(res["f1"]),
                } for uid, res in results.items()
            }
        }, f, indent=4)

    print(f"Resultados guardados en {out_file}")
