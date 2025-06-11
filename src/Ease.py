import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from multiprocessing import Pool, cpu_count
import os
import json

class EASE:
    def __init__(self, lambda_=0.5, implicit=True):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()
        self.lambda_ = lambda_
        self.implicit = implicit
        self.B = None
        self.pred = None
        self.X = None

    def _get_users_and_items(self, df):
        users = self.user_enc.fit_transform(df['user_id'])
        items = self.item_enc.fit_transform(df['venue_id'])
        return users, items

    def fit(self, df):
        users, items = self._get_users_and_items(df)
        values = np.ones(df.shape[0]) if self.implicit else df['score'].to_numpy() / df['score'].max()
        X = csr_matrix((values, (users, items)))
        self.X = X
        G = X.T.dot(X).toarray()
        diag_indices = np.diag_indices(G.shape[0])
        G[diag_indices] += self.lambda_
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diag_indices] = 0
        self.B = B
        self.pred = X.dot(B)

    def predict(self, train, users, items, k):
        # Filtrar items para incluir solo aquellos vistos durante el entrenamiento
        seen_items = set(self.item_enc.classes_)
        filtered_items = [item for item in items if item in seen_items]
        if not filtered_items:  # Si no hay ítems válidos, devolver DataFrame vacío
            return pd.DataFrame(columns=['user_id', 'venue_id', 'score'])
        items_encoded = self.item_enc.transform(filtered_items)
        
        dd = train.loc[train.user_id.isin(users)]
        dd['ci'] = self.item_enc.transform(dd.venue_id)
        dd['cu'] = self.user_enc.transform(dd.user_id)
        g = dd.groupby('cu')
        with Pool(cpu_count()) as p:
            user_preds = p.starmap(
                self.predict_for_user,
                [(user, group, self.pred[user, :], items_encoded, k) for user, group in g],
            )
        if not user_preds:  # Si no hay predicciones, devolver DataFrame vacío
            return pd.DataFrame(columns=['user_id', 'venue_id', 'score'])
        df = pd.concat(user_preds)
        df['venue_id'] = self.item_enc.inverse_transform(df['item_id'])
        df['user_id'] = self.user_enc.inverse_transform(df['user_id'])
        return df

    @staticmethod
    def predict_for_user(user, group, pred, items, k):
        watched = set(group['ci'])
        candidates = [item for item in items if item not in watched]
        if not candidates:
            return pd.DataFrame()  # No hay candidatos para recomendar
        pred_scores = np.take(pred, candidates)
        res = np.argpartition(pred_scores, -k)[-k:]
        r = pd.DataFrame(
            {
                "user_id": [user] * len(res),
                "item_id": np.take(candidates, res),
                "score": np.take(pred_scores, res),
            }
        ).sort_values('score', ascending=False)
        return r

def preprocess_ease(train_file, test_file, use_imputed=False, imputed_file=None):
    if use_imputed and imputed_file is not None:
        train_df = pd.read_csv(imputed_file)
    else:
        train_df = pd.read_csv(train_file)
    
    test_df = pd.read_csv(test_file)
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    user_ids = combined_df["user_id"].unique()
    venue_ids = combined_df["venue_id"].unique()
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    venue_to_idx = {vid: idx for vid, idx in enumerate(venue_ids)}
    idx_to_venue = {idx: vid for vid, idx in venue_to_idx.items()}
    idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}
    
    train_df["user_idx"] = train_df["user_id"].map(user_to_idx)
    train_df["venue_idx"] = train_df["venue_id"].map(venue_to_idx)
    test_df["user_idx"] = test_df["user_id"].map(user_to_idx)
    test_df["venue_idx"] = test_df["venue_id"].map(venue_to_idx)
    
    train_visited_real = defaultdict(set)
    if use_imputed:
        for _, row in train_df.iterrows():
            user_idx = row["user_idx"]
            venue_idx = row["venue_idx"]
            source = row.get("source", "N")
            if source == 'N':
                train_visited_real[user_idx].add(venue_idx)
    else:
        for _, row in train_df.iterrows():
            user_idx = row["user_idx"]
            venue_idx = row["venue_idx"]
            train_visited_real[user_idx].add(venue_idx)
    
    test_data = defaultdict(set)
    for _, row in test_df.iterrows():
        test_data[row["user_idx"]].add(row["venue_idx"])
    
    return train_df, test_df, train_visited_real, test_data, len(user_ids), len(venue_ids), user_to_idx, venue_to_idx, idx_to_user, idx_to_venue

def evaluate_ease(model, train_df, test_data, train_visited_real, num_users, num_items, idx_to_user, idx_to_venue, top_k=10):
    results = {}
    recall_total = precision_total = f1_total = num_test_users = 0
    users = [idx_to_user[user_idx] for user_idx in range(num_users) if user_idx in test_data]
    items = [idx_to_venue[idx] for idx in range(num_items)]
    pred_df = model.predict(train_df, users, items, top_k)
    
    if pred_df.empty:
        print("No se pudieron generar predicciones para los usuarios dados.")
        return {}, 0, 0, 0
    
    pred_grouped = pred_df.groupby('user_id')
    for user_idx in range(num_users):
        if user_idx not in test_data:
            continue
        pos_items = test_data[user_idx]
        if not pos_items:
            continue
        num_test_users += 1
        user_id = idx_to_user[user_idx]
        recommended_df = pred_grouped.get_group(user_id) if user_id in pred_grouped.groups else pd.DataFrame()
        recommended_indices = set(model.item_enc.transform(recommended_df['venue_id'])) if not recommended_df.empty else set()
        recommended_venues = recommended_df['venue_id'].tolist() if not recommended_df.empty else []
        relevant = pos_items
        hits = len(recommended_indices.intersection(relevant))
        recall = hits / len(relevant) if len(relevant) > 0 else 0
        precision = hits / top_k if top_k > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        results[user_id] = {
            "recommended_venues": recommended_venues,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "relevant_items": [idx_to_venue[idx] for idx in relevant]
        }
        recall_total += recall
        precision_total += precision
        f1_total += f1
    
    avg_recall = recall_total / num_test_users if num_test_users > 0 else 0
    avg_precision = precision_total / num_test_users if num_test_users > 0 else 0
    avg_f1 = f1_total / num_test_users if num_test_users > 0 else 0
    print(f"\nMétricas promedio para todos los usuarios de prueba con top_k={top_k}:")
    print(f"Recall@{top_k}: {avg_recall:.4f}, Precision@{top_k}: {avg_precision:.4f}, F1@{top_k}: {avg_f1:.4f}")
    return results, avg_recall, avg_precision, avg_f1

if __name__ == "__main__":
    city = "London"
    train_file = f"data/Checkins_{city}_train.csv"
    test_file = f"data/Checkins_{city}_test.csv"
    imputed_algo = "BPRMF"
    imputed_file = f"data/{city}_{imputed_algo}_combined.csv"
    use_imputed = False
    
    train_df, test_df, train_visited_real, test_data, num_users, num_items, user_to_idx, venue_to_idx, idx_to_user, idx_to_venue = preprocess_ease(train_file, test_file, use_imputed, imputed_file)
    
    lambda_ = 0.5
    top_k = 5
    
    model = EASE(lambda_=lambda_, implicit=True)
    model.fit(train_df)
    
    results, avg_recall, avg_precision, avg_f1 = evaluate_ease(model, train_df, test_data, train_visited_real, num_users, num_items, idx_to_user, idx_to_venue, top_k)
    
    result_dir = f"results/{city}/EASE"
    os.makedirs(result_dir, exist_ok=True)
    
    if use_imputed:
        output_file = f"{result_dir}/imputed/{city}_EASE_{imputed_algo}_results_topk_{top_k}.json"
        os.makedirs(f"{result_dir}/imputed", exist_ok=True)
    else:
        output_file = f"{result_dir}/{city}_EASE_results_topk_{top_k}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "params": {"lambda": float(lambda_), "top_k": int(top_k)},
            "metrics": {
                "avg_recall": float(avg_recall),
                "avg_precision": float(avg_precision),
                "avg_f1": float(avg_f1)
            },
            "results": {
                str(user_id): {
                    "recommended_venues": [int(venue) for venue in result["recommended_venues"]],
                    "recall": float(result["recall"]),
                    "precision": float(result["precision"]),
                    "f1": float(result["f1"]),
                    "relevant_items": [int(venue) for venue in result["relevant_items"]]
                } for user_id, result in results.items()
            }
        }, f, indent=4)
    print(f"Resultados guardados en {output_file}")