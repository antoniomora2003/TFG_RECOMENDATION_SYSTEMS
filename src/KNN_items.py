import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import normalize          # solo para L2 por‐usuario
import json, os


# ────────────────────────────────────────────────────────────────
# 1. Item-KNN con similitud Jaccard
# ────────────────────────────────────────────────────────────────
class ItemKNNRecommenderImproved:
    """
    K vecinos más parecidos según la similitud Jaccard entre columnas (ítems).
    """
    def __init__(self, k: int, num_users: int, num_items: int, normalize: bool = True):
        self.k          = k
        self.num_users  = num_users
        self.num_items  = num_items
        self.normalize  = normalize

        self.user_venue_matrix = None   # R  (usuarios × ítems)
        self.similarity_matrix = None   # S  (ítems × ítems)

    # ...........................................................
    def _compute_jaccard(self, bin_mat: np.ndarray) -> np.ndarray:
        """
        Calcula la matriz de similitud Jaccard entre columnas de una matriz binaria.
        """
        col_sum      = bin_mat.sum(axis=0, dtype=np.int32)           # |U(i)|
        intersection = bin_mat.T @ bin_mat                          # |U(i) ∩ U(j)|
        union        = col_sum[:, None] + col_sum[None, :] - intersection
        with np.errstate(divide="ignore", invalid="ignore"):
            jaccard = intersection / union
            jaccard[union == 0] = 0.0
        return jaccard.astype(np.float32)

    # ...........................................................
    def fit(self, train_df, user_to_idx, venue_to_idx, rating_col: str | None = None):
        """Crea R y S (Jaccard)."""
        self.user_venue_matrix = np.zeros((self.num_users, self.num_items), dtype=np.float32)

        for _, row in train_df.iterrows():
            u = user_to_idx[row["user_id"]]
            v = venue_to_idx[row["venue_id"]]
            val = row[rating_col] if rating_col and rating_col in row else 1.0
            self.user_venue_matrix[u, v] = val

        # Normalización L2 por usuario (opcional, no afecta a Jaccard)
        if self.normalize:
            self.user_venue_matrix = normalize(self.user_venue_matrix, axis=1, norm="l2", copy=False)

        # Similitud Jaccard (se calcula sobre presencia binaria)
        bin_mat = (self.user_venue_matrix > 0).astype(np.uint8)
        self.similarity_matrix = self._compute_jaccard(bin_mat)

    # ...........................................................
    def predict(self, user_idx: int, n: int, already_seen: set[int]):
        """Top-n ítems no visitados con mayor score para `user_idx`."""
        if self.similarity_matrix is None:
            raise RuntimeError("Debes llamar a fit() antes de predecir")

        user_vec = self.user_venue_matrix[user_idx]
        scores   = np.zeros(self.num_items, dtype=np.float32)

        consumed = np.nonzero(user_vec)[0]
        for i in consumed:
            neigh_idx = np.argpartition(self.similarity_matrix[i], -self.k)[-self.k:]
            scores[neigh_idx] += self.similarity_matrix[i, neigh_idx] * user_vec[i]

        scores[list(already_seen)] = -np.inf
        top_n = np.argpartition(scores, -n)[-n:]
        return top_n[np.argsort(scores[top_n])][::-1].tolist()


# ────────────────────────────────────────────────────────────────
# 2. Preproceso y evaluación (sin cambios)
# ────────────────────────────────────────────────────────────────
def preprocess_knn(train_file, test_file, use_imputed=False, imputed_file=None):
    if use_imputed and imputed_file:
        train_df = pd.read_csv(imputed_file)
    else:
        train_df = pd.read_csv(train_file)

    test_df = pd.read_csv(test_file)
    all_df  = pd.concat([train_df, test_df], ignore_index=True)

    user_ids = all_df["user_id"].unique()
    venue_ids = all_df["venue_id"].unique()

    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    venue_to_idx = {v: i for i, v in enumerate(venue_ids)}
    idx_to_user   = {i: u for u, i in user_to_idx.items()}
    idx_to_venue  = {i: v for v, i in venue_to_idx.items()}

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

    return (train_df, test_df, train_positive_items, train_visited_real, test_data,
            len(user_ids), len(venue_ids),
            user_to_idx, venue_to_idx, idx_to_user, idx_to_venue)


def evaluate_knn(model, test_data, train_visited_real,
                 num_users, num_items, idx_to_user, idx_to_venue, top_k=10):
    results = {}
    rec_sum = prec_sum = f1_sum = users_eval = 0

    for u in range(num_users):
        relevant = test_data.get(u, set())
        if not relevant:
            continue
        users_eval += 1

        rec_idx = model.predict(u, top_k, train_visited_real.get(u, set()))
        rec_set = set(rec_idx)

        hits = len(rec_set & relevant)
        recall = hits / len(relevant)
        precision = hits / top_k
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

        results[idx_to_user[u]] = {
            "recommended_venues": [idx_to_venue[i] for i in rec_idx],
            "relevant_items":     [idx_to_venue[i] for i in relevant],
            "recall": recall, "precision": precision, "f1": f1,
        }
        rec_sum  += recall
        prec_sum += precision
        f1_sum   += f1

    avg_rec  = rec_sum  / users_eval if users_eval else 0
    avg_prec = prec_sum / users_eval if users_eval else 0
    avg_f1   = f1_sum   / users_eval if users_eval else 0

    print(f"\nPromedios (k = {model.k}, top-k = {top_k})")
    print(f"Recall@{top_k}:    {avg_rec:.4f}")
    print(f"Precision@{top_k}: {avg_prec:.4f}")
    print(f"F1@{top_k}:         {avg_f1:.4f}")
    return results, avg_rec, avg_prec, avg_f1


# ────────────────────────────────────────────────────────────────
# 3. MAIN (sin cambios, tal como lo pediste)
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    city = "London"
    train_file = f"data/Checkins_{city}_train.csv"
    test_file  = f"data/Checkins_{city}_test.csv"
    imputed_algo  = "BPRMF"
    imputed_file  = f"data/{city}_{imputed_algo}_combined.csv"
    use_imputed   = False

    (train_df, test_df, train_positive_items, train_visited_real, test_data,
     num_users, num_items, user_to_idx, venue_to_idx,
     idx_to_user, idx_to_venue) = preprocess_knn(train_file, test_file,
                                                 use_imputed, imputed_file)

    k = 10
    top_k = 5

    best_model = ItemKNNRecommenderImproved(k, num_users, num_items, normalize=True)
    best_model.fit(train_df, user_to_idx, venue_to_idx)

    results, avg_rec, avg_prec, avg_f1 = evaluate_knn(
        best_model, test_data, train_visited_real,
        num_users, num_items, idx_to_user, idx_to_venue, top_k
    )

    result_dir = f"results/{city}/ItemKNN"
    os.makedirs(result_dir, exist_ok=True)

    if use_imputed:
        output_file = (
            f"{result_dir}/imputed/{city}_ItemKNN_{imputed_algo}_results_topk_{top_k}.json"
        )
        os.makedirs(f"{result_dir}/imputed", exist_ok=True)
    else:
        output_file = f"{result_dir}/{city}_ItemKNN_results_topk_{top_k}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "params": {"k": int(k), "top_k": int(top_k)},
                "metrics": {
                    "avg_recall":    float(avg_rec),
                    "avg_precision": float(avg_prec),
                    "avg_f1":        float(avg_f1),
                },
                "results": {
                    str(uid): {
                        "recommended_venues": [int(v) for v in res["recommended_venues"]],
                        "relevant_items":     [int(v) for v in res["relevant_items"]],
                        "recall":    float(res["recall"]),
                        "precision": float(res["precision"]),
                        "f1":        float(res["f1"]),
                    }
                    for uid, res in results.items()
                },
            },
            f,
            indent=4,
        )

    print(f"Resultados guardados en {output_file}")

