import numpy as np
import pandas as pd
from collections import defaultdict
import json
import os

# Clase para el modelo de recomendación por popularidad
class PopularityRecommender:
    def __init__(self):
        self.popular_items = None
        self.user_visited = None

    def fit(self, train_df):
        """Entrena el modelo calculando la popularidad de los venues."""
        if "score" in train_df.columns:
            popularity = train_df.groupby("venue_idx")["score"].sum()
        else:
            popularity = train_df.groupby("venue_idx").size()
        popularity = popularity.sort_values(ascending=False)
        self.popular_items = popularity.index.tolist()
        self.user_visited = defaultdict(set)
        for _, row in train_df.iterrows():
            user_idx = row["user_idx"]
            venue_idx = row["venue_idx"]
            self.user_visited[user_idx].add(venue_idx)

    def predict(self, user_idx, top_k=5, train_visited_real=None):
        """Genera top_k recomendaciones excluyendo venues visitados reales."""
        if self.popular_items is None:
            raise ValueError("El modelo no ha sido entrenado aún.")
        visited = train_visited_real.get(user_idx, set())  # Usar solo venues reales visitados
        recommended = []
        for venue_idx in self.popular_items:
            if venue_idx not in visited:
                recommended.append(venue_idx)
                if len(recommended) >= top_k:
                    break
        return recommended

# Preprocesamiento de datos desde archivos CSV pre-divididos
def preprocess_popularity(train_file, test_file, use_imputed=False, imputed_file=None):
    """Carga y preprocesa los datos, asignando índices a usuarios y venues."""
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
    idx_to_venue = {idx: vid for vid, idx in venue_to_idx.items()}
    idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}
    
    train_df["user_idx"] = train_df["user_id"].map(user_to_idx)
    train_df["venue_idx"] = train_df["venue_id"].map(venue_to_idx)
    test_df["user_idx"] = test_df["user_id"].map(user_to_idx)
    test_df["venue_idx"] = test_df["venue_id"].map(venue_to_idx)
    
    test_data = defaultdict(set)
    for _, row in test_df.iterrows():
        user = row["user_idx"]
        item = row["venue_idx"]
        test_data[user].add(item)
    
    train_visited_real = defaultdict(set)
    if use_imputed:
        for _, row in train_df.iterrows():
            user_idx = row["user_idx"]
            venue_idx = row["venue_idx"]
            source = row.get("source", 'N')  # 'N' para real, 'S' para imputado
            if source == 'N':
                train_visited_real[user_idx].add(venue_idx)
    else:
        for _, row in train_df.iterrows():
            user_idx = row["user_idx"]
            venue_idx = row["venue_idx"]
            train_visited_real[user_idx].add(venue_idx)
    
    return train_df, test_df, test_data, len(user_ids), len(venue_ids), user_to_idx, venue_to_idx, idx_to_user, idx_to_venue, train_visited_real

# Evaluación del modelo con F1-score añadido
def evaluate_popularity(model, test_data, train_visited_real, num_users, idx_to_user, idx_to_venue, top_k=5):
    """Evalúa el modelo calculando recall, precision y F1-score por usuario."""
    results = {}
    recall_total = precision_total = f1_total = num_test_users = 0
    
    for user_idx in range(num_users):
        if user_idx not in test_data:
            continue
        pos_items = test_data[user_idx]
        if not pos_items:
            continue
        num_test_users += 1
        recommended_indices = model.predict(user_idx, top_k, train_visited_real)
        recommended_venues = [int(idx_to_venue[idx]) for idx in recommended_indices]
        recommended = set(recommended_indices)
        relevant = pos_items
        hits = len(recommended.intersection(relevant))
        recall = hits / len(relevant) if len(relevant) > 0 else 0
        precision = hits / top_k if top_k > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        user_id = idx_to_user[user_idx]
        results[user_id] = {
            "recommended_venues": recommended_venues,
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1),
            "relevant_items": [int(idx_to_venue[idx]) for idx in relevant]
        }
        recall_total += recall
        precision_total += precision
        f1_total += f1
    
    avg_recall = recall_total / num_test_users if num_test_users > 0 else 0
    avg_precision = precision_total / num_test_users if num_test_users > 0 else 0
    avg_f1 = f1_total / num_test_users if num_test_users > 0 else 0
    
    return results, avg_recall, avg_precision, avg_f1

# Ejecución principal
if __name__ == "__main__":
    city = "Tokyo"  # Cambiar a la ciudad deseada
    algo_imputed = "BPRMF"
    algo = "popularity"
    train_file = f"data/Checkins_{city}_train.csv"
    test_file = f"data/Checkins_{city}_test.csv"
    imputed_file = f"data/{city}_{algo_imputed}_combined.csv"  # Ajustar según el archivo imputado real
    use_imputed = False  # Cambiar a True para usar datos imputados
    
    # Preprocesar datos
    train_df, test_df, test_data, num_users, num_items, user_to_idx, venue_to_idx, idx_to_user, idx_to_venue, train_visited_real = preprocess_popularity(train_file, test_file, use_imputed, imputed_file)
    
    # Inicializar y entrenar el modelo
    pop_recommender = PopularityRecommender()
    pop_recommender.fit(train_df)
    
    # Establecer un valor fijo para top_k
    top_k = 5
    
    # Evaluar el modelo con top_k fijo
    results, avg_recall, avg_precision, avg_f1 = evaluate_popularity(pop_recommender, test_data, train_visited_real, num_users, idx_to_user, idx_to_venue, top_k)
    
    # Guardar resultados y métricas en JSON con nombramiento consistente
    if use_imputed:
        output_dir = f"results/{city}/{algo}/imputed"
        output_file = f"{output_dir}/{city}_{algo}_{algo_imputed}_results_topk_{top_k}.json"
    else:
        output_dir = f"results/{city}/{algo}"
        output_file = f"{output_dir}/{city}_{algo}_results_topk_{top_k}.json"
    
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({
            "top_k": int(top_k),
            "metrics": {
                "avg_recall": float(avg_recall),
                "avg_precision": float(avg_precision),
                "avg_f1": float(avg_f1)
            },
            "results": {str(k): v for k, v in results.items()}
        }, f, indent=4)
    print(f"Resultados guardados en {output_file}")