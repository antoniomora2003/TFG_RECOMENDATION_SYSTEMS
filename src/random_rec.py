import numpy as np
import pandas as pd
from collections import defaultdict
import json
import os

# Clase para el recomendador aleatorio
class RandomRecommender:
    def __init__(self, num_users, num_items):
        """Inicializa el modelo con el número de usuarios e ítems."""
        self.num_users = num_users
        self.num_items = num_items

    def fit(self, train_df, user_to_index, venue_to_index):
        """No se necesita entrenamiento para un recomendador aleatorio."""
        pass

    def predict(self, user_idx, n, train_visited_real):
        """Genera las top-n recomendaciones aleatorias, excluyendo solo visitas reales."""
        all_items = set(range(self.num_items))
        visited_items = train_visited_real.get(user_idx, set())
        candidate_items = list(all_items - visited_items)
        if len(candidate_items) < n:
            raise ValueError(f"No hay suficientes ítems no visitados para recomendar {n} ítems.")
        recommended_indices = np.random.choice(candidate_items, n, replace=False).tolist()
        return recommended_indices

# Preprocesamiento con datos pre-divididos y soporte para imputación
def preprocess_random(train_file, test_file, use_imputed=False, imputed_file=None):
    """Carga y prepara datos desde archivos CSV pre-divididos, con opción de imputación."""
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
    
    train_visited_real = defaultdict(set)
    
    if use_imputed:
        for _, row in train_df.iterrows():
            user_idx = row["user_idx"]
            venue_idx = row["venue_idx"]
            source = row["source"]
            if source == 'N':  # Solo considerar visitas reales
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

# Evaluación con F1-score
def evaluate_random(model, test_data, train_visited_real, num_users, num_items, idx_to_user, idx_to_venue, top_k):
    """Evalúa el modelo aleatorio y retorna métricas incluyendo F1-score, excluyendo solo visitas reales."""
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
        recommended_venues = [idx_to_venue[idx] for idx in recommended_indices]
        recommended = set(recommended_indices)
        relevant = pos_items
        hits = len(recommended.intersection(relevant))
        recall = hits / len(relevant) if len(relevant) > 0 else 0
        precision = hits / top_k if top_k > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        user_id = idx_to_user[user_idx]
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
    print(f"\nMétricas promedio para todos los usuarios de prueba:")
    print(f"Recall@{top_k}: {avg_recall:.4f}, Precision@{top_k}: {avg_precision:.4f}, F1@{top_k}: {avg_f1:.4f}")
    return results, avg_recall, avg_precision, avg_f1

# Ejecución principal
if __name__ == "__main__":
    # Cargar y preprocesar datos desde archivos pre-divididos
    city = "London"  # Cambiar a la ciudad deseada
    train_file = f"data/Checkins_{city}_train.csv"
    test_file = f"data/Checkins_{city}_test.csv"
    imputed_algo = "BPRMF"  # Algoritmo de imputación
    imputed_file = f"data/{city}_{imputed_algo}_combined.csv"
    use_imputed = False  # Cambiar a True para usar datos imputados
    train_df, test_df, train_visited_real, test_data, num_users, num_items, user_to_idx, venue_to_idx, idx_to_user, idx_to_venue = preprocess_random(train_file, test_file, use_imputed, imputed_file)
    
    # Valores fijos para top_k
    top_k = 5  # Número de recomendaciones
    
    # Instanciar el modelo
    model = RandomRecommender(num_users, num_items)
    
    # No se necesita entrenar para un recomendador aleatorio
    
    # Evaluar el modelo
    results, avg_recall, avg_precision, avg_f1 = evaluate_random(model, test_data, train_visited_real, num_users, num_items, idx_to_user, idx_to_venue, top_k)
    
    # Crear directorio de resultados si no existe
    result_dir = f"results/{city}/Random"
    os.makedirs(result_dir, exist_ok=True)
    
    # Guardar resultados en un archivo JSON con consistencia en el nombramiento
    if use_imputed:
        output_file = f"{result_dir}/imputed/{city}_Random_{imputed_algo}_results_topk_{top_k}.json"
        os.makedirs(f"{result_dir}/imputed", exist_ok=True)
    else:
        output_file = f"{result_dir}/{city}_Random_results_topk_{top_k}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "params": {"top_k": int(top_k)},
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