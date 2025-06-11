import json
import numpy as np
from collections import Counter
import pandas as pd

def load_json(file_path):
    """Carga el archivo JSON desde el path especificado."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_train_data(train_file):
    """Carga el conjunto de entrenamiento y calcula pop_max, pop_i y total_items."""
    train_df = pd.read_csv(train_file)
    # Asegurarse de que user_id y venue_id sean strings
    train_df['user_id'] = train_df['user_id'].astype(str)
    train_df['venue_id'] = train_df['venue_id'].astype(str)
    # Calcular pop_max: número de usuarios únicos
    pop_max = train_df['user_id'].nunique()
    # Calcular pop_i: número de usuarios que han visitado cada venue
    pop_i = train_df.groupby('venue_id')['user_id'].nunique().to_dict()
    # Calcular total_items: número de venues únicos
    total_items = len(pop_i)
    print(f"Total items (venues): {total_items}")
    return pop_max, pop_i, total_items

def calculate_metrics(data, pop_max, pop_i, total_items):
    """Calcula las métricas EPC, Novelty y Diversidad a partir de los datos del JSON, pop_i y total_items."""
    num_users = len(data['results'])
    
    # Calcular EPC
    epc_total = 0
    for user, user_data in data['results'].items():
        recommended = list(map(str, user_data['recommended_venues']))
        user_epc = 0
        for item in recommended:
            pop_item = pop_i.get(item, 0)  # Si el ítem no está en entrenamiento, pop_i = 0
            user_epc += (1 - pop_item / pop_max)  # EPC original
        if len(recommended) > 0:
            user_epc /= len(recommended)
        epc_total += user_epc
    epc = epc_total / num_users if num_users > 0 else 0

    # Calcular Novelty con log2
    novelty_total = 0
    for user, user_data in data['results'].items():
        recommended = list(map(str, user_data['recommended_venues']))
        user_novelty = 0
        for item in recommended:
            pop_item = pop_i.get(item, 0)
            if pop_item > 0:
                novelty_score = -np.log2(pop_item / pop_max)
            else:
                novelty_score = -np.log2(1 / (pop_max + 1))  # Asume máxima novedad para ítems no vistos
            user_novelty += novelty_score
        if len(recommended) > 0:
            user_novelty /= len(recommended)
        novelty_total += user_novelty
    novelty = novelty_total / num_users if num_users > 0 else 0

    # Calcular Diversidad: porcentaje de ítems únicos recomendados sobre el total de ítems
    all_recommended = set()
    for user_data in data['results'].values():
        all_recommended.update(map(str, user_data['recommended_venues']))
    unique_recommended = len(all_recommended)
    print(f"Unique recommended venues: {unique_recommended}")
    diversity = unique_recommended

    return epc, novelty, diversity

def save_metrics(epc, novelty, diversity, output_path):
    """Guarda las métricas EPC, Novelty y Diversidad en un nuevo archivo JSON."""
    metrics = {
        "EPC": epc,
        "Novelty": novelty,
        "Diversity": diversity
    }
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    city = 'Tokyo'  # Cambia a la ciudad deseada
    algorithm = "GNN"  # Cambia al algoritmo deseado
    imputed_algo = "BPRMF"  # Cambia al algoritmo de imputación deseado
    k = 10  # Cambia a la cantidad de recomendaciones deseada
    imputed_file = f"data/{city}_BPRMF_combined.csv"  # Ajusta según la ubicación real
    train_file = f"data/Checkins_{city}_train.csv"  # Ajusta según la ubicación real
    imputed = True # Cambia a True si se desea usar el archivo imputado
    if not imputed: 
        input_path_name = f"results/{city}/{algorithm}/{city}_{algorithm}_results_topk_{k}_manual.json"
        output_path_name = f"results/{city}/{algorithm}/{city}_{algorithm}_{k}_adtional_metrics.json"
    else:
        input_path_name = f"results/{city}/{algorithm}/imputed/{city}_{algorithm}_{imputed_algo}_results_topk_{k}.json"
        output_path_name = f"results/{city}/{algorithm}/imputed/{city}_{algorithm}_{imputed_algo}_{k}_adtional_metrics.json"
   
    # Cargar conjunto de entrenamiento para calcular pop_max, pop_i y total_items
    pop_max, pop_i, total_items = load_train_data(train_file)
    
    # Cargar JSON de resultados
    data = load_json(input_path_name)
    
    # Calcular métricas
    epc, novelty, diversity = calculate_metrics(data, pop_max, pop_i, total_items)
    
    # Guardar métricas
    save_metrics(epc, novelty, diversity, output_path_name)
    
    print(f"EPC: {epc:.4f}")
    print(f"Novelty: {novelty:.4f}")
    print(f"Diversidad: {diversity:.2f}")
    print(f"Métricas guardadas en {output_path_name}")
