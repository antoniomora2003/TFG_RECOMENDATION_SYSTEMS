import pandas as pd
import json
import numpy as np

def load_json(file_path):
    """Carga el archivo JSON desde el path especificado."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_csv(file_path):
    """Carga un archivo CSV y devuelve un DataFrame."""
    return pd.read_csv(file_path)

def calculate_user_mean_score(train_df, user_id):
    """Calcula la media de los scores para un usuario en el conjunto de entrenamiento."""
    user_scores = train_df[train_df['user_id'] == user_id]['score']
    if len(user_scores) > 0:
        return user_scores.mean()
    else:
        return 0  # Valor por defecto si no hay scores

def get_venue_info(venue_id, train_df, test_df):
    """Busca la información de un venue en train o test."""
    venue_info = train_df[train_df['venue_id'] == venue_id]
    if venue_info.empty:
        venue_info = test_df[test_df['venue_id'] == venue_id]
    if not venue_info.empty:
        return venue_info.iloc[0]
    else:
        return None

def impute_recommendations(json_data, train_df, test_df):
    """Crea nuevas interacciones imputadas a partir de las recomendaciones."""
    imputed_data = []
    for user_id, user_data in json_data['results'].items():
        user_id = int(user_id)  # Asegurarse de que sea int
        mean_score = calculate_user_mean_score(train_df, user_id)
        for venue_id in user_data['recommended_venues']:
            venue_id = int(venue_id)  # Asegurarse de que sea int
            venue_info = get_venue_info(venue_id, train_df, test_df)
            if venue_info is not None:
                new_entry = {
                    'user_id': user_id,
                    'venue_id': venue_id,
                    'latitude': venue_info['latitude'],
                    'longitude': venue_info['longitude'],
                    'category_name': venue_info['category_name'],
                    'country_code': venue_info['country_code'],
                    'nearest_city': venue_info['nearest_city'],
                    'datetime': None,  # Nulo como se solicitó
                    'score': mean_score
                }
                imputed_data.append(new_entry)
    return pd.DataFrame(imputed_data)
    


if __name__ == "__main__":
    algo = 'GNN'  # Algoritmo a usar, puede ser BPRMF, BPR, etc.
    city = 'Tokyo'  # Cambia según la ciudad que estés usando
    json_file = f"results/{city}/{algo}/{city}_{algo}_results_topk_10_manual.json"  # Ajusta la ruta al archivo JSON
    train_file = f"data/Checkins_{city}_train.csv"    # Ajusta la ruta al archivo train
    test_file = f"data/Checkins_{city}_test.csv"      # Ajusta la ruta al archivo test
    output_file = f"data/{city}_{algo}_imputed_train.csv"    # Ruta del archivo de salida

    # Cargar los datos
    json_data = load_json(json_file)
    train_df = load_csv(train_file)
    test_df = load_csv(test_file)

    # Asegurarse de que los tipos de datos sean consistentes
    train_df['user_id'] = train_df['user_id'].astype(int)
    train_df['venue_id'] = train_df['venue_id'].astype(int)
    test_df['user_id'] = test_df['user_id'].astype(int)
    test_df['venue_id'] = test_df['venue_id'].astype(int)

    # Generar las interacciones imputadas
    imputed_df = impute_recommendations(json_data, train_df, test_df)

    # Guardar el nuevo CSV
    imputed_df.to_csv(output_file, index=False)
    print(f"Nuevo archivo CSV guardado en {output_file}")

   