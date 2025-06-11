import pandas as pd
import random

# Supongamos que 'df' es tu DataFrame con las columnas mencionadas
# Si lo lees desde un archivo: df = pd.read_csv('tu_archivo.csv')

def per_user_split(df, test_ratio=0.2):
    from collections import defaultdict
    
    # Asegurarnos de que 'datetime' esté en formato datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Crear un diccionario de usuario a lista de venues
    user_venue_dict = df.groupby('user_id')['venue_id'].apply(list).to_dict()
    
    # Listas para almacenar filas de train y test
    train_rows = []
    test_rows = []
    
    for user_id in user_venue_dict:
        venues = user_venue_dict[user_id]
        num_venues = len(venues)
        num_test = max(1, int(num_venues * test_ratio))  # Asegurar al menos 1 para test si posible
        
        if num_test >= 1 and num_venues > 1:  # Si hay suficientes venues
            # Ordenar por datetime para split temporal (opcional, descomenta si lo prefieres)
            user_df = df[df['user_id'] == user_id].sort_values('datetime')
            venues_sorted = user_df['venue_id'].tolist()
            test_venues = venues_sorted[-num_test:]  # Los más recientes para test
            train_venues = venues_sorted[:-num_test]  # Los anteriores para train
        else:
            # Si solo hay 1 venue, todo a train
            train_venues = venues
            test_venues = []
        
        # Añadir filas a train
        for v in train_venues:
            row = df[(df['user_id'] == user_id) & (df['venue_id'] == v)].iloc[0]
            train_rows.append(row)
        
        # Añadir filas a test
        for v in test_venues:
            row = df[(df['user_id'] == user_id) & (df['venue_id'] == v)].iloc[0]
            test_rows.append(row)
    
    # Convertir listas a DataFrames
    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)
    
    return train_df, test_df

# Ejemplo de uso
path_file = 'data/Checkins_New_York.csv'
df = pd.read_csv(path_file)  # Ajusta la ruta
train_df, test_df = per_user_split(df, test_ratio=0.2)

# Guardar los splits con el nombre del pathfile mas test o train 
train_df.to_csv(path_file.replace('.csv', '_train.csv'), index=False)
test_df.to_csv(path_file.replace('.csv', '_test.csv'), index=False)

