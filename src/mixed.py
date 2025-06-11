import pandas as pd
import math

# Leer los archivos CSV
path_train = "data/Checkins_Tokyo_train.csv"
train_df = pd.read_csv(path_train)
path_imputed = "data/Tokyo_GNN_imputed_train.csv"
imputed_df = pd.read_csv(path_imputed)

# Agregar una columna 'source' a los DataFrames
train_df['source'] = 'N'  # 'N' para entrenamiento
imputed_df['source'] = 'Y'  # 'Y' para imputado

# Contar el número de entradas por usuario en el conjunto de entrenamiento
train_counts = train_df.groupby('user_id').size()

# Función para calcular cuántas entradas imputadas añadir
def calculate_imputed_to_add(n):
    if n == 1:
        return 1
    else:
        return min(math.floor(0.2 * n), 10)

# Calcular el número de entradas imputadas a añadir por usuario
imputed_to_add = train_counts.apply(calculate_imputed_to_add)

# Agrupar los datos imputados por usuario
imputed_grouped = imputed_df.groupby('user_id')

# Seleccionar las entradas imputadas según las reglas
selected_imputed = []
for user_id, m in imputed_to_add.items():
    if user_id in imputed_grouped.groups:
        group = imputed_grouped.get_group(user_id)
        selected = group.head(m)  # Tomar las primeras m entradas
        selected_imputed.append(selected)
    else:
        print(f"Advertencia: el usuario {user_id} no tiene entradas en imputed.csv")

# Combinar las entradas imputadas seleccionadas en un solo DataFrame
selected_imputed_df = pd.concat(selected_imputed, ignore_index=True)

# Combinar los datos de entrenamiento con las entradas imputadas seleccionadas
combined_df = pd.concat([train_df, selected_imputed_df], ignore_index=True)

# Ordenar el DataFrame combinado por 'user_id' y 'venue_id'
combined_df = combined_df.sort_values(by=['user_id', 'venue_id'])

output_path = "data/Tokyo_GNN_combined.csv"
# Guardar el resultado en un nuevo archivo CSV
combined_df.to_csv(output_path, index=False)

print(f"Proceso completado. El archivo {output_path} ha sido generado.")