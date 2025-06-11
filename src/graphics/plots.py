import matplotlib.pyplot as plt
import pandas as pd


def plot_number_neighbors(k, best_number_neighbors, city_name, model_name):
    """
    Genera un gráfico de líneas para visualizar el número óptimo de vecinos
    para diferentes valores de K en un modelo de recomendación.

    Parámetros:
    k (list): Lista de valores de K.
    best_number_neighbors (list): Lista del mejor número de vecinos para cada K.
    city_name (str): Nombre de la ciudad.
    model_name (str): Nombre del modelo de recomendación.

    Retorna:
    None: Muestra el gráfico directamente.
    """
    # Configurar el tamaño del gráfico
    plt.figure(figsize=(10, 6))
    
    # Graficar el número óptimo de vecinos
    plt.plot(k, best_number_neighbors, marker='o', label='Mejor Número de Vecinos')
    
    # Añadir etiquetas a los ejes
    plt.xlabel('K : Número de recomendaciones')
    plt.ylabel('Número óptimo de vecinos')
    
    # Establecer el título dinámico con el nombre del modelo y la ciudad
    plt.title(f'Número Óptimo de Vecinos para {model_name} en {city_name}')
    
    # Añadir leyenda y cuadrícula
    plt.legend()
    plt.grid(True)
    
    # guardar el grafico 
    plt.savefig(f"results/plots/{model_name}_{city_name}_neighbors.pgf")

def plot_metrics(k, avg_recall, avg_precision, avg_f1, city_name, model_name):
    """
    Genera un gráfico de líneas para visualizar la evolución de las métricas de recomendación
    (recall, precision y F1) para diferentes valores de K.

    Parámetros:
    k (list): Lista de valores de K.
    avg_recall (list): Lista de valores de recall promedio para cada K.
    avg_precision (list): Lista de valores de precision promedio para cada K.
    avg_f1 (list): Lista de valores de F1 promedio para cada K.
    city_name (str): Nombre de la ciudad.
    model_name (str): Nombre del modelo de recomendación.

    Retorna:
    None: Muestra el gráfico directamente.
    """
    # Configurar el tamaño del gráfico
    plt.figure(figsize=(10, 6))
    
    # Graficar cada métrica con una línea y un marcador
    plt.plot(k, avg_recall, marker='o', label='Recall')
    plt.plot(k, avg_precision, marker='o', label='Precision')
    plt.plot(k, avg_f1, marker='o', label='F1')
    
    # Añadir etiquetas a los ejes
    plt.xlabel('K : Número de recomendaciones')
    plt.ylabel('Valor de la métrica')
    
    # Establecer el título dinámico con el nombre del modelo y la ciudad
    plt.title(f'Evolución de Métricas de Recomendación para {model_name} en {city_name}')
    
    # Añadir leyenda y cuadrícula
    plt.legend()
    plt.grid(True)
    
    # guardar el grafico 
    plt.savefig(f"results/plots/{model_name}_{city_name}.png")

def plot_all_cities (k, avg_recall_1, avg_precision_1, avg_f1_1,
                   avg_recall_2, avg_precision_2, avg_f1_2,
                   avg_recall_3, avg_precision_3, avg_f1_3,
                   city_name_1, city_name_2, city_name_3,
                   model_name):
    """
    Genera un gráfico de líneas para visualizar la evolución de las métricas de recomendación
    (recall, precision y F1) para diferentes valores de K en tres ciudades.

    Parámetros:
    k (list): Lista de valores de K.
    avg_recall (list): Lista de valores de recall promedio para cada K.
    avg_precision (list): Lista de valores de precision promedio para cada K.
    avg_f1 (list): Lista de valores de F1 promedio para cada K.
    city_name (str): Nombre de la ciudad.
    model_name (str): Nombre del modelo de recomendación.

    Retorna:
    None: Muestra el gráfico directamente.
    """
    # Configurar el tamaño del gráfico
    plt.figure(figsize=(10, 6))
    
    # Graficar cada métrica con una línea y un marcador
    plt.plot(k, avg_recall_1, marker='o', label=city_name_1 + " Recall")
    plt.plot(k, avg_recall_2, marker='o', label=city_name_2 + " Recall")
    plt.plot(k, avg_recall_3, marker='o', label=city_name_3 + " Recall")
    plt.plot(k, avg_precision_1, marker='x', label=city_name_1 + " Precision")
    plt.plot(k, avg_precision_2, marker='x', label=city_name_2 + " Precision")
    plt.plot(k, avg_precision_3, marker='x', label=city_name_3 + " Precision")
    plt.plot(k, avg_f1_1, marker='^', label=city_name_1 + " F1")
    plt.plot(k, avg_f1_2, marker='^', label=city_name_2 + " F1")
    plt.plot(k, avg_f1_3, marker='^', label=city_name_3 + " F1")

    # ponme el caracter de un triangulo 

    # Añadir etiquetas a los ejes
    plt.xlabel('K')
    plt.ylabel('Valor de la métrica')
    
    # Establecer el título dinámico con el nombre del modelo y la ciudad
    plt.title(f'Evolución de Métricas de Recomendación para {model_name} en varias ciudades')
    
    # Añadir leyenda ue indique que métrica es cada una

    
    plt.legend()
    plt.grid(True)

    # guardar el grafico
    plt.savefig(f"results/plots/{model_name}_all_cities.png")
  

    
if __name__ == "__main__":
    k = [5,10,15,20]
    avg_recall_london_pop = [0.0648650973233329, 0.08910364990601712,0.1069136563873355,0.1394100018741126]
    avg_precision_london_pop = [0.020278712157616563,0.014320038443056295,0.011757168028191545, 0.011004324843825169]
    avg_f1_london_pop =  [0.028053826751765962, 0.022765671184982822, 0.019786663478477143,0.019368984929858085]
    avg_recall_new_york_pop = [0.10950935125943607,0.142005868827536, 0.1607273796273503,0.17206566592820588]
    avg_precision_new_york_pop = [0.032828919714165565, 0.02200504413619136, 0.01693989071038262,0.013713745271122034]
    avg_f1_new_york_pop = [0.04642990756790323, 0.035573644715710416, 0.028944142249787988, 0.024182976308802766]
    avg_recall_tokyo_pop = [ 0.08685028421078704, 0.10900786451992958, 0.1208922935052289, 0.1357534090085363]
    avg_precision_tokyo_pop = [0.053720626631852515, 0.03297976501305393, 0.024597476066144115, 0.020536879895560783]
    avg_f1_tokyo_pop = [0.055956612693808715,  0.04403604833983085, 0.03638882433395658, 0.0323136919154357]
    avg_recall_london_knn = [0.01319649435939681,0.015971001074316787,0.019514888506040953, 0.0253696536351934, ]
    avg_precision_london_knn = [0.003652090341182126, 0.002787121576165303, 0.0022104757328207606, 0.0021383950024026874 ]
    avg_f1_london_knn = [0.005258599588248796,  0.004309205132792242, 0.0037081406945004003, 0.0037350407356947354]
    best_number_neighbors_london_knn = [20,15,3,3]
    avg_recall_new_york_knn = []
    avg_precision_new_york_knn = []
    avg_f1_new_york_knn = []
    avg_recall_tokyo_knn = []
    avg_precision_tokyo_knn = []
    avg_f1_tokyo_knn = []
    best_number_neighbors_new_york_knn = []
    best_number_neighbors_tokyo_knn = []
    avg_recall_london_BPRMF = []
    avg_precision_london_BPRMF = []
    avg_f1_london_BPRMF = []
    avg_recall_new_york_BPRMF = []
    avg_precision_new_york_BPRMF = []
    avg_f1_new_york_BPRMF = []
    avg_recall_tokyo_BPRMF = []
    avg_precision_tokyo_BPRMF = []
    avg_f1_tokyo_BPRMF = []
    avg_recall_london_GeoBPRMF = []
    avg_precision_london_GeoBPRMF = []
    avg_f1_london_GeoBPRMF = []
    avg_recall_new_york_GeoBPRMF = []
    avg_precision_new_york_GeoBPRMF = []
    avg_f1_new_york_GeoBPRMF = []
    avg_recall_tokyo_GeoBPRMF = []
    avg_precision_tokyo_GeoBPRMF = []
    avg_f1_tokyo_GeoBPRMF = []
    # Plot para Londres
    plot_metrics(k, avg_recall_london_pop, avg_precision_london_pop, avg_f1_london_pop, "Londres", "Popularidad")
    # Plot para Nueva York
    plot_metrics(k, avg_recall_new_york_pop, avg_precision_new_york_pop, avg_f1_new_york_pop, "Nueva York", "Popularidad")
    # Plot para Tokio
    plot_metrics(k, avg_recall_tokyo_pop, avg_precision_tokyo_pop, avg_f1_tokyo_pop, "Tokio", "Popularidad")

    # Plot para todas las ciudades
    plot_all_cities(k, avg_recall_london_pop, avg_precision_london_pop, avg_f1_london_pop,
                    avg_recall_new_york_pop, avg_precision_new_york_pop, avg_f1_new_york_pop,
                    avg_recall_tokyo_pop, avg_precision_tokyo_pop, avg_f1_tokyo_pop,
                    "Londres", "Nueva York", "Tokio", "Popularidad")
    # Plot para Londres con KNN
    plot_metrics(k, avg_recall_london_knn, avg_precision_london_knn, avg_f1_london_knn, "Londres", "KNN")
    # Plot del número óptimo de vecinos para Londres con KNN
    plot_number_neighbors(k, best_number_neighbors_london_knn, "Londres", "KNN")