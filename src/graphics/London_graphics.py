import matplotlib.pyplot as plt
import numpy as np

# Datos extraídos de la tabla
k = 5
city = 'London'
algoritmos = ['Popularidad', 'BPRMF', 'GeoBPRMF', 'KNN', 'GNN']
precision_no = [0.020, 0.017, 0.017, 0.004, 0.017]
precision_si = [0.020, 0.019, 0.020, 0.004, 0.019]
recall_no = [0.065, 0.056, 0.054, 0.013, 0.056]
recall_si = [0.062, 0.063, 0.062, 0.018, 0.058]

# Cálculo del crecimiento porcentual
def crecimiento_porcentual(antes, despues):
    return [(b - a) / a * 100 if a > 0 else 0 for a, b in zip(antes, despues)]

precision_growth = crecimiento_porcentual(precision_no, precision_si)
recall_growth = crecimiento_porcentual(recall_no, recall_si)

# Gráficos
x = np.arange(len(algoritmos))
width = 0.6

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

# Gráfico de precisión
axes[0].bar(x, precision_growth, color='skyblue')
axes[0].set_xticks(x)
axes[0].set_xticklabels(algoritmos, rotation=45)
axes[0].set_ylabel('Crecimiento en Precisión (%)')
axes[0].set_title('Aumento porcentual de Precisión con imputación')
axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)

# Gráfico de recall
axes[1].bar(x, recall_growth, color='salmon')
axes[1].set_xticks(x)
axes[1].set_xticklabels(algoritmos, rotation=45)
axes[1].set_ylabel('Crecimiento en Recall (%)')
axes[1].set_title('Aumento porcentual de Recall con imputación')
axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.show()