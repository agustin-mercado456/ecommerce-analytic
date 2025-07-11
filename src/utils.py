import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
import mlflow
import mlflow.sklearn
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import math

def kmeans_clustering_mlflow(data, columnas_rfm, max_clusters=10, n_clusters=3, experiment_name="KMeans_RFM"):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)

    # Cerrar run activo si existe para evitar conflicto
    if mlflow.active_run() is not None:
        mlflow.end_run()

    with mlflow.start_run(run_name=f"Final_Clustering_k={n_clusters}"):
        kmeans = KMeans(init='k-means++', n_init=2, n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(data[columnas_rfm])
        data['Cluster'] = clusters

        # Métricas de evaluación
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_metric("Silhouette Score", silhouette_score(data[columnas_rfm], clusters))
        mlflow.log_metric("Calinski-Harabasz Score", calinski_harabasz_score(data[columnas_rfm], clusters))
        mlflow.log_metric("Davies-Bouldin Score", davies_bouldin_score(data[columnas_rfm], clusters))

        # Combinaciones de pares de columnas para graficar
        combinaciones = [(i, j) for i in range(len(columnas_rfm)) for j in range(i + 1, len(columnas_rfm))]
        n = len(combinaciones)
        c = math.ceil(math.sqrt(n))
        f = math.ceil(n / c)

        plt.figure(figsize=(6 * c, 5 * f))
        for idx, (x_idx, y_idx) in enumerate(combinaciones):
            x_col = columnas_rfm[x_idx]
            y_col = columnas_rfm[y_idx]

            plt.subplot(f, c, idx + 1)
            sns.scatterplot(
                x=data[x_col],
                y=data[y_col],
                hue=clusters,
                palette='Set1',
                s=60,
                alpha=0.7
            )
            plt.scatter(
                kmeans.cluster_centers_[:, x_idx],
                kmeans.cluster_centers_[:, y_idx],
                c='black',
                s=200,
                marker='X',
                label='Centroides'
            )
            plt.title(f'{x_col} vs {y_col}')
            plt.xlabel(f'{x_col} (escalada)')
            plt.ylabel(f'{y_col} (escalada)')
            plt.legend()

        plot_path = "cluster_pairs_plot.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        # Limpiar archivo temporal
        try:
            os.remove(plot_path)
        except Exception:
            pass

        print(f"Run ID final clustering: {mlflow.active_run().info.run_id}")
        return data


def transform_pca(data, n_components=2):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)
    
    return data_pca

def silhouette_analysis(X, range_n_clusters):
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    for n_clusters in range_n_clusters:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        clusterer = KMeans(init='k-means++',n_init=2 ,n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        print(f"Para n_clusters = {n_clusters}, el silhouette promedio es: {silhouette_avg:.4f}")

        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        ax1.set_title("Silhouette plot para diferentes clusters")
        ax1.set_xlabel("Coeficiente de silueta")
        ax1.set_ylabel("Etiqueta de clúster")
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])
        ax1.set_xticks(np.arange(-0.1, 1.1, 0.2))

        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")

        centers = clusterer.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], marker="o", c="white", alpha=1, s=200, edgecolor="k")

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker=f"${i}$", alpha=1, s=50, edgecolor="k")

        ax2.set_title("Visualización de los clústeres")
        ax2.set_xlabel("1ª característica")
        ax2.set_ylabel("2ª característica")

        plt.suptitle(
            f"Análisis de silueta para KMeans con n_clusters = {n_clusters}",
            fontsize=14,
            fontweight="bold",
        )

    plt.show()

    return silhouette_avg

def elbow_method(X, k_range):
    inertias = []

    for k in k_range:
        kmeans = KMeans(init='k-means++',n_init=2 ,n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Inercia (SSE)')
    plt.title('Método del Codo para seleccionar k óptimo')
    plt.grid(True)
    plt.show()

    return inertias


def plot_davies_bouldin_scores(X, k_range):
    db_scores = []

    for k in k_range:
        kmeans = KMeans(init='k-means++',n_init=2 ,n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        db = davies_bouldin_score(X, labels)
        db_scores.append(db)

    # Graficar los valores
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, db_scores, marker='o', linestyle='-', color='purple')
    plt.title("Índice de Davies-Bouldin para diferentes k")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Davies-Bouldin Score")
    plt.grid(True)
    plt.show()

    return dict(zip(k_range, db_scores))


def evaluar_calinski_harabasz(X, k_range):
    """
    Calcula el índice de Calinski-Harabasz para los números de clusters dados en k_range
    usando KMeans y grafica los resultados.
    
    Parámetros:
    - X: array-like, datos a clusterizar
    - k_range: iterable de enteros, números de clusters a evaluar
    - random_state: int, semilla para reproducibilidad
    
    Retorna:
    - lista con los valores del índice para cada k en k_range
    """
    scores = []
    
    for k in k_range:
        kmeans = KMeans(init='k-means++',n_init=2 ,n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = calinski_harabasz_score(X, labels)
        scores.append(score)
    
    # Graficar
    plt.figure(figsize=(8,5))
    plt.plot(list(k_range), scores, marker='o')
    plt.title('Índice de Calinski-Harabasz para distintos números de clusters')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Calinski-Harabasz Score')
    plt.grid(True)
    plt.show()
    
    return scores


def run_dbscan(X, eps, min_samples):
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    return model.labels_

def run_agglomerative(X, k, linkage='ward'):
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit(X)
    return model.labels_

def run_gmm(X, k):
    model = GaussianMixture(n_components=k, random_state=42).fit(X)
    return model.predict(X)

def run_spectral(X, k):
    model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42).fit(X)
    return model.labels_

def evaluate_clustering(X, labels):
    return {
        "silhouette": silhouette_score(X, labels) if len(set(labels))>1 else None,
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels)
    }
