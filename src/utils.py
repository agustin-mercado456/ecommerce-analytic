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
    # Set experiment
    mlflow.set_experiment(experiment_name)

    # Cerrar run activo si existe para evitar conflicto
    if mlflow.active_run() is not None:
        mlflow.end_run()

    # Escalar datos una sola vez
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[columnas_rfm])

   

    # Lista para guardar inercia de cada k
    inertias = []

    # Run padre para el método del codo
    with mlflow.start_run(run_name="Elbow_Method"):
        for k in range(1, max_clusters + 1):
            # Run anidado por cada k para loggear parámetro y métrica
            with mlflow.start_run(nested=True, run_name=f"K={k}"):
                km = KMeans(n_clusters=k, random_state=42)
                km.fit(data_scaled)
                inertia = km.inertia_
                inertias.append(inertia)
                mlflow.log_param("n_clusters", k)
                mlflow.log_metric("inertia", inertia)

        # Graficar y loggear artefacto del codo
        plt.figure()
        plt.plot(range(1, max_clusters + 1), inertias, marker='o')
        plt.title("Método del Codo (Elbow)")
        plt.xlabel("Número de Clusters")
        plt.ylabel("Inercia")
        elbow_path = "elbow_plot.png"
        plt.savefig(elbow_path)
        mlflow.log_artifact(elbow_path)
        plt.close()

        # Limpiar archivo temporal
        try:
            os.remove(elbow_path)
        except Exception:
            pass

    # Run final para clustering con n_clusters
    with mlflow.start_run(run_name=f"Final_Clustering_k={n_clusters}"):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(data_scaled)
        data['Cluster'] = clusters

        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_metric("Silhouette Score", silhouette_score(data_scaled, clusters))
        mlflow.log_metric("Calinski-Harabasz Score", calinski_harabasz_score(data_scaled, clusters))
        mlflow.log_metric("Davies-Bouldin Score", davies_bouldin_score(data_scaled, clusters))

        # Graficar pares de variables con clusters y centroides
        combinaciones = [(i, j) for i in range(len(columnas_rfm)) for j in range(i + 1, len(columnas_rfm))]
        n = len(combinaciones)
        c = math.ceil(math.sqrt(n))
        f = math.ceil(n / c)

        plt.figure(figsize=(6 * c, 5 * f))
        for idx, (x_idx, y_idx) in enumerate(combinaciones):
            plt.subplot(f, c, idx + 1)
            sns.scatterplot(
                x=data_scaled[:, x_idx],
                y=data_scaled[:, y_idx],
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
            plt.title(f'{columnas_rfm[x_idx]} vs {columnas_rfm[y_idx]}')
            plt.xlabel(f'{columnas_rfm[x_idx]} (escalada)')
            plt.ylabel(f'{columnas_rfm[y_idx]} (escalada)')
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
