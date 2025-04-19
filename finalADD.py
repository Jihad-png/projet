import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from io import BytesIO
import base64
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

# Configuration de la page
st.set_page_config(
    page_title="Analyse de Clustering Avancée",
    page_icon="📊",
    layout="wide"
)

# Style CSS amélioré
st.markdown("""
<style>
    .title {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
    }
    .sub-title {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1.2rem;
        font-weight: bold;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 0.3rem;
    }
    .method-title {
        font-size: 1.5rem;
        color: #1E88E5;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .method-card {
        background-color: #e3f2fd;
        border-left: 5px solid #1E88E5;
        padding: 20px;
        margin-bottom: 25px;
        border-radius: 0 10px 10px 0;
        transition: transform 0.3s;
    }
    .method-card:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .pros-cons {
        background-color: #f0f7ff;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
    }
    .pros {
        color: #2e7d32;
    }
    .cons {
        color: #c62828;
    }
    .algorithm-steps {
        background-color: #fff8e1;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
    }
    .sidebar-title {
        font-size: 1.3rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0D47A1;
        color: white;
        transform: scale(1.05);
    }
    .use-case {
        background-color: #e8f5e9;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
    }
    .math-formula {
        font-family: monospace;
        background-color: #f5f5f5;
        padding: 8px;
        border-radius: 4px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<div class="title">📊 Exploration des Méthodes de Clustering</div>', unsafe_allow_html=True)

# Fonction pour charger les données
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Format de fichier non supporté. Veuillez importer un fichier Excel (.xlsx) ou CSV.")
            return None
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier: {e}")
        return None

# Implémentation personnalisée de K-means
class CustomKMeans:
    def __init__(self, n_clusters=3, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None
    
    def fit_predict(self, X):
        np.random.seed(self.random_state)
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]
        
        for _ in range(self.max_iter):
            distances = self._calculate_distances(X)
            self.labels = np.argmin(distances, axis=1)
            
            new_centroids = np.array([X[self.labels == i].mean(axis=0) if np.sum(self.labels == i) > 0 
                                     else self.centroids[i] for i in range(self.n_clusters)])
            
            if np.allclose(self.centroids, new_centroids):
                break
                
            self.centroids = new_centroids
            
        return self.labels
    
    def _calculate_distances(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.sum((X - centroid) ** 2, axis=1)
        return distances

# Fonction pour créer un graphique à barres coloré
def plot_cluster_distribution(cluster_counts, n_clusters):
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    bars = ax.bar(cluster_counts['Cluster'], cluster_counts['Nombre'], color=colors)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('Distribution des observations par cluster', fontsize=14)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Nombre d\'observations', fontsize=12)
    ax.set_xticks(cluster_counts['Cluster'])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    return fig

# Fonction pour normaliser les données
def standardize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    return (X - mean) / std

# Fonction pour prétraiter les données
def preprocess_data(df, features, normalize=True):
    data = df[features].copy()
    data = data.dropna()
    X = data.values
    if normalize:
        X = standardize_data(X)
    return data, X

# Fonction pour générer le lien de téléchargement
def get_table_download_link(df, filename="resultats.xlsx"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Résultats')
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Télécharger les résultats</a>'
    return href

# Sidebar - Configuration
with st.sidebar:
    st.markdown('<div class="sidebar-title">📂 Import des Données</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choisir un fichier Excel ou CSV", type=["xlsx", "csv"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.success("Fichier chargé avec succès!")
            
            # Sélection des colonnes
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_cols = st.multiselect(
                "Sélectionnez les colonnes numériques", 
                numeric_cols,
                default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
            )
            
            # Paramètres généraux
            st.markdown('<div class="sidebar-title">⚙️ Paramètres Généraux</div>', unsafe_allow_html=True)
            normalize = st.checkbox("Normaliser les données", value=True)
            random_state = st.slider("Random state", 0, 100, 42)
            
            # Choix de la méthode
            st.markdown('<div class="sidebar-title">📊 Méthode de Clustering</div>', unsafe_allow_html=True)
            method = st.radio(
                "Choisissez une méthode",
                ["K-means", "Classification Hiérarchique"],
                label_visibility="collapsed"
            )
            
            if method == "K-means":
                n_clusters = st.slider("Nombre de clusters", 2, 10, 3)
                max_iter = st.slider("Nombre maximum d'itérations", 10, 500, 100)
            else:
                n_clusters = st.slider("Nombre de clusters", 2, 10, 3)
                linkage_method = st.selectbox(
                    "Méthode de liaison",
                    ["ward", "complete", "average", "single"],
                    format_func=lambda x: {
                        "ward": "Ward (minimise la variance)", 
                        "complete": "Complete (distance maximum)",
                        "average": "Average (distance moyenne)",
                        "single": "Single (distance minimum)"
                    }.get(x, x)
                )
                distance_metric = st.selectbox(
                    "Métrique de distance",
                    ["euclidean", "manhattan", "cosine"],
                    format_func=lambda x: {
                        "euclidean": "Euclidienne", 
                        "manhattan": "Manhattan",
                        "cosine": "Cosinus"
                    }.get(x, x)
                )
                show_dendrogram = st.checkbox("Afficher le dendrogramme", value=True)
            
            # Bouton d'exécution
            run_analysis = st.button("Exécuter l'analyse")

# Page d'accueil 
if uploaded_file is None:
    st.title("📊 Exploration des Méthodes de Clustering")
    st.write("""
    Cette application vous permet d'analyser vos données en utilisant deux algorithmes de clustering puissants.
    Importez vos données pour commencer l'analyse ou découvrez les méthodes ci-dessous.
    """)
    
    # Section K-means
    st.header("1. Algorithme K-means", divider='blue')
    
    with st.expander("📝 **Principe algorithmique**", expanded=True):
        st.write("""
        **Algorithme itératif :**
        1. Choisir aléatoirement K centroïdes initiaux
        2. Assigner chaque point au centroïde le plus proche
        3. Recalculer la position des centroïdes
        4. Répéter jusqu'à convergence
        """)
        st.code("Objectif : Minimiser ∑∑||xᵢ - μⱼ||²\nOù μⱼ est le centroïde du cluster j", language="math")
    
    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ **Avantages**")
        st.write("""
        - Rapide et efficace (O(n))
        - Simple à implémenter
        - Résultats interprétables
        - Scalable aux grands datasets
        """)
    
    with col2:
        st.error("❌ **Limitations**")
        st.write("""
        - Nombre de clusters à définir
        - Sensible aux initialisations
        - Clusters sphériques seulement
        - Sensible aux outliers
        """)
    
    st.info("🔍 **Applications typiques :** Segmentation client, Analyse d'images, Détection d'anomalies")
    
    
    # Section Classification Hiérarchique
    st.header("2. Classification Hiérarchique", divider='blue')
    
    with st.expander("📝 **Approche hiérarchique**", expanded=True):
        st.write("""
        **Algorithme agglomératif :**
        1. Calculer la matrice de distance
        2. Fusionner les clusters les plus proches
        3. Mettre à jour la matrice
        4. Construire le dendrogramme
        """)
        st.code("Méthodes de liaison :\n- Ward (variance)\n- Complete (max)\n- Average (moyenne)", language="text")
    
    col3, col4 = st.columns(2)
    with col3:
        st.success("✅ **Avantages**")
        st.write("""
        - Pas de K prédéfini nécessaire
        - Visualisation par dendrogramme
        - Flexible (différentes métriques)
        - Hiérarchie naturelle
        """)
    
    with col4:
        st.error("❌ **Limitations**")
        st.write("""
        - Complexité élevée (O(n³))
        - Sensible au choix de distance
        - Difficile sur grands datasets
        - Interprétation complexe
        """)
    
    st.info("🔍 **Applications typiques :** Biologie (arbres phylogénétiques), Analyse textuelle, Génomique")
    
    
    # Guide d'utilisation
    st.header("📌 Guide d'utilisation", divider='blue')
    tab1, tab2 = st.tabs(["Pour K-means", "Pour le Hiérarchique"])
    
    with tab1:
        st.write("""
        **Recommandations :**
        - Standardisez vos données
        - Utilisez la méthode du coude
        - Lancez plusieurs initialisations
        - Valeur typique pour max_iter : 300
        """)
        
    
    with tab2:
        st.write("""
        **Recommandations :**
        - Privilégiez Ward pour clusters denses
        - Complete pour clusters séparés
        - Limitez à 500 observations pour le dendrogramme
        - Euclidean distance pour Ward
        """)
    
    # Instructions rapides
    st.info("""
    **Pour commencer :**
    1. Importer un fichier (Excel/CSV) via le panneau latéral
    2. Sélectionner les colonnes numériques
    3. Choisir la méthode et paramètres
    4. Lancer l'analyse
    5. Explorer et exporter les résultats
    """)

# Analyse des données si fichier importé
elif uploaded_file is not None and df is not None and 'run_analysis' in locals() and run_analysis:
    if len(selected_cols) < 2:
        st.warning("Veuillez sélectionner au moins 2 colonnes numériques pour le clustering.")
    else:
        # Préparation des données
        X = df[selected_cols].values
        scaler = StandardScaler() if normalize else None
        X_scaled = scaler.fit_transform(X) if normalize else X
        
        # K-means clustering
        if method == "K-means":
            kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state)
            clusters = kmeans.fit_predict(X_scaled)
            centroids = kmeans.cluster_centers_
            
            st.markdown('<div class="sub-title">Résultats du Clustering K-means</div>', unsafe_allow_html=True)
            
            # Métriques
            inertia = kmeans.inertia_
            silhouette = silhouette_score(X_scaled, clusters)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Inertie", f"{inertia:.2f}")
            with col2:
                st.metric("Score de silhouette", f"{silhouette:.2f}")
            
            # Distribution des clusters
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Distribution des clusters")
            
            cluster_counts = pd.DataFrame({'Cluster': range(n_clusters)})
            cluster_counts['Nombre'] = [sum(clusters == i) for i in range(n_clusters)]
            
            fig = plot_cluster_distribution(cluster_counts, n_clusters)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualisation 2D
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Visualisation des clusters")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.6)
            
            if scaler:
                centroids_original = scaler.inverse_transform(centroids)
                ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.8, marker='X')
            else:
                ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.8, marker='X')
            
            ax.set_xlabel(selected_cols[0])
            ax.set_ylabel(selected_cols[1])
            ax.set_title("Visualisation 2D des clusters")
            plt.colorbar(scatter, label='Cluster')
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Résultats dans un dataframe
            result_df = df.copy()
            result_df['Cluster'] = clusters
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Données avec clusters")
            st.dataframe(result_df)
            st.markdown(get_table_download_link(result_df), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Classification hiérarchique
        else:
            if len(X_scaled) > 500 and show_dendrogram:
                st.warning("Le dendrogramme est désactivé pour plus de 500 observations pour des raisons de performance.")
                show_dendrogram = False
            
            distances = pdist(X_scaled, metric=distance_metric)
            Z = linkage(distances, method=linkage_method)
            clusters = fcluster(Z, t=n_clusters, criterion='maxclust') - 1
            
            st.markdown('<div class="sub-title">Résultats de la Classification Hiérarchique</div>', unsafe_allow_html=True)
            
            # Distribution des clusters
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Distribution des clusters")
            
            cluster_counts = pd.DataFrame({'Cluster': range(n_clusters)})
            cluster_counts['Nombre'] = [sum(clusters == i) for i in range(n_clusters)]
            
            fig = plot_cluster_distribution(cluster_counts, n_clusters)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Dendrogramme
            if show_dendrogram:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Dendrogramme")
                
                fig = Figure(figsize=(12, 6))
                ax = fig.add_subplot(1, 1, 1)
                
                threshold = Z[-(n_clusters-1), 2]
                dendrogram(
                    Z,
                    ax=ax,
                    orientation='top',
                    color_threshold=threshold,
                    above_threshold_color='gray'
                )
                
                ax.axhline(y=threshold, color='r', linestyle='--', label=f'Seuil pour {n_clusters} clusters')
                ax.set_title('Dendrogramme de la Classification Hiérarchique')
                ax.set_xlabel('Index des échantillons')
                ax.set_ylabel('Distance')
                ax.legend()
                
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualisation 2D
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Visualisation des clusters")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.6)
            ax.set_xlabel(selected_cols[0])
            ax.set_ylabel(selected_cols[1])
            ax.set_title("Visualisation 2D des clusters")
            plt.colorbar(scatter, label='Cluster')
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Résultats dans un dataframe
            result_df = df.copy()
            result_df['Cluster'] = clusters
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Données avec clusters")
            st.dataframe(result_df)
            st.markdown(get_table_download_link(result_df), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)