
# ============================================
# 1. IMPORTATION DES BIBLIOTHEQUES
# ============================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================
# 2. CONFIGURATION DE L'APPLICATION
# ============================================

st.set_page_config(
    page_title="Application de prédiction - MIS",
    page_icon="📊",
    layout="wide"
)

# Chemin du dossier contenant app.py
BASE_DIR = Path(__file__).resolve().parent

# Chemins des fichiers
DATA_PATH = BASE_DIR / "prostate_dataset.txt"
MODEL_PATH = BASE_DIR / "Model_MIS.pkl"

# Ordre exact des variables utilisées à l'entraînement
FEATURES = [
    "lcavol", "lweight", "age", "lbph",
    "svi", "lcp", "gleason", "pgg45"
]


# ============================================
# 3. CHARGEMENT DES DONNEES ET DU MODELE
# ============================================

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH, sep="\t")
    data = data.drop(columns=["col", "train"])
    return data


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    return model


# Vérifier que les fichiers sont disponibles
if not DATA_PATH.exists():
    st.error("Le fichier prostate_dataset.txt est introuvable.")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Le modèle Model_MIS.pkl est introuvable.")
    st.info("Exécute d'abord la commande : python train.py")
    st.stop()

# Charger les fichiers
data = load_data()
model = load_model()


# ============================================
# 4. PREPARATION DES DONNEES POUR L'EVALUATION
# ============================================

X = data[FEATURES]
y = data["lpsa"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Prédictions sur les données de test
y_pred = model.predict(X_test)

# Calcul des métriques
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# ============================================
# 5. TITRE ET MENU DE NAVIGATION
# ============================================

st.title("📊 Application de Machine Learning")
st.subheader("Prédiction de la valeur lpsa avec une régression linéaire")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choisissez une page :",
    ["Tableau de bord", "Faire une prédiction", "À propos"]
)


# ============================================
# 6. PAGE TABLEAU DE BORD
# ============================================

if page == "Tableau de bord":

    st.header("Présentation du jeu de données")

    col1, col2, col3 = st.columns(3)

    col1.metric("Nombre d'observations", data.shape[0])
    col2.metric("Nombre de variables", data.shape[1])
    col3.metric("Valeurs manquantes", int(data.isna().sum().sum()))

    st.subheader("Aperçu des données")
    st.dataframe(data, use_container_width=True)

    st.subheader("Statistiques descriptives")
    st.dataframe(data.describe(), use_container_width=True)

    st.subheader("Relation entre l'âge et lpsa")
    st.scatter_chart(data, x="age", y="lpsa")

    st.divider()

    st.header("Performance du modèle")

    c1, c2, c3 = st.columns(3)

    c1.metric("RMSE", f"{rmse:.3f}")
    c2.metric("MAE", f"{mae:.3f}")
    c3.metric("R²", f"{r2:.3f}")

    st.subheader("Valeurs réelles et valeurs prédites")

    comparaison = pd.DataFrame({
        "Valeurs réelles": y_test.values,
        "Valeurs prédites": y_pred
    })

    st.scatter_chart(
        comparaison,
        x="Valeurs réelles",
        y="Valeurs prédites"
    )


# ============================================
# 7. PAGE DE PREDICTION
# ============================================

elif page == "Faire une prédiction":

    st.header("Effectuer une nouvelle prédiction")

    st.write(
        "Renseignez les caractéristiques ci-dessous "
        "pour obtenir une estimation de la variable lpsa."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            lcavol = st.number_input(
                "lcavol", value=1.45, step=0.1
            )

            lweight = st.number_input(
                "lweight", value=3.62, step=0.1
            )

            age = st.number_input(
                "age", min_value=18, max_value=100, value=65
            )

            lbph = st.number_input(
                "lbph", value=0.30, step=0.1
            )

        with col2:
            svi = st.selectbox(
                "svi", options=[0, 1]
            )

            lcp = st.number_input(
                "lcp", value=-0.80, step=0.1
            )

            gleason = st.selectbox(
                "gleason", options=[6, 7, 8, 9]
            )

            pgg45 = st.number_input(
                "pgg45", min_value=0, max_value=100, value=15
            )

        submitted = st.form_submit_button(
            "Lancer la prédiction"
        )

    if submitted:

        # Construire un DataFrame avec les variables attendues
        nouvelle_donnee = pd.DataFrame(
            [[lcavol, lweight, age, lbph,
              svi, lcp, gleason, pgg45]],
            columns=FEATURES
        )

        # Utiliser le modèle sauvegardé pour prédire
        prediction = model.predict(nouvelle_donnee)

        resultat = float(prediction[0])

        st.success("Prédiction réalisée avec succès !")

        st.metric(
            "Valeur estimée de lpsa",
            f"{resultat:.3f}"
        )

        st.subheader("Données envoyées au modèle")
        st.dataframe(nouvelle_donnee, use_container_width=True)


# ============================================
# 8. PAGE A PROPOS
# ============================================

elif page == "À propos":

    st.header("À propos de cette application")

    st.write("""
    Cette application a été développée avec Streamlit
    dans le cadre d'un projet pédagogique de Machine Learning.

    Elle utilise un modèle de régression linéaire entraîné
    sur le jeu de données prostate_dataset.txt.

    Le modèle est sauvegardé dans un fichier .pkl,
    puis chargé par l'application pour réaliser de nouvelles prédictions.
    """)

    st.subheader("Technologies utilisées")

    st.write("- Python")
    st.write("- Pandas et NumPy")
    st.write("- Scikit-learn")
    st.write("- Joblib")
    st.write("- Streamlit")
