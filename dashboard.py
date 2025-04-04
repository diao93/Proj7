# Import des librairies
import streamlit as st
from PIL import Image
import shap
import requests
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de la page
st.set_page_config(page_title="Dashboard Prêt à dépenser", layout="wide")

# URL de l'API
API_URL = "https://gclkktbtunarfsy4xjuinj.streamlit.app/"

# Chargement des datasets
data_train = pd.read_csv('train_df_sample.csv')
data_test = pd.read_csv('test_df_sample.csv')

# Fonctions de preprocessing
def minmax_scale(df, scaler):
    cols = df.select_dtypes(['float64']).columns
    df_scaled = df.copy()
    scal = MinMaxScaler() if scaler == 'minmax' else StandardScaler()
    df_scaled[cols] = scal.fit_transform(df[cols])
    return df_scaled

data_train_mm = minmax_scale(data_train, 'minmax')
data_test_mm = minmax_scale(data_test, 'minmax')

# Fonction API prédiction avec gestion des erreurs
def get_prediction(client_id):
    url_get_pred = API_URL + "prediction/" + str(client_id)
    response = requests.get(url_get_pred)

    if response.status_code != 200:
        st.error(f"Erreur lors de la récupération de la réponse de l'API. Code d'erreur : {response.status_code}")
        return None, None

    # Debug: Afficher la réponse complète pour mieux comprendre le problème
    st.write("Réponse complète de l'API :", response.text)

    try:
        proba_default = round(float(response.content), 3)
    except ValueError as e:
        st.error(f"Erreur de conversion: {e}")
        return None, None
    
    decision = "Refusé" if proba_default >= 0.54 else "Accordé"
    return proba_default, decision

# Jauge de score
def jauge_score(proba):
    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=proba * 100,
        mode="gauge+number+delta",
        title={'text': "Jauge de score"},
        delta={'reference': 54},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': "MidnightBlue"},
            'steps': [
                {'range': [0, 20], 'color': "Green"},
                {'range': [20, 45], 'color': "LimeGreen"},
                {'range': [45, 54], 'color': "Orange"},
                {'range': [54, 100], 'color': "Red"}
            ],
            'threshold': {'line': {'color': "brown", 'width': 4}, 'thickness': 1, 'value': 54}
        }
    ))
    st.plotly_chart(fig)

# SHAP local
def get_shap_val_local(client_id):
    url = API_URL + "shaplocal/" + str(client_id)
    response = requests.get(url)
    res = json.loads(response.content)
    shap_val = res['shap_values']
    base_value = res['base_value']
    feat_values = res['data']
    feat_names = res['feature_names']
    explanation = shap.Explanation(
        np.reshape(np.array(shap_val, dtype='float'), (1, -1)),
        base_value,
        data=np.reshape(np.array(feat_values, dtype='float'), (1, -1)),
        feature_names=feat_names
    )
    return explanation[0]

# SHAP global
def get_shap_val():
    url = API_URL + "shap/"
    response = requests.get(url)
    content = json.loads(response.content)
    shap_val_glob_0 = content['shap_values_0']
    shap_val_glob_1 = content['shap_values_1']
    return np.array([shap_val_glob_0, shap_val_glob_1])

# Voisins
def df_voisins(id_client):
    url = API_URL + "clients_similaires/" + str(id_client)
    response = requests.get(url)
    return pd.read_json(eval(response.content))

# Distribution
def distribution(feature, id_client, df):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.hist(df[df['TARGET'] == 0][feature], bins=30, label='accordé')
    ax.hist(df[df['TARGET'] == 1][feature], bins=30, label='refusé')
    obs_val = data_test.loc[data_test['SK_ID_CURR'] == id_client][feature].values
    ax.axvline(obs_val, color='green', linestyle='dashed', linewidth=2, label='Client')
    ax.set_title(f'Distribution de "{feature}"')
    ax.legend()
    st.pyplot(fig)

# Scatter plot
def scatter(id_client, feature_x, feature_y, df):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[df['TARGET'] == 0][feature_x], df[df['TARGET'] == 0][feature_y], color='blue', alpha=0.5, label='accordé')
    ax.scatter(df[df['TARGET'] == 1][feature_x], df[df['TARGET'] == 1][feature_y], color='red', alpha=0.5, label='refusé')
    obs = data_test.loc[data_test['SK_ID_CURR'] == id_client]
    ax.scatter(obs[feature_x], obs[feature_y], marker='*', s=200, color='black', label='Client')
    ax.set_title('Analyse bivariée')
    ax.legend()
    st.pyplot(fig)

# Boxplot
def boxplot_graph(id_client, feat, df_vois):
    df_box = data_train_mm.melt(id_vars=['TARGET'], value_vars=feat, var_name="variables", value_name="values")
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.boxplot(data=df_box, x='variables', y='values', hue='TARGET', ax=ax)
    df_voisins_scaled = minmax_scale(df_vois, 'minmax')
    df_voisins_box = df_voisins_scaled.melt(id_vars=['TARGET'], value_vars=feat, var_name="var", value_name="val")
    sns.swarmplot(data=df_voisins_box, x='var', y='val', hue='TARGET', size=8, palette=['green', 'red'], ax=ax)
    data_client = data_test_mm.loc[data_test['SK_ID_CURR'] == id_client][feat]
    for i, cat in enumerate(feat):
        plt.scatter(i, data_client.iloc[0, i], marker='*', s=250, color='blueviolet', label='Client')
    st.pyplot(fig)

# Sidebar
with st.sidebar:
    logo = Image.open('img/logo pret à dépenser.png')
    st.image(logo, width=200)
    page = st.selectbox('Navigation', ["Home", "Information du client", "Interprétation locale", "Interprétation globale"])
    list_id_client = list(data_test['SK_ID_CURR'])
    list_id_client.insert(0, '<Select>')
    id_client_dash = st.selectbox("ID Client", list_id_client)
    st.write('Client sélectionné : ' + str(id_client_dash))
    st.markdown("""---""")
    st.write("Created by Océane Youyoutte")

# Pages
if page == "Home":
    st.title("🏠 Dashboard Prêt à dépenser - Accueil")
    st.markdown(""" 
    Ce dashboard explique les raisons d'approbation ou de refus d'une demande de crédit à l'aide d'un modèle **LightGBM**.
    
    **Pages disponibles :**
    - 🧍 *Information du client*
    - 🔍 *Interprétation locale*
    - 🌍 *Interprétation globale*

    [Jeu de données original](https://www.kaggle.com/c/home-credit-default-risk/data)
    """)

elif page == "Information du client":
    st.title("🧍 Information du client")
    if st.button("Statut de la demande"):
        if id_client_dash != '<Select>':
            proba, decision = get_prediction(id_client_dash)
            if proba is not None:
                st.markdown("### Résultat de la demande :")
                st.success("✅ Crédit accordé") if decision == "Accordé" else st.error("❌ Crédit refusé")
                jauge_score(proba)

    with st.expander("Afficher les informations du client", expanded=False):
        st.write(data_test.loc[data_test['SK_ID_CURR'] == id_client_dash])

elif page == "Interprétation locale":
    st.title("🔍 Interprétation locale")
    if st.checkbox("Afficher l'interprétation locale"):
        shap_val = get_shap_val_local(id_client_dash)
        nb_feat = st.slider('Nombre de variables à afficher', 1, 20, 10)
        fig = shap.waterfall_plot(shap_val, max_display=nb_feat, show=False)
        st.pyplot(fig)

elif page == "Interprétation globale":
    st.title("🌍 Interprétation globale")
    voisins = df_voisins(id_client_dash)

    if st.checkbox("Importance globale"):
        shap_values = get_shap_val()
        test_std = minmax_scale(data_test.drop('SK_ID_CURR', axis=1), 'std')
        nb_feat = st.slider("Variables à afficher", 1, 20, 10)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values[1], test_std, plot_type='bar', max_display=nb_feat, show=False)
        st.pyplot(fig)

    if st.checkbox("Comparaison des distributions"):
        scope = st.radio("Comparaison avec :", ("Tous", "Clients similaires"))
        list_features = list(data_train.columns)
        list_features.remove('SK_ID_CURR')
        col1, col2 = st.columns(2)
        with col1:
            f1 = st.selectbox("Caractéristique 1", list_features, index=5)
        with col2:
            f2 = st.selectbox("Caractéristique 2", list_features, index=4)
        
        if scope == "Tous":
            scatter(id_client_dash, f1, f2, data_test)
        else:
            scatter(id_client_dash, f1, f2, voisins)

    if st.checkbox("Boxplot avec variables sélectionnées"):
        list_feat = st.multiselect("Sélectionnez des variables", list(data_train.columns), default=['AMT_INCOME_TOTAL', 'DAYS_BIRTH'])
        boxplot_graph(id_client_dash, list_feat, voisins)

