import cv2 as cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import streamlit as st
import urllib.request

sns.set_style("darkgrid")

# /!\ Utiliser le thème "Dark" (Settings -> Theme) /!\

# import des données
# os.chdir('C:\\Users\\marti\\Notebooks\\OC_Projet_7')
# os.chdir('https://raw.githubusercontent.com/martinletouvet/oc_projet7')
data_dashboard = pd.read_csv('https://raw.githubusercontent.com/martinletouvet/oc_projet7/master/data_dashboard.csv?raw=true')
data_dashboard = data_dashboard.sort_values('client', ascending=True)

# import des images
# -------------------------------------------------
# barre
url = 'https://raw.githubusercontent.com/martinletouvet/oc_projet7/master/barre.png'
url_response = urllib.request.urlopen(url)
img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
barre = cv2.imdecode(img_array, -1)

# genre M
url = 'https://raw.githubusercontent.com/martinletouvet/oc_projet7/master/gender_M.png'
url_response = urllib.request.urlopen(url)
img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
gender_M = cv2.imdecode(img_array, -1)

# genre F
url = 'https://raw.githubusercontent.com/martinletouvet/oc_projet7/master/gender_F.png'
url_response = urllib.request.urlopen(url)
img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
gender_F = cv2.imdecode(img_array, -1)
# -------------------------------------------------

# définition des listes/curseurs de sélection
age_category_list = (sorted(data_dashboard['categorie_age'].unique()))
age_category_list.append('Toutes')
revenus_max = data_dashboard['revenus'].max().astype(int).item()

st.set_page_config(page_title="Home Credit - Client Scoring",
                   page_icon=":money_with_wings:",
                   layout='wide')
# --- Titre ---
st.write("""# Dashboard HOME CREDIT""")
st.markdown("-------------------------------------------------")

# --- Sidebar ---
st.sidebar.header('Sélection et filtres :')

# inputs :
def inputs():
    age_category = st.sidebar.selectbox("Catégorie d'age", age_category_list, 4)
    revenus_limit = st.sidebar.slider('Revenus limites :', 0, revenus_max, (0, revenus_max))
    return age_category, revenus_limit

age_category, revenus_limit = inputs()

st.sidebar.markdown("-------------------------------------------------")

# stockage des 2 selections dans 2 filtres
filtre_revenus = ((data_dashboard['revenus'] >= revenus_limit[0]) & (data_dashboard['revenus'] <= revenus_limit[1]))
if (age_category != 'Toutes'):
    filtre_age = (data_dashboard['categorie_age'] == age_category)
else:
    filtre_age = filtre_revenus

client_list = list(data_dashboard[filtre_age & filtre_revenus]['client'])
client = st.sidebar.selectbox('Client :', client_list)

st.markdown("""
<style>
.big-font {
    font-size:18px !important;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

st.write("""# CLIENT :""")
st.markdown('<p class="big-font">N° {}</p>'.format(client), unsafe_allow_html=True)

st.sidebar.markdown("-------------------------------------------------")

# --- Variables client et catégorie ---
# variables majeures
client_proba = round((data_dashboard[data_dashboard['client'] == client]['proba'])*100, 0).astype(int).item()
genre = data_dashboard[data_dashboard['client'] == client]['genre'].astype(str).item()
age = data_dashboard[data_dashboard['client'] == client]['age'].astype(int).item()
duree_carriere = data_dashboard[data_dashboard['client'] == client]['duree_emploi'].astype(int).item()

# infos complémentaires
proprietaire = data_dashboard[data_dashboard['client'] == client]['proprietaire'].astype(str).item()
montant_credit = data_dashboard[data_dashboard['client'] == client]['montant_credit'].astype(int).item()
enfants = data_dashboard[data_dashboard['client'] == client]['enfants'].astype(int).item()

# variables sur la catégorie de clients selectionnée
moyenne_proba = round((data_dashboard[filtre_age & filtre_revenus]['proba'].mean())*100, 0).astype(int).item()

# affichage variables majeures : séparation des colonnes
col1, col2, col3, col4, col5, col6 = st.beta_columns([0.3, 0.05, 1, 0.05, 1, 1])

# # col1 - GENRE
if (genre == 'M'):
    image1 = gender_M
else:
    image1 = gender_F
col1.image(image1, caption=genre, width=60)

# col2 - BARRE png
col2.image(barre, width=25)

# col 3 - AGE
col3.markdown("""
<style>
.client {
    font-size:18px !important;
    text-align: center !important;
}
</style>
""", unsafe_allow_html=True)

col3.markdown('<p class="client">AGE</p>', unsafe_allow_html=True)
col3.markdown('<p class="client">{} ans</p>'.format(age), unsafe_allow_html=True)

# col4 - BARRE png
col4.image(barre, width=25)

# col 5 - CARRIERE
col5.markdown("") # pour aligner avec AGE

col5.markdown('<p class="client">CARRIERE</p>', unsafe_allow_html=True)
col5.markdown('<p class="client">{} ans</p>'.format(duree_carriere), unsafe_allow_html=True)


# --- JAUGE ---
fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = client_proba,
    mode = "gauge+number",
    title = {'text': "Probabilité de défaut de paiement (%)"},
    delta = {'reference': 45},
    gauge = {'axis': {'range': [None, 100]},
             'steps' : [
                 {'range': [0, 45], 'color': "green"},
                 {'range': [45, 80], 'color': "orange"},
                 {'range': [80, 100], 'color': "red"}],
             'bar': {'color': "lightgray"},
             'threshold' : {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 45}}))
st.plotly_chart(fig)

# --- Texte prediction ---
st.markdown("""
<style>
.prediction {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)

col21, col22 = st.beta_columns([2,6])

col21.markdown('<p class="prediction">Prédiction : </p>', unsafe_allow_html=True)

if (client_proba < 45):
    col22.markdown('<p class="prediction"><span style="color:green">remboursement du prêt</span></p>', unsafe_allow_html=True)
if ((client_proba >= 45) & (client_proba < 80)):
    col22.markdown('<p class="prediction"><span style="color:orange">risque modéré de défaut de paiement</span></p>', unsafe_allow_html=True)
if (client_proba >= 80):
    col22.markdown('<p class="prediction"><span style="color:red">risque de défaut de paiement</span></p>', unsafe_allow_html=True)

st.markdown('<p class="big-font"><i>Probabilité de défaut de paiement moyenne de la catégorie selectionnée : {} %</i></p>'.format(moyenne_proba), unsafe_allow_html=True)

st.markdown("-------------------------------------------------")


# --- infos complémentaires ---
st.markdown("""
<style>
.compl {
    font-size:14px;
    text-align:left;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Informations complémentaires client :</p>', unsafe_allow_html=True)

col31, col32, col33 = st.beta_columns([0.5, 1, 1])

col31.markdown("<p class='compl'>Montant du crédit :</p>", unsafe_allow_html=True)
col32.markdown('<p class="compl"><span style="color:green">{} $</span></p>'.format(montant_credit), unsafe_allow_html=True)

proprietaire_french = 'Oui' if proprietaire == 'Y' else 'Non'
col31.markdown('<p class="compl">Propriétaire :</p>', unsafe_allow_html=True)
col32.markdown('<p class="compl"><span style="color:green">{}</span></p>'.format(proprietaire_french), unsafe_allow_html=True)

col31.markdown("<p class='compl'>Nombre d'enfants :</p>", unsafe_allow_html=True)
col32.markdown('<p class="compl"><span style="color:green">{}</span></p>'.format(enfants), unsafe_allow_html=True)

# --- Histogramme âges ---
if st.sidebar.checkbox("Histogramme des âges",value=True):

    st.markdown("-------------------------------------------------")
    st.markdown("<p class='big-font'>Histogramme des âges de la catégorie sélectionnée :</p>", unsafe_allow_html=True)

    y = data_dashboard[filtre_age & filtre_revenus]['age']
    fig, ax = plt.subplots(figsize=(8,3))
    fig.patch.set_facecolor('None')
    ax.patch.set_facecolor('None')
    ax.set(xlabel = 'Age')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.xaxis.label.set_color('white')
    ax.hist(y, bins=20, color="green")
    st.pyplot(fig)


# --- Histogramme revenus
if st.sidebar.checkbox("Histogramme des revenus",value=True):

    st.markdown("-------------------------------------------------")
    st.markdown("<p class='big-font'>Histogramme des revenus de la catégorie selectionnée :</p>", unsafe_allow_html=True)

    y = data_dashboard[filtre_age & filtre_revenus]['revenus']
    fig, ax = plt.subplots(figsize=(8,3))
    fig.patch.set_facecolor('None')
    ax.patch.set_facecolor('None')
    ax.set(xlabel = 'Revenus')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.xaxis.label.set_color('white')
    ax.hist(y, bins=100, color="blue")
    st.pyplot(fig)

st.markdown("-------------------------------------------------")

# --- Data
st.markdown("<p class='big-font'>Données :</p>", unsafe_allow_html=True)
st.write(data_dashboard[filtre_age & filtre_revenus])
