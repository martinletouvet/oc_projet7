import streamlit as st
import numpy as np
import pandas as pd
import cv2 as cv2
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import os

sns.set_style('dark')

os.chdir('C:\\Users\\marti\\Notebooks\\OC_Projet_7')

data_dashboard = pd.read_csv('data_dashboard.csv')
data_dashboard = data_dashboard.sort_values('client', ascending=True)

# définition des listes
age_category_list = (sorted(data_dashboard['categorie_age'].unique()))
revenus_range = data_dashboard['revenus'].max().astype(int).item()
age_category_list.append('Toutes')

# --- Titre
st.write("""# Dashboard HOME CREDIT""")
st.markdown("-------------------------------------------------")

# --- Sidebar
st.sidebar.header('Sélection et filtres :')

def inputs():
    age_category = st.sidebar.selectbox("Catégorie d'age", age_category_list, 4)
    revenus_limit = st.sidebar.slider('Revenus max', 0, revenus_range, revenus_range)
    return age_category, revenus_limit

age_category, revenus_limit = inputs()

st.sidebar.markdown("-------------------------------------------------")

filtre_revenus = (data_dashboard['revenus'] <= revenus_limit)
if (age_category != 'Toutes'):
    filtre_age = (data_dashboard['categorie_age'] == age_category)
else:
    filtre_age = filtre_revenus

client_list = list(data_dashboard[filtre_age & filtre_revenus]['client'])
client = st.sidebar.selectbox('Client :', client_list)

st.write("""# CLIENT :""")
st.markdown('<p class="big-font">N° {}</p>'.format(client), unsafe_allow_html=True)

st.sidebar.markdown("-------------------------------------------------")

# --- Variables client
client_proba = round((data_dashboard[data_dashboard['client'] == client]['proba'])*100, 0).astype(int).item()
genre = data_dashboard[data_dashboard['client'] == client]['genre'].astype(str).item()
age = data_dashboard[data_dashboard['client'] == client]['age'].astype(int).item()
duree_carriere = data_dashboard[data_dashboard['client'] == client]['duree_emploi'].astype(int).item()


# infos complémentaires
proprietaire = data_dashboard[data_dashboard['client'] == client]['proprietaire'].astype(str).item()
montant_credit = data_dashboard[data_dashboard['client'] == client]['montant_credit'].astype(int).item()
enfants = data_dashboard[data_dashboard['client'] == client]['enfants'].astype(int).item()

# variables catégorie de client_list
moyenne_proba = round((data_dashboard[filtre_age & filtre_revenus]['proba'].mean())*100, 0).astype(int).item()

# --- Séparation des colonnées
col1, col2, col3, col4, col5, col6 = st.beta_columns([0.3, 0.05, 1, 0.05, 1, 1])

# ----- col1
if (genre == 'M'):
    image = cv2.imread('gender_M.png', cv2.IMREAD_UNCHANGED)
else:
    image = cv2.imread('gender_F.png', cv2.IMREAD_UNCHANGED)
cv2.imshow('Image', image)
col1.image(image, caption=genre, width=60)

# ----- col2 - BARRE

barre = cv2.imread('barre.png', cv2.IMREAD_UNCHANGED)
cv2.imshow('Image', barre)
col2.image(barre, width=25)

# ---- col 3  AGE ----
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

# ----- col4 - BARRE
barre = cv2.imread('barre.png', cv2.IMREAD_UNCHANGED)
cv2.imshow('Image', barre)
col4.image(barre, width=25)


# ---- col 5 CARRIERE ----
col5.markdown("""
<style>
.infos {
    font-size:18px !important;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

col5.markdown('<p class="client">CARRIERE</p>', unsafe_allow_html=True)
col5.markdown('<p class="client">{} ans</p>'.format(duree_carriere), unsafe_allow_html=True)



# ----- JAUGE
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
             'bar': {'color': "white"},
             'threshold' : {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 45}}))
st.plotly_chart(fig)

# ----- Texte prediction

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

st.markdown('<p class="infos"><i>Probabilité de défaut de paiement moyenne de la catégorie selectionnée : {} %</i></p>'.format(moyenne_proba), unsafe_allow_html=True)


st.markdown("-------------------------------------------------")


# infos complémentaires
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




# --- Histogramme âges


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

# Data
st.markdown("<p class='big-font'>Données :</p>", unsafe_allow_html=True)
st.write(data_dashboard[filtre_age & filtre_revenus])
