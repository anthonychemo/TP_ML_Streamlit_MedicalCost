import streamlit as st
import pandas as pd 
import numpy as np
import pickle
from sklearn.preprocessing  import LabelEncoder
import time

#chargement du modèle
with open('reg.pkl','rb') as file:
  model=pickle.load(file)

#Titre et mise en 
st.set_page_config(page_title="Predicteur de Charges Medicales")
st.title("Prédiction des charges medicales")
st.markdown("Remplis les informations ci-dessous pour predire les charges medicales")

#Ajout d'animations
with st.spinner("chargement du modele"):
    time.sleep(1)

#Entrées utilisateur
col1,col2=st.columns(2)
with col1:
   age=st.slider('Age',18,100,30)

with col2:
   sex=st.selectbox('Sexe',["male","female"])

col3,col4=st.columns(2)
with col3:
   bmi=st.number_input("BMI 5Indice de masse corporelle)",10,50,25)

with col4:
   children=st.slider("Nombre d'enfants",0,5,1)

col5,col6=st.columns(2)
with col5:
   smoker=st.selectbox('Fumeur',["yes","no"])

with col6:
   region=st.selectbox('Region',["southwest","southeast","northwest","northeast"])


#ENCODAGE
sex_encoded=1 if sex=="male" else 0
smoker_encoded=1 if smoker=="yes" else 0
region_dict = {"southwest": 0.24308153, "southeast":0.27225131, "northwest":0.24233358, "northeast":0.27225131}
region_encoded = region_dict[region]

#Preparation des données
input_data=[[age,sex_encoded,bmi,children,smoker_encoded,region_encoded]]

#Prediction
if st.button("Prediction des charges medicales"):
   with st.spinner("Calcul en cours ... "):
        prediction=model.predict(input_data)
        time.sleep(1)
   st.success("Prediction Terminee")
   st.markdown(f"### Charges medicales Estimees: **${prediction[0]:,.2f}**")
   st.balloons()
