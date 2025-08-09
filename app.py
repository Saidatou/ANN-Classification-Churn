import streamlit as st
import numpy as np
import pandas as pd
import os

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de Désabonnement Client",
    page_icon="📊",
    layout="wide"
)

# Fonction pour charger les modèles avec gestion d'erreur
@st.cache_resource
def load_models():
    """Charge les modèles et encodeurs avec gestion d'erreurs"""
    try:
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
        import pickle
        
        # Vérification de l'existence des fichiers
        required_files = {
            'model.h5': 'Modèle TensorFlow',
            'label_encoder_gender.pkl': 'Encodeur de genre',
            'onehot_encoder_geo.pkl': 'Encodeur géographique',
            'scaler.pkl': 'Normalisateur de données'
        }
        
        missing_files = []
        for file_path, description in required_files.items():
            if not os.path.exists(file_path):
                missing_files.append(f"{description} ({file_path})")
        
        if missing_files:
            st.error("❌ Fichiers manquants pour l'application :")
            for missing_file in missing_files:
                st.error(f"• {missing_file}")
            st.info("💡 Assurez-vous que tous les fichiers de modèle sont présents dans le repository GitHub")
            return None
        
        # Chargement des modèles
        with st.spinner("🔄 Chargement du modèle..."):
            model = tf.keras.models.load_model('model.h5')
            
            with open('label_encoder_gender.pkl', 'rb') as file:
                label_encoder_gender = pickle.load(file)
            
            with open('onehot_encoder_geo.pkl', 'rb') as file:
                onehot_encoder_geo = pickle.load(file)
            
            with open('scaler.pkl', 'rb') as file:
                scaler = pickle.load(file)
        
        st.success("✅ Modèles chargés avec succès!")
        return model, label_encoder_gender, onehot_encoder_geo, scaler
        
    except ImportError as e:
        st.error(f"❌ Erreur d'importation : {str(e)}")
        st.info("💡 Vérifiez que toutes les dépendances sont installées correctement")
        return None
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des modèles : {str(e)}")
        return None

# Interface principale
st.title('🎯 Prédiction de Désabonnement Client Bancaire')
st.markdown("---")

# Chargement des modèles
models = load_models()

if models is None:
    st.stop()

model, label_encoder_gender, onehot_encoder_geo, scaler = models

# Interface utilisateur organisée en colonnes
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Informations géographiques et démographiques")
    geographie = st.selectbox('Géographie', onehot_encoder_geo.categories_[0])
    genre = st.selectbox('Genre', label_encoder_gender.classes_)
    age = st.slider('Âge', 18, 92, value=40)

with col2:
    st.subheader("💰 Informations financières")
    score_credit = st.number_input('Score de Crédit', min_value=300, max_value=850, value=650)
    solde = st.number_input('Solde du Compte (€)', min_value=0.0, value=50000.0, step=1000.0)
    salaire_estime = st.number_input('Salaire Estimé (€)', min_value=0.0, value=50000.0, step=1000.0)

st.subheader("🏦 Informations bancaires")
col3, col4, col5 = st.columns(3)

with col3:
    anciennete = st.slider('Ancienneté (années)', 0, 10, value=5)

with col4:
    nb_produits = st.slider('Nombre de Produits', 1, 4, value=2)

with col5:
    carte_credit = st.selectbox('Possède une Carte de Crédit', [0, 1], index=1, 
                                format_func=lambda x: 'Oui' if x == 1 else 'Non')
    membre_actif = st.selectbox('Membre Actif', [0, 1], index=1,
                                format_func=lambda x: 'Oui' if x == 1 else 'Non')

# Bouton de prédiction
if st.button('🔮 Prédire le Risque de Désabonnement', type="primary", use_container_width=True):
    try:
        # Affichage du statut de préparation
        with st.spinner("🔄 Préparation des données client..."):
            # Préparation des données (garde les noms originaux pour le modèle)
            input_data = pd.DataFrame({
                'Score de Crédit': [score_credit],
                'Genre': [label_encoder_gender.transform([genre])[0]],
                'Age': [age],
                'Encienneté': [anciennete],
                'Solde': [solde],
                'Nombre de Produits': [nb_produits],
                'Passède Carte de Crédit': [carte_credit],
                'Est un Client Actif': [membre_actif],
                'Estimation de Salaire': [salaire_estime]
            })
            # input_data = pd.DataFrame({
            #     'CreditScore': [score_credit],
            #     'Gender': [label_encoder_gender.transform([genre])[0]],
            #     'Age': [age],
            #     'Tenure': [anciennete],
            #     'Balance': [solde],
            #     'NumOfProducts': [nb_produits],
            #     'HasCrCard': [carte_credit],
            #     'IsActiveMember': [membre_actif],
            #     'EstimatedSalary': [salaire_estime]
            # })

        # Encodage géographique
        with st.spinner("🌍 Traitement des données géographiques..."):
            geo_encoded = onehot_encoder_geo.transform([[geographie]]).toarray()
            geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

        # Combinaison des données
        with st.spinner("🔗 Assemblage des caractéristiques..."):
            input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

        # Normalisation
        with st.spinner("⚖️ Normalisation des données..."):
            input_data_scaled = scaler.transform(input_data)

        # Prédiction
        with st.spinner("🤖 Calcul de la prédiction..."):
            prediction = model.predict(input_data_scaled)
            prediction_proba = prediction[0][0]

        # Affichage des résultats
        st.markdown("---")
        st.subheader("📊 Résultats de la prédiction")
        
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            st.metric("Probabilité de Désabonnement", f"{prediction_proba:.1%}")
        
        with col_result2:
            if prediction_proba > 0.5:
                st.error("⚠️ Le client risque fortement de partir")
                risk_level = "ÉLEVÉ"
                color = "red"
            elif prediction_proba > 0.3:
                st.warning("⚡ Risque modéré de départ")
                risk_level = "MODÉRÉ"
                color = "orange"
            else:
                st.success("✅ Client fidèle")
                risk_level = "FAIBLE"
                color = "green"
        
        # Barre de progression
        st.progress(float(prediction_proba))
        
        # Recommandations
        st.subheader("💡 Recommandations")
        if prediction_proba > 0.5:
            st.markdown("""
            - 🎁 Proposer des offres de fidélisation personnalisées
            - 📞 Contact proactif du conseiller clientèle
            - 💰 Révision des conditions tarifaires
            - 🌟 Programme de récompenses exclusif
            - 🤝 Entretien de satisfaction approfondi
            """)
        elif prediction_proba > 0.3:
            st.markdown("""
            - 📧 Campagne de rétention ciblée
            - 🔍 Analyse approfondie des besoins
            - 📈 Proposition d'amélioration de services
            - 📊 Suivi renforcé de satisfaction
            """)
        else:
            st.markdown("""
            - 🚀 Opportunité de vente croisée (cross-selling)
            - 📢 Programme de parrainage
            - ⭐ Sollicitation d'avis et témoignages
            - 💎 Proposer des services premium
            """)
            
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {str(e)}")
        st.info("💡 Vérifiez que tous les champs sont remplis correctement")

# Section récapitulatif des données saisies
with st.expander("📋 Récapitulatif des informations client"):
    if 'score_credit' in locals():
        st.markdown(f"""
        **Profil Client :**
        - **Géographie :** {geographie}
        - **Genre :** {genre}
        - **Âge :** {age} ans
        - **Score de crédit :** {score_credit}
        - **Solde du compte :** {solde:,.0f} €
        - **Salaire estimé :** {salaire_estime:,.0f} €
        - **Ancienneté :** {anciennete} ans
        - **Nombre de produits :** {nb_produits}
        - **Carte de crédit :** {'Oui' if carte_credit == 1 else 'Non'}
        - **Membre actif :** {'Oui' if membre_actif == 1 else 'Non'}
        """)

# Informations sur l'application
with st.expander("ℹ️ À propos de cette application"):
    st.markdown("""
    Cette application utilise un modèle de machine learning pour prédire la probabilité qu'un client quitte la banque.
    
    **Modèle utilisé :** Réseau de neurones artificiels (TensorFlow/Keras)
    
    **Variables d'entrée :**
    - **Informations démographiques :** âge, genre, géographie
    - **Données financières :** score de crédit, solde, salaire estimé
    - **Historique bancaire :** ancienneté, nombre de produits, possession carte de crédit, statut d'activité
    
    **Interprétation des résultats :**
    - **🔴 Risque élevé (>50%) :** Action immédiate requise
    - **🟡 Risque modéré (30-50%) :** Surveillance et actions préventives
    - **🟢 Risque faible (<30%) :** Client stable, opportunités de développement
    """)

# Footer
st.markdown("---")
st.markdown("*💼 Application développée pour l'analyse prédictive du churn bancaire*")