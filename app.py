import streamlit as st
import numpy as np
import pandas as pd
import os

# Configuration de la page
st.set_page_config(
    page_title="Customer Churn Prediction",
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
st.title('🎯 Customer Churn Prediction')
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
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92, value=40)

with col2:
    st.subheader("💰 Informations financières")
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
    balance = st.number_input('Balance', min_value=0.0, value=50000.0, step=1000.0)
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0, step=1000.0)

st.subheader("🏦 Informations bancaires")
col3, col4, col5 = st.columns(3)

with col3:
    tenure = st.slider('Tenure (années)', 0, 10, value=5)

with col4:
    num_of_products = st.slider('Number of Products', 1, 4, value=2)

with col5:
    has_cr_card = st.selectbox('Has Credit Card', [0, 1], index=1)
    is_active_member = st.selectbox('Is Active Member', [0, 1], index=1)

# Bouton de prédiction
if st.button('🔮 Prédire le Churn', type="primary", use_container_width=True):
    try:
        # Préparation des données
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        # Encodage géographique
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

        # Combinaison des données
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

        # Normalisation
        input_data_scaled = scaler.transform(input_data)

        # Prédiction
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]

        # Affichage des résultats
        st.markdown("---")
        st.subheader("📊 Résultats de la prédiction")
        
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            st.metric("Probabilité de Churn", f"{prediction_proba:.1%}")
        
        with col_result2:
            if prediction_proba > 0.5:
                st.error("⚠️ Le client risque de partir")
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
        st.progress(prediction_proba)
        
        # Recommandations
        st.subheader("💡 Recommandations")
        if prediction_proba > 0.5:
            st.markdown("""
            - 🎁 Proposer des offres de fidélisation
            - 📞 Contact proactif du service client
            - 💰 Révision des conditions tarifaires
            - 🌟 Programme de récompenses personnalisé
            """)
        elif prediction_proba > 0.3:
            st.markdown("""
            - 📧 Campagne de rétention ciblée
            - 🔍 Analyse des besoins clients
            - 📈 Proposition d'upgrade de services
            """)
        else:
            st.markdown("""
            - 🚀 Opportunité d'upselling
            - 📢 Programme de parrainage
            - ⭐ Solliciter des avis clients
            """)
            
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {str(e)}")
        st.info("💡 Vérifiez que tous les champs sont remplis correctement")

# Informations sur l'application
with st.expander("ℹ️ À propos de cette application"):
    st.markdown("""
    Cette application utilise un modèle de machine learning pour prédire la probabilité qu'un client quitte la banque.
    
    **Modèle utilisé :** Réseau de neurones (TensorFlow/Keras)
    
    **Variables d'entrée :**
    - Informations démographiques (âge, sexe, géographie)
    - Score de crédit et informations financières
    - Historique bancaire (ancienneté, produits, activité)
    """)

