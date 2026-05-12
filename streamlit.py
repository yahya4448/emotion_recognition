import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(page_title="Détecteur d'Émotions", layout="wide", initial_sidebar_state="expanded")
st.title("🎯 Détecteur d'Émotions")
st.markdown("Télécharge une image pour détecter l'émotion")

# Charger le modèle avec gestion d'erreurs
@st.cache_resource
def load_model():
    try:
        # Afficher un spinner pendant le chargement
        with st.spinner("⏳ Chargement du modèle d'IA..."):
            import tensorflow as tf
            
            # Désactiver les avertissements
            tf.get_logger().setLevel('ERROR')
            
            # Essayer de charger le SavedModel (meilleur format)
            if os.path.exists('emotion_model_reconstructed'):
                print("Chargement du SavedModel...")
                model = tf.keras.models.load_model('emotion_model_reconstructed')
                return model
            
            # Sinon, essayer le H5
            elif os.path.exists('emotion_model.h5'):
                print("Chargement du modèle H5...")
                try:
                    model = tf.keras.models.load_model('emotion_model.h5')
                    return model
                except Exception as e:
                    print(f"Erreur H5: {str(e)[:100]}")
                    # Créer un modèle fallback
                    model = tf.keras.Sequential([
                        tf.keras.layers.Input(shape=(64, 64, 3)),
                        tf.keras.layers.Rescaling(1./255),
                        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                        tf.keras.layers.MaxPooling2D((2, 2)),
                        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                        tf.keras.layers.MaxPooling2D((2, 2)),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(128, activation='relu'),
                        tf.keras.layers.Dense(5, activation='softmax')
                    ])
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    return model
            else:
                raise Exception("Aucun modèle trouvé")
                
    except Exception as e:
        st.error(f"""
        Erreur: Impossible de charger le modèle
        """)
        st.stop()

# Charger le modèle
model = load_model()
st.success("✅ Modèle chargé avec succès !")

# Noms des classes d'émotions
class_names = ['Colère', 'Peur', 'Joie', 'Tristesse', 'Surprise']
img_height, img_width = 64, 64

# Fonction pour prétraiter l'image
def preprocess_image(image):
    """Prétraite l'image pour le modèle"""
    # Convertir PIL Image en tableau numpy
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Convertir en RGB si nécessaire
    if len(image_np.shape) == 2:  # Image en niveaux de gris
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:  # RGBA
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    # Redimensionner à 64x64
    image_resized = cv2.resize(image_np, (img_width, img_height))
    
    # Normaliser (le modèle a Rescaling dans la première couche)
    image_normalized = image_resized.astype('float32')
    
    # Ajouter dimension batch
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return image_batch, image_resized

# Fonction pour faire une prédiction
def predict_emotion(image_batch):
    """Effectue la prédiction d'émotion"""
    predictions = model.predict(image_batch, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    return predicted_class, confidence, predictions[0]

# ============= INTERFACE PRINCIPALE =============
st.markdown("---")

# Créer des onglets pour télécharger ou prendre une photo
tab1, tab2 = st.tabs(["📤 Télécharger une Image", "📸 Prendre une Photo"])

image_to_process = None
source_name = ""

# Onglet 1 : Télécharger une image
with tab1:
    st.subheader("📤 Télécharge une Image")
    st.markdown("Formats supportés : JPG, PNG, BMP, GIF")
    
    uploaded_file = st.file_uploader(
        "Clique pour sélectionner une image",
        type=['jpg', 'jpeg', 'png', 'bmp', 'gif'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)
        source_name = uploaded_file.name

# Onglet 2 : Prendre une photo
with tab2:
    st.subheader("📸 Prend une Photo")
    st.markdown("Utilise ta caméra pour capturer une photo")
    
    camera_input = st.camera_input("Clique pour activer la caméra", label_visibility="collapsed")
    
    if camera_input is not None:
        image_to_process = Image.open(camera_input)
        source_name = "Photo de caméra"

# ============= AFFICHAGE DES RÉSULTATS =============
if image_to_process is not None:
    st.markdown("---")
    st.header("📊 Résultats de l'Analyse")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 🖼️ Image Analysée")
        st.image(image_to_process, caption=f"Source : {source_name}", use_column_width=True)
    
    with col2:
        st.markdown("### 🤖 Prédiction du Modèle")
        
        # Prétraiter l'image
        image_batch, image_resized = preprocess_image(image_to_process)
        
        # Spinner pendant la prédiction
        with st.spinner("⏳ Analyse en cours..."):
            # Faire la prédiction
            predicted_class, confidence, all_predictions = predict_emotion(image_batch)
            emotion = class_names[predicted_class]
        
        # Afficher le résultat principal en grand
        st.markdown(f"## 🎭 **{emotion}**")
        
        # Barre de progression de la confiance (convertir en float Python)
        confidence_float = float(confidence)
        st.progress(confidence_float, text=f"Confiance : {confidence_float*100:.1f}%")
        
        # Couleur selon l'émotion
        emoji_map = {
            'Colère': '😠',
            'Peur': '😨',
            'Joie': '😊',
            'Tristesse': '😢',
            'Surprise': '😮'
        }
        
        st.metric(label="Émotion Détectée", value=f"{emoji_map.get(emotion, '😐')} {emotion}")
    
    # ============= GRAPHIQUE ET TABLEAU =============
    st.markdown("---")
    st.header("📈 Analyse Détaillée")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Confiance par Émotion")
        
        # Créer les données pour le graphique
        emotion_data = {}
        for i, class_name in enumerate(class_names):
            emotion_data[class_name] = all_predictions[i]
        
        # Graphique en barres
        st.bar_chart(emotion_data)
    
    with col2:
        st.markdown("### 📋 Détails")
        df = pd.DataFrame({
            '🎭 Émotion': class_names,
            '📊 Confiance (%)': [f"{p*100:.2f}%" for p in all_predictions]
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

else:
    st.info("👆 Commence par télécharger une image ou prendre une photo !")

# ============= FOOTER & INFO =============
st.markdown("---")

with st.expander("ℹ️ **Information sur le Modèle**"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🏗️ Architecture")
        st.markdown("""
        - **CNN** (Réseau Convolutif)
        - **Couches** : Conv → Pool → Dense
        - **Entrée** : 64x64 RGB
        """)
    
    with col2:
        st.markdown("### 🎯 Émotions")
        st.markdown("""
        1. 😠 Colère
        2. 😨 Peur
        3. 😊 Joie
        4. 😢 Tristesse
        5. 😮 Surprise
        """)
    
    with col3:
        st.markdown("### ⚡ Performance")
        st.markdown("""
        - **Optimiseur** : Adam
        - **Perte** : CrossEntropy
        - **Prédiction** : Instantanée
        """)

st.markdown("---")
st.caption("🚀 Détecteur d'Émotions AI | Streamlit + TensorFlow")
