"""
Application Streamlit Professionnelle - Détection de Poubelles avec YOLOv11
============================================================================

Cette application permet la détection automatique de poubelles dans des images 
et des vidéos en utilisant un modèle YOLOv11 personnalisé.

Auteur: Votre Nom
Date: 21 novembre 2025
Version: 2.0.0
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2


# ============================================================================
# CONFIGURATION
# ============================================================================

# Constantes
MODEL_PATH = "runs/detect/train2/weights/best.pt"
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_VIDEO_SIZE_MB = 200
SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "bmp", "webp"]
SUPPORTED_VIDEO_FORMATS = ["mp4", "avi", "mov", "mkv"]

# Classes de détection (à adapter selon votre modèle)
CLASS_NAMES = {
    0: "Poubelle Pleine",
    1: "Poubelle Vide"
}


# ============================================================================
# CONFIGURATION DE LA PAGE
# ============================================================================

st.set_page_config(
    page_title="Détection de Poubelles - YOLOv11",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/ultralytics/ultralytics',
        'Report a bug': None,
        'About': "Application de détection de poubelles utilisant YOLOv11"
    }
)


# ============================================================================
# STYLES CSS PERSONNALISÉS
# ============================================================================

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

@st.cache_resource
def load_model(model_path: str) -> Optional[YOLO]:
    """
    Charge le modèle YOLO avec mise en cache.
    
    Args:
        model_path: Chemin vers le fichier de poids du modèle
        
    Returns:
        Modèle YOLO chargé ou None en cas d'erreur
    """
    try:
        if not os.path.exists(model_path):
            st.error(f"Le modèle n'a pas été trouvé: {model_path}")
            return None
        
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {str(e)}")
        return None


def validate_file_size(file, max_size_mb: int = MAX_VIDEO_SIZE_MB) -> bool:
    """
    Valide la taille du fichier uploadé.
    
    Args:
        file: Fichier uploadé
        max_size_mb: Taille maximale en MB
        
    Returns:
        True si la taille est valide, False sinon
    """
    file_size_mb = file.size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        st.error(f"Le fichier est trop volumineux ({file_size_mb:.2f} MB). Limite: {max_size_mb} MB")
        return False
    return True


def get_detection_stats(results) -> dict:
    """
    Extrait les statistiques de détection.
    
    Args:
        results: Résultats de la prédiction YOLO
        
    Returns:
        Dictionnaire contenant les statistiques
    """
    boxes = results[0].boxes
    stats = {
        "total_detections": len(boxes),
        "classes": {},
        "avg_confidence": 0.0
    }
    
    if len(boxes) > 0:
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        
        stats["avg_confidence"] = float(np.mean(confidences))
        
        for cls in np.unique(classes):
            class_name = CLASS_NAMES.get(int(cls), f"Classe {int(cls)}")
            stats["classes"][class_name] = int(np.sum(classes == cls))
    
    return stats


def process_image(model: YOLO, image: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Traite une image et retourne l'image annotée avec les statistiques.
    
    Args:
        model: Modèle YOLO
        image: Image à traiter
        conf: Seuil de confiance
        iou: Seuil IOU
        
    Returns:
        Tuple (image annotée, statistiques)
    """
    results = model.predict(
        source=image,
        # conf=conf,
        # iou=iou,
        verbose=False
    )
    
    annotated_image = results[0].plot()
    stats = get_detection_stats(results)
    
    return annotated_image, stats


def process_video(model: YOLO, video_path: str, 
                  progress_bar, status_text) -> Tuple[Optional[str], dict]:
    """
    Traite une vidéo et retourne le chemin de la vidéo annotée.
    
    Args:
        model: Modèle YOLO
        video_path: Chemin de la vidéo d'entrée
        conf: Seuil de confiance
        iou: Seuil IOU
        progress_bar: Barre de progression Streamlit
        status_text: Texte de statut Streamlit
        
    Returns:
        Tuple (chemin de la vidéo de sortie, statistiques globales)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Impossible d'ouvrir la vidéo")
            return None, {}
        
        # Propriétés de la vidéo
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Créer un fichier temporaire pour la vidéo intermédiaire
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        # Utiliser mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            st.error("Impossible de créer le fichier vidéo de sortie")
            return None, {}
        
        # Statistiques globales
        global_stats = {
            "total_frames": total_frames,
            "processed_frames": 0,
            "total_detections": 0,
            "frames_with_detections": 0
        }
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Prédiction sur la frame
            results = model.predict(
                source=frame,
                verbose=False
            )
            
            # Frame annotée
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            
            # Mise à jour des statistiques
            detections = len(results[0].boxes)
            global_stats["total_detections"] += detections
            if detections > 0:
                global_stats["frames_with_detections"] += 1
            
            frame_count += 1
            global_stats["processed_frames"] = frame_count
            
            # Mise à jour de la barre de progression
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Traitement: Frame {frame_count}/{total_frames} ({progress*100:.1f}%)")
        
        cap.release()
        out.release()
        
        # Convertir en MP4 avec H.264 en utilisant FFmpeg
        status_text.text("Conversion de la vidéo en format compatible navigateur...")
        try:
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-y', '-i', temp_path,
                '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '23', '-pix_fmt', 'yuv420p',
                output_path
            ], capture_output=True, text=True)
            
            # Nettoyer le fichier temporaire
            try:
                os.unlink(temp_path)
            except:
                pass
            
            if result.returncode != 0:
                st.warning("Conversion FFmpeg échouée, utilisation de la vidéo originale")
                return temp_path, global_stats
                
        except FileNotFoundError:
            st.warning("FFmpeg non disponible, utilisation du format MJPEG")
            return temp_path, global_stats
        except Exception as e:
            st.warning(f"Erreur lors de la conversion: {str(e)}")
            return temp_path, global_stats
        
        return output_path, global_stats
        
    except Exception as e:
        st.error(f" Erreur lors du traitement de la vidéo: {str(e)}")
        return None, {}


def display_detection_stats(stats: dict):
    """
    Affiche les statistiques de détection dans une mise en page élégante.
    
    Args:
        stats: Dictionnaire des statistiques
    """
    st.markdown("### Statistiques de Détection")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Détections",
            value=stats.get("total_detections", 0)
        )
    
    with col2:
        avg_conf = stats.get("avg_confidence", 0.0)
        st.metric(
            label="Confiance Moyenne",
            value=f"{avg_conf:.2%}" if avg_conf > 0 else "N/A"
        )
    
    with col3:
        classes = stats.get("classes", {})
        st.metric(
            label=" Classes Détectées",
            value=len(classes)
        )
    
    # Détails par classe
    if classes:
        st.markdown("#### Détections par Classe")
        for class_name, count in classes.items():
            st.write(f"- **{class_name}**: {count} détection(s)")


# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================

def main():
    """Fonction principale de l'application."""
    
    # En-tête
    st.markdown('<p class="main-header">Détection de Poubelles avec YOLOv11</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Application professionnelle de détection d\'objets par intelligence artificielle</p>', unsafe_allow_html=True)
    
    # Sidebar - Configuration
    with st.sidebar:
        # st.header("⚙️ Configuration")
        
        # Paramètres du modèle
        # st.subheader("Paramètres de Détection")
        # confidence = st.slider(
        #     "Seuil de Confiance",
        #     min_value=0.0,
        #     max_value=1.0,
        #     value=CONFIDENCE_THRESHOLD,
        #     step=0.05,
        #     help="Seuil minimum de confiance pour considérer une détection"
        # )
        
        # iou = st.slider(
        #     "Seuil IOU (Non-Maximum Suppression)",
        #     min_value=0.0,
        #     max_value=1.0,
        #     value=IOU_THRESHOLD,
        #     step=0.05,
        #     help="Seuil pour éliminer les détections redondantes"
        # )
        
        # st.markdown("---")
        
        # Informations sur le modèle
        st.subheader(" Informations")
        st.info(f"""
        **Modèle**: YOLOv11  
        **Version**: 2.0.0  
        **Classes**: {', '.join(CLASS_NAMES.values())}
        """)
        
        # Téléchargement du modèle
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                st.download_button(
                    label=" Télécharger le Modèle",
                    data=f,
                    file_name="yolo_poubelle_detection.pt",
                    mime="application/octet-stream",
                    help="Télécharger le modèle YOLO entraîné"
                )
        
        st.markdown("---")
        
        # Guide d'utilisation
        with st.expander(" Guide d'Utilisation"):
            st.markdown("""
            1. **Uploadez** une image ou une vidéo
            2. **Ajustez** les paramètres si nécessaire
            3. **Visualisez** les résultats avec les détections
            4. **Téléchargez** le résultat traité
            
            **Formats supportés**:
            - Images: JPG, PNG, BMP, WEBP
            - Vidéos: MP4, AVI, MOV, MKV
            """)
    
    # Chargement du modèle
    with st.spinner(" Chargement du modèle..."):
        model = load_model(MODEL_PATH)
    
    if model is None:
        st.error(" Impossible de charger le modèle. Vérifiez que le fichier existe.")
        st.info(f" Chemin attendu: `{MODEL_PATH}`")
        st.stop()
    
    # st.success(" Modèle chargé avec succès!")
    
    # Zone de upload
    st.markdown("### Chargement du Fichier")
    
    upload_type = st.radio(
        "Type de fichier:",
        ["Image", "Vidéo"],
        horizontal=True
    )
    
    if upload_type == "Image":
        uploaded_file = st.file_uploader(
            "Choisissez une image",
            type=SUPPORTED_IMAGE_FORMATS,
            help="Formats supportés: " + ", ".join(SUPPORTED_IMAGE_FORMATS)
        )
        
        if uploaded_file is not None:
            # Validation
            if not validate_file_size(uploaded_file, 10):
                st.stop()
            
            # Affichage de l'image originale
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Image Originale")
                image = Image.open(uploaded_file)
                st.image(image, width="stretch")
            
            # Traitement
            with st.spinner(" Détection en cours..."):
                image_np = np.array(image)
                start_time = time.time()
                annotated_image, stats = process_image(model, image_np)
                processing_time = time.time() - start_time
            
            # Affichage du résultat
            with col2:
                st.markdown("#### Image avec Détections")
                st.image(annotated_image, width="stretch")
            
            # Statistiques
            st.markdown("---")
            display_detection_stats(stats)
            
            st.info(f" Temps de traitement: {processing_time:.2f} secondes")
            
            # Téléchargement
            result_image = Image.fromarray(annotated_image)
            buf = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            result_image.save(buf.name, format='JPEG')
            
            with open(buf.name, 'rb') as f:
                st.download_button(
                    label=" Télécharger l'Image Annotée",
                    data=f,
                    file_name=f"detection_{uploaded_file.name}",
                    mime="image/jpeg"
                )
    
    else:  # Vidéo
        uploaded_file = st.file_uploader(
            "Choisissez une vidéo",
            type=SUPPORTED_VIDEO_FORMATS,
            help="Formats supportés: " + ", ".join(SUPPORTED_VIDEO_FORMATS)
        )
        
        if uploaded_file is not None:
            # Validation
            if not validate_file_size(uploaded_file):
                st.stop()
            
            # Sauvegarde temporaire
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_file.read())
                video_path = tfile.name
            
            # Affichage vidéo originale
            st.markdown("#### Vidéo Originale")
            st.video(video_path)
            
            # Traitement automatique
            st.markdown("---")
            st.markdown("### 🔄 Traitement en cours...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            start_time = time.time()
            
            output_path, video_stats = process_video(
                model, video_path,
                progress_bar, status_text
            )
            
            processing_time = time.time() - start_time
            
            if output_path:
                st.success(" Traitement terminé!")
                
                # Affichage de la vidéo traitée
                st.markdown("#### Vidéo avec Détections")
                st.video(output_path)
                
                # Statistiques vidéo
                st.markdown("### Statistiques de la Vidéo")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(" Frames Totales", video_stats.get("total_frames", 0))
                
                with col2:
                    st.metric(" Frames Traitées", video_stats.get("processed_frames", 0))
                
                with col3:
                    st.metric(" Détections Totales", video_stats.get("total_detections", 0))
                
                with col4:
                    frames_with_det = video_stats.get("frames_with_detections", 0)
                    total = video_stats.get("total_frames", 1)
                    percentage = (frames_with_det / total * 100) if total > 0 else 0
                    st.metric(" Frames avec Détections", f"{percentage:.1f}%")
                
                st.info(f" Temps de traitement: {processing_time:.2f} secondes")
                
                # Téléchargement
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label=" Télécharger la Vidéo Annotée",
                        data=f,
                        file_name=f"detection_{uploaded_file.name}",
                        mime="video/mp4"
                    )
                
                # Nettoyage
                try:
                    os.unlink(output_path)
                except:
                    pass
            
            # Nettoyage du fichier temporaire
            try:
                os.unlink(video_path)
            except:
                pass
    
    # Footer
    # st.markdown("---")
    # st.markdown("""
    # <div style='text-align: center; color: #666; padding: 2rem;'>
    #     <p>Développé avec ❤️ en utilisant Streamlit et YOLOv11</p>
    #     <p>© 2025 - Tous droits réservés</p>
    # </div>
    # """, unsafe_allow_html=True)


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    main()