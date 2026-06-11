from transformers import pipeline
import soundfile as sf
import librosa
import time
import os

# =====================================================================
# ⚙️ CONFIGURATION DU PROJET (Version automatique pour le prof)
# =====================================================================
MODELE_IA = "openai/whisper-tiny" 
AUDIO_A_TRAITER = "audio.m4a"  # Mets le nom du fichier du prof ici (.wav, .mp3, .m4a...)
FICHIER_SORTIE = "texte_ia.txt"
# =====================================================================

def transcrire_projet():
    print(f"🔄 Étape 1 : Chargement de l'IA Whisper ({MODELE_IA})...")
    traducteur = pipeline("automatic-speech-recognition", model=MODELE_IA)
    
    if not os.path.exists(AUDIO_A_TRAITER):
        print(f"❌ Erreur : Le fichier '{AUDIO_A_TRAITER}' est introuvable.")
        return

    print(f"🎙️ Étape 2 : Chargement sécurisé de l'audio...")
    try:
        # On force la conversion de l'audio en 16000Hz (le standard de l'IA Whisper)
        # Cela évite TOUTES les erreurs de formats malformés ou de codecs Windows
        audio_data, samplerate = librosa.load(AUDIO_A_TRAITER, sr=16000)
    except Exception as e:
        print(f"❌ Impossible de lire le fichier via librosa, tentative avec soundfile...")
        audio_data, samplerate = sf.read(AUDIO_A_TRAITER)
        if samplerate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=16000)

    print(f"🚀 Étape 3 : Transcription automatique (Détection de la langue active)...")
    debut = time.time()
    
    # On passe directement les données numériques brutes ("raw") à l'IA. 
    # "task": "transcribe" force l'IA à écrire dans la langue détectée sans traduire en anglais.
    resultat = traducteur(
        {"raw": audio_data, "sampling_rate": 16000},
        generate_kwargs={"task": "transcribe"}
    )
    
    fin = time.time()
    texte_final = resultat['text']
    
    print("\n=== ✨ RÉSULTATS DU SCRIPT ===")
    print(f"⏱️ Temps de calcul : {fin - debut:.2f} secondes")
    print(f"📝 Texte trouvé : {texte_final}")
    
    # 💾 Sauvegarde automatique pour l'Élève 3
    with open(FICHIER_SORTIE, "w", encoding="utf-8") as f:
        f.write(texte_final)
    print(f"💾 Fichier '{FICHIER_SORTIE}' créé avec succès pour le groupe !")

if __name__ == "__main__":
    transcrire_projet()