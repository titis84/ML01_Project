#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import time
import os
import glob

NOM_MODELE = "openai/whisper-tiny"

# C'est ici que l'Étudiant 1 doit déposer ses 3 dossiers
DOSSIERS_ENTREE = {
    "brut": "audios_brut",         # Dossier pour les fichiers originaux
    "denoise": "audios_denoise",   # Dossier pour les fichiers après débruitage
    "vad": "audios_vad"            # Dossier pour les fichiers après VAD
}

# Dossier principal où seront sauvegardés tous tes résultats
DOSSIER_SORTIE_BASE = "resultats_transcriptions"

def traiter_dossier(dossier_entree, suffixe):
    """Transcrit TOUS les fichiers audio d'un dossier et sauvegarde les résultats."""
    dossier_sortie = os.path.join(DOSSIER_SORTIE_BASE, suffixe)
    os.makedirs(dossier_sortie, exist_ok=True)

    # Cherche tous les fichiers audio dans le dossier (wav, mp3, etc.)
    extensions = ["*.wav", "*.mp3", "*.m4a", "*.flac"]
    fichiers = []
    for ext in extensions:
        fichiers.extend(glob.glob(os.path.join(dossier_entree, ext)))

    if not fichiers:
        print(f"⚠️ ATTENTION : Le dossier '{dossier_entree}' est vide. Crée-le et ajoutes-y des fichiers audio.")
        return

    print(f"\n--- TRAITEMENT DU PIPELINE : {suffixe.upper()} ---")
    print(f"📁 Dossier source : {dossier_entree}")
    print(f"📄 {len(fichiers)} fichier(s) trouvé(s).")

    # Charge le modèle UNE SEULE FOIS pour ce dossier (optimisation)
    try:
        processor = WhisperProcessor.from_pretrained(NOM_MODELE)
        model = WhisperForConditionalGeneration.from_pretrained(NOM_MODELE)
    except Exception as e:
        print(f"❌ Impossible de charger le modèle Whisper. Vérifie ton installation. Erreur : {e}")
        return

    for i, chemin_audio in enumerate(fichiers, 1):
        nom_fichier = os.path.basename(chemin_audio)
        nom_sans_ext = os.path.splitext(nom_fichier)[0]
        chemin_sortie_txt = os.path.join(dossier_sortie, f"{nom_sans_ext}_{suffixe}.txt")

        try:
            print(f"  → Traitement ({i}/{len(fichiers)}) : {nom_fichier}")

            # --- 1. Chargement de l'audio ---
            # Le paramètre 'sr=16000' est CRUCIAL pour Whisper, qui attend cette fréquence
            audio_data, _ = librosa.load(chemin_audio, sr=16000)

            # Ignorer les fichiers trop courts (par exemple, moins d'1/2 seconde)
            if len(audio_data) < 8000:
                raise ValueError("Fichier audio trop court (moins de 0.5 seconde).")

            # --- 2. Transcription avec Whisper ---
            debut = time.time()
            input_features = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features

            # Le paramètre 'no_speech_threshold' est la solution à ton problème de "Thank you"
            predicted_ids = model.generate(
                input_features,
                task="transcribe",          # Whisper détecte automatiquement la langue (anglais, français, chinois)
                num_beams=4,                # Beam Search pour une meilleure précision
                temperature=0.0,            # Pour des résultats déterministes, sans "inventer"
                no_speech_threshold=0.4,   # 🔥 C'est ce qui empêche les hallucinations !
                max_length=448
            )

            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            fin = time.time()

            # Nettoyage post-traitement : si "Thank you" est court, on le considère comme du silence
            if len(transcription.strip()) < 10 and any(mot in transcription.lower() for mot in ["thank", "thanks"]):
                transcription = "[SILENCE - AUCUNE PAROLE DETECTEE]"

            # --- 3. Sauvegarde des résultats pour l'Étudiant 3 ---
            with open(chemin_sortie_txt, "w", encoding="utf-8") as f:
                f.write(f"--- RAPPORT DE TRANSCRIPTION ---\n")
                f.write(f"Fichier source : {nom_fichier}\n")
                f.write(f"Pipeline appliqué : {suffixe}\n")
                f.write(f"Modèle utilisé : {NOM_MODELE}\n")
                f.write(f"Temps d'inférence : {fin - debut:.2f} secondes\n")
                f.write(f"\n--- TEXTE TRANSCRIT ---\n{transcription}\n")

            print(f"    ✅ Succès ! Transcription sauvegardée dans '{chemin_sortie_txt}'")

        except Exception as e:
            # Crée un fichier d'erreur pour que l'Étudiant 3 sache que le fichier a été ignoré
            with open(chemin_sortie_txt, "w", encoding="utf-8") as f:
                f.write(f"--- RAPPORT D'ERREUR ---\n")
                f.write(f"Fichier source : {nom_fichier}\n")
                f.write(f"Pipeline : {suffixe}\n")
                f.write(f"ERREUR : {e}\n")
            print(f"    ❌ Échec pour {nom_fichier}. Vérifie le fichier. Erreur : {e}")

def main():
    print("=== LANCEMENT DU PIPELINE DE RECHERCHE ASR ===")
    print(f"Recherche des dossiers : {', '.join(DOSSIERS_ENTREE.values())}")

    for suffixe, dossier_entree in DOSSIERS_ENTREE.items():
        if os.path.exists(dossier_entree):
            traiter_dossier(dossier_entree, suffixe)
        else:
            print(f"⚠️ Le dossier '{dossier_entree}' est introuvable. L'étudiant 1 doit le créer.")
            print("   Il doit contenir des fichiers audio pour le pipeline :", suffixe)

    print("\n\n=== PIPELINE TERMINE ===")
    print(f"Tous les résultats ont été sauvegardés dans le dossier : '{DOSSIER_SORTIE_BASE}/'")

if __name__ == "__main__":
    main()
