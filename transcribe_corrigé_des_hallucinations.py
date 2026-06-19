#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import time
import os
import glob

# ============================================================
# FONCTION DE CORRECTION PAR LLM (GEMINI)
# ============================================================

def corriger_avec_gemini(texte_a_corriger):
    """
    Envoie la transcription à Gemini pour qu'elle corrige les erreurs.
    """
    try:
        from google import genai
    except ImportError:
        print("⚠️ La librairie 'google-genai' n'est pas installée.")
        print("   Pour l'installer : pip install google-genai")
        return texte_a_corriger

    # 🔑 REMPLACE par ta clé API Gemini (obtenue sur Google AI Studio)
    import os
    API_KEY = os.environ.get("GEMINI_API_KEY", "TA_CLE_API_GEMINI")
    if not API_KEY or API_KEY == "TA_CLE_API_GEMINI":
        print("⚠️ Clé API Gemini manquante. Correction ignorée.")
        return texte_a_corriger

    # Construction de la demande pour l'IA
    prompt = f"""
    Corrige cette transcription automatique.

    Règles :
    1. Ne corrige que les erreurs évidentes (mots mal coupés, fautes, homophones).
    2. Ne réécris pas des phrases entières si elles sont correctes.
    3. Ne change pas le sens, n'ajoute pas d'informations.
    4. Si la transcription est déjà parfaite, ne la change pas.

    Transcription à corriger :
    ---
    {texte_a_corriger}
    ---

    RÉPONDS UNIQUEMENT AVEC LE TEXTE CORRIGÉ, RIEN D'AUTRE.
    """

    try:
        client = genai.Client(api_key=API_KEY)
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )
        texte_corrige = response.text

        if "---" in texte_corrige:
            texte_corrige = texte_corrige.split("---")[-1]
        return texte_corrige.strip()

    except Exception as e:
        print(f"❌ Erreur avec Gemini : {e}")
        return texte_a_corriger


# ============================================================
# CONFIGURATION
# ============================================================

NOM_MODELE = "openai/whisper-tiny"

DOSSIERS_ENTREE = {
    "brut": "audios_brut",
    "denoise": "audios_denoise",
    "vad": "audios_vad"
}
DOSSIER_SORTIE_BASE = "resultats_transcriptions"


# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def traiter_dossier(dossier_entree, suffixe):
    """Transcrit TOUS les fichiers audio d'un dossier."""
    dossier_sortie = os.path.join(DOSSIER_SORTIE_BASE, suffixe)
    os.makedirs(dossier_sortie, exist_ok=True)

    extensions = ["*.wav", "*.mp3", "*.m4a", "*.flac"]
    fichiers = []
    for ext in extensions:
        fichiers.extend(glob.glob(os.path.join(dossier_entree, ext)))

    if not fichiers:
        print(f"⚠️ Le dossier '{dossier_entree}' est vide. Ignoré.")
        return

    print(f"\n📁 Traitement du pipeline : {suffixe.upper()}")
    print(f"   Dossier : {dossier_entree} ({len(fichiers)} fichier(s))")

    # Charger le modèle une seule fois
    processor = WhisperProcessor.from_pretrained(NOM_MODELE)
    model = WhisperForConditionalGeneration.from_pretrained(NOM_MODELE)

    for i, chemin_audio in enumerate(fichiers, 1):
        nom_fichier = os.path.basename(chemin_audio)
        nom_sans_ext = os.path.splitext(nom_fichier)[0]

        try:
            print(f"  [{i}/{len(fichiers)}] {nom_fichier}...")

            # Chargement audio
            audio_data, _ = librosa.load(chemin_audio, sr=16000)

            if len(audio_data) < 8000:
                raise ValueError("Fichier trop court (<0.5s)")

            # Transcription Whisper
            debut = time.time()
            input_features = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features

            predicted_ids = model.generate(
                input_features,
                max_length=448,
                #return_dict_in_generate=True
            )

            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            fin = time.time() 

            # Remplace ton anti-hallucination actuel par :
            mots_hallucination = ["thank", "thanks", "万税", "ご視聴", "subtitles"]
            if len(transcription.strip()) < 20 or any(mot in transcription for mot in mots_hallucination):
                transcription = "[SILENCE - AUCUNE PAROLE DETECTEE]"


            # --- Sauvegarde ORIGINALE ---
            chemin_original = os.path.join(dossier_sortie, f"{nom_sans_ext}_{suffixe}_original.txt")
            with open(chemin_original, "w", encoding="utf-8") as f:
                f.write(f"--- TRANSCRIPTION ORIGINALE (Whisper) ---\n")
                f.write(f"Pipeline : {suffixe}\n")
                f.write(f"Temps d'inférence : {fin - debut:.2f} secondes\n")
                f.write(f"\n--- TEXTE ---\n{transcription}\n")
            print(f"    ✅ Original sauvegardé")

            # --- Sauvegarde CORRIGÉE (Gemini) ---
            print(f"    🤖 Correction par Gemini...")
            transcription_corrigee = corriger_avec_gemini(transcription)

            chemin_corrige = os.path.join(dossier_sortie, f"{nom_sans_ext}_{suffixe}_corrige.txt")
            with open(chemin_corrige, "w", encoding="utf-8") as f:
                f.write(f"--- TRANSCRIPTION CORRIGÉE (Gemini) ---\n")
                f.write(f"Pipeline : {suffixe}\n")
                f.write(f"\n--- TEXTE CORRIGÉ ---\n{transcription_corrigee}\n")
            print(f"    ✅ Corrigé sauvegardé")

        except Exception as e:
            print(f"    ❌ Erreur : {e}")

def main():
    print("=== PIPELINE ASR AVEC CORRECTION GEMINI ===")
    for suffixe, dossier in DOSSIERS_ENTREE.items():
        if os.path.exists(dossier):
            traiter_dossier(dossier, suffixe)
        else:
            print(f"⚠️ Dossier '{dossier}' introuvable. Crée-le !")
    print("\n✅ Terminé ! Résultats dans :", DOSSIER_SORTIE_BASE)

if __name__ == "__main__":
    main()
