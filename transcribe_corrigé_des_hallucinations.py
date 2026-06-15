from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import time
import os
import glob

NOM_MODELE = "openai/whisper-tiny"

# Dossiers pour les 3 pipelines
DOSSIERS_ENTREE = {
    "brut": "audios_brut",
    "denoise": "audios_denoise",
    "vad": "audios_vad"
}
DOSSIER_SORTIE_BASE = "textes_pour_eleve3"

def traiter_dossier(dossier_entree, suffixe):
    """Traite tous les audios d'un dossier et écrit les transcriptions dans un sous-dossier."""
    dossier_sortie = os.path.join(DOSSIER_SORTIE_BASE, suffixe)
    os.makedirs(dossier_sortie, exist_ok=True)

    extensions = ["*.mp3", "*.wav", "*.m4a", "*.flac"]
    fichiers_audio = []
    for ext in extensions:
        fichiers_audio.extend(glob.glob(os.path.join(dossier_entree, ext)))

    if not fichiers_audio:
        print(f"⚠️ Dossier '{dossier_entree}' vide -> ignoré.")
        return

    print(f"\n📁 Traitement du dossier: {dossier_entree} ({len(fichiers_audio)} fichiers)")
    
    for i, chemin_audio in enumerate(fichiers_audio, 1):
        nom_fichier = os.path.basename(chemin_audio)
        nom_sans_ext = os.path.splitext(nom_fichier)[0]
        fichier_texte = os.path.join(dossier_sortie, f"{nom_sans_ext}_{suffixe}.txt")

        try:
            debut = time.time()
            # Chargement à 16kHz
            audio_data, _ = librosa.load(chemin_audio, sr=16000)
            
            # Vérification : si le fichier est trop court (<0.5s), on ignore
            if len(audio_data) < 8000:  # 0.5s à 16kHz
                with open(fichier_texte, "w", encoding="utf-8") as f:
                    f.write("[TROP COURT - IGNORÉ]")
                print(f"  ⏭️ {nom_fichier}: fichier trop court (<0.5s)")
                continue
                
            processor = WhisperProcessor.from_pretrained(NOM_MODELE)
            model = WhisperForConditionalGeneration.from_pretrained(NOM_MODELE)
            
            input_features = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features
            
            # Génération avec anti-hallucination
            predicted_ids = model.generate(
                input_features,
                task="transcribe",      # auto-détection de la langue (anglais, français, chinois)
                num_beams=4,
                temperature=0.0,
                no_speech_threshold=0.4,   # évite le "Thank you" sur silence
                max_length=448
            )
            
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # Nettoyage post-traitement : si transcription courte = "thank you" ou "thanks", on vide
            if len(transcription.strip()) < 10 and any(word in transcription.lower() for word in ["thank", "thanks"]):
                transcription = "[SILENCE - AUCUNE PAROLE]"
            
            fin = time.time()
            
            # Écriture du fichier avec métadonnées pour l'étudiant 3
            with open(fichier_texte, "w", encoding="utf-8") as f:
                f.write(f"FILE: {nom_fichier}\n")
                f.write(f"PIPELINE: {suffixe}\n")
                f.write(f"MODEL: {NOM_MODELE}\n")
                f.write(f"INFERENCE_TIME: {fin - debut:.2f}s\n")
                f.write(f"TRANSCRIPTION:\n{transcription}\n")
            
            print(f"  ✅ {nom_fichier} -> {fin - debut:.2f}s")
            
        except Exception as e:
            print(f"  ❌ Erreur sur {nom_fichier}: {e}")

def main():
    print("=== Pipeline ASR pour projet ML ===")
    print("Traitement des trois dossiers : brut, denoise, vad")
    
    for suffixe, dossier in DOSSIERS_ENTREE.items():
        if os.path.exists(dossier):
            traiter_dossier(dossier, suffixe)
        else:
            print(f"⚠️ Dossier {dossier} n'existe pas, crée-le avec les fichiers de l'étudiant 1.")
    
    print("\n✅ Terminé ! Tous les résultats sont dans le dossier:", DOSSIER_SORTIE_BASE)

if __name__ == "__main__":
    main()
