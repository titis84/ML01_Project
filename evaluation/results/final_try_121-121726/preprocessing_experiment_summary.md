# Experiment Summary: Effect of Audio Preprocessing on ASR Performance

## 1. Research Objective

The goal of this experiment is to evaluate whether local audio preprocessing can improve the performance of automatic speech recognition (ASR) using Whisper.

We focus on one main research question:

**Does audio preprocessing, such as denoising and voice activity detection, reduce transcription errors compared with using the original audio directly?**

To answer this question, we compare multiple audio processing pipelines and measure their transcription accuracy using Word Error Rate (WER).

## 2. Experimental Design

We evaluate three audio input pipelines:

| Pipeline | Description |
|---|---|
| `brut` | Original untreated audio |
| `denoise` | Audio after denoising |
| `vad` | Audio after denoising and VAD preprocessing |

For each pipeline, we evaluate two transcription versions:

| Version | Description |
|---|---|
| `original` | Raw Whisper transcription |
| `corrige` | Transcription after LLM correction |

This gives six scenarios in total:

| Scenario | Meaning |
|---|---|
| `brut_without_llm` | Original audio with raw Whisper output |
| `denoise_without_llm` | Denoised audio with raw Whisper output |
| `vad_without_llm` | Denoised + VAD audio with raw Whisper output |
| `brut_with_llm` | Original audio with LLM correction |
| `denoise_with_llm` | Denoised audio with LLM correction |
| `vad_with_llm` | Denoised + VAD audio with LLM correction |

## 3. Evaluation Metric

For the English samples with available ground-truth references, we use **Word Error Rate (WER)**.

WER is defined as:

```text
WER = (Substitutions + Deletions + Insertions) / Number of reference words
```

A lower WER means that the ASR output is closer to the human reference transcript.

## 4. Quantitative Results

The quantitative evaluation is based on 15 English audio samples with manually provided reference transcripts.

| Scenario | Mean WER |
|---|---:|
| Original audio, without LLM correction | 0.088 |
| Denoised audio, without LLM correction | 0.088 |
| Denoised + VAD audio, without LLM correction | 0.088 |
| Original audio, with LLM correction | 0.088 |
| Denoised audio, with LLM correction | 0.088 |
| Denoised + VAD audio, with LLM correction | 0.088 |

The average WER is the same for all six scenarios.

## 5. Analysis of Audio Preprocessing

The results show that audio preprocessing did not produce a measurable improvement on this English dataset.

The original audio, denoised audio, and denoised + VAD audio all achieved the same mean WER. This suggests that the original English samples were already clean enough for Whisper to recognize them accurately.

In this situation, denoising does not provide much additional benefit because there is little background noise to remove. Similarly, VAD does not improve recognition because the audio samples are short and do not contain enough silence or irrelevant segments for VAD to make a meaningful difference.

This result shows that preprocessing is not automatically useful in every ASR pipeline. Its value depends strongly on the quality and structure of the input audio.

## 6. Analysis of LLM Correction

LLM correction also did not reduce WER in this experiment.

This is likely because the raw Whisper transcriptions were already close to the reference texts. When the ASR output is already accurate, the LLM has limited room to improve the result.

LLM correction may be more useful for longer, noisier, or less fluent transcriptions, where it can improve readability and fix obvious language-level errors. However, for short and clean English utterances, it did not provide measurable improvement in this experiment.

## 7. Qualitative Observations on Complex Audio

The final transcription package also contains Chinese overlapped speech samples and meeting-like samples. These samples do not currently have human reference transcripts, so WER or CER cannot be computed fairly for them.

However, qualitative inspection shows that these samples are much more difficult for Whisper. In particular, the outputs show:

- recognition errors in multi-speaker conditions;
- hallucination-like content in Chinese transcriptions;
- reduced readability when speakers overlap;
- unstable behavior on meeting-style audio.

These observations suggest that audio preprocessing may be more valuable in complex real-world scenarios than in clean short English samples. However, a fair quantitative evaluation requires manually created reference transcripts.

For Chinese audio, **Character Error Rate (CER)** would be more appropriate than WER because Chinese text does not have explicit word boundaries.

## 8. Limitations

This experiment has several limitations:

- The quantitative WER evaluation is only available for the English samples with reference transcripts.
- The Chinese overlapped and meeting-like samples do not yet have human references.
- The English samples are relatively clean and simple, so they may not fully demonstrate the potential benefit of preprocessing.
- WER measures transcription accuracy, but it does not fully capture readability or speaker separation quality.

## 9. Conclusion

In this experiment, audio preprocessing did not improve Whisper ASR accuracy on the clean English dataset. All six scenarios achieved the same mean WER of 0.088.

The main conclusion is that preprocessing is not a universal solution for improving ASR. For clean and simple audio, Whisper can already perform well without denoising or VAD. In such cases, preprocessing may add complexity without improving recognition accuracy.

However, this does not mean preprocessing is useless. For noisy, long, multi-speaker, or overlapped speech, preprocessing may still be important. The qualitative results on Chinese and meeting-like samples suggest that complex audio remains challenging for Whisper.

Therefore, the value of preprocessing should be evaluated case by case using objective metrics such as WER for English and CER for Chinese.
