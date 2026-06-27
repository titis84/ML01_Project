# Final Evaluation Notes

## Dataset with references

The quantitative WER evaluation uses the 15 English samples from:

`evaluation_input/121-121726.trans.txt`

These are the only samples in the current package with available ground-truth references.

## Final WER result

| Scenario | Mean WER |
|---|---:|
| brut without LLM | 0.088 |
| denoise without LLM | 0.088 |
| VAD without LLM | 0.088 |
| brut with LLM | 0.088 |
| denoise with LLM | 0.088 |
| VAD with LLM | 0.088 |

The final version is stable compared with the previous latest version. It keeps the improved WER of 0.088, while the earlier second try had a mean WER of 0.244.

## Interpretation

The parameter changes improved Whisper performance on the simple English dataset. In particular, the short utterances that were previously detected as silence are now transcribed correctly.

However, the six scenarios have exactly the same WER on this dataset. This means that, for clean and simple English speech, denoising, VAD, and LLM correction do not provide measurable additional improvement.

## Samples without references

The package also contains overlapped Chinese samples and two additional `R800...` samples. These files do not currently have human reference transcripts, so WER/CER cannot be computed fairly for them yet.

They can still be used for qualitative analysis. The Chinese/multi-speaker outputs show clear recognition limitations and hallucination-like behavior, which is useful to discuss in the report as a limitation of Whisper in difficult multi-speaker conditions.

## Next step

For quantitative evaluation of the Chinese overlapped or meeting-like samples, create manual reference transcripts first. For Chinese, CER is more appropriate than WER because Chinese text has no explicit word boundaries.
