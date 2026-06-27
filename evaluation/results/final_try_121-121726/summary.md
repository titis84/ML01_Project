# WER Evaluation Summary

This evaluation uses the 15 simple English samples from `121-121726.trans.txt`.
The first overlapped-dialogue batch still needs manually prepared references before WER can be computed fairly.

## Average WER

| Scenario | Mean WER |
|---|---:|
| brut_with_llm | 0.088 |
| brut_without_llm | 0.088 |
| denoise_with_llm | 0.088 |
| denoise_without_llm | 0.088 |
| vad_with_llm | 0.088 |
| vad_without_llm | 0.088 |

Best scenario on this dataset: `brut_without_llm` with mean WER `0.088`.
