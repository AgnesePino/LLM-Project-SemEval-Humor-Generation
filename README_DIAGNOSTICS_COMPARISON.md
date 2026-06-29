# Diagnostics Comparison: Single Ministral vs Ensemble

This README compares the diagnostics in:

- `diagnostics_single_ministral/`
- `diagnostics_ensemble/`

The reading follows the criterion described in `RERANKING.md`: these metrics are not an official challenge metric, but internal proxies for estimating which joke would win a human pairwise preference. The main signals are:

- `judge_score`: LLM judge score from 0 to 10 for constraints, brevity, surprise, specificity, naturalness, and humor;
- `humor_score`: classifier humor probability, used as an auxiliary signal;
- `score`: final combination used to sort candidates before the pairwise tournament;
- `winner`: candidate selected after reranking.

## Summary

The ensemble produces winners that are stronger on average according to the judge, but the gap is not very large. The single Ministral run is competitive on the combined `score` and slightly stronger on the auxiliary humor classifier, while the ensemble has better validity and a clearer advantage on judge-perceived quality.

The most important difference is:

| winner metric | single Ministral | ensemble | ensemble delta |
| --- | ---: | ---: | ---: |
| avg `score` | 7.452 | 7.517 | +0.065 |
| avg `judge_score` | 7.752 | 8.053 | +0.301 |
| avg `humor_score` | 0.627 | 0.537 | -0.090 |
| avg chars | 176.1 | 172.7 | -3.4 |
| median chars | 164.0 | 163.0 | -1.0 |
| invalid winner | 1 | 0 | -1 |

Interpretation: the ensemble gains mostly from judge-perceived quality (`judge_score`), especially specificity, surprise, and connection to the headline. However, the advantage is moderate rather than overwhelming. The single run has a higher average `humor_score`, which suggests that the humor classifier rewards somewhat different traits from the LLM judge.

## Coverage And Validity

Both strategies cover the same 300 inputs:

| metric | single Ministral | ensemble |
| --- | ---: | ---: |
| ids | 300 | 300 |
| news headline | 275 | 275 |
| word inclusion | 25 | 25 |
| total candidates | 1500 | 2697 |
| valid candidates | 1390 | 2578 |
| invalid candidates | 110 | 119 |
| invalid rate | 7.3% | 4.4% |

The single run generates 5 candidates per ID, all from `ministral-3-14b-reasoning`. The ensemble generates almost 9 candidates per ID, using `gemma-12b-qat`, `qwen3-14b`, and `ministral-3-14b-reasoning`.

Although the ensemble has more invalid candidates in absolute terms, it has a lower invalid rate because the pool is larger. It also does not select invalid winners. In the single run, there is one problematic case:

- `en_2287`: invalid winner for `missing_required_words:shake`, with `score = 0.0` and `judge_score = 0.0`.

## Direct Comparison On The 300 Winners

The winners are identical in 67 cases and different in 233 cases.

| direct comparison | count |
| --- | ---: |
| ensemble with higher `score` | 121 |
| single with higher `score` | 118 |
| tied on `score` | 61 |
| ensemble with higher `judge_score` | 98 |
| single with higher `judge_score` | 50 |
| tied on `judge_score` | 152 |
| ensemble shorter | 114 |
| single shorter | 118 |
| tied on length | 68 |

This table captures a tight trade-off. On final `score`, the comparison is almost perfectly balanced, with many ties caused by shared or equivalent winners. On `judge_score`, the ensemble wins more often than the single run, but most cases are tied. Length is also essentially balanced, so compactness is not a clear advantage for either strategy.

## Style Differences

| winner pattern | single Ministral | ensemble |
| --- | ---: | ---: |
| starts with `Why did` | 39 | 16 |
| starts with `I tried` | 15 | 36 |
| starts with `I asked` | 4 | 7 |
| starts with `I told` | 10 | 5 |
| contains `turns out` | 57 | 36 |
| contains quotation marks | 113 | 91 |
| over 200 characters | 84 | 79 |
| over 300 characters | 14 | 6 |

The single Ministral run uses more classic `Why did...` openings than the ensemble, but its dominant style is not only the short question-answer joke. It also produces many narrative or dialogue-like winners, with more `turns out`, more quotation marks, and more very long outputs than the ensemble.

The ensemble is more varied across models and uses more `I tried` and `I asked` openings. Its main advantage is not compactness by a large margin, but a slightly better balance between judge score, validity, and model diversity. Both strategies need style cleanup, especially for repeated openings, quoted dialogue, and overlong jokes.

## Models In The Ensemble

| model | candidates | valid | winners | win share |
| --- | ---: | ---: | ---: | ---: |
| `ministral-3-14b-reasoning` | 900 | 848 | 114 | 38.0% |
| `gemma-12b-qat` | 900 | 873 | 98 | 32.7% |
| `qwen3-14b` | 897 | 857 | 88 | 29.3% |

`ministral-3-14b-reasoning` is the model with the most winners in the ensemble, but the distribution is fairly balanced. This is a good sign: the ensemble is not merely replicating one model, but actually drawing from different styles.

## Pairwise Tournament Behavior

In the single run, the winner is often in the middle or lower part of the sorted candidate pool:

| winner rank in sorted pool | single Ministral |
| --- | ---: |
| 1 | 49 |
| 2 | 38 |
| 3 | 67 |
| 4 | 82 |
| 5 | 64 |

In the ensemble, the winner very often comes from the lower part of the top-k:

| winner rank in sorted pool | ensemble |
| --- | ---: |
| 1 | 15 |
| 2 | 16 |
| 3 | 28 |
| 4 | 39 |
| 5 | 58 |
| 6 | 144 |

This indicates that the pairwise tournament changes the final choice substantially compared with pure numerical sorting. The effect is strongest in the ensemble, where the top-k is larger and the pairwise judge often prefers candidates that do not have the highest initial `score`.

## Practical Reading

The ensemble is the better strategy if the goal is to maximize the proxy closest to human evaluation: the LLM judgment on specificity, surprise, and joke quality. The average `+0.301` advantage on `judge_score` is meaningful, even though the two systems are close on the combined `score`.

The single Ministral run is a strong baseline. It has a higher classifier `humor_score` and is very close on final `score`, but it does not provide a clearly shorter or cleaner alternative. It also keeps one invalid winner and shows stronger exposure to `turns out`, quotation marks, and very long outputs.

The operational recommendation is to use the ensemble as the incumbent, then apply targeted refinement to its style risks and to the overlapping risks visible in the single run:

1. shorten winners over 200 characters;
2. penalize repeated openings such as `I tried`, `I asked`, and `I told`;
3. penalize `turns out` when it does not add a real surprise;
4. check cases with many quotation marks;
5. keep the conservative pairwise comparison before replacing an already valid incumbent.

In short: the ensemble wins on estimated judge quality and validity, while the single run is close on the combined score and stronger on the humor classifier. For the final submission, it is better to start from the ensemble and clean up its style, rather than using the pure single-model run.
