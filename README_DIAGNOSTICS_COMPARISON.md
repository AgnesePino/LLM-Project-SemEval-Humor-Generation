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

The ensemble produces winners that are stronger on average according to the judge, but also longer and more exposed to repeated style patterns. The single Ministral run is more compact and much more likely to use classic joke formulas such as `Why did...`, but it has less variety and selects one invalid winner.

The most important difference is:

| winner metric | single Ministral | ensemble | ensemble delta |
| --- | ---: | ---: | ---: |
| avg `score` | 7.303 | 7.517 | +0.214 |
| avg `judge_score` | 7.250 | 8.053 | +0.802 |
| avg `humor_score` | 0.754 | 0.537 | -0.214 |
| avg chars | 145.7 | 172.7 | +27.0 |
| median chars | 133.0 | 163.0 | +30.0 |
| invalid winner | 1 | 0 | -1 |

Interpretation: the ensemble gains mostly from judge-perceived quality (`judge_score`), not from the humor classifier. In fact, its average `humor_score` is lower. This happens because the judge often rewards specificity, surprise, and connection to the headline, while the classifier seems to favor shorter and more conventional forms.

## Coverage And Validity

Both strategies cover the same 300 inputs:

| metric | single Ministral | ensemble |
| --- | ---: | ---: |
| ids | 300 | 300 |
| news headline | 275 | 275 |
| word inclusion | 25 | 25 |
| total candidates | 1500 | 2697 |
| valid candidates | 1393 | 2578 |
| invalid candidates | 107 | 119 |
| invalid rate | 7.1% | 4.4% |

The single run generates 5 candidates per ID, all from `ministral-3-14b-reasoning`. The ensemble generates almost 9 candidates per ID, using `gemma-12b-qat`, `qwen3-14b`, and `ministral-3-14b-reasoning`.

Although the ensemble has more invalid candidates in absolute terms, it has a lower invalid rate because the pool is larger. It also does not select invalid winners. In the single run, there is one problematic case:

- `en_2287`: invalid winner for `missing_required_words:clothe`, with `score = 0.0` and `judge_score = 0.0`.

## Direct Comparison On The 300 Winners

The winners are different in all 300 cases: there is no identical final text between the two strategies.

| direct comparison | count |
| --- | ---: |
| ensemble with higher `score` | 157 |
| single with higher `score` | 143 |
| ensemble with higher `judge_score` | 184 |
| single with higher `judge_score` | 32 |
| tied on `judge_score` | 84 |
| ensemble shorter | 96 |
| single shorter | 203 |

This table captures the trade-off. On final `score`, the comparison is almost even because the ensemble's lower `humor_score` offsets part of its judge advantage. On `judge_score`, however, the ensemble wins clearly.

## Style Differences

| winner pattern | single Ministral | ensemble |
| --- | ---: | ---: |
| starts with `Why did` | 127 | 16 |
| starts with `I tried` | 0 | 36 |
| starts with `I asked` | 0 | 7 |
| starts with `I told` | 0 | 5 |
| contains `turns out` | 6 | 36 |
| contains quotation marks | 74 | 145 |
| over 200 characters | 34 | 79 |
| over 300 characters | 7 | 6 |

The single Ministral run tends to produce shorter jokes and traditional joke structures. This makes the output cleaner and more predictable, but also less specific to the headlines.

The ensemble is more varied across models and often more specific, but it introduces more narrative phrasing, more quotation marks, more `turns out`, and many more jokes above 200 characters. These patterns are risky for human pairwise preference because they can feel templated or too verbose.

## Models In The Ensemble

| model | candidates | valid | winners | win share |
| --- | ---: | ---: | ---: | ---: |
| `ministral-3-14b-reasoning` | 900 | 848 | 114 | 38.0% |
| `gemma-12b-qat` | 900 | 873 | 98 | 32.7% |
| `qwen3-14b` | 897 | 857 | 88 | 29.3% |

`ministral-3-14b-reasoning` is the model with the most winners in the ensemble, but the distribution is fairly balanced. This is a good sign: the ensemble is not merely replicating one model, but actually drawing from different styles.

## Pairwise Tournament Behavior

In the single run, the winner is often among the first candidates sorted by `score`:

| winner rank in sorted pool | single Ministral |
| --- | ---: |
| 1 | 76 |
| 2 | 49 |
| 3 | 57 |
| 4 | 54 |
| 5 | 64 |

In the ensemble, however, the winner very often comes from the lower part of the top-k:

| winner rank in sorted pool | ensemble |
| --- | ---: |
| 1 | 15 |
| 2 | 16 |
| 3 | 28 |
| 4 | 39 |
| 5 | 58 |
| 6 | 144 |

This indicates that the pairwise tournament changes the final choice substantially compared with pure numerical sorting. In the ensemble, the top-k is more competitive, and the pairwise judge seems to prefer candidates that do not always have the highest initial `score`.

## Practical Reading

The ensemble is the better strategy if the goal is to maximize the proxy closest to human evaluation: the LLM judgment on specificity, surprise, and joke quality. The average `+0.802` advantage on `judge_score` is substantial.

The single Ministral run remains useful as a conservative baseline: it is shorter, more uniform, and less exposed to long narrative patterns. However, it loses clearly on judge score and has less candidate diversity, so it has fewer opportunities to find specific and surprising jokes.

The operational recommendation is to use the ensemble as the incumbent, then apply targeted refinement to its style risks:

1. shorten winners over 200 characters;
2. penalize repeated openings such as `I tried`, `I asked`, and `I told`;
3. penalize `turns out` when it does not add a real surprise;
4. check cases with many quotation marks;
5. keep the conservative pairwise comparison before replacing an already valid incumbent.

In short: the ensemble wins on estimated quality, while the single run wins on compactness. For the final submission, it is better to start from the ensemble and clean up its style, rather than returning to the pure single-model run.
