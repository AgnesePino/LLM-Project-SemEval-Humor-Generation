# MWAHAHA Task A EN - Ensemble Report

Generated: 2026-06-28 16:25:57

## Executive Summary

- Submission: `submission/task-a-en.ensemble.tsv`
- Input rows: `300`
- Output rows: `300`
- Diagnostics files: `300`
- Formal validation: `OK`
- Winner matches missing in diagnostics: `0`

The Gemma rerun is now active in the ensemble: Gemma candidates are non-empty and appear in the final diagnostics/winners. The ranking still uses the local judge score plus optional humor classifier score, so the numbers below are a proxy for likely pairwise preference, not an official leaderboard metric.

## Candidate Pool Health

| model | records | ids | valid | invalid | avg valid chars | median valid chars |
| --- | --- | --- | --- | --- | --- | --- |
| gemma-12b-qat | 900 | 300 | 873 | 27 | 155.5 | 154 |
| ministral-3-14b-reasoning | 900 | 300 | 848 | 52 | 144.2 | 134.0 |
| qwen3-14b | 900 | 300 | 860 | 40 | 135.8 | 133.0 |

### Invalid Candidate Reasons

| model | invalid | reasons |
| --- | --- | --- |
| gemma-12b-qat | 27 | weak_headline_overlap: 21, missing_required_words:blend: 2, missing_required_words:hammer: 1, missing_required_words:drill: 1, missing_required_words:fridge: 1, missing_required_words:pepper: 1 |
| ministral-3-14b-reasoning | 52 | weak_headline_overlap: 32, missing_required_words:measure: 5, missing_required_words:spray: 4, missing_required_words:pepper: 2, missing_required_words:clothe: 2, missing_required_words:blend: 2, missing_required_words:hammer: 1, missing_required_words:drill: 1, missing_required_words:roll: 1, missing_required_words:shake: 1, missing_required_words:bicycle: 1 |
| qwen3-14b | 40 | weak_headline_overlap: 36, missing_required_words:spray: 2, missing_required_words:egg: 1, missing_required_words:shake: 1 |

## Model Win Statistics

| model | pool records | pool valid | evaluated candidates | wins | win share | avg candidate score | avg winner score | avg winner judge | avg winner humor | avg winner chars |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma-12b-qat | 900 | 873 | 900 | 98 | 32.7% | 6.862 | 7.345 | 8.212 | 0.387 | 175.1 |
| ministral-3-14b-reasoning | 900 | 848 | 900 | 114 | 38.0% | 7.167 | 7.697 | 7.970 | 0.660 | 188.8 |
| qwen3-14b | 900 | 860 | 897 | 88 | 29.3% | 7.163 | 7.476 | 7.982 | 0.545 | 149.2 |

### Wins By Task Kind

| model | news_headline wins | word_inclusion wins |
| --- | --- | --- |
| gemma-12b-qat | 94 | 4 |
| ministral-3-14b-reasoning | 105 | 9 |
| qwen3-14b | 76 | 12 |

### Head-To-Head Best Candidate Score

For each item, this compares the best scored candidate from each model before the final winner is selected.

| model A | model B | A wins | B wins | ties | A rate | B rate |
| --- | --- | --- | --- | --- | --- | --- |
| gemma-12b-qat | ministral-3-14b-reasoning | 86 | 214 | 0 | 28.7% | 71.3% |
| gemma-12b-qat | qwen3-14b | 93 | 207 | 0 | 31.0% | 69.0% |
| ministral-3-14b-reasoning | qwen3-14b | 190 | 110 | 0 | 63.3% | 36.7% |

## Ranking Behavior

### Winner Rank Before Final Selection

| rank among scored candidates | count | share |
| --- | --- | --- |
| 1 | 15 | 5.0% |
| 2 | 16 | 5.3% |
| 3 | 28 | 9.3% |
| 4 | 39 | 13.0% |
| 5 | 58 | 19.3% |
| 6 | 144 | 48.0% |

### Winning Styles

| model | style | wins |
| --- | --- | --- |
| ministral-3-14b-reasoning | absurd literalism | 58 |
| gemma-12b-qat | reversal | 41 |
| gemma-12b-qat | absurd literalism | 39 |
| qwen3-14b | absurd literalism | 39 |
| ministral-3-14b-reasoning | reversal | 35 |
| qwen3-14b | reversal | 25 |
| qwen3-14b | wordplay | 24 |
| ministral-3-14b-reasoning | wordplay | 21 |
| gemma-12b-qat | wordplay | 18 |

## Output Style Metrics

| pattern | count | share |
| --- | --- | --- |
| I tried | 39 | 13.0% |
| I asked | 7 | 2.3% |
| I told | 9 | 3.0% |
| turns out | 36 | 12.0% |
| quote marks | 105 | 35.0% |
| >200 chars | 79 | 26.3% |
| >300 chars | 6 | 2.0% |
| Joke: | 1 | 0.3% |

### Output Style By Winning Model

| model | wins | turns out | quote marks | >200 chars | avg chars |
| --- | --- | --- | --- | --- | --- |
| gemma-12b-qat | 98 | 3 | 25 | 20 | 175.1 |
| ministral-3-14b-reasoning | 114 | 24 | 57 | 45 | 188.8 |
| qwen3-14b | 88 | 9 | 23 | 14 | 149.2 |

### Most Common Openings

| opening | count |
| --- | --- |
| I tried to | 29 |
| Why did the | 10 |
| I told my | 5 |
| I asked the | 3 |
| I asked my | 2 |
| Why did I | 2 |
| The school board | 2 |
| The government is | 2 |
| I followed the | 2 |
| They said the | 2 |
| I went to | 2 |
| China was so | 1 |
| Liverpool's manager said | 1 |
| Experts warn that | 1 |
| At Dick Cheney's | 1 |

## Judge Reason Signals

Most frequent meaningful words in judge reasons for winning candidates. This is a coarse explanation of what the judge rewarded.

| reason token | count |
| --- | --- |
| surprising | 222 |
| specific | 219 |
| twist | 170 |
| concise | 161 |
| clever | 106 |
| headline | 72 |
| creative | 60 |
| natural | 58 |
| ties | 52 |
| wordplay | 46 |
| somewhat | 44 |
| context | 38 |
| specificity | 38 |
| forced | 36 |
| lacks | 36 |
| funny | 35 |
| punchline | 35 |
| surprise | 33 |
| reference | 31 |
| feels | 28 |

## Best Winners

| model | id | score | judge | humor | chars | text preview |
| --- | --- | --- | --- | --- | --- | --- |
| gemma-12b-qat | en_2288 | 8.791 | 8.50 | 0.996 | 154 | The barber asked if I wanted a trim, but I told him I couldn't let him measure my hair because I'm currently undercover  |
| gemma-12b-qat | en_2287 | 8.789 | 8.50 | 0.994 | 133 | I tried to put a new clothe on the horse to make it look fancy, but every time I tried to shake it, it just turned into  |
| gemma-12b-qat | en_2244 | 8.785 | 8.50 | 0.993 | 130 | I asked the grocery store why my bill was so high, and they said it’s because the prices are on a roll—specifically, a b |
| gemma-12b-qat | en_2239 | 8.776 | 8.50 | 0.988 | 191 | I spent years worrying about the secret list of banned words on the internet, only to realize the real secret is that th |
| gemma-12b-qat | en_2022 | 8.770 | 8.50 | 0.985 | 152 | I asked my therapist if I was cut out for non-monogamy, and she said, "Well, since you're technically only one person, y |
| ministral-3-14b-reasoning | en_2286 | 8.794 | 8.50 | 0.997 | 108 | Why did the banana roll out of my cart during the grocery store race? Because it knew I had a *peel* to win! |
| ministral-3-14b-reasoning | en_2281 | 8.793 | 8.50 | 0.997 | 164 | Why did my drill laptop explode? Because I tried to power it with a battery that had *drilled* through its own case—turn |
| ministral-3-14b-reasoning | en_2058 | 8.792 | 8.50 | 0.996 | 203 | The VPN sales were so aggressive during Black Friday that they accidentally routed all my Thanksgiving leftovers through |
| ministral-3-14b-reasoning | en_2283 | 8.789 | 8.50 | 0.995 | 96 | I went to spray my pumpkin with pesticide, but then I realized—it was already a pumpkin *spray*! |
| ministral-3-14b-reasoning | en_2278 | 8.786 | 8.50 | 0.993 | 173 | I was building a bookshelf with my laptop and hammer when suddenly it started singing opera—turns out, I'd accidentally  |
| qwen3-14b | en_2282 | 8.784 | 8.50 | 0.992 | 108 | I tried to spray paint my banana, but it slipped and went down the drain — now I've got a peel of a problem. |
| qwen3-14b | en_2081 | 8.784 | 8.50 | 0.992 | 178 | I tried eating like a pro in Kentucky — now my teeth are permanently stained, my pants smell like a bourbon barrel, and  |
| qwen3-14b | en_2259 | 8.782 | 8.50 | 0.991 | 171 | I tried following a recipe for "everyday cooking," but now my kitchen looks like a special occasion — complete with a po |
| qwen3-14b | en_2059 | 8.781 | 8.50 | 0.991 | 245 | I tried to have a slumber party with my friend, but he insisted on bringing his camping gear, a tent, and a survival gui |
| qwen3-14b | en_2213 | 8.779 | 8.50 | 0.990 | 162 | I tried to pop the AI bubble with a pin, but apparently Nvidia’s earnings were so strong, the bubble just handed me a st |

## Weakest Winners

| model | id | score | judge | humor | chars | text preview |
| --- | --- | --- | --- | --- | --- | --- |
| gemma-12b-qat | en_2154 | 5.792 | 6.50 | 0.296 | 216 | The world's most influential philanthropy is changing its name to "The Charity of Getting Richer" because they finally r |
| gemma-12b-qat | en_2128 | 5.955 | 6.50 | 0.378 | 168 | Heimir Hallgrímsson spent all morning carefully assessing the World Cup play-off draw, only to realize he’d accidentally |
| gemma-12b-qat | en_2044 | 6.035 | 7.50 | 0.018 | 175 | The school board announced that since the thundersnow would be closing all classes, the children were officially allowed |
| gemma-12b-qat | en_2024 | 6.040 | 7.50 | 0.020 | 185 | The government finally admitted their response was too little, too late, so they issued a formal apology and a small ban |
| gemma-12b-qat | en_2219 | 6.066 | 7.50 | 0.033 | 193 | The Amazonian tribe held a sacred food ritual where they offered a feast to the forest spirits, but the ritual was ruine |
| ministral-3-14b-reasoning | en_2030 | 5.402 | 6.50 | 0.101 | 251 | House Speaker Mike Johnson dramatically avoided a censure vote by hiding behind a giant poster that read 'You're a disgr |
| ministral-3-14b-reasoning | en_2122 | 5.738 | 6.50 | 0.269 | 151 | Marseille’s new luxury restaurant charges €1 per meal, but beware—you’ll get so full of fine dining, you’ll forget what  |
| ministral-3-14b-reasoning | en_2019 | 6.119 | 7.50 | 0.060 | 137 | A hurling match in Hanoi got so intense, a local vendor started selling "emergency earrings" for players who needed quic |
| ministral-3-14b-reasoning | en_2091 | 6.143 | 6.50 | 0.472 | 159 | Queen Elizabeth I wasn't hated—she was just really good at hiding the fact that she loved her dog more than her people,  |
| ministral-3-14b-reasoning | en_2179 | 6.333 | 7.50 | 0.167 | 208 | The market was finally calmed down after the AI chipmaker’s strong results—just like a toddler after eating their brocco |
| qwen3-14b | en_2103 | 5.819 | 6.50 | 0.309 | 144 | They’re building houses with no walls—just open spaces and a sign that says “We’ve met all labor standards, but we’re st |
| qwen3-14b | en_2231 | 5.900 | 6.50 | 0.350 | 148 | The town of Affordably was thrilled to be named Britain’s most affordable—until they realized it was a typo and everyone |
| qwen3-14b | en_2236 | 5.909 | 6.50 | 0.354 | 127 | Joshua finally answered the difficult Paul question after realizing he was fighting a literal Paul shaped like a questio |
| qwen3-14b | en_2255 | 6.046 | 7.50 | 0.023 | 120 | The teams were thrilled until they realized the World Cup play-off was just a game of Twister with the UK as the center. |
| qwen3-14b | en_2141 | 6.119 | 7.50 | 0.060 | 180 | Labour MPs tried to take a comfort break, but the approach was too literal — they were forced to sit on a couch labeled  |

## Interpretation

- The strongest generator by final wins is `ministral-3-14b-reasoning` with `114` wins.
- `win share` is the most useful model-level metric here because every model generated the same number of candidates per item.
- `avg winner judge` reflects how much the Qwen judge liked the selected jokes on compliance, specificity, surprise and concision.
- `avg winner humor` is only a weak auxiliary signal: it helps filter obviously non-humorous text, but it should not be treated as the competition objective.
- Long outputs, quote-heavy jokes and repeated patterns remain the main stylistic risk because human pairwise preference usually rewards compact, specific punchlines.

## Recommended Next Steps

1. Run a targeted refinement pass on winners over 200 characters, quote-heavy winners, and repeated `turns out` jokes.
2. Keep the current ensemble as incumbent and replace only when the challenger wins a conservative pairwise vote.
3. Re-run this report after refinement and compare model win shares plus style metrics, especially `>200 chars`, quote marks and `turns out`.
4. Manually inspect the weakest winners table before final submission; these are the rows with the highest expected ROI.

