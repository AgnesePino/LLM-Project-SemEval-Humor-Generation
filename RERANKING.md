# Reranking Pipeline - MWAHAHA Task A EN

Questo documento spiega come vengono ordinati e scelti i candidati generati da `mwahaha_task_a_en.py`.

## Overview

Il reranking non usa una singola metrica automatica ufficiale. La scelta finale cerca di approssimare il criterio della challenge: preferenza umana pairwise tra battute.

La pipeline usa tre livelli:

1. validazione dei vincoli;
2. scoring numerico con LLM judge e classifier humor opzionali;
3. torneo pairwise tra i migliori candidati.

Nel comando `refine` c'e anche un quarto controllo conservativo: il nuovo candidato sostituisce la battuta esistente solo se batte l'incumbent in un confronto pairwise ripetuto.

## 1. Validazione

Prima del rerank vero e proprio, ogni candidato viene validato.

Un candidato viene penalizzato o escluso se:

- e vuoto;
- supera 900 caratteri;
- contiene tab o newline;
- contiene boilerplate tipo `Joke:`, `Here is a joke:`, `Final joke:`;
- per `word_inclusion`, non contiene entrambe le parole richieste verbatim;
- per `news_headline`, non mantiene almeno un legame minimo con la headline.

Solo i candidati validi passano allo scoring.

Funzioni principali:

- `validate_candidate`
- `headline_related`
- `contains_verbatim`

## 2. Score Numerico

Ogni candidato valido viene giudicato da Qwen in modalita judge.

Il prompt chiede uno score da `0` a `10` considerando:

- rispetto dei vincoli;
- brevita;
- sorpresa;
- specificita;
- inglese naturale;
- comicita.

Il modello ritorna un JSON del tipo:

```json
{"score": 7.5, "reason": "short reason"}
```

Questo valore viene salvato come:

```text
judge_score
```

Funzioni principali:

- `score_candidates`
- `build_score_prompt`
- `parse_score`

## 3. Humor Classifier Opzionale

Se vengono passati uno o piu `--humor-model`, lo script calcola anche una probabilita di humor:

```text
humor_score = P(humor)
```

Il valore e tra `0` e `1`, poi viene scalato a `0-10`.

Lo score finale diventa:

```text
final_score = (1 - humor_weight) * judge_score + humor_weight * (10 * humor_score)
```

Con il comando usato nella run originale:

```text
--humor-weight 0.25
```

la formula era:

```text
final_score = 0.75 * judge_score + 0.25 * (10 * humor_score)
```

Con il comando `refine` consigliato:

```text
--humor-weight 0.20
```

la formula diventa:

```text
final_score = 0.80 * judge_score + 0.20 * (10 * humor_score)
```

Esempio:

```text
judge_score = 8.0
humor_score = 0.75
humor_weight = 0.20

final_score = 0.80 * 8.0 + 0.20 * 7.5
final_score = 7.9
```

Quindi i classifier non decidono da soli. Servono solo come segnale ausiliario.

## 4. Ordinamento Dei Candidati

Dopo lo scoring, i candidati validi vengono ordinati per:

```text
candidate.score
```

dal piu alto al piu basso.

I candidati non validi vengono messi in fondo.

Questa e la prima scrematura.

## 5. Torneo Pairwise

Dopo lo score numerico, lo script non prende automaticamente il candidato con score piu alto.

Prende invece i primi `--rerank-top-k` candidati e fa un mini torneo pairwise.

Esempio con:

```text
--rerank-top-k 4
```

Il flusso e:

```text
champion = candidato #1

champion vs candidato #2 -> nuovo champion
champion vs candidato #3 -> nuovo champion
champion vs candidato #4 -> winner finale
```

In questa fase Qwen riceve due battute e deve scegliere quale vincerebbe una human pairwise humor battle.

Il prompt chiede un JSON:

```json
{"winner": "A", "reason": "short reason"}
```

Funzioni principali:

- `pairwise_tournament`
- `judge_pair`

## 6. Differenza Tra `run` E `refine`

### `run`

Nel comando `run`, la battuta finale per ogni ID e il vincitore del torneo pairwise tra i migliori candidati generati da zero.

Schema:

```text
generate candidates
validate
judge score
humor classifier score
sort by final_score
pairwise tournament among top-k
write winner
```

### `refine`

Nel comando `refine`, esiste gia una battuta corrente, chiamata incumbent.

Lo script genera nuovi challenger solo per ID considerati deboli, poi sceglie il migliore challenger con lo stesso sistema:

```text
generate challenger candidates
validate
judge score
humor classifier score
style penalties
sort
pairwise tournament among top-k
best challenger
```

Poi confronta:

```text
incumbent vs best challenger
```

Il challenger sostituisce l'incumbent solo se vince abbastanza voti.

Con:

```text
--pairwise-votes 3
--replace-votes 2
```

serve almeno:

```text
2 vittorie su 3
```

Questo rende il refine conservativo: in caso di dubbio, resta la battuta gia valida.

## 7. Penalita Specifiche Del `refine`

Nel refine viene applicata una penalita stilistica dopo lo score numerico.

La penalita riduce `candidate.score` se il testo contiene pattern che erano troppo frequenti nella submission:

- inizio con `I tried`;
- inizio con `I asked`;
- inizio con `I told`;
- presenza di `turns out`;
- virgolette non necessarie;
- testo troppo lungo;
- frasi generiche tipo `the kind of headline`, `sounds like the news`, `my coffee`, `press secretary`.

Esempio concettuale:

```text
score iniziale = 8.4
penalty per "I tried" = 2.0
penalty per "turns out" = 1.25

score finale refine = 8.4 - 3.25 = 5.15
```

Funzioni principali:

- `refine_style_penalty`
- `apply_refine_style_penalties`

## 8. Cosa Guardare Nei Diagnostics

Nei file JSON di diagnostics trovi per ogni candidato:

```json
{
  "text": "...",
  "style": "...",
  "seed": 123,
  "temperature": 0.92,
  "score": 7.9,
  "judge_score": 8.0,
  "judge_raw": "...",
  "humor_score": 0.75,
  "valid": true,
  "invalid_reason": ""
}
```

Campi piu importanti:

- `score`: score finale usato per ordinare i candidati;
- `judge_score`: voto Qwen da 0 a 10;
- `humor_score`: probabilita di humor del classifier;
- `valid`: se il candidato rispetta i vincoli;
- `invalid_reason`: motivo di esclusione;
- `judge_raw`: spiegazione grezza del judge.

Nei diagnostics del refine trovi anche:

```json
{
  "incumbent": "...",
  "challenger": "...",
  "challenger_votes": 2,
  "vote_trace": ["B", "A", "B"],
  "replaced": true
}
```

Qui:

- `A` = incumbent;
- `B` = challenger;
- `replaced: true` significa che il challenger ha sostituito la battuta precedente.

## Sintesi

La metrica principale non e BLEU, ROUGE o similarita testuale.

La metrica pratica e:

```text
preferenza stimata da un LLM judge in una human pairwise humor battle
```

supportata da:

- validazione dei vincoli;
- classifier humor locali;
- penalita anti-template nel refine;
- confronto conservativo contro l'incumbent.

