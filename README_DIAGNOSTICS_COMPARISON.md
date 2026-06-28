# Confronto diagnostics: single Ministral vs ensemble

Questo README confronta i diagnostics in:

- `diagnostics_single_ministral/`
- `diagnostics_ensemble/`

La lettura segue il criterio descritto in `RERANKING.md`: le metriche non sono una metrica ufficiale della challenge, ma proxy interni per stimare quale battuta vincerebbe una preferenza umana pairwise. I segnali principali sono:

- `judge_score`: voto LLM judge da 0 a 10 su vincoli, brevita, sorpresa, specificita, naturalezza e comicita;
- `humor_score`: probabilita di humor del classifier, usata come segnale ausiliario;
- `score`: combinazione finale usata per ordinare i candidati prima del torneo pairwise;
- `winner`: candidato selezionato dopo il reranking.

## Sintesi

L'ensemble produce winner mediamente piu forti per il judge, ma anche piu lunghi e piu esposti a pattern stilistici ripetuti. Il single Ministral e piu compatto e usa molto piu spesso formule classiche tipo `Why did...`, ma ha meno varieta e in un caso seleziona un winner non valido.

La differenza piu importante e questa:

| metrica winner | single Ministral | ensemble | delta ensemble |
| --- | ---: | ---: | ---: |
| avg `score` | 7.303 | 7.517 | +0.214 |
| avg `judge_score` | 7.250 | 8.053 | +0.802 |
| avg `humor_score` | 0.754 | 0.537 | -0.214 |
| avg chars | 145.7 | 172.7 | +27.0 |
| median chars | 133.0 | 163.0 | +30.0 |
| invalid winner | 1 | 0 | -1 |

Interpretazione: l'ensemble guadagna soprattutto per qualita percepita dal judge (`judge_score`), non per il classifier humor. Anzi, il suo `humor_score` medio e piu basso. Questo succede perche il judge premia spesso specificita, sorpresa e legame con la headline, mentre il classifier sembra favorire piu facilmente forme brevi e convenzionali.

## Copertura e validita

Entrambe le strategie coprono gli stessi 300 input:

| metrica | single Ministral | ensemble |
| --- | ---: | ---: |
| ids | 300 | 300 |
| news headline | 275 | 275 |
| word inclusion | 25 | 25 |
| candidati totali | 1500 | 2697 |
| candidati validi | 1393 | 2578 |
| candidati invalidi | 107 | 119 |
| invalid rate | 7.1% | 4.4% |

Il single genera 5 candidati per id, tutti da `mistral-7b-instruct-v03`. L'ensemble genera quasi 9 candidati per id, usando `gemma-12b-qat`, `qwen3-14b` e `ministral-3-14b-reasoning`.

Anche se l'ensemble ha piu invalidi in valore assoluto, ha un tasso di invalidita piu basso perche il pool e piu grande. Inoltre non seleziona winner invalidi. Nel single c'e un caso problematico:

- `en_2287`: winner invalido per `missing_required_words:clothe`, con `score = 0.0` e `judge_score = 0.0`.

## Confronto diretto sui 300 winner

I winner sono diversi in tutti i 300 casi: non c'e nessun testo finale identico tra le due strategie.

| confronto diretto | count |
| --- | ---: |
| ensemble con `score` piu alto | 157 |
| single con `score` piu alto | 143 |
| ensemble con `judge_score` piu alto | 184 |
| single con `judge_score` piu alto | 32 |
| pari su `judge_score` | 84 |
| ensemble piu corto | 96 |
| single piu corto | 203 |

Questa tabella spiega bene il tradeoff. Sullo `score` finale la gara e quasi pari, perche il `humor_score` piu basso dell'ensemble compensa parte del vantaggio del judge. Sul `judge_score`, invece, l'ensemble vince nettamente.

## Differenze stilistiche

| pattern sui winner | single Ministral | ensemble |
| --- | ---: | ---: |
| inizia con `Why did` | 127 | 16 |
| inizia con `I tried` | 0 | 36 |
| inizia con `I asked` | 0 | 7 |
| inizia con `I told` | 0 | 5 |
| contiene `turns out` | 6 | 36 |
| contiene virgolette | 74 | 145 |
| oltre 200 caratteri | 34 | 79 |
| oltre 300 caratteri | 7 | 6 |

Il single Ministral tende a produrre battute piu corte e strutture da joke tradizionale. Questo rende l'output piu pulito e prevedibile, ma anche meno specifico sulle headline.

L'ensemble e piu vario nei modelli e spesso piu specifico, pero introduce piu frasi narrative, piu virgolette, piu `turns out` e molte piu battute sopra i 200 caratteri. Questi pattern sono rischiosi per la preferenza umana pairwise, perche possono sembrare template o troppo verbosi.

## Modelli nell'ensemble

| modello | candidati | validi | winner | win share |
| --- | ---: | ---: | ---: | ---: |
| `ministral-3-14b-reasoning` | 900 | 848 | 114 | 38.0% |
| `gemma-12b-qat` | 900 | 873 | 98 | 32.7% |
| `qwen3-14b` | 897 | 857 | 88 | 29.3% |

`ministral-3-14b-reasoning` e il modello con piu winner nell'ensemble, ma la distribuzione e abbastanza bilanciata. Questo e un buon segnale: l'ensemble non sta solo replicando un singolo modello, ma sta davvero pescando da stili diversi.

## Comportamento del torneo pairwise

Nel single, il winner e spesso tra i primi candidati ordinati per `score`:

| rank winner nel pool ordinato | single Ministral |
| --- | ---: |
| 1 | 76 |
| 2 | 49 |
| 3 | 57 |
| 4 | 54 |
| 5 | 64 |

Nell'ensemble, invece, il winner arriva molto spesso dalla parte bassa della top-k:

| rank winner nel pool ordinato | ensemble |
| --- | ---: |
| 1 | 15 |
| 2 | 16 |
| 3 | 28 |
| 4 | 39 |
| 5 | 58 |
| 6 | 144 |

Questo indica che il torneo pairwise sta cambiando parecchio la scelta finale rispetto al puro ordinamento numerico. Nell'ensemble la top-k e piu competitiva e il judge pairwise sembra preferire candidati che non sempre hanno il massimo `score` iniziale.

## Lettura pratica

L'ensemble e la strategia migliore se l'obiettivo e massimizzare il proxy piu vicino alla valutazione umana, cioe il giudizio LLM su specificita, sorpresa e qualita della battuta. Il vantaggio medio di `+0.802` su `judge_score` e sostanziale.

Il single Ministral resta utile come baseline conservativa: e piu breve, piu uniforme e meno soggetto a pattern narrativi lunghi. Pero perde nettamente sul judge e ha meno diversita di candidati, quindi ha meno occasioni di trovare battute specifiche e sorprendenti.

La raccomandazione operativa e usare l'ensemble come incumbent, ma applicare un refine mirato sui suoi rischi stilistici:

1. ridurre winner oltre 200 caratteri;
2. penalizzare aperture ripetute come `I tried`, `I asked`, `I told`;
3. penalizzare `turns out` quando non aggiunge una vera sorpresa;
4. controllare i casi con molte virgolette;
5. mantenere il confronto pairwise conservativo prima di sostituire un incumbent gia valido.

In breve: l'ensemble vince sulla qualita stimata, il single vince sulla compattezza. Per la submission finale conviene partire dall'ensemble e fare una pulizia stilistica, non tornare al single puro.
