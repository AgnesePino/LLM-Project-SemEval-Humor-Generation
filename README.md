# MWAHAHA Task A English Local Pipeline

Pipeline locale per generare, validare, rifinire e impacchettare la submission `task-a-en.tsv` per il Task A English di MWAHAHA.

Il flusso e pensato per hardware locale con RTX 4070 12GB VRAM e 32GB RAM, usando un modello open-source servito da LM Studio, llama.cpp server o Ollama.

## Stato Del Progetto

File principali:

| File o directory | Ruolo |
|---|---|
| `mwahaha_task_a_en.py` | Script principale con comandi `run`, `validate`, `evaluate`, `refine` |
| `data/input/task-a-en.tsv` | Input test set Task A English |
| `submission/task-a-en.tsv` | Submission corrente validata |
| `submission/task-a-en.refined.tsv` | Submission rifinita, se generata |
| `submission.zip` | Archivio finale da caricare su CodaBench |
| `diagnostics/` | Diagnostics della run principale |
| `refine_diagnostics/` | Diagnostics della run di refinement |
| `refine_report.json` | Report aggregato del refinement |
| `REPORT.md` | Report locale sulla submission |
| `RERANKING.md` | Spiegazione dettagliata del reranking |

Output richiesto dalla challenge:

```text
id	text
en_2001	...
en_2002	...
```

Il file finale deve chiamarsi `task-a-en.tsv` dentro lo ZIP.

## Requisiti

- Python 3.10+.
- Un server LLM locale:
  - LM Studio o llama.cpp server OpenAI-compatible su `http://localhost:1234/v1`;
  - oppure Ollama su `http://localhost:11434`.
- Modello generativo consigliato:
  - `Qwen3-14B` GGUF quantizzato, idealmente `Q3_K_M`, `Q4_K_M` o simile in base alla VRAM disponibile.
- Per i classifier humor opzionali:

```powershell
pip install -r requirements.txt
```

Lo script base funziona senza librerie esterne oltre alla standard library Python. Il notebook usa `tqdm`; i classifier richiedono `torch` e `transformers`.

## Setup Del Modello

### LM Studio

1. Scarica un GGUF di `Qwen3-14B`.
2. Caricalo in LM Studio.
3. Avvia il server OpenAI-compatible sulla porta `1234`.
4. Controlla il nome esatto del modello:

```powershell
Invoke-WebRequest -Uri http://localhost:1234/v1/models -UseBasicParsing
```

Usa quel nome in `--model`. Nel progetto e stato usato:

```text
qwen/qwen3-14b
```

### Ollama

Esempio:

```powershell
python .\mwahaha_task_a_en.py run `
  --input .\data\input\task-a-en.tsv `
  --output .\submission\task-a-en.tsv `
  --backend ollama `
  --base-url http://localhost:11434 `
  --model qwen3:14b
```

## Input Supportato

Lo script accetta TSV, CSV, JSON e JSONL.

Inferisce automaticamente:

- `id` da colonne come `id`, `ID`, `sample_id`, `example_id`;
- `word_inclusion` da `word1`/`word2` o da una colonna `words`;
- `news_headline` da colonne come `headline`, `title`, `news_title`, `prompt`, `text`, `context`, `input`.

Valori come `-`, `nan`, `none`, `null`, `n/a` sono trattati come mancanti.

## Comandi Disponibili

```powershell
python .\mwahaha_task_a_en.py --help
```

Comandi:

| Comando | Scopo |
|---|---|
| `run` | Genera una submission da input |
| `validate` | Valida un TSV gia generato |
| `evaluate` | Confronta la submission contro una baseline generata localmente |
| `refine` | Rigenera solo righe deboli e sostituisce in modo conservativo |

## Generazione Principale

Comando consigliato per generare la submission base:

```powershell
python .\mwahaha_task_a_en.py run `
  --input .\data\input\task-a-en.tsv `
  --output .\submission\task-a-en.tsv `
  --backend openai `
  --base-url http://localhost:1234/v1 `
  --model qwen/qwen3-14b `
  --variants-per-style 2 `
  --rerank-top-k 3 `
  --timeout 300 `
  --diagnostics-dir .\diagnostics `
  --resume `
  --humor-model Humor-Research/humor-detection-comb-23 `
  --humor-model mohameddhiab/humor-no-humor `
  --humor-weight 0.25 `
  --humor-device -1
```

Note:

- `--resume` rilegge l'output gia presente e salta gli ID gia generati.
- `--diagnostics-dir` salva i candidati e gli score per ogni riga.
- `--variants-per-style 2` genera 12 candidati per riga, perche gli stili base sono 6.
- `--rerank-top-k 3` fa il torneo finale tra i migliori 3.
- `--humor-device -1` usa CPU per i classifier, evitando pressione sulla VRAM gia occupata da Qwen.

## Generazione Sequenziale Multi-Modello

Se non puoi tenere piu modelli in VRAM contemporaneamente, usa il workflow a candidate pool. Ogni modello viene caricato manualmente in LM Studio, genera i suoi candidati e li salva su disco. Alla fine ricarichi Qwen come judge e fai il ranking globale.

### 1. Genera candidati con Qwen

```powershell
python .\mwahaha_task_a_en.py generate-candidates `
  --input .\data\input\task-a-en.tsv `
  --pool-dir .\candidate_pools\ensemble_v1 `
  --backend openai `
  --base-url http://localhost:1234/v1 `
  --model qwen/qwen3-14b `
  --model-alias qwen3-14b `
  --variants-per-entry 3 `
  --resume
```

### 2. Cambia modello in LM Studio e genera con Gemma

```powershell
python .\mwahaha_task_a_en.py generate-candidates `
  --input .\data\input\task-a-en.tsv `
  --pool-dir .\candidate_pools\ensemble_v1 `
  --backend openai `
  --base-url http://localhost:1234/v1 `
  --model GEMMA_12B_QAT_MODEL_NAME `
  --model-alias gemma-12b-qat `
  --variants-per-entry 3 `
  --resume
```

### 3. Cambia modello in LM Studio e genera con Ministral

```powershell
python .\mwahaha_task_a_en.py generate-candidates `
  --input .\data\input\task-a-en.tsv `
  --pool-dir .\candidate_pools\ensemble_v1 `
  --backend openai `
  --base-url http://localhost:1234/v1 `
  --model mistralai/ministral-3-14b-reasoning `
  --model-alias ministral-3-14b-reasoning `
  --variants-per-entry 3 `
  --resume
```

### 4. Ricarica Qwen e ranka tutti i candidati

```powershell
python .\mwahaha_task_a_en.py rank-candidates `
  --input .\data\input\task-a-en.tsv `
  --pool-dir .\candidate_pools\ensemble_v1 `
  --output .\submission\task-a-en.ensemble.tsv `
  --backend openai `
  --base-url http://localhost:1234/v1 `
  --model qwen/qwen3-14b `
  --rerank-top-k 6 `
  --diagnostics-dir .\diagnostics_ensemble `
  --humor-model Humor-Research/humor-detection-comb-23 `
  --humor-model mohameddhiab/humor-no-humor `
  --humor-weight 0.20 `
  --humor-device -1 `
  --resume
```

`generate-candidates --resume` non duplica righe gia presenti per lo stesso `--model-alias`. `rank-candidates` legge tutti i file `*.jsonl` nella pool, deduplica globalmente, valida i candidati e usa il normale reranking con Qwen judge.

## Strategie Di Generazione

Per ogni input vengono generati candidati usando stili diversi:

- `wordplay`;
- `reversal`;
- `absurd literalism`;
- `deadpan understatement`;
- `topical punchline`;
- `incongruity`.

Per `word_inclusion`, il prompt chiede una battuta che includa entrambe le parole verbatim e faccia dipendere la punchline da entrambe.

Per `news_headline`, il prompt chiede una battuta ispirata alla headline, mantenendo un legame chiaro senza limitarsi a riassumerla.

## Validazione

Comando:

```powershell
python .\mwahaha_task_a_en.py validate `
  --input .\data\input\task-a-en.tsv `
  --output .\submission\task-a-en.tsv
```

Controlli:

- header esatto `id	text`;
- tutti gli ID presenti una sola volta;
- nessun ID extra;
- testo non vuoto;
- massimo 900 caratteri;
- nessun tab o newline nel campo `text`;
- nessun boilerplate tipo `Joke:` o `Here is a joke:`;
- parole obbligatorie presenti per `word_inclusion`;
- overlap minimo con la headline per `news_headline`.

## Reranking

Il reranking cerca di approssimare la metrica reale della challenge: preferenza umana pairwise tra battute.

Non usa BLEU, ROUGE o similarita testuale.

### Livello 1: filtro di validita

I candidati non validi vengono messi in fondo o esclusi.

### Livello 2: score numerico

Qwen judge assegna uno score `0-10` considerando:

- rispetto dei vincoli;
- brevita;
- sorpresa;
- specificita;
- inglese naturale;
- comicita.

Se sono attivi i classifier humor, lo score finale e:

```text
final_score = (1 - humor_weight) * judge_score + humor_weight * (10 * humor_score)
```

Esempio con `--humor-weight 0.25`:

```text
final_score = 0.75 * judge_score + 0.25 * (10 * humor_score)
```

### Livello 3: torneo pairwise

Dopo lo score numerico, lo script prende i migliori `--rerank-top-k` candidati e li confronta in mini torneo:

```text
champion = candidato #1
champion vs candidato #2
winner vs candidato #3
winner vs candidato #4
...
```

Il winner del torneo e la battuta finale.

Approfondimento: [RERANKING.md](RERANKING.md).

## Refinement Mirato

Il comando `refine` serve quando la submission e valida ma stilisticamente ripetitiva.

Non rigenera tutto. Seleziona solo righe ad alto rischio, genera nuovi challenger e sostituisce la battuta esistente solo se il challenger vince un confronto pairwise.

Target automatici:

- inizio con `I tried`;
- inizio con `I asked`;
- inizio con `I told`;
- presenza di `turns out`;
- testo oltre 200 caratteri;
- virgolette;
- score basso nei diagnostics;
- righe non valide, se presenti.

Comando consigliato:

```powershell
python .\mwahaha_task_a_en.py refine `
  --input .\data\input\task-a-en.tsv `
  --incumbent-output .\submission\task-a-en.tsv `
  --output .\submission\task-a-en.refined.tsv `
  --backend openai `
  --base-url http://localhost:1234/v1 `
  --model qwen/qwen3-14b `
  --variants-per-style 2 `
  --rerank-top-k 4 `
  --timeout 300 `
  --diagnostics-dir .\diagnostics `
  --refine-diagnostics-dir .\refine_diagnostics `
  --report .\refine_report.json `
  --resume `
  --max-targets 120 `
  --pairwise-votes 3 `
  --replace-votes 2 `
  --humor-model Humor-Research/humor-detection-comb-23 `
  --humor-model mohameddhiab/humor-no-humor `
  --humor-weight 0.20 `
  --humor-device -1
```

Note:

- `--incumbent-output` e la submission valida da battere.
- `--output` e il nuovo TSV refined; non sovrascrive automaticamente l'incumbent.
- `--resume` riprende una run interrotta.
- `--refine-diagnostics-dir` permette di saltare gli ID gia processati.
- `--pairwise-votes 3` e `--replace-votes 2` richiedono almeno 2 vittorie su 3 contro l'incumbent.

### Dry Run Del Refinement

Per vedere quali righe verrebbero selezionate senza generare:

```powershell
python .\mwahaha_task_a_en.py refine `
  --input .\data\input\task-a-en.tsv `
  --incumbent-output .\submission\task-a-en.tsv `
  --output .\submission\task-a-en.refined.tsv `
  --backend openai `
  --diagnostics-dir .\diagnostics `
  --refine-diagnostics-dir .\refine_diagnostics `
  --resume `
  --dry-run
```

Il dry-run non contatta il modello se non deve generare.

### Ripresa Dopo Interruzione

Se la run si interrompe:

1. non cancellare `submission/task-a-en.refined.tsv`;
2. non cancellare `refine_diagnostics/`;
3. rilancia lo stesso comando con `--resume`.

Lo script:

- rilegge il refined parziale;
- salta gli ID gia presenti in `refine_diagnostics`;
- continua dai target mancanti;
- aggiorna progressivamente `refine_report.json`.

## Diagnostics

Ogni file in `diagnostics/` contiene:

```json
{
  "input": {
    "id": "en_2001",
    "kind": "news_headline",
    "headline": "..."
  },
  "winner": "...",
  "candidates": [
    {
      "text": "...",
      "style": "wordplay",
      "seed": 123,
      "temperature": 0.92,
      "score": 7.9,
      "judge_score": 8.0,
      "judge_raw": "...",
      "humor_score": 0.75,
      "valid": true,
      "invalid_reason": ""
    }
  ]
}
```

Campi utili:

- `score`: score finale usato per ordinare i candidati;
- `judge_score`: voto LLM judge;
- `humor_score`: probabilita stimata dai classifier;
- `valid`: rispetto dei vincoli;
- `invalid_reason`: motivo di scarto;
- `winner`: battuta scelta.

I file in `refine_diagnostics/` contengono anche:

```json
{
  "incumbent": "...",
  "challenger": "...",
  "challenger_votes": 2,
  "vote_trace": ["B", "A", "B"],
  "replaced": true
}
```

Dove:

- `A` = incumbent;
- `B` = challenger;
- `replaced: true` = il challenger ha sostituito l'incumbent.

## Quality Check Locale

Il comando `evaluate` genera una baseline locale nello stile dei prompt ufficiali e giudica submission vs baseline.

```powershell
python .\mwahaha_task_a_en.py evaluate `
  --input .\data\input\task-a-en.tsv `
  --output .\submission\task-a-en.tsv `
  --backend openai `
  --base-url http://localhost:1234/v1 `
  --model qwen/qwen3-14b `
  --timeout 300 `
  --min-win-rate 0.55
```

Interpretazione:

- `WIN`: la submission batte la baseline locale;
- `LOSS`: la baseline locale viene preferita;
- sotto `55%` il comando ritorna errore.

Questo non e scoring ufficiale CodaBench. Serve solo come sanity check.

## Baseline Ufficiali

`data/baseline/task-a-en-baseline.tsv` contiene baseline della fase dev/trial. Non ha gli stessi ID del test finale, quindi non va confrontata riga per riga con `data/input/task-a-en.tsv`.

Uso consigliato:

- analisi stilistica;
- confronto di lunghezza media;
- confronto di pattern ripetitivi;
- ispirazione per prompt e stili.

Uso sconsigliato:

- copiare righe;
- assumere che gli ID corrispondano al test set finale;
- usarla come ground truth.

## Creazione ZIP Finale

Dopo avere scelto il file finale, assicurati che dentro lo ZIP ci sia solo `task-a-en.tsv`.

Se vuoi usare la refined come finale:

```powershell
Copy-Item .\submission\task-a-en.refined.tsv .\submission\task-a-en.tsv
Compress-Archive -Path .\submission\task-a-en.tsv -DestinationPath .\submission.zip -Force
```

Verifica contenuto ZIP:

```powershell
tar -tf .\submission.zip
```

Output atteso:

```text
task-a-en.tsv
```

## Smoke Test Senza Modello

Run completa mock su tutto il dataset, scrivendo solo in `C:\tmp`:

```powershell
python .\mwahaha_task_a_en.py run `
  --input .\data\input\task-a-en.tsv `
  --output C:\tmp\mwahaha_mock_output.tsv `
  --backend mock `
  --diagnostics-dir C:\tmp\mwahaha_mock_diagnostics
```

Validazione mock:

```powershell
python .\mwahaha_task_a_en.py validate `
  --input .\data\input\task-a-en.tsv `
  --output C:\tmp\mwahaha_mock_output.tsv
```

Evaluate mock:

```powershell
python .\mwahaha_task_a_en.py evaluate `
  --input .\data\input\task-a-en.tsv `
  --output C:\tmp\mwahaha_mock_output.tsv `
  --backend mock
```

Refine mock:

```powershell
python .\mwahaha_task_a_en.py refine `
  --input .\data\input\task-a-en.tsv `
  --incumbent-output C:\tmp\mwahaha_mock_output.tsv `
  --output C:\tmp\mwahaha_refined_mock.tsv `
  --backend mock `
  --diagnostics-dir C:\tmp\mwahaha_mock_diagnostics `
  --refine-diagnostics-dir C:\tmp\mwahaha_refine_diagnostics_mock `
  --report C:\tmp\mwahaha_refine_report_mock.json `
  --limit 2 `
  --resume
```

## Troubleshooting

### Il comando sembra bloccato

Possibili cause:

- il modello locale sta generando lentamente;
- LM Studio e occupato o non risponde;
- il timeout e alto (`--timeout 300`);
- i classifier su CPU stanno rallentando il batch.

Controlli utili:

```powershell
Get-Process python -ErrorAction SilentlyContinue
Get-ChildItem .\refine_diagnostics | Sort-Object LastWriteTime -Descending | Select-Object -First 5
Get-Item .\submission\task-a-en.refined.tsv
```

Se la run si e interrotta, rilancia con `--resume`.

### Device set to use cpu

Messaggio normale quando `--humor-device -1`.

Significa che i classifier Hugging Face girano su CPU. Questo evita di saturare la VRAM usata dal modello generativo.

### Validation fallisce con `weak_headline_overlap`

La battuta non contiene abbastanza segnali lessicali della headline. Soluzioni:

- rigenerare l'ID;
- usare `refine --target-ids en_xxxx`;
- aumentare specificita del prompt.

### Troppe battute iniziano con `I tried`

Usa `refine`. Il comando penalizza automaticamente:

- `I tried`;
- `I asked`;
- `I told`;
- `turns out`;
- testi lunghi;
- virgolette.

### Voglio rifinire solo pochi ID

Usa `--target-ids`:

```powershell
python .\mwahaha_task_a_en.py refine `
  --input .\data\input\task-a-en.tsv `
  --incumbent-output .\submission\task-a-en.tsv `
  --output .\submission\task-a-en.refined.tsv `
  --backend openai `
  --base-url http://localhost:1234/v1 `
  --model qwen/qwen3-14b `
  --target-ids en_2150,en_2075,en_2062 `
  --resume
```

`--target-ids` puo anche puntare a un file di testo con ID separati da spazi, virgole o newline.

## Workflow Consigliato

1. Avvia LM Studio con Qwen.
2. Esegui `run` con diagnostics e resume.
3. Valida `submission/task-a-en.tsv`.
4. Genera o aggiorna `REPORT.md`.
5. Esegui `refine --dry-run` per vedere i target.
6. Esegui `refine --resume`.
7. Valida `submission/task-a-en.refined.tsv`.
8. Confronta metriche e diagnostics.
9. Se la refined e migliore, copiala come `submission/task-a-en.tsv`.
10. Ricrea `submission.zip`.

## Note Sulla Qualita

Per questa challenge la qualita dipende piu da preferenza umana che da metriche testuali.

Priorita pratiche:

- battute brevi;
- punchline specifica;
- legame chiaro con headline o parole obbligatorie;
- meno template ripetuti;
- niente spiegazioni;
- niente prefissi tipo `Joke:`;
- evitare output troppo lunghi;
- evitare contenuto offensivo o rischioso.
