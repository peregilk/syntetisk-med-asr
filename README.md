# Syntetisk datasett for medisinsk ASR
Dette er et repo for syntetisk ASR datasett-delen av prosjektet "Talegjenkjenning og redusert dokumentasjonsbyrde med bruk av kunstig intelligens".

## Mål
Etablere et stort, åpent norsk **medisinsk tale-til-tekst-treningssett** for å trene og forbedre KI-DOK- og ASR-modeller i spesialisthelsetjenesten, samt stimulere konkurranse og kvalitet i markedet.

## Omfang og format
- **Størrelse:** 100–1000 timer lyd  
- **Innhold:** Syntetiske lydopptak med tilhørende tekst/transkripsjoner (parallelle tekst–lyd-par) egnet for modelltrening

## Produksjon (datagenerering)
- Kildemateriale: **syntetisk tekstbasert** medisinsk datasett
- Tale genereres på to måter:
  - Opplesning fra manus (studenter/innlesere)
  - Syntetisk tale (TTS) der det er hensiktsmessig

## Juridikk og personvern
- **Syntetisk** datasett for å redusere personvernkonsekvenser
- Pilotering meldes som **kvalitetsprosjekter (PVO)**; **REK-godkjenning** kreves ikke
- Forankres i **KI-forordningen (AI Act)**, **MDR** og **GDPR**

## Tilgjengeliggjøring
- Åpnes og publiseres internasjonalt via **Hugging Face** for både kommersielle aktører og FoU-miljøer

## Videre bruk
- **Påbegynne trening av en norsk medisinsk Whisper-modell**
- Grunnlag for videre utvikling av forbedrede **KI-DOK-løsninger**

## Forventet effekt
- Raskere utvikling og **rettferdig (fair) evaluering** av medisinske tale-til-tekst-løsninger
- Potensial for **lavere kostnader** og **bedre kvalitet** i anskaffelser og klinisk bruk

## Bruk

### Oppsett

Installer avhengigheter:

```bash
uv sync
```

For API-basert generering (iterativ pipeline) må `DEEPINFRA_API_KEY` være satt:

```bash
export DEEPINFRA_API_KEY=<din-api-nokkel>
```

### Oversikt over datagenereringspipeline

1) Preprosessering: SNOMED-termer og -uttrykk renses/rankes til `data/preprocessed/snomed.jsonl`.
2) Plan: `data/terms_to_use.jsonl` styrer mål per term.
3) Prompt-generering: JSONL med prompts bygges fra plan + template.
4) LLM-generering: hver prompt får et output i output-filen.
5) Planoppdatering: `used_terms` fylles inn, og `target_remaining` oppdateres.
6) Filtrering/pruning ved behov.

Canonical output-schema etter planoppdatering:

`{"id": ..., "template": ..., "prompt": ..., "result": ..., "used_terms": [...]}`

### 1) Preprosessering
Kjør SNOMED-preprosessering for å lage filer i data/preprocessed/.
Henter ut nøkkelbegreper fra SNOMED, samt uttrykk der de brukes.

Eksempel:

```bash
uv run python prompt_creation/processing.py \
  --terms-file data/snomed_ct_norwegian_terms.jsonl \
  --preprocessed-dir data/preprocessed \
  --word-occurrence-dir data/word_occurences \
  --rank-cap 10000 \
  --max-corpus-count 99
```

Viktig parameter:
- `--rank-cap`: angir *omtrent* hvor mange SNOMED-termer som tas med (mest relevante først). Lavere verdi gir mindre datasett og raskere kjøring.


### 2a) Standard default pipeline (iterativ)
Dette er hovedløpet i repoet. Det kjører prompt-generering + LLM + planoppdatering i loop til målet er nådd.

```bash
uv run python main.py \
  --init-plan \
  --snomed-file data/preprocessed/snomed.jsonl \
  --plan-file data/terms_to_use.jsonl \
  --template templates/a.txt \
  --prompt-file prompts/generated_prompts.jsonl \
  --output-file data/outputs/output.jsonl \
  --target-count 100 \
  --max-iterations 100 \
  --reasoning-effort low
```

Viktige parametere:
- `--target-count`: bestemmer hvor mange ganger hver term totalt skal forekomme.
- `--max-iterations`: sikkerhetsgrense for hvor mange runder pipelinen skal kjøre.
- `--reasoning-effort`: chain-of-thought test-time compute budsjett. `low` gir flere brukte begreper per output enn `none` (6.1 vs 4.5). `medium` og `high` ga ikke merkbart økning.

Filtrering i loop (aktiv som standard):
- `--disable-output-filter`: skru av filtrering i loop.
- `--filter-min-used-terms`: minimum antall `used_terms` (standard `4`).
- `--filter-min-chars`: minimum antall tegn i `result["text"]` (standard `400`).
- `--filter-max-chars`: maksimum antall tegn i `result["text"]` (standard `600`).
- `--rejected-output-file`: valgfri JSONL-fil med avviste records og `filter_reason`.

Etter hver iterasjon gjør pipelinen nå:
1) Generering av outputs.
2) Oppdatering av plan (`used_terms`, presence-basert bidrag per term).
3) Filtrering av output-fil med kravene over.
4) Ny planoppdatering på filtrert output.

### Partitionerte JSONL-kataloger for store filer

For disse path-argumentene kan du gi enten en `.jsonl`-fil eller en katalog:
- prompt-filer
- output-filer
- rejected-filer

Hvis path ikke ender med `.jsonl`, behandles den som en partition-katalog.

Regler:
- Filer navngis som `part_00.jsonl`, `part_01.jsonl`, osv.
- Maks størrelse per del er 60 MB.
- Ved append skrives nye records til siste del som fortsatt har plass, ellers opprettes neste del.
- Ved lesing forventes sammenhengende sekvens uten hull i part-indekser.

Eksempel:

```bash
uv run python main.py \
  --prompt-file prompts/post_bulk_balance \
  --output-file data/outputs/medical_results_balanced \
  --rejected-output-file data/outputs/medical_results_rejected
```

### 2b) Sette opp for bulk-run eksternt (lokalt i dette repoet)
For bulk-run lager du først hele prompt-settet lokalt, og kjører så selve LLM-jobben eksternt.

1) Initialiser plan:

```bash
uv run python generate_prompts.py init-plan \
  --plan-file data/terms_to_use.jsonl \
  --snomed-file data/preprocessed/snomed.jsonl \
  --target-count 100
```

2) Generer alle prompts i én kjøring:

```bash
uv run python generate_prompts.py generate \
  --plan-file data/terms_to_use.jsonl \
  --snomed-file data/preprocessed/snomed.jsonl \
  --template templates/a.txt \
  --output-file prompts/all_prompts_none.jsonl \
  --optional-count 10 \
  --reasoning-effort none \
  --generate-all
```

Notat:
- `generate --generate-all` estimerer først prompts per term og skriver estimatet til `target_remaining` i planfilen.
- `--target-count` er del av `init-plan`, ikke `generate`.
- Promptfila (`prompts/all_prompts_none.jsonl`) brukes videre i ekstern bulk-kjøring.

### 3) Fortsette etter bulk-run: filtrering, pruning og balansering
Etter at bulk-resultat er tilbake i repoet (f.eks. `data/outputs/medical_results_deepseek.jsonl`), kjør disse stegene i rekkefølge.

#### 3.1 Oppdatere plan fra bulk-resultat

```bash
uv run python generate_prompts.py update-plan \
  --output-file data/outputs/medical_results_deepseek.jsonl \
  --plan-file data/terms_to_use.jsonl \
  --evaluate-mode all
```

#### 3.2 Filtrere bulk-output manuelt

```bash
uv run python generate_prompts.py filter-output \
  --input-file data/outputs/medical_results_deepseek.jsonl \
  --output-file data/outputs/medical_results_deepseek_filtered.jsonl \
  --rejected-file data/outputs/medical_results_deepseek_rejected.jsonl \
  --min-used-terms 4 \
  --min-chars 400 \
  --max-chars 600 \
  --overwrite
```

Record avvises dersom:
- `used_terms` har færre enn 4 elementer.
- `result` ikke kan parses med `ast.literal_eval(...)` til en dict.
- `result["text"]` mangler/er tom.
- `result["text"]` har færre enn 400 tegn eller flere enn 600 tegn.

#### 3.3 Flytt obsolete outputs til rejected (pruning)
Når enkelte outputs er overflødige i forhold til måloppnåelse per term, kan de flyttes til rejected-poolen.

```bash
uv run python generate_prompts.py prune-obsolete \
  --plan-file data/terms_to_use.jsonl \
  --input-file data/outputs/medical_results_deepseek_filtered.jsonl \
  --output-file data/outputs/medical_results_deepseek_pruned.jsonl \
  --rejected-file data/outputs/medical_results_deepseek_rejected.jsonl \
  --overwrite
```

Pruning-regel (safe mode):
- En record flyttes kun hvis alle termene i `used_terms` fortsatt er dekket av de gjenværende recordene etter flytting.
- Bidrag regnes presence-basert: maks 1 bidrag per term per record.
- Records uten gyldig `used_terms` flyttes til rejected med `filter_reason="obsolete_terms_saturated"`.

#### 3.4 Oppdatere plan etter filtrering/pruning

```bash
uv run python generate_prompts.py update-plan \
  --output-file data/outputs/medical_results_deepseek_pruned.jsonl \
  --plan-file data/terms_to_use.jsonl \
  --evaluate-mode all
```

#### 3.5 Fortsette balansering med iterativ pipeline (ved behov)
Hvis planfila fortsatt har termer med `target_remaining > 0`, fortsett med iterativ kjøring via `main.py`:

```bash
uv run python main.py \
  --snomed-file data/preprocessed/snomed.jsonl \
  --plan-file data/terms_to_use.jsonl \
  --template templates/a.txt \
  --prompt-file prompts/generated_prompts.jsonl \
  --output-file data/outputs/output.jsonl \
  --target-count 100 \
  --max-iterations 100 \
  --reasoning-effort none
```

Når alle termer har `target_remaining = 0`, er den tekstlige balanseringen ferdig.

Se `logg.md` for historiske tall/erfaringer fra tidligere bulk-kjøringer.


## Arbeidspakkeplan
- [Midlertidig arbeidspakkeplan](arbeidspakke.md)
