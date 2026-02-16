# Syntetisk datasett for medisinsk ASR
Dette er et repo for syntestisk ASR datasett-delen av prosjektet "Talegjenkjenning og redusert dokumentasjonsbyrde med bruk av kunstig intelligens".

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

### Oversikt over datagenereringspipeline
1) Preprosessering: SNOMED-termer og -uttrykk renses og rangeres til `data/preprocessed/snomed.jsonl`.
2) Plan: `data/terms_to_use.jsonl` opprettes og styrer hvor mange ganger hvert begrep skal forekomme.
3) Prompt-generering: `prompts/generated_prompts.jsonl` bygges fra planen + template.
4) LLM-generering: hver prompt får et output i `data/outputs/output.jsonl`.
5) Planoppdatering: brukt terminologi telles, `used_terms` lagres per output, og `target_remaining` justeres.
6) Loop: steg 3–5 gjentas til alle termer når målet eller `--max-iterations` stopper kjøringen.

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


### 2a) Kjør hele pipelinen
Kjører prompt-generering, LLM-generering og planoppdatering i en loop til ønsket antall begrepsbruk er nådd.

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
- `--max-iterations`: sikkerhetsgrense for hvor mange runder pipelinen skal kjøre. `--max-iterations 1` gir git en output per begrep.
- `--reasoning-effort`: chain-of-thought test-time compute budsjett. `low` gir flere brukte begreper per output enn `none` (6.1 vs 4.5). `medium` og `high` ga ikke merkbart økning.

### 2b) Generer alle prompts i én kjøring
For å generere alle prompts uten iterativ løkke må vi
1. Lage en datagenereringsplan i `data/terms_to_use.jsonl` som styrer hvor mange ganger hvert begrep skal brukes.
2. Estimere antall nødvendige prompts og generere dem. 
   - Estimatet skrives til `target_remaining` i `data/terms_to_use.jsonl`.
   - Generere prompts basert på oppdatert plan.

#### Generere plan
```bash
# 1. Initialize plan
uv run python generate_prompts.py init-plan \
  --plan-file data/terms_to_use.jsonl \
  --snomed-file data/preprocessed/snomed.jsonl \
  --target-count 100

```

Viktig parameter:
- `--target-count`: ønsket total forekomst per term (inkludert corpus og genererte outputs).

#### Generere prompts

```bash
# 2. Generate all prompts
uv run python generate_prompts.py generate \
  --plan-file data/terms_to_use.jsonl \
  --snomed-file data/preprocessed/snomed.jsonl \
  --template templates/a.txt \
  --output-file prompts/generated_prompts.jsonl \
  --optional-count 10 \
  --reasoning-effort low \
  --target-count 100 \
  --generate-all
```

Viktige parametere:
- `--reasoning-effort`: styrer estimatet. `none` antar 4.5 termer per output,
  alle andre verdier bruker 6.1 (gjennomsnittsverdier fra DeepSeek V3.2 ved 10 optionals).
- `--optional-count`: skalerer estimatet lineært i forhold til 10 optionals.


## Arbeidspakkeplan
- [Midlertidig arbeidspakkeplan](arbeidspakke.md)
