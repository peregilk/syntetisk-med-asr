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

### 1) Preprosessering
Kjør SNOMED-preprosessering for å lage filer i data/preprocessed/.
Henter ut nøkkelbegreper fra SNOMED, samt uttrykk der de brukes.

Eksempel:

```bash
uv run python prompt_creation/processing.py \
  --terms-file data/snomed_ct_norwegian_terms.jsonl \
  --preprocessed-dir data/preprocessed \
  --word-occurrence-dir data/word_occurences \
  --rank-cap 10000
```

Viktig parameter:
- `--rank-cap`: angir *omtrent* hvor mange SNOMED-termer som tas med (mest relevante først). Lavere verdi gir mindre datasett og raskere kjøring.


### 2) Kjør hele pipelinen
Kjører prompt-generering, LLM-generering og planoppdatering i en loop til mål er nådd.

```bash
uv run python main.py \
  --init-plan \
  --snomed-file data/preprocessed/snomed.jsonl \
  --plan-file data/terms_to_use.jsonl \
  --template templates/a.txt \
  --prompt-file prompts/generated_prompts.jsonl \
  --output-file data/outputs/output.jsonl \
  --target-count 100 \
  --max-iterations 1000
```

Viktige parametere:
- `--target-count`: bestemmer hvor mange ganger hver term totalt skal forekomme.
- `--max-iterations`: sikkerhetsgrense for hvor mange runder pipelinen skal kjøre.

#### Generer plan manuelt
En datagenereringsplan lages automatisk i pipelinen, men kan også lages manuelt.

Planen lagres i data/terms_to_use.jsonl og brukes til å styre hvor mange ganger hvert begrep skal forekomme.

```bash
uv run python generate_prompts.py init-plan \
  --snomed-file data/preprocessed/snomed.jsonl \
  --plan-file data/terms_to_use.jsonl \
  --target-count 100
```

Viktig parameter:
- `--target-count`: ønsket total forekomst per term (inkludert corpus og genererte outputs).


## Arbeidspakkeplan
- [Midlertidig arbeidspakkeplan](arbeidspakke.md)
