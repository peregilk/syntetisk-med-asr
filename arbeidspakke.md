# Midlertidig arbeidspakkeplan
*Syntetisk datasett for medisinsk ASR – generering av tekst og tale (100–1000 timer)*

## 1. Formål og resultatmål
- **Formål:** Generere et åpent, norsk **syntetisk** medisinsk ASR-treningssett (tekst ↔ lyd) som styrker utvikling og “fair” evaluering av KI-DOK/ASR i spesialisthelsetjenesten.
- **Resultatmål (MVP):**
  - Minst **1 000 000 ord** syntetisk tekst, segmentert i **≤ 75 ord** per segment.
  - **100–1000 timer** tale (TTS og/eller manuell innlesning) i **WAV PCM 16 kHz, mono, 16-bit**.
  - Komplett datasett klar for publisering på **Hugging Face** med datasettkort, lisens og dokumentasjon.

---

## 2. Arbeidsstrømmer og leveranser

### 2.1 Scenarioer (10–100 realistiske)
**Aktivitet:**
- Utarbeide 10–100 **realistiske scenariobeskrivelser** hvor ASR er nyttig i medisinsk bruk (psykiatri, somatikk, akutt, radiologi, adm. m.m.).
- **Fagfellevurdering** av scenariene (kliniske fagpersoner).

**Eksempler (2–3 stk):**
1. **Akuttmottak – innkomstnotat (norsk/engelsk):**  
   Voksen turist med ankelskade. Tolkebehov. Mål: strukturert innkomstnotat (Anamnese, Status, Vurdering, Plan) til DIPS-stil.
2. **Radiologi – CT abdomen:**  
   Mistanke om spesifikk diagnose. Mål: kort, presis radiologibeskrivelse med standard fraseologi (funn/konklusjon).
3. **BUP – poliklinisk time:**  
   Samtale med lege og pasient der diagnose fra tekst under er mistenkt. Mål: kort oppsummering til journal, fokus på tiltak/plan.

**Leveranser:**
- `scenarios.json` (strukturert: id, teksttype, scenario).

---

### 2.2 Kildeinnhenting av medisinsk tekst
**Aktivitet:**
- Ekstrahere åpne tekstutdrag (enkeltavsnitt) fra bl.a. **finnkode.helsedirektoratet.no** og annen **åpen** faglitteratur.
- Loggføre **kilde**, **lisens** og **sitater** pr. avsnitt.

**Leveranser:**
- `sources/paragraphs.jsonl` (felt: `source_url`, `license`, `section`, `text`).

> **Merknad:** Følg nettstedets vilkår/robots.txt. Lagre lisens- og kilde-metadata for hvert avsnitt. Vi bør kun benytte kilder som er åpne.

---

### 2.3 Stoppliste (5 000 vanligste ord) og domeneord
**Aktivitet:**
- Lage liste over **5 000 vanligste norske ord** (stoppliste).
- Fra innhentede avsnitt: ekstrahere ord **ikke** i stopplisten ⇒ **kandidater til domeneord**.
- Normalisering: lower-case, tegnsetting, tallhåndtering, lemmatisering (der hensiktsmessig), sammenskrivingsstøtte.

**Kriterier:**
- Ekskluder personnavn/identifiserbare uttrykk.
- Bevar medisinske termer og flerstavs-sammensetninger.

**Leveranser:**
- `stopwords_nb_5000.txt`
- `domain_terms.json` (felt: `term`, `lemma`, `freq`, `source_refs`)

---

### 2.4 Prompt-mal og tekstgenerering (LLM)
**Aktivitet:**
- Kombinere scenarioer + avsnittstekster + domeneord for å **prompt’e** LLM til å skrive **realistiske** segmenter (≤ 75 ord) med **høy tetthet av ikke-stoppord**.
- Variere stil: innkomstnotat, radiologibeskrivelse, poliklinisk oppsummering, henvisning, administrativt referat.

**Prompt-mal (eksempel):**
```text
[SYSTEM]
Du er medisinsk skribent. Skriv norsk (Bokmål). Teksten skal være anonym og uten identifiserende detaljer.

[KRAV]
- Teksttype: {teksttype}  (f.eks. "innkomstnotat", "radiologibeskrivelse")
- Setting: {scenario}     
- 50-70 ord. Korte, presise setninger i oppgitt stil.
- Bruk mange av disse domeneordene naturlig og korrekt: {target_terms generert ved vasking av grunntekst mot stoppord}
- Unngå pasientnavn, fødselsdata, adresser, arbeidsgiver o.l.
- Følg norsk klinisk stil (f.eks. "Anamnese/Status/Vurdering/Plan" der relevant).

[INPUT]
Utdrag fra åpen fagtekst: 
{kilde_avsnitt}

[OUTPUT]
Kun selve teksten, ingen forklaring.
```
### 2.5 Script for generering av prompt
Lag Python script som kan gjøres for å traversere grunnlagsdokumentene og genere run 1 million ord i 50-70-ord segmenter (ca 15.000 segmenter).

### 2.6 Generere og publisere
Generere datasettet og gjøre dette klart for publisering.
