# Guida Completa al Progetto di Traffic Monitoring

> **Per chi non sa nulla di Computer Vision**

---

## 1. In Due Parole: Di Cosa Si Tratta?

Immagina di avere una telecamera puntata su una strada. Questo progetto insegna al computer a **guardare quella strada e capire cosa sta succedendo**: quante auto passano, quanti camion, quanti pedoni attraversano, quante biciclette. E lo fa in **tempo reale**, cioè mentre le cose stanno succedendo, non dopo.

Il progetto usa un metodo "moderno", basato sull'intelligenza artificiale (deep learning), che impara a riconoscere i veicoli guardando migliaia di esempi. È come quando un bambino impara a riconoscere gli animali: gli fai vedere tante foto finché non impara da solo.

---

## 2. Perché Monitorare il Traffico?

### Il Problema

Le città moderne sono piene di macchine. Gestire il traffico è difficile perché:
- **Non sappiamo quanti veicoli passano** in ogni strada
- **Gli incidenti** vengono notati troppo tardi
- **I semafori** spesso non si adattano al traffico reale
- **L'inquinamento** dipende da quanto traffico c'è

### La Soluzione: La Telecamera Intelligente

Invece di mettere sensori sotto l'asfalto (costosi da installare), usiamo una **semplice telecamera** (anche quella del computer) e un programma che "capisce" cosa vede. È economico, si installa in 5 minuti e può contare, classificare e persino seguire i veicoli.

---

## 3. Come Funziona? Il Cervello Artificiale

Il progetto usa una "mente" molto potente basata sull'intelligenza artificiale.

---

## 4. La Mente Moderna: YOLO + CNN (Deep Learning)

### L'idea di base

Questo metodo funziona come un **bambino prodigio** che impara guardando migliaia di esempi. Non ci sono regole scritte: il computer impara da solo cos'è un'auto e cos'è un camion.

### Passo 1: YOLO — "L'occhio che vede tutto in un colpo"

**YOLO** sta per *You Only Look Once* (Guardi Solo Una Volta).

**Analogia semplice:**
Immagina di entrare in una stanza piena di oggetti. Invece di guardare ogni oggetto uno per uno, fai una **fotografia mentale** di tutta la stanza e immediatamente dici: "C'è un gatto sul divano, un libro sul tavolo, una lampada sul soffitto". YOLO fa esattamente questo con le immagini della strada.

Prima dell'intelligenza artificiale, i computer dovevano scorrere l'immagine pezzetto per pezzetto (come cercare un ago in un pagliaio). YOLO invece guarda **tutta l'immagine insieme** e in un solo "colpo d'occhio" dice:
- "Qui c'è un'auto (al 95% di sicurezza)"
- "Qui c'è un pedone (al 88% di sicurezza)"
- "Qui c'è un camion (al 92% di sicurezza)"

Tecnicamente:
- Divide l'immagine in una griglia
- Per ogni quadratino della griglia predice: "c'è un oggetto?", "che oggetto è?", "dove si trova esattamente?"
- Unisce le predizioni sovrapposte usando la **NMS** (Non-Maximum Suppression), cioè elimina i doppioni

### Passo 2: CNN — "Il Cervello Artificiale"

**CNN** sta per *Convolutional Neural Network* (Rete Neurale Convoluzionale).

**Analogia semplice:**
Immagina il cervello umano. Quando vedi un'auto, i tuoi neuroni si attivano in un certo modo: alcuni rilevano le ruote, altri il parabrezza, altri la carrozzeria. Una CNN è un **cervello artificiale** composto da milioni di "neuroni" software che fanno la stessa cosa.

Come impara?
- Gli mostriamo **migliaia di foto** di auto, camion, pedoni, biciclette
- All'inizio il cervello artificiale sbaglia miseramente
- Ad ogni errore, corregge leggermente i suoi neuroni
- Dopo migliaia di tentativi, diventa bravissimo

**Perché è così potente?**
- Il computer **scopre da solo** quali pattern sono importanti
- Trova cose che noi non avremmo mai pensato (es. la texture dell'asfalto intorno all'auto, l'ombra sotto il veicolo)
- Migliora continuamente con più dati

### Vantaggi del Deep Learning
- **Molto preciso**: raggiunge il 95-99% di accuratezza
- **Robusto**: funziona con pioggia, nebbia, buio (entro certi limiti)
- **Automatico**: non serve scrivere regole a mano
- **Versatile**: impara a riconoscere qualsiasi oggetto, basta dargli esempi

### Svantaggi
- **Assetato di potenza**: richiede una scheda grafica (GPU) potente per allenarsi
- **Scatola nera**: quando sbaglia, è difficile capire esattamente perché
- **Dati, dati, dati**: ha bisogno di migliaia di foto annotate

---

## 5. Il Conteggio dei Veicoli: Come "Contare" Senza Dita

Ora che il computer vede i veicoli, come fa a **contarli**? Non può usare le dita!

### Il Trucco della Linea Virtuale

Immagina di disegnare una **linea rossa** sul video della strada, a metà schermo. Ogni volta che un veicolo attraversa quella linea, il computer fa "ding!" e aumenta il contatore.

Ma attenzione: il video scorre a 30 fotogrammi al secondo. Se un'auto attraversa la linea, compare in 15 fotogrammi consecutivi. Se contassimo ogni fotogramma, conteremmo la stessa auto 15 volte!

### Il Tracker: "Segui quell'auto!"

Per evitare di contare la stessa auto più volte, usiamo un **tracker** (inseguitore).

**Come funziona (semplificato):**
1. Al primo fotogramma, l'auto viene vista alle coordinate (x=100, y=200)
2. Al fotogramma successivo, il computer cerca un'auto simile vicino a quella posizione (x=105, y=205)
3. Se la trova, capisce che è **la stessa auto** che si è spostata
4. L'auto ha un **ID** (come un nome: "Auto#7")
5. Quando "Auto#7" attraversa la linea rossa, viene contata UNA sola volta

È come quando segui una persona in mezzo alla folla: non la perdi di vista perché sai che non può teletrasportarsi.

---

## 6. Il Dataset: La "Scuola" del Computer

Per insegnare al computer, servono **esempi**. Tante foto di strade con le auto evidenziate.

### VisDrone2019-DET

Il dataset usato in questo progetto si chiama **VisDrone2019-DET** ed è stato creato da ricercatori cinesi. Contiene:
- **10.209 foto** scattate da droni e telecamere fisse
- Strade cittadine, autostrade, incroci
- Ogni foto ha un file di testo che dice: "qui c'è un'auto, qui c'è un pedone..."

### Perché proprio questo dataset?
1. È **gratuito**
2. È **già pronto**: le foto sono già annotate (qualcuno ha già disegnato i rettangoli intorno ai veicoli)
3. È **famoso**: tutti i ricercatori lo usano, quindi possiamo confrontare i nostri risultati con quelli degli altri

### Le 7 Classi che Riconosciamo

| Icona | Classe | Esempio | Difficoltà |
|---|---|---|---|
| 🚶 | Pedone | Persone che camminano | Media (si muovono, sono piccoli) |
| 🚲 | Bicicletta | Bici e ciclisti | Alta (piccole, sottili) |
| 🚗 | Auto | Le macchine normali | Bassa (grandi, ben visibili) |
| 🚐 | Furgone | Van, furgoncini | Media (simili alle auto ma più alti) |
| 🚚 | Camion | Tir, autocarri | Bassa (molto grandi) |
| 🚌 | Autobus | Bus pubblici | Bassa (enormi e colorati) |
| 🛵 | Moto | Scooter, motocicli | Alta (piccole e veloci) |

Le classi ignorate (perché poco utili o rappresentate male nel dataset): "regioni ignorate", "persone" (troppo generico), "tricicli", "tricicli con tetto", "altro".

---

## 7. Glossario per Non-Addetti ai Lavori

| Termine | Spiegazione Semplice |
|---|---|
| **Pixel** | Il puntino più piccolo di un'immagine digitale. Una foto HD ne ha 2 milioni. |
| **Bounding Box** | Il rettangolo che disegniamo intorno a un oggetto per dire "eccolo qui". |
| **Feature** | Una caratteristica dell'immagine (es. "ha tanti bordi verticali", "è rosso"). |
| **Classifier** | Il componente che dice "questo è un'auto" o "questo è un pedone". |
| **Detector** | Il componente che trova gli oggetti nell'immagine e dice DOVE sono. |
| **Training** | La fase in cui il computer impara guardando esempi. |
| **Inference** | La fase in cui il computer applica ciò che ha imparato a una nuova foto. |
| **FPS** | *Frames Per Second* — fotogrammi al secondo. 30 FPS = fluido come un film. |
| **IoU** | *Intersection over Union* — misura quanto due rettangoli si sovrappongono. 1.0 = perfetto, 0.0 = nessuna sovrapposizione. |
| **NMS** | *Non-Maximum Suppression* — elimina i rettangoli doppioni che indicano lo stesso oggetto. |
| **Pretrained** | Un modello già allenato su milioni di foto generiche (es. ImageNet), che poi affiniamo sul nostro dataset. |

---

## 8. Struttura del Progetto (Semplificata)

```
computer_vision_exam/
├── data/
│   ├── raw/visdrone/           ← Le foto scaricate da internet
│   └── processed/
│       ├── yolo/               ← Foto + etichette per YOLO
│       └── classifier/         ← Ritagli di veicoli per la CNN
├── models/
│   ├── vehicle_cnn.pt          ← Il cervello moderno (file grosso)
│   └── yolov8n_traffic.pt      ← L'occhio di YOLO (file enorme)
├── src/                        ← Il codice sorgente
│   ├── deep_learning/          ← YOLO + CNN
│   ├── postprocessing/         ← Tracker, IoU, NMS
│   ├── webcam/                 ← La telecamera live
│   └── utils/                  ← Disegnare i rettangoli, i contatori...
├── tests/                      ← I compiti in classe del computer
└── scripts/                    ← I pulsanti "play" del progetto
```

---

## 9. Come Si Usa (Guida Passo-Passo)

### Per un Utente Normale

1. **Installa il programma** (serve Python e qualche libreria)
2. **Scarica le foto di esempio** (`python scripts/download_visdrone.py`)
3. **Fai imparare al computer** (`python scripts/train_yolo.py`)
4. **Accendi la webcam** (`python scripts/run_webcam.py`)
5. **Guarda lo schermo**: vedrai i rettangoli colorati intorno ai veicoli e i contatori che aumentano

### Cosa Si Vede sullo Schermo?

Quando il programma è attivo, vedi:
- **Rettangoli colorati** intorno a ogni veicolo (verde per le auto, rosso per i camion, ecc.)
- **Etichette** sopra i rettangoli: "car: 0.95" significa "auto, sono sicuro al 95%"
- **Una linea arancione**: la "linea del conteggio"
- **I contatori** in alto a destra: "TOTAL: 45, car: 30, truck: 5..."
- **FPS**: quante volte al secondo il computer analizza l'immagine

I tasti da premere:
- **s** → scatta una foto dello schermo
- **q** → esci

---

## 10. Limiti e Cose da Sapere

### Cosa FUNZIONA bene
- Giorno soleggiato
- Strade con traffico moderato
- Veicoli ben visibili

### Cosa NON funziona bene
- **Nebbia fitta**: il computer vede peggio di noi
- **Notte buia**: senza infrarossi, è cieco
- **Crowding**: se 50 persone sono ammassate, i rettangoli si sovrappongono
- **Velocità estreme**: un aereo o una rondine sono troppo veloci

### Il problema dei pregiudizi (Bias)

Il dataset VisDrone è stato raccolto principalmente in Cina. Questo significa che:
- Le auto cinesi sono molto presenti
- I pedoni hanno abiti tipici di quella zona
- Le strade possono avere segnaletica diversa

Se usi questo programma in Italia o in Brasile, potrebbe essere leggermente meno preciso all'inizio. È come se il computer avesse imparato a riconoscere le persone guardando solo un paese: quando vede qualcuno con abiti diversi, impiega un attimo ad abituarsi.

---

## 11. Perché Questo Progetto è Utile?

1. **Smart Cities**: le città intelligenti usano telecamere per regolare i semafori in tempo reale
2. **Sicurezza stradale**: rilevamento automatico di incidenti (se una macchina si ferma improvvisamente)
3. **Ambientale**: contare il traffico per calcolare l'inquinamento
4. **Parcheggi intelligenti**: sapere quanti posti liberi ci sono
5. **Tasse e pedaggi**: contare i veicoli per il pedaggio autostradale

---

## 12. Riepilogo Finale

Questo progetto insegna a un computer a **guardare una strada e capire cosa c'è**.

Usa **YOLO + CNN**, un metodo moderno e potentissimo basato sull'intelligenza artificiale. Il computer impara da solo a riconoscere auto, camion, bus, pedoni, biciclette e moto guardando migliaia di esempi.

Il computer impara da un dataset chiamato **VisDrone**, che contiene migliaia di foto di strade con i veicoli già evidenziati.

Alla fine, il programma può:
- **Vedere** auto, camion, bus, pedoni, biciclette e moto
- **Contarli** usando una linea virtuale e un tracker
- **Farlo in tempo reale**, anche da una webcam

È un esempio perfetto di come la Computer Vision stia trasformando il mondo intorno a noi, rendendo le macchine capaci di "vedere" come noi.

---

*Scritto per chi non sa nulla di Computer Vision, ma ha curiosità di capire.*
