# PROJECT PLAN — Real-Time Traffic Monitoring Framework

> **Tema:** Real-Time Traffic Monitoring (7 classi)  
> **Paradigma:** Deep Learning (YOLOv8 Multi-Class Detection + CNN Classifier)  
> **Dataset:** VisDrone2019-DET (`pedestrian`, `bicycle`, `car`, `van`, `truck`, `bus`, `motor`)  
> **Branching:** GitHub Flow (`main` stabile, feature branch brevi, conventional scoped commits).

---

## 1. Current State vs Target

Il repository è stato migrato completamente dal progetto precedente (Hand Gesture Recognition — MediaPipe + SVM e YOLOv8 single-class) alla nuova struttura Traffic Monitoring basata esclusivamente su Deep Learning. Tutti i moduli core sono implementati e testati.

---

## 2. Stato Moduli

| Modulo | File | Stato | Note |
|--------|------|-------|------|
| Post-processing IoU | `src/postprocessing/iou.py` | ✅ DONE | Supporta `xywh`/`xyxy`, test passano |
| Post-processing NMS | `src/postprocessing/nms.py` | ✅ DONE | `manual_nms` + `opencv_nms`, test passano |
| Post-processing Tracker | `src/postprocessing/tracker.py` | ✅ DONE | Simple IOU tracker + counting line |
| Evaluation metrics | `src/evaluation/metrics.py` | ✅ DONE | mAP, AP, confusion matrix, P, R, F1 |
| Evaluation benchmark | `src/evaluation/benchmark.py` | ✅ DONE | Latenza & throughput |
| Evaluation compare | `src/evaluation/compare.py` | ✅ DONE | Valutazione pipeline DL |
| Visualization base | `src/utils/visualization.py` | ✅ DONE | Draw bbox multi-class, contatori, linea |
| Config loader | `src/config.py` | ✅ DONE | Config centralizzato con classi traffic, path, iperparametri |
| DL YOLO detector | `src/deep_learning/yolo_detector.py` | ✅ DONE | Wrapper Ultralytics YOLOv8n multi-class |
| DL vehicle classifier | `src/deep_learning/vehicle_net.py` | ✅ DONE | ResNet18 PyTorch (7 classi) |
| Webcam pipeline | `src/webcam/pipeline.py` | ✅ DONE | Loop live con conteggio veicoli |
| Tests IoU/NMS | `tests/test_iou.py`, `tests/test_nms.py` | ✅ DONE | 10 test passano |
| Tests DL/classifier | `tests/test_classifier.py` | ✅ DONE | 3 test passano |
| Tests pipeline | `tests/test_pipeline.py` | ✅ DONE | 8 test passano |

---

## 3. Roadmap Pratica

### FASE A — Cleanup & Scaffold ✅ COMPLETATA
Obiettivo: Eliminare codice legacy hand-gesture e creare la struttura per traffic monitoring DL.

- [x] Rimuovere `src/classical/` (hand_detector, feature_extractor, gesture_classifier)
- [x] Rimuovere `src/deep_learning/gesture_net.py`
- [x] Creare `src/deep_learning/vehicle_net.py`, adattare `yolo_detector.py` per multi-class
- [x] Creare `src/postprocessing/tracker.py`
- [x] Scrivere `src/config.py` con classi VisDrone, path modelli traffic
- [x] Aggiornare `src/utils/visualization.py` per multi-class traffic
- [x] Creare `scripts/download_visdrone.py`
- [x] Aggiornare test suite per riflettere la nuova struttura

### FASE B — DL Pipeline: YOLOv8 + CNN ✅ COMPLETATA
Obiettivo: Multi-class vehicle detection + classification con deep learning.

- [x] `scripts/build_yolo_dataset.py` — Converte VisDrone in formato YOLO (7 classi)
- [x] `scripts/build_classifier_dataset.py` — Estrae crop 224×224 per CNN
- [x] `src/deep_learning/yolo_detector.py` — Wrapper YOLOv8n multi-class
- [x] `src/deep_learning/vehicle_net.py` — ResNet18/CNN custom (7 classi)
- [x] `scripts/train_yolo.py` — Fine-tuning YOLOv8n (MPS/CUDA/CPU support)
- [x] `scripts/train_classifier.py` — Training CNN (MPS/CUDA/CPU support)
- [x] `tests/test_classifier.py` — Test CNN output shape

### FASE C — Integration, Webcam & Benchmark ✅ COMPLETATA
Obiettivo: Unificare le pipeline in un'applicazione live con conteggio veicoli.

- [x] `src/webcam/streamer.py` — Gestione VideoCapture
- [x] `src/webcam/pipeline.py` — Loop live con conteggio + tracker
- [x] `scripts/run_webcam.py` — Entrypoint CLI
- [x] `src/evaluation/compare.py` — Valutazione pipeline DL su traffic
- [x] `tests/test_pipeline.py` — Test end-to-end per nuova pipeline

### FASE D — Documentation & Polish 🔄 IN CORSO
Obiettivo: Documentazione academic-grade.

- [ ] `docs/technical_analysis.md` — Methodology, results, failure analysis, ethics
- [ ] `README.md` — Aggiornare con istruzioni nuove (setup, training, inference)
- [ ] Notebook demo in `notebooks/`
- [ ] Pulizia codice morto, TODO, verifica `.gitignore`
- [ ] **Merge:** `docs(project): finalize technical analysis and release v1.0.0`

---

## 4. Deliverables Checklist (Exam)

| Deliverable | Location | Peso | Stato |
|-------------|----------|------|-------|
| Source code modulare | `src/`, `tests/`, `scripts/` | Obbligatorio | ✅ Completato |
| `requirements.txt` / `environment.yml` | Root | Obbligatorio | ✅ Aggiornati |
| Dataset script (download & prep) | `scripts/download_visdrone.py` | Obbligatorio | ✅ Creato e funzionante |
| Modello DL (YOLO + CNN weights) | `models/yolov8n_traffic.pt`, `models/vehicle_cnn.pt` | Obbligatorio | ⚠️ Da addestrare |
| Test suite pytest | `tests/` | Obbligatorio | ✅ Test aggiornati |
| Notebook demo | `notebooks/` | Consigliato | ⚠️ Da creare |
| Technical Analysis PDF | `docs/technical_analysis.pdf` | Obbligatorio | ⚠️ Da creare |
| README completo | `README.md` | Obbligatorio | ✅ Aggiornato |
| Demo video / webcam GIF | `docs/assets/demo.gif` | Extra points | ❌ Da creare |

---

## 5. Quick Start Commands

```bash
# 1. Clone e setup
git clone <repo>
cd computer_vision_exam
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Scarica dataset VisDrone (~1.5GB, non in repo)
python scripts/download_visdrone.py --output data/raw/visdrone

# 3. Build dataset strutturati
python scripts/build_yolo_dataset.py
python scripts/build_classifier_dataset.py

# 4. Training
python scripts/train_yolo.py
python scripts/train_classifier.py

# 5. Evaluation
python -m src.evaluation.compare --mode dataset --output docs/results/

# 6. Webcam live
python scripts/run_webcam.py --source 0
```

---

## 6. Note & Decisioni Architetturali

| Decisione | Rationale |
|-----------|-----------|
| **VisDrone2019-DET** | Dataset gratuito, ben documentato, usato in letteratura YOLO, formato semplice da convertire. |
| **YOLOv8n COCO pre-trained** | Riconosce già car/truck/bus/motorcycle/bicycle/person. Usabile out-of-the-box per MVP, poi fine-tunable su VisDrone. |
| **Tracker semplice (centroid/IOU)** | Per un progetto di esame, un tracker complesso (DeepSORT) è eccessivo. IOU tracker è sufficiente per conteggio. |
| **CNN separato opzionale** | YOLOv8n già classifica. Il CNN separato serve per confrontare detection puro vs detection+classification. |
| **MPS (Apple Silicon GPU)** | Utilizzato per training YOLO e CNN, riduce drasticamente i tempi rispetto a CPU. |
