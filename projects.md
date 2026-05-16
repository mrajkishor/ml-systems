# AI/ML Engineering — 16 Systems Built End to End

> Each project is tagged by Phase covered, Difficulty, and real-world Use Case.
> Build these end-to-end: data → model → API → deployment → monitoring.

---

## How This Is Organised

- **Tier 1 (Core Systems)** — Classical ML, data pipelines, and first deployments.
- **Tier 2 (Specialised Systems)** — Deep Learning, NLP, Computer Vision.
- **Tier 3 (Production Systems)** — LLMs, agents, real-time ML, full MLOps.
- **Tier 4 (Deep Specialisation)** — RL, alignment, audio, safety, tabular DL.
- Every project ships with a **live demo, architecture diagram, and write-up**.

---

## TIER 1 — Core Systems

---

### Project 1: House Price Prediction — Full ML Pipeline
**Phases Covered:** 1 (Math), 2 (Python), 3 (Classical ML)
**Difficulty:** ⭐☆☆☆☆
**Use Case:** 🟡 Medium (common, but good baseline)

**Technical Concepts Applied:**
- §1.2 Calculus — gradient descent applied in linear/ridge/lasso regression
- §1.3 Probability & Statistics — hypothesis testing, confidence intervals in EDA
- §2.1 Python — full project structure, OOP, modules
- §2.2 Scientific Python Stack — NumPy, Pandas, Matplotlib, Plotly, SciPy
- §2.3 Software Engineering — Git, clean code, pytest, logging
- §3.1 Core Concepts — bias-variance tradeoff, cross-validation, regression metrics (MAE, RMSE, R²)
- §3.2 Supervised Learning — Linear Regression, Ridge, Lasso, Random Forest, XGBoost
- §3.4 Feature Engineering — missing value imputation, encoding, scaling, outlier treatment
- §3.5 Hyperparameter Tuning — Optuna (Bayesian optimisation)
- §8.3 Model Serving — FastAPI REST endpoint
- §8.4 Containerisation — Docker, deploy on Railway/Render

**What to Build:**
- End-to-end regression pipeline on Kaggle's Ames Housing dataset
- Exploratory Data Analysis (EDA) with statistical tests
- Feature engineering: handling missing values, encoding, scaling
- Train 5+ models: Linear Regression, Ridge, Lasso, Random Forest, XGBoost
- Hyperparameter tuning with Optuna
- SHAP values for model explainability
- FastAPI endpoint to serve predictions
- Dockerize and deploy on Railway or Render (free)

**Key Skills Demonstrated:** EDA, feature engineering, ensemble models, model explainability, basic deployment

---

### Project 2: Customer Churn Prediction with Imbalanced Data
**Phases Covered:** 2, 3
**Difficulty:** ⭐⭐☆☆☆
**Use Case:** 🟡 Medium

**Technical Concepts Applied:**
- §2.1–2.2 Python, Pandas, Matplotlib — data wrangling and visualisation
- §3.1 Core Concepts — classification metrics (Precision, Recall, F1, AUC-ROC, PR Curve), stratified k-fold
- §3.2 Supervised Learning — Logistic Regression, Random Forest, LightGBM, CatBoost
- §3.4 Feature Engineering — imbalanced datasets (SMOTE, class weights, threshold tuning)
- §3.5 Hyperparameter Tuning — grid/random search, learning curves

**What to Build:**
- Binary classification on telecom or bank churn dataset
- Handle class imbalance: SMOTE, class weights, threshold tuning
- Compare models: Logistic Regression, Random Forest, LightGBM, CatBoost
- Full cross-validation with stratified k-fold
- Precision-Recall curves, AUC-ROC analysis
- Business interpretation: cost-sensitive evaluation
- Streamlit dashboard showing predictions + explanations

**Key Skills Demonstrated:** Imbalanced learning, business framing, model evaluation depth

---

### Project 3: Sales Forecasting System
**Phases Covered:** 2, 3, 10 (Time Series)
**Difficulty:** ⭐⭐☆☆☆
**Use Case:** 🟠 High (retail, supply chain, FMCG demand planning)

**Technical Concepts Applied:**
- §2.2 Pandas — time-indexed DataFrames, resampling, rolling windows
- §2.2 Plotly — interactive forecast visualisation dashboard
- §2.4 Data Engineering — ETL pipeline for time series data
- §3.2 Supervised Learning — LightGBM with lag features
- §3.4 Feature Engineering — lag features, rolling statistics, calendar features
- §4.6 RNNs — LSTM for sequence forecasting
- §10.1 Time Series & Forecasting — ARIMA, SARIMA, Holt-Winters, Temporal Fusion Transformer, MAPE/RMSE/MASE evaluation

**What to Build:**
- Time series forecasting on M5 Competition or Rossmann dataset
- Classical methods: ARIMA, SARIMA, Holt-Winters
- ML approach: LightGBM with lag features, rolling statistics
- DL approach: LSTM or Temporal Fusion Transformer
- Multi-horizon forecasting (7-day, 30-day, 90-day)
- Evaluation with MAPE, RMSE, MASE
- Interactive Plotly dashboard for forecast visualization

**Key Skills Demonstrated:** Time series feature engineering, multiple forecasting paradigms, evaluation

---

## TIER 2 — Specialised Systems

---

### Project 4: Image Classification + Deployment Pipeline
**Phases Covered:** 4 (Deep Learning), 6 (Computer Vision), 8 (MLOps)
**Difficulty:** ⭐⭐⭐☆☆
**Use Case:** 🟠 High

**Technical Concepts Applied:**
- §4.1 NN Fundamentals — forward/backprop, activation functions, loss functions
- §4.2 Optimization — Adam, learning rate schedules, warmup
- §4.3 Regularisation — Dropout, BatchNorm, data augmentation, early stopping
- §4.4 Deep Learning Frameworks — PyTorch (DataLoader, nn.Module, training loop, AMP)
- §4.5 CNNs — ResNet, EfficientNet; transfer learning and fine-tuning
- §6.1 Image Fundamentals — preprocessing, augmentation (Albumentations)
- §8.1 ML Project Lifecycle — problem framing through deployment
- §8.2 Experiment Tracking — MLflow (tracking, model registry)
- §8.3 Model Serving — FastAPI, ONNX export, TorchScript
- §8.4 Containerisation — Docker, deploy on GCP Cloud Run
- §8.7 Monitoring — Evidently AI for data drift
- §8.8 Model Optimisation — ONNX, TorchScript conversion

**What to Build:**
- Multi-class image classifier (plant disease, food classification, or medical imaging)
- Train from scratch: ResNet / EfficientNet baseline
- Transfer learning and fine-tuning comparison
- Data augmentation pipeline with Albumentations
- Grad-CAM visualizations for interpretability
- ONNX export + TorchScript conversion
- FastAPI + Docker + deploy on GCP Cloud Run
- MLflow for experiment tracking
- Basic drift monitoring with Evidently AI

**Key Skills Demonstrated:** CNNs, transfer learning, model optimization, production deployment

---

### Project 5: Real-Time Object Detection System
**Phases Covered:** 4, 6
**Difficulty:** ⭐⭐⭐☆☆
**Use Case:** 🟠 High (robotics, smart cities, safety surveillance)

**Technical Concepts Applied:**
- §4.5 CNNs — object detection architectures (YOLO, Faster R-CNN concepts), mAP evaluation
- §6.1 Image Fundamentals — video frame processing, OpenCV
- §8.3 Model Serving — FastAPI video/image endpoint, TensorRT optimisation
- §8.8 Model Optimisation — TensorRT for inference speed, edge deployment

**What to Build:**
- Custom object detection with YOLOv8 on a domain-specific dataset
- Label your own dataset using Roboflow or Label Studio
- Train, evaluate with mAP metrics
- Video inference pipeline (frame-by-frame)
- FastAPI endpoint accepting image/video input
- Optimize with TensorRT for inference speed
- Deploy on edge (Raspberry Pi or Jetson Nano if available)

**Key Skills Demonstrated:** Custom training, labeling, deployment, optimization

---

### Project 6: Sentiment Analysis & Review Intelligence Platform
**Phases Covered:** 5 (NLP), 4, 8
**Difficulty:** ⭐⭐⭐☆☆
**Use Case:** 🟠 High (e-commerce, brand monitoring, consumer insights)

**Technical Concepts Applied:**
- §5.1 Text Preprocessing — tokenisation, lemmatisation, normalisation
- §5.2 Text Representations — TF-IDF, word embeddings
- §5.3 Transformer Architecture — self-attention, encoder-only models (BERT)
- §5.4 Pre-trained Language Models — BERT, DistilBERT fine-tuning
- §5.5 Core NLP Tasks — text classification, aspect-based sentiment, NER
- §5.6 Fine-tuning & Efficient Adaptation — LoRA on domain-specific data
- §8.2 Experiment Tracking — W&B
- §8.6 CI/CD for ML — A/B test two model versions in production
- §8.7 Monitoring — model performance tracking, A/B testing

**What to Build:**
- Multi-class sentiment analysis on Amazon/Yelp reviews
- Compare: TF-IDF + ML → BERT fine-tuning → DistilBERT
- Aspect-based sentiment analysis (which product feature is positive/negative)
- Named Entity Recognition to extract product names
- Fine-tune with LoRA on domain-specific data
- REST API + Streamlit UI with confidence scores
- Track experiments with W&B
- A/B test two model versions in production

**Key Skills Demonstrated:** NLP pipeline, BERT fine-tuning, LoRA, A/B testing

---

### Project 7: Recommendation Engine
**Phases Covered:** 3, 4, 5, 10 (Recommender Systems)
**Difficulty:** ⭐⭐⭐⭐☆
**Use Case:** 🔴 Very High (streaming, e-commerce, content discovery)

**Technical Concepts Applied:**
- §3.2 Supervised Learning — matrix factorisation (ALS)
- §4.1 Neural Networks — Neural Collaborative Filtering (NCF)
- §5.2 Text Representations — item and user embeddings
- §8.7 Monitoring — feature store (Feast), offline evaluation metrics
- §10.2 Recommender Systems — collaborative filtering, content-based, two-tower model, SASRec, cold-start, NDCG/MAP/Hit Rate

**What to Build:**
- Content-based + collaborative filtering baseline (MovieLens or Instacart dataset)
- Matrix factorization with ALS
- Neural Collaborative Filtering (NCF)
- Two-tower model with user and item embeddings
- Sequential recommendation: SASRec
- Cold-start handling strategy
- Real-time serving with feature store (use Feast locally)
- Offline evaluation: NDCG, MAP, Hit Rate
- A/B test simulation

**Key Skills Demonstrated:** End-to-end RecSys, two-tower architecture, feature store, evaluation

---

## TIER 3 — Production Systems

---

### Project 8: Production RAG System — Document Q&A
**Phases Covered:** 5, 9 (LLMs, RAG), 8 (MLOps)
**Difficulty:** ⭐⭐⭐⭐☆
**Use Case:** 🔴 Very High (most in-demand project in 2025–26)

**Technical Concepts Applied:**
- §5.3 Transformer Architecture — attention, encoder models for embeddings
- §5.4 Pre-trained Language Models — LLaMA 3, Mistral (decoder-only), BGE/E5 embedding models
- §9.3 Prompt Engineering — query rewriting, structured output
- §9.4 RAG — vector DB (Qdrant/pgvector), chunking, dense+BM25 hybrid retrieval, cross-encoder re-ranking, HyDE, multi-hop reasoning, RAGAS evaluation
- §9.5 LLM Application Development — LangChain, LlamaIndex, chat interface
- §8.4 Containerisation — Docker Compose
- §8.5 Cloud Platforms — AWS EC2 with auto-scaling
- §8.7 Monitoring — Prometheus + Grafana

**What to Build:**
- Ingest: PDF, Word, web pages → chunk → embed → store in vector DB (Qdrant/pgvector)
- Embedding model comparison: OpenAI ada, BGE, E5, Nomic
- Retrieval: dense search + BM25 hybrid + re-ranking (cross-encoder)
- LLM: use open-source (Mistral, LLaMA 3) via Ollama or Groq API
- Advanced RAG: query rewriting, HyDE, multi-hop reasoning
- Evaluation pipeline with RAGAS (faithfulness, answer relevancy, context recall)
- Chat interface with LangChain/LlamaIndex + Streamlit
- Docker Compose with monitoring (Prometheus + Grafana)
- Deploy on AWS EC2 with auto-scaling

**Key Skills Demonstrated:** RAG pipeline depth, evaluation, production deployment, LLM integration

---

### Project 9: LLM Fine-Tuning Pipeline — Domain Expert Model
**Phases Covered:** 5, 9, 8
**Difficulty:** ⭐⭐⭐⭐☆
**Use Case:** 🔴 Very High

**Technical Concepts Applied:**
- §5.6 Fine-tuning & Efficient Adaptation — QLoRA, LoRA, PEFT library
- §9.1 LLM Architecture — autoregressive modelling, scaling laws, tokenisation
- §9.2 LLM Training — SFT, instruction tuning, QLoRA, distributed training concepts
- §9.6 LLM Evaluation — MT-Bench, LLM-as-Judge, domain-specific benchmarks, hallucination detection
- §8.8 Model Optimisation — GGUF quantisation, llama.cpp serving, vLLM

**What to Build:**
- Choose a domain: legal, medical, finance, or coding
- Curate or use existing dataset (OpenHermes, Alpaca, FinGPT datasets)
- Supervised Fine-Tuning (SFT) with LLaMA 3 / Mistral using QLoRA
- Instruction template formatting
- Training on free GPU: Google Colab Pro / Kaggle / RunPod
- Merge and quantize (GGUF for llama.cpp serving)
- Evaluate: MT-Bench, domain-specific benchmarks, LLM-as-Judge
- Serve with vLLM or Ollama
- Compare base vs fine-tuned on 20 test prompts with LLM judge scoring

**Key Skills Demonstrated:** QLoRA fine-tuning, dataset curation, evaluation, model serving

---

### Project 10: AI Agent System — Autonomous Research Assistant
**Phases Covered:** 5, 9 (Agents), 8
**Difficulty:** ⭐⭐⭐⭐⭐
**Use Case:** 🔴 Very High (cutting-edge, differentiator)

**Technical Concepts Applied:**
- §9.3 Prompt Engineering — ReAct framework, Chain-of-Thought, structured output
- §9.5 LLM Application Development — LangGraph / CrewAI multi-agent orchestration, tool use, memory systems (short-term + long-term vector DB + episodic), streaming, code execution
- §8.3 Model Serving — FastAPI backend, WebSocket streaming
- §8.5 Cloud Platforms — deployment with rate limiting and auth
- §8.7 Monitoring — LangSmith / Arize Phoenix tracing and observability

**What to Build:**
- Multi-agent system using LangGraph or CrewAI
- Agents: Researcher (web search) + Analyst (data) + Writer (report) + Critic (review)
- Tools: web search, code execution, PDF reader, calculator, SQL query
- Memory: short-term (conversation) + long-term (vector DB) + episodic
- Human-in-the-loop checkpoints
- Streaming responses with WebSocket
- Full observability: LangSmith or Arize Phoenix tracing
- Deploy as a web app with FastAPI backend + React frontend
- Rate limiting, auth, and cost tracking per user

**Key Skills Demonstrated:** Agentic architectures, tool use, observability, full-stack deployment

---

### Project 11: MLOps Platform — End-to-End ML System
**Phases Covered:** 8 (MLOps), 3, 4
**Difficulty:** ⭐⭐⭐⭐⭐
**Use Case:** 🔴 Very High (AI infrastructure, reliable production ML systems)

**Technical Concepts Applied:**
- §2.4 Data Engineering — DVC + S3 versioning, Airflow/Prefect orchestration
- §8.1 ML Project Lifecycle — full lifecycle from data to monitoring
- §8.2 Experiment Tracking — MLflow hosted on EC2, model registry (staging/prod/archived)
- §8.4 Containerisation — Docker, Kubernetes basics, deploy to ECS
- §8.5 Cloud Platforms — AWS ECR, ECS, S3
- §8.6 CI/CD for ML — GitHub Actions → test → Docker build → ECR push → ECS deploy, canary deployment (10% traffic split)
- §8.7 Monitoring — Evidently (data drift), Prometheus + Grafana (performance), Feast feature store (online + offline), PagerDuty/Slack alerting
- §10.6 ML System Design — Lambda/Kappa architecture patterns, real-time feature serving

**What to Build:**
- ML pipeline for fraud detection (or any classification problem)
- Data versioning with DVC + S3
- Feature store with Feast (online + offline store)
- Training pipeline orchestrated with Airflow or Prefect
- Experiment tracking with MLflow (hosted on EC2)
- Model registry with staging/production/archived states
- CI/CD: GitHub Actions → test → build Docker → push to ECR → deploy to ECS
- Canary deployment: route 10% traffic to new model
- Monitoring: data drift (Evidently) + performance metrics (Prometheus + Grafana)
- Alerting: PagerDuty or Slack webhook on drift detection

**Key Skills Demonstrated:** Full MLOps lifecycle, CI/CD, monitoring, feature store — this is a complete ML platform

---

### Project 12: Multimodal AI App — Vision + Language
**Phases Covered:** 5, 6, 9
**Difficulty:** ⭐⭐⭐⭐⭐
**Use Case:** 🔴 Very High (product companies building AI features)

**Technical Concepts Applied:**
- §6.2 Vision Architectures — ViT, CLIP (vision-language pretraining)
- §6.3 Generative Vision Models — Stable Diffusion, ControlNet, image conditioning
- §6.4 Multimodal Learning — LLaVA / InternVL (vision-language model), image captioning, Visual QA
- §9.5 LLM Application Development — voice input integration (Whisper), full-stack FastAPI + React
- §10.4 Audio & Speech (partial) — Whisper for speech-to-text input

**What to Build:**
- Upload image → AI describes, answers questions, extracts text (OCR)
- Use LLaVA or InternVL (open-source vision-language model)
- Add voice input (Whisper for speech-to-text)
- Generate images from text (Stable Diffusion with ControlNet)
- Build a product: AI-powered invoice reader, medical image explainer, or visual shopping assistant
- FastAPI backend + React frontend
- Deploy on GCP with GPU instance
- Add user authentication, usage tracking, and cost monitoring

**Key Skills Demonstrated:** Multimodal models, full-stack AI app, production-grade system

---

### Project 13: Fraud Detection System — Real-Time ML
**Phases Covered:** 3, 8, 10
**Difficulty:** ⭐⭐⭐⭐☆
**Use Case:** 🔴 Very High (fintech, payments, digital banking)

**Technical Concepts Applied:**
- §1.4 Discrete Mathematics — graph theory (nodes, edges) applied via GNN
- §2.4 Data Engineering — Kafka stream simulation, Redis real-time feature serving
- §3.2 Supervised Learning — XGBoost + neural network ensemble
- §3.3 Unsupervised Learning — anomaly detection techniques
- §3.4 Feature Engineering — extreme class imbalance (0.1% fraud), velocity features, behavioural patterns
- §8.3 Model Serving — FastAPI with <10ms p99 latency
- §8.7 Monitoring — real-time fraud rate dashboard, SHAP explainability (regulatory)
- §8.8 Model Optimisation — latency optimisation for real-time inference
- §10.3 Graph Neural Networks — GNN for transaction network features (GCN, message passing)
- §10.6 ML System Design — real-time streaming architecture, feature store patterns

**What to Build:**
- Train on IEEE-CIS or synthetic credit card fraud dataset
- Handle extreme class imbalance (0.1% fraud)
- Feature engineering: velocity features, behavioural patterns
- Models: XGBoost + Neural network ensemble
- Real-time feature serving with Redis
- Kafka stream simulation for real-time transactions
- Model served via FastAPI with <10ms p99 latency
- Graph features using GNN (transaction network)
- Monitoring: real-time fraud rate dashboard
- Explainability: SHAP for every prediction (regulatory requirement)

**Key Skills Demonstrated:** Real-time ML, streaming, latency optimization, GNN, explainability

---

## TIER 4 — Deep Specialisation

---

### Project 14: Reinforcement Learning System — Game AI + RLHF Alignment
**Phases Covered:** 7 (Reinforcement Learning), 9 (RLHF / DPO)
**Difficulty:** ⭐⭐⭐⭐☆
**Use Case:** 🟠 High (AI alignment, robotics, autonomous systems)

**Technical Concepts Applied:**
- §7.1 RL Fundamentals — MDPs, states, actions, rewards, policy, value function, Bellman equations, exploration vs exploitation
- §7.2 Model-Free RL — DQN (replay buffer, target network, epsilon-greedy), Double DQN, Dueling DQN, TD learning
- §7.3 Policy Gradient Methods — REINFORCE from scratch, A2C, PPO
- §7.4 RL from Human Feedback — reward model from preference pairs (Anthropic HH-RLHF dataset), PPO loop via TRL library, Direct Preference Optimization (DPO), Constitutional AI concepts
- §7.5 Advanced RL — OpenAI Gym / Gymnasium environments (CartPole, LunarLander)
- §9.2 LLM Training — SFT → RLHF → DPO pipeline, KL divergence from reference model

**What to Build:**
- **Part A — Classic RL:** Train a DQN agent on CartPole / LunarLander (OpenAI Gym)
  - Implement replay buffer, target network, epsilon-greedy exploration
  - Upgrade to Double DQN + Dueling DQN; compare learning curves
  - Train a PPO agent on the same environments; compare DQN vs PPO
- **Part B — Policy Gradient deep dive:** Implement REINFORCE from scratch, then A2C
- **Part C — RLHF pipeline (the critical part):**
  - Fine-tune a small LLM (GPT-2 / LLaMA 3.2-1B) with Supervised Fine-Tuning (SFT)
  - Train a reward model from human preference pairs (use Anthropic HH-RLHF dataset)
  - Run PPO loop against the reward model using TRL library
  - Implement Direct Preference Optimization (DPO) as a cleaner alternative
  - Compare base → SFT → RLHF → DPO outputs on 20 test prompts with LLM-as-Judge
- Visualise training: reward curves, KL divergence from reference model, win-rate
- Deploy as a side-by-side comparison web app (FastAPI + React)

**Key Skills Demonstrated:** MDPs, DQN, PPO, REINFORCE, reward modeling, RLHF, DPO — full Phase 7 + Phase 9.2 alignment coverage

---

### Project 15: Audio & Speech Intelligence Platform
**Phases Covered:** 10 (Audio/Speech), 5 (NLP), 8 (MLOps)
**Difficulty:** ⭐⭐⭐☆☆
**Use Case:** 🟠 High (voice AI, accessibility, multilingual transcription)

**Technical Concepts Applied:**
- §10.4 Audio & Speech — audio signal processing (FFT, MFCC, Mel spectrograms via Librosa), ASR (Whisper fine-tuning on Indian languages via IndicSUPERB/AI4Bharat), Wav2Vec 2.0 concepts, TTS (VITS / Coqui TTS, MOS evaluation), speaker diarization (pyannote.audio)
- §5.5 Core NLP Tasks — summarisation of transcripts using LLM
- §8.3 Model Serving — real-time WebSocket streaming transcription (faster-whisper), FastAPI backend
- §8.4 Containerisation — Docker, deploy on HuggingFace Spaces

**What to Build:**
- **Audio preprocessing pipeline:** load audio → FFT → Mel spectrograms → MFCCs using Librosa
- **Speech Recognition (ASR):**
  - Use Whisper (base/small) out of the box on English audio; measure WER
  - Fine-tune Whisper on an Indian language dataset (Hindi/Telugu via IndicSUPERB or AI4Bharat)
  - Compare fine-tuned vs base Whisper on domain-specific audio (medical, legal, call centre)
- **Speaker Diarization:** integrate pyannote.audio — "who spoke when" on multi-speaker recordings
- **Text-to-Speech (TTS):** generate synthetic speech with VITS / Coqui TTS; evaluate MOS score
- **End-to-end pipeline:**
  - Upload audio file → transcribe → diarize → summarize (using LLM) → export structured JSON
  - Real-time streaming transcription via WebSocket (Whisper + faster-whisper)
- **FastAPI backend + Streamlit UI**, Dockerised and deployed on HuggingFace Spaces
- Monitoring: WER tracking per language, latency dashboard

**Key Skills Demonstrated:** Audio signal processing (FFT, MFCC, Mel), ASR fine-tuning, speaker diarization, TTS, real-time streaming — full Phase 10.4 coverage

---

### Project 16: Safe AI Platform — Tabular Deep Learning + LLM Guardrails
**Phases Covered:** 10.5 (Tabular DL), 9.7 (LLM Safety), 1 (Math foundations applied)
**Difficulty:** ⭐⭐⭐⭐☆
**Use Case:** 🟠 High (regulated industries: fintech, healthcare, enterprise AI)

**Technical Concepts Applied:**
- §1.1 Linear Algebra — PCA from SVD implemented in NumPy from scratch
- §1.2 Calculus & Optimisation — gradient descent implemented from scratch (no frameworks)
- §1.3 Probability & Statistics — Bayes' theorem visualised, KL divergence between distributions, information theory applied
- §9.7 LLM Safety & Alignment — NeMo Guardrails (topic/fact-checking/jailbreak rails), Llama Guard 3 (input/output classification), red-teaming (10 attack types documented), differential privacy (DP-SGD, privacy-utility tradeoff), federated learning concepts, responsible AI frameworks
- §10.5 Tabular Deep Learning — TabNet (attention-based), FT-Transformer (feature tokenisation), SAINT (intersample attention), self-supervised pretraining on tabular data (SCARF/VIME), when DL beats tree-based models

**What to Build:**
- **Part A — Tabular Deep Learning (covers Phase 10.5):**
  - Take the churn or fraud dataset from Projects 2/13
  - Benchmark tree models (XGBoost, LightGBM) vs tabular DL models:
    - TabNet (attention-based tabular DL)
    - FT-Transformer (feature tokenization transformer)
    - SAINT (intersample attention)
  - Self-supervised pretraining on unlabelled tabular data with SCARF / VIME
  - Analyse when DL beats tree-based and why (document the finding)
- **Part B — LLM Safety & Guardrails (covers Phase 9.7):**
  - Deploy a small LLM (Mistral 7B via Ollama) as a chatbot
  - Add NeMo Guardrails: topic rails, fact-checking rails, jailbreak prevention
  - Integrate Llama Guard 3 for input/output content classification
  - Red-team the system: document 10 attack types (prompt injection, jailbreaks, PII leakage) and test defenses
  - Implement differential privacy concepts: add DP-SGD noise during fine-tuning; measure privacy-utility tradeoff
  - Build a safety evaluation dashboard: toxicity scores (Perspective API), hallucination rate (RAGAS), refusal rate
- **Part C — Math Foundations Applied (covers Phase 1 explicitly):**
  - Jupyter notebook series: implement gradient descent from scratch (numpy only), PCA from SVD, Bayes' theorem visualised, KL divergence between distributions
  - Attach these as an "Under the Hood" appendix to the project README — shows depth to interviewers
- Full deployment: FastAPI with guardrails middleware, Grafana safety dashboard

**Key Skills Demonstrated:** TabNet, FT-Transformer, SAINT, NeMo Guardrails, Llama Guard, red-teaming, DP-SGD, safety dashboards, math from scratch — closes all remaining gaps

---

## Technical Coverage Across All 16 Systems

| Domain | Coverage |
|---|---|
| Phase 1 — Math & Stats | ✅ Applied in Project 16 (Part C), implicit across all |
| Phase 2 — Programming & SWE | ✅ Projects 1–3 |
| Phase 3 — Classical ML | ✅ Projects 1, 2, 3, 7, 13 |
| Phase 4 — Deep Learning | ✅ Projects 4, 5, 6, 7 |
| Phase 5 — NLP | ✅ Projects 6, 7, 8, 9, 10 |
| Phase 6 — Computer Vision | ✅ Projects 4, 5, 12 |
| Phase 7 — Reinforcement Learning | ✅ Project 14 (DQN, PPO, RLHF, DPO) |
| Phase 8 — MLOps | ✅ Projects 4, 6, 8, 9, 10, 11 |
| Phase 9 — Generative AI & LLMs | ✅ Projects 8, 9, 10, 12, 14 |
| Phase 9.7 — LLM Safety | ✅ Project 16 (Part B) |
| Phase 10.1 — Time Series | ✅ Project 3 |
| Phase 10.2 — Recommenders | ✅ Project 7 |
| Phase 10.3 — GNN | ✅ Project 13 |
| Phase 10.4 — Audio & Speech | ✅ Project 15 |
| Phase 10.5 — Tabular Deep Learning | ✅ Project 16 (Part A) |

**16 systems. Every domain. Every layer of the stack.**

---

## Project Build Order

| Project | Domain | Difficulty | Impact | Start When |
|---|---|---|---|---|
| 1. House Price Prediction | Proptech / Finance | Easy | 🟡 Foundational | Month 1 |
| 2. Churn Prediction | SaaS / Telecom | Easy | 🟡 Foundational | Month 1 |
| 3. Sales Forecasting | Retail / Supply Chain | Easy-Med | 🟠 High | Month 2 |
| 4. Image Classifier + MLOps | Agriculture / Healthcare | Medium | 🟠 High | Month 3 |
| 5. Object Detection | Safety / Smart Cities | Medium | 🟠 High | Month 4 |
| 6. Sentiment Platform | E-commerce / Brand | Medium | 🟠 High | Month 4 |
| 7. Recommendation Engine | Streaming / Commerce | Hard | 🔴 Very High | Month 5 |
| 8. RAG System | Enterprise Knowledge | Hard | 🔴 Very High | Month 6 |
| 9. LLM Fine-Tuning | Domain AI | Hard | 🔴 Very High | Month 7 |
| 10. AI Agent System | Automation / Research | Hard | 🔴 Very High | Month 8 |
| 11. MLOps Platform | AI Infrastructure | Expert | 🔴 Very High | Month 9 |
| 12. Multimodal AI App | Accessibility / Vision | Expert | 🔴 Very High | Month 10 |
| 13. Fraud Detection RT | Fintech / Payments | Hard | 🔴 Very High | Month 9 |
| 14. RL + RLHF System | AI Alignment / Robotics | Hard | 🟠 High | Month 11 |
| 15. Audio & Speech Platform | Voice AI / Accessibility | Medium | 🟠 High | Month 11 |
| 16. Safe AI + Tabular DL | Regulated AI / Safety | Medium | 🟠 High | Month 12 |

🔴 = Very High Impact | 🟠 = High Impact | 🟡 = Foundational

---

## What Makes a Project Stand Out

### ✅ Do This
- Write a clear README with problem statement, architecture diagram, and results
- Include a **live demo** (even a free-tier deployment)
- Show **before/after metrics** (baseline vs your model)
- Add a **monitoring dashboard** — shows you think about production, not just training
- Write a **blog post** or article explaining your thinking and decisions
- Record a **2-minute demo video**

### ❌ Avoid This
- Notebooks with no deployment
- Projects copied 1:1 from tutorials with no twist
- Missing evaluation metrics or business context
- No version control or messy commit history
- Projects with no README

---

## Recommended Learning Paths

### Interested in Computer Vision & MLOps
Pick: **Projects 4 + 5 + 11 + 13**

### Interested in NLP, LLMs & Agents
Pick: **Projects 7 + 8 + 10 + 12**

### Interested in AI Safety & Alignment
Pick: **Projects 9 + 14 + 16** — covers RLHF, DPO, guardrails, and alignment

### Fast Track — Core AI/ML in one sweep
Pick: **Projects 1 + 6 + 8** — classical ML, NLP, and LLMs end-to-end

---

## Full Tech Stack Across All 16 Projects

| Category | Technologies |
|---|---|
| **Languages** | Python, SQL |
| **Scientific Stack** | NumPy, Pandas, Matplotlib, Seaborn, Plotly, SciPy, Librosa |
| **Classical ML** | Scikit-learn, XGBoost, LightGBM, CatBoost, imbalanced-learn |
| **Hyperparameter Tuning** | Optuna, Hyperopt |
| **Explainability** | SHAP, Grad-CAM |
| **Deep Learning** | PyTorch, TensorFlow / Keras, JAX |
| **Computer Vision** | OpenCV, Albumentations, torchvision, YOLOv8, ResNet, EfficientNet, ViT, CLIP |
| **NLP / Transformers** | Hugging Face Transformers, BERT, DistilBERT, RoBERTa, T5 |
| **LLMs** | LLaMA 3, Mistral, Falcon, GPT-2 |
| **Embeddings** | Sentence-BERT, BGE, E5, Nomic, Word2Vec, GloVe, FastText |
| **Fine-tuning** | LoRA, QLoRA, PEFT, TRL |
| **LLM Frameworks** | LangChain, LlamaIndex, LangGraph, CrewAI |
| **LLM Evaluation** | RAGAS, MT-Bench, LLM-as-Judge |
| **Vision-Language** | LLaVA, InternVL, BLIP |
| **Image Generation** | Stable Diffusion, ControlNet, DDPM |
| **Audio & Speech** | Whisper, faster-whisper, Wav2Vec 2.0, VITS, Coqui TTS, pyannote.audio |
| **LLM Serving** | Ollama, vLLM, llama.cpp, Groq API |
| **RL Environments** | OpenAI Gym, Gymnasium |
| **RL Algorithms** | DQN, Double DQN, PPO, A2C, REINFORCE, TRL (RLHF, DPO) |
| **Vector DBs** | Qdrant, pgvector, FAISS, Chroma, Pinecone, Weaviate |
| **Retrieval** | BM25, dense retrieval, hybrid search, cross-encoder re-ranking |
| **Tabular DL** | TabNet, FT-Transformer, SAINT, SCARF, VIME |
| **LLM Safety** | NeMo Guardrails, Llama Guard 3, DP-SGD, Perspective API |
| **Experiment Tracking** | MLflow, Weights & Biases (W&B) |
| **Data & Pipelines** | DVC, Apache Airflow, Prefect, Feast (feature store) |
| **Monitoring** | Evidently AI, Prometheus, Grafana, LangSmith, Arize Phoenix |
| **Model Export** | ONNX, TorchScript, TensorRT, OpenVINO |
| **APIs & Serving** | FastAPI, TorchServe, Triton Inference Server, ONNX Runtime |
| **Containerisation** | Docker, Docker Compose, Kubernetes, Helm |
| **Cloud** | AWS (EC2, S3, ECR, ECS), GCP (Cloud Run, Vertex AI) |
| **CI/CD** | GitHub Actions |
| **Streaming & Cache** | Kafka, Redis, WebSocket |
| **Graph Neural Networks** | PyTorch Geometric, DGL, GCN, GraphSAGE, GAT |
| **Frontend & UI** | Streamlit, React |
| **Free Deployment** | Railway, Render, Hugging Face Spaces |

---

## Free Resources & Tools

| Need | Free Option |
|---|---|
| GPU Training | Kaggle (30 hrs/week), Google Colab, Lightning.ai |
| LLM API | Groq (fast + free), Together AI, Ollama (local) |
| Vector DB | Qdrant Cloud free tier, Chroma (local) |
| Deployment | Railway, Render, Hugging Face Spaces, GCP free tier |
| Experiment Tracking | MLflow (self-host), W&B free tier |
| Data Labeling | Label Studio (free), Roboflow (free tier) |
| Monitoring | Evidently AI (open source), Prometheus + Grafana |

---

*Last Updated: May 2026*
