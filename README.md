# AI/ML Engineering — Building Intelligent Systems

16 end-to-end AI/ML systems built from scratch — data pipelines, model training, APIs, deployment, and monitoring. Every layer of the stack. Every domain that matters.

---

## What's in This Repo

| File | Purpose |
|---|---|
| [`concepts.md`](concepts.md) | Every concept, algorithm, and system design pattern that powers these projects |
| [`projects.md`](projects.md) | 16 systems with full build specs, technical depth, and real-world use cases |
| [`project-map.html`](project-map.html) | Interactive map — real-world impact and societal value of each project |

---

## Technical Domains

| # | Domain | Core Technologies |
|---|---|---|
| 1 | Mathematics & Statistics | Linear Algebra, Calculus, Probability, Information Theory |
| 2 | Programming & Software Engineering | Python, SQL, Git, Docker, pytest |
| 3 | Classical Machine Learning | XGBoost, LightGBM, Scikit-learn, SHAP |
| 4 | Deep Learning & Neural Networks | PyTorch, TensorFlow, JAX |
| 5 | Natural Language Processing | Transformers, BERT, LLaMA, Mistral |
| 6 | Computer Vision | ResNet, ViT, YOLO, Stable Diffusion |
| 7 | Reinforcement Learning | DQN, PPO, RLHF, DPO |
| 8 | MLOps & Production Systems | MLflow, Airflow, Kubernetes, Prometheus |
| 9 | Generative AI & LLMs | RAG, Agents, Fine-tuning, LangGraph |
| 10 | Specialised Domains | Time Series, RecSys, GNN, Audio, Tabular DL, AI Safety |

---

## The 16 Projects

### Tier 1 — Core Systems

| # | Project | Complexity | Use Case |
|---|---|---|---|
| 1 | House Price Prediction | ⭐ | Proptech / Finance |
| 2 | Customer Churn Prediction | ⭐⭐ | SaaS / Telecom |
| 3 | Sales Forecasting System | ⭐⭐ | Retail / Supply Chain |

### Tier 2 — Specialised Systems

| # | Project | Complexity | Use Case |
|---|---|---|---|
| 4 | Image Classification + MLOps | ⭐⭐⭐ | Agriculture / Healthcare |
| 5 | Real-Time Object Detection | ⭐⭐⭐ | Safety / Smart Cities |
| 6 | Sentiment & Review Intelligence | ⭐⭐⭐ | E-commerce / Brand |
| 7 | Recommendation Engine | ⭐⭐⭐⭐ | Streaming / Commerce |

### Tier 3 — Production Systems

| # | Project | Complexity | Use Case |
|---|---|---|---|
| 8 | Production RAG System | ⭐⭐⭐⭐ | Enterprise Knowledge |
| 9 | LLM Fine-Tuning Pipeline | ⭐⭐⭐⭐ | Domain AI |
| 10 | AI Agent System | ⭐⭐⭐⭐⭐ | Automation / Research |
| 11 | MLOps Platform | ⭐⭐⭐⭐⭐ | AI Infrastructure |
| 12 | Multimodal AI App | ⭐⭐⭐⭐⭐ | Accessibility / Vision |
| 13 | Real-Time Fraud Detection | ⭐⭐⭐⭐ | Fintech / Payments |

### Tier 4 — Deep Specialisation

| # | Project | Concepts | Use Case |
|---|---|---|---|
| 14 | RL System + RLHF Alignment | DQN, PPO, RLHF, DPO | AI Alignment / Robotics |
| 15 | Audio & Speech Platform | Whisper, ASR, TTS, Diarization | Voice AI / Accessibility |
| 16 | Safe AI + Tabular Deep Learning | Guardrails, TabNet, DP-SGD | Regulated AI / Safety |

---

## Technical Coverage

| Domain | Projects |
|---|---|
| Math & Statistics | 16 (from scratch), 1 |
| Programming & SWE | 1, 2, 3 |
| Classical ML | 1, 2, 3, 7, 13 |
| Deep Learning | 4, 5, 6, 7 |
| NLP | 6, 7, 8, 9, 10 |
| Computer Vision | 4, 5, 12 |
| Reinforcement Learning | 14 |
| MLOps | 4, 6, 8, 9, 10, 11 |
| Generative AI & LLMs | 8, 9, 10, 12, 14 |
| LLM Safety | 16 |
| Time Series | 3 |
| Recommenders | 7 |
| Graph Neural Networks | 13 |
| Audio & Speech | 15 |
| Tabular Deep Learning | 16 |

---

## Where to Start

**Computer Vision & MLOps** → Projects 4 + 5 + 11 + 13

**NLP, LLMs & Agents** → Projects 7 + 8 + 10 + 12

**AI Safety & Alignment** → Projects 9 + 14 + 16

**End-to-end in one sweep** → Projects 1 + 6 + 8

---

## What Every Project Includes

- ✅ Architecture diagram
- ✅ Live deployed demo
- ✅ Before/after metrics — baseline vs final model
- ✅ Monitoring dashboard
- ✅ Write-up explaining decisions and tradeoffs
- ✅ Demo video

---

## Full Tech Stack

### Languages & Core
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat&logo=postgresql&logoColor=white)

| Category | Technologies |
|---|---|
| **Languages** | Python, SQL |
| **Scientific Stack** | NumPy, Pandas, Matplotlib, Seaborn, Plotly, SciPy, Librosa |

### Classical ML
| Category | Technologies |
|---|---|
| **Frameworks** | Scikit-learn, XGBoost, LightGBM, CatBoost |
| **Imbalanced Data** | imbalanced-learn (SMOTE, class weights) |
| **Hyperparameter Tuning** | Optuna, Hyperopt |
| **Explainability** | SHAP, Grad-CAM |

### Deep Learning
| Category | Technologies |
|---|---|
| **Frameworks** | PyTorch, TensorFlow / Keras, JAX |
| **Computer Vision** | torchvision, Albumentations, OpenCV, YOLOv8 (Ultralytics) |
| **Architectures** | ResNet, EfficientNet, ViT, CLIP, YOLO, U-Net |
| **Model Export** | ONNX, TorchScript, TensorRT, OpenVINO |

### NLP & LLMs
| Category | Technologies |
|---|---|
| **Transformers** | Hugging Face Transformers, BERT, DistilBERT, RoBERTa |
| **LLMs** | LLaMA 3, Mistral, Falcon, GPT-2 |
| **Embeddings** | Sentence-BERT, BGE, E5, Nomic, Word2Vec, GloVe, FastText |
| **Fine-tuning** | LoRA, QLoRA, PEFT, TRL |
| **Frameworks** | LangChain, LlamaIndex, LangGraph, CrewAI |
| **Evaluation** | RAGAS, MT-Bench, LLM-as-Judge |

### Generative AI
| Category | Technologies |
|---|---|
| **Vision-Language** | LLaVA, InternVL, BLIP |
| **Image Generation** | Stable Diffusion, ControlNet, DDPM |
| **Speech** | Whisper, faster-whisper, Wav2Vec 2.0, VITS, Coqui TTS, pyannote.audio |
| **LLM Serving** | Ollama, vLLM, llama.cpp, Groq API, Together AI |

### Reinforcement Learning
| Category | Technologies |
|---|---|
| **Environments** | OpenAI Gym, Gymnasium |
| **Algorithms** | DQN, Double DQN, PPO, A2C, REINFORCE (from scratch) |
| **Alignment** | TRL (RLHF, DPO), Anthropic HH-RLHF dataset |
| **Libraries** | Stable Baselines3 |

### RAG & Vector Databases
| Category | Technologies |
|---|---|
| **Vector DBs** | Qdrant, pgvector, FAISS, Chroma, Pinecone, Weaviate |
| **Retrieval** | BM25 (sparse), dense retrieval, hybrid search |
| **Re-ranking** | Cross-encoder models |

### Tabular Deep Learning
| Category | Technologies |
|---|---|
| **Models** | TabNet, FT-Transformer, SAINT |
| **Self-supervised** | SCARF, VIME |

### LLM Safety
| Category | Technologies |
|---|---|
| **Guardrails** | NeMo Guardrails, Llama Guard 3 |
| **Privacy** | DP-SGD (differential privacy) |
| **Evaluation** | Perspective API, RAGAS (hallucination), red-teaming |

### MLOps & Production
| Category | Technologies |
|---|---|
| **Experiment Tracking** | MLflow, Weights & Biases (W&B) |
| **Data Versioning** | DVC |
| **Feature Store** | Feast |
| **Orchestration** | Apache Airflow, Prefect |
| **Monitoring** | Evidently AI, NannyML, Prometheus, Grafana |
| **Observability** | LangSmith, Arize Phoenix |
| **CI/CD** | GitHub Actions |
| **Alerting** | PagerDuty, Slack webhooks |

### Serving & Deployment
| Category | Technologies |
|---|---|
| **APIs** | FastAPI, Flask |
| **Model Serving** | TorchServe, Triton Inference Server, ONNX Runtime |
| **Containerisation** | Docker, Docker Compose |
| **Orchestration** | Kubernetes, Helm |
| **Cloud — AWS** | EC2, S3, ECR, ECS, Lambda, SageMaker |
| **Cloud — GCP** | Cloud Run, Vertex AI, GCS, BigQuery |
| **Free Deployment** | Railway, Render, Hugging Face Spaces |
| **Streaming** | WebSocket, Kafka |
| **Cache / Store** | Redis |

### Frontend & UI
| Category | Technologies |
|---|---|
| **Dashboards** | Streamlit |
| **Web Apps** | React, FastAPI |

### Graph & Networks
| Category | Technologies |
|---|---|
| **GNN** | PyTorch Geometric, DGL |
| **Algorithms** | GCN, GraphSAGE, GAT |

---

## References

- *Mathematics for Machine Learning* — Deisenroth, Faisal, Ong
- *Designing Machine Learning Systems* — Chip Huyen
- *Deep Learning* — Goodfellow, Bengio, Courville
- *Building Machine Learning Powered Applications* — Emmanuel Ameisen

---

*Last Updated: May 2026*
