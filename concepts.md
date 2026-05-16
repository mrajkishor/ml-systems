# AI/ML Engineering — Technical Depth

> Every concept, algorithm, and system design pattern that powers the 16 projects in this repo — from linear algebra to LLM alignment.

---

## Table of Contents

1. [Phase 1 — Mathematics & Statistics Foundations](#phase-1)
2. [Phase 2 — Programming & Software Engineering](#phase-2)
3. [Phase 3 — Classical Machine Learning](#phase-3)
4. [Phase 4 — Deep Learning & Neural Networks](#phase-4)
5. [Phase 5 — Natural Language Processing (NLP)](#phase-5)
6. [Phase 6 — Computer Vision](#phase-6)
7. [Phase 7 — Reinforcement Learning](#phase-7)
8. [Phase 8 — MLOps & Production Systems](#phase-8)
9. [Phase 9 — Generative AI & LLMs](#phase-9)
10. [Phase 10 — Specialized Domains & Capstone](#phase-10)

---

## Phase 1 — Mathematics & Statistics Foundations {#phase-1}

### 1.1 Linear Algebra
> **Covered by Projects:** 16 (Part C — PCA from SVD in NumPy), implicit in 4, 5, 6, 7, 8, 9

- Scalars, Vectors, Matrices, and Tensors
- Matrix operations: addition, multiplication, transpose, inverse
- Determinants and trace
- Eigenvalues and Eigenvectors
- Singular Value Decomposition (SVD)
- Principal Component Analysis (PCA) — mathematical basis
- Norms: L1, L2, Frobenius
- Orthogonality and projections
- Change of basis

### 1.2 Calculus & Optimization
> **Covered by Projects:** 16 (Part C — gradient descent from scratch), 1 (applied in regression training)

- Derivatives and partial derivatives
- Chain rule and product rule
- Multivariable calculus — gradients, Jacobians, Hessians
- Taylor series approximations
- Gradient descent and its variants
- Convex vs non-convex functions
- Lagrange multipliers and constrained optimization
- Automatic differentiation (conceptual)

### 1.3 Probability & Statistics
> **Covered by Projects:** 16 (Part C — Bayes, KL divergence notebooks), 1 (hypothesis testing, confidence intervals in EDA)

- Sample spaces, events, probability axioms
- Conditional probability and Bayes' theorem
- Random variables — discrete and continuous
- Common distributions: Bernoulli, Binomial, Poisson, Gaussian, Exponential
- Expectation, Variance, Covariance, Correlation
- Law of Large Numbers and Central Limit Theorem
- Maximum Likelihood Estimation (MLE)
- Maximum A Posteriori (MAP) estimation
- Hypothesis testing: p-values, t-tests, chi-square tests
- Confidence intervals
- Information theory: Entropy, KL divergence, Cross-entropy, Mutual Information

### 1.4 Discrete Mathematics
> **Covered by Projects:** 13 (graph theory applied via GNN — nodes, edges, transaction networks)

- Set theory and logic
- Graph theory basics (nodes, edges, paths, trees)
- Combinatorics and counting
- Big-O notation and complexity analysis

---

## Phase 2 — Programming & Software Engineering {#phase-2}

### 2.1 Python for AI/ML
> **Covered by Projects:** 1, 2, 3 (full project structure, OOP, modules, type hints)

- Python fundamentals: data types, control flow, functions, OOP
- List comprehensions, generators, decorators, context managers
- File I/O and exception handling
- Modules and package management (pip, conda, virtualenv)
- Type hints and dataclasses

### 2.2 Scientific Python Stack
> **Covered by Projects:** 1 (NumPy, Pandas, Matplotlib, Plotly, SciPy), 2 (Pandas, Matplotlib), 3 (Pandas, Plotly)

- **NumPy**: arrays, broadcasting, vectorized operations, linear algebra
- **Pandas**: DataFrames, indexing, groupby, merging, time series
- **Matplotlib / Seaborn**: static visualizations, plots, heatmaps
- **Plotly / Bokeh**: interactive visualizations
- **SciPy**: statistical tests, signal processing, optimization

### 2.3 Software Engineering Best Practices
> **Covered by Projects:** 1 (Git, pytest, logging, clean code), all projects (version control, Docker, APIs)

- Git and version control workflows (branching, PRs, rebasing)
- Clean code principles (SOLID, DRY, KISS)
- Writing unit tests and integration tests (pytest)
- Code profiling and performance optimization
- Logging and debugging
- Documentation (docstrings, Sphinx)
- Design patterns relevant to ML (Factory, Strategy, Pipeline)

### 2.4 Data Engineering Basics
> **Covered by Projects:** 3 (ETL for time series), 11 (DVC + S3, Airflow/Prefect), 13 (Kafka streaming, Redis)

- SQL fundamentals and advanced queries (joins, window functions, CTEs)
- NoSQL concepts (key-value, document, graph databases)
- Data pipelines and ETL concepts
- Apache Spark basics for large-scale data
- Introduction to Airflow for workflow orchestration
- Data formats: CSV, JSON, Parquet, Avro, HDF5
- Data versioning with DVC

---

## Phase 3 — Classical Machine Learning {#phase-3}

### 3.1 Core Concepts
> **Covered by Projects:** 1 (bias-variance, cross-validation, regression metrics), 2 (classification metrics, stratified k-fold), 3 (time series CV)

- Types of ML: Supervised, Unsupervised, Semi-supervised, Self-supervised
- Bias-variance tradeoff
- Underfitting and overfitting
- Training, validation, and test splits
- Cross-validation strategies (k-fold, stratified, time-series split)
- Evaluation metrics:
  - Classification: Accuracy, Precision, Recall, F1, AUC-ROC, PR Curve
  - Regression: MAE, MSE, RMSE, R², MAPE
  - Clustering: Silhouette score, Davies-Bouldin

### 3.2 Supervised Learning
> **Covered by Projects:** 1 (Linear/Ridge/Lasso/RF/XGBoost), 2 (Logistic/RF/LightGBM/CatBoost), 3 (LightGBM for forecasting), 7 (matrix factorisation/ALS), 13 (XGBoost + NN ensemble)

- **Linear Regression**: OLS, ridge, lasso, elastic net
- **Logistic Regression**: binary and multinomial
- **Decision Trees**: CART, pruning, impurity measures
- **Random Forests**: bagging, feature importance
- **Gradient Boosting**: XGBoost, LightGBM, CatBoost
- **Support Vector Machines (SVM)**: kernel trick, soft margin
- **k-Nearest Neighbors (kNN)**
- **Naive Bayes**: Gaussian, Multinomial, Bernoulli
- Ensemble methods: Bagging, Boosting, Stacking, Blending

### 3.3 Unsupervised Learning
> **Covered by Projects:** 13 (anomaly detection — Isolation Forest concepts), 16 (self-supervised pretraining on tabular data)

- **Clustering**: k-Means, DBSCAN, Hierarchical clustering, Gaussian Mixture Models
- **Dimensionality Reduction**: PCA, t-SNE, UMAP, LDA
- **Anomaly Detection**: Isolation Forest, One-Class SVM, Autoencoders
- **Association Rules**: Apriori, FP-Growth

### 3.4 Feature Engineering & Preprocessing
> **Covered by Projects:** 1 (imputation, encoding, scaling, outliers), 2 (SMOTE, class weights, threshold tuning), 3 (lag features, rolling stats, calendar features), 13 (velocity features, behavioural patterns, extreme imbalance)

- Handling missing data (imputation strategies)
- Encoding categorical variables (one-hot, label, target, ordinal)
- Feature scaling (StandardScaler, MinMaxScaler, RobustScaler)
- Feature selection (filter, wrapper, embedded methods)
- Handling imbalanced datasets (SMOTE, class weights, undersampling)
- Outlier detection and treatment
- Feature crosses and polynomial features

### 3.5 Hyperparameter Tuning
> **Covered by Projects:** 1 (Optuna — Bayesian optimisation), 2 (grid/random search, learning curves)

- Grid search and random search
- Bayesian optimization (Optuna, Hyperopt)
- Early stopping
- Learning curves analysis

---

## Phase 4 — Deep Learning & Neural Networks {#phase-4}

### 4.1 Neural Network Fundamentals
> **Covered by Projects:** 4 (forward/backprop, activations, loss functions), 7 (NCF — neural collaborative filtering)

- Biological vs artificial neurons
- Perceptron and multi-layer perceptron (MLP)
- Activation functions: Sigmoid, Tanh, ReLU, Leaky ReLU, GELU, Swish, Softmax
- Forward propagation
- Backpropagation and chain rule in depth
- Loss functions: BCE, CCE, MSE, Huber, Focal Loss
- Weight initialization strategies (Xavier, He, Glorot)

### 4.2 Optimization for Deep Learning
> **Covered by Projects:** 4 (Adam, LR schedules, warmup), 5 (TensorRT inference optimisation), 6 (AdamW for BERT fine-tuning)

- Stochastic Gradient Descent (SGD)
- Momentum and Nesterov momentum
- AdaGrad, RMSProp, Adam, AdamW, Lion
- Learning rate schedules: step decay, cosine annealing, warmup
- Gradient clipping

### 4.3 Regularization Techniques
> **Covered by Projects:** 4 (Dropout, BatchNorm, Albumentations augmentation, early stopping), 6 (label smoothing, data augmentation for NLP)

- L1 / L2 weight regularization
- Dropout and DropConnect
- Batch Normalization, Layer Normalization, Group Normalization
- Data augmentation
- Early stopping and model checkpointing
- Label smoothing

### 4.4 Deep Learning Frameworks
> **Covered by Projects:** 4 (PyTorch — DataLoader, nn.Module, AMP, training loop), 5 (PyTorch), 6 (Transformers + PyTorch), 7 (PyTorch NCF)

- **PyTorch** (primary):
  - Tensors, autograd, `nn.Module`
  - Custom datasets and DataLoaders
  - Training loops, optimizers, schedulers
  - Mixed precision training (AMP)
  - Distributed training (DDP, FSDP)
- **TensorFlow / Keras**:
  - Sequential and functional API
  - Custom layers and training loops
  - TensorBoard visualization
- **JAX** (advanced): JIT, vmap, pmap, Flax/Haiku

### 4.5 Convolutional Neural Networks (CNNs)
> **Covered by Projects:** 4 (ResNet, EfficientNet, transfer learning, Grad-CAM), 5 (YOLOv8, mAP evaluation, TensorRT)

- Convolution operation, padding, stride, dilation
- Pooling layers
- CNN architectures: LeNet, AlexNet, VGG, ResNet, InceptionNet, DenseNet, EfficientNet
- Transfer learning and fine-tuning
- Object detection: YOLO, SSD, Faster R-CNN, DETR
- Semantic segmentation: FCN, U-Net, DeepLab

### 4.6 Recurrent Neural Networks (RNNs)
> **Covered by Projects:** 3 (LSTM for multi-horizon sales forecasting)

- Vanilla RNN and vanishing gradient problem
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)
- Bidirectional RNNs
- Sequence-to-sequence models
- Attention mechanism basics

---

## Phase 5 — Natural Language Processing (NLP) {#phase-5}

### 5.1 Text Preprocessing
> **Covered by Projects:** 6 (tokenisation, lemmatisation, normalisation, regex), 15 (transcript preprocessing)

- Tokenization (word, character, subword)
- Stemming and Lemmatization
- Stop word removal
- Text normalization
- Regular expressions for NLP

### 5.2 Text Representations
> **Covered by Projects:** 6 (TF-IDF + ML baseline, word embeddings), 7 (user/item embeddings for RecSys)

- Bag of Words (BoW) and TF-IDF
- Word embeddings: Word2Vec (CBOW, Skip-gram), GloVe, FastText
- Contextual embeddings
- Subword tokenization: BPE, WordPiece, SentencePiece, Unigram

### 5.3 Transformer Architecture (Deep Dive)
> **Covered by Projects:** 6 (BERT self-attention, encoder-only), 7 (two-tower embeddings), 8 (full transformer for RAG embeddings)

- Self-attention mechanism
- Multi-head attention
- Positional encoding (sinusoidal, learned, RoPE, ALiBi)
- Feed-forward layers
- Encoder-only, Decoder-only, Encoder-Decoder architectures
- Pre-LayerNorm vs Post-LayerNorm
- Key-Value caching

### 5.4 Pre-trained Language Models
> **Covered by Projects:** 6 (BERT, DistilBERT fine-tuning), 7 (Sentence-BERT embeddings), 8 (LLaMA 3, Mistral, BGE/E5), 9 (LLaMA 3 / Mistral QLoRA fine-tuning)

- BERT, RoBERTa, DistilBERT, ALBERT, DeBERTa (encoder models)
- GPT series, LLaMA, Mistral, Falcon (decoder models)
- T5, BART, mT5 (seq2seq models)
- Sentence-BERT and bi-encoders for embeddings

### 5.5 Core NLP Tasks
> **Covered by Projects:** 6 (text classification, aspect-based sentiment, NER), 8 (question answering, information extraction), 15 (transcript summarisation)

- Text classification and sentiment analysis
- Named Entity Recognition (NER)
- Part-of-Speech tagging
- Machine translation
- Text summarization (extractive and abstractive)
- Question answering (extractive, abstractive, generative)
- Information extraction and relation extraction
- Coreference resolution

### 5.6 Fine-tuning & Efficient Adaptation
> **Covered by Projects:** 6 (LoRA on domain-specific reviews), 9 (QLoRA / full QLoRA pipeline on LLaMA 3), 14 (SFT before RLHF)

- Full fine-tuning
- LoRA (Low-Rank Adaptation) and QLoRA
- Prefix tuning, Prompt tuning, P-Tuning
- Adapter layers
- PEFT library usage

---

## Phase 6 — Computer Vision {#phase-6}

### 6.1 Image Fundamentals
> **Covered by Projects:** 4 (Albumentations augmentation pipeline, OpenCV), 5 (video frame processing, OpenCV)

- Image representation (pixels, channels, color spaces)
- Image preprocessing and augmentation (albumentations, torchvision)
- OpenCV for image processing

### 6.2 Vision Architectures
> **Covered by Projects:** 12 (ViT, CLIP for vision-language pretraining)

- Vision Transformer (ViT) and variants (DeiT, Swin, BEiT)
- CLIP (Contrastive Language-Image Pretraining)
- Image segmentation: SAM (Segment Anything Model)
- Depth estimation and 3D vision
- Optical flow

### 6.3 Generative Vision Models
> **Covered by Projects:** 12 (Stable Diffusion text-to-image, ControlNet image conditioning)

- Variational Autoencoders (VAE)
- Generative Adversarial Networks (GANs): DCGAN, StyleGAN, CycleGAN, Pix2Pix
- Diffusion models: DDPM, DDIM, Score matching
- Latent diffusion models (Stable Diffusion)
- ControlNet and image conditioning

### 6.4 Multimodal Learning
> **Covered by Projects:** 12 (LLaVA / InternVL vision-language model, image captioning, Visual QA)

- Vision-Language models (BLIP, LLaVA, GPT-4V)
- Image captioning and Visual QA
- Contrastive learning (SimCLR, MoCo, DINO)

---

## Phase 7 — Reinforcement Learning {#phase-7}

### 7.1 RL Fundamentals
> **Covered by Projects:** 14 (MDPs, states/actions/rewards/policy, value function, Bellman equations, exploration vs exploitation)

- Markov Decision Processes (MDPs)
- States, actions, rewards, policy, value function
- Exploration vs exploitation tradeoff
- Bellman equations
- Dynamic programming

### 7.2 Model-Free RL
> **Covered by Projects:** 14 (DQN on CartPole/LunarLander — replay buffer, target network, epsilon-greedy; Double DQN, Dueling DQN)

- Monte Carlo methods
- Temporal Difference learning (TD, SARSA, Q-learning)
- Deep Q-Networks (DQN) and improvements (Double DQN, Dueling DQN, PER)

### 7.3 Policy Gradient Methods
> **Covered by Projects:** 14 (REINFORCE from scratch, A2C, PPO — all implemented and compared)

- REINFORCE algorithm
- Actor-Critic methods (A2C, A3C)
- Proximal Policy Optimization (PPO)
- Trust Region Policy Optimization (TRPO)
- Soft Actor-Critic (SAC)

### 7.4 RL from Human Feedback (RLHF)
> **Covered by Projects:** 14 (reward model from HH-RLHF dataset, PPO loop via TRL, DPO as alternative, base→SFT→RLHF→DPO comparison)

- Reward modeling from human preferences
- PPO for LLM alignment
- Direct Preference Optimization (DPO)
- Constitutional AI

### 7.5 Advanced RL Topics
> **Covered by Projects:** 14 (OpenAI Gym / Gymnasium environments, multi-environment training)

- Multi-agent RL
- Model-based RL
- Hierarchical RL
- RL environments: OpenAI Gym, MuJoCo, Isaac Gym

---

## Phase 8 — MLOps & Production Systems {#phase-8}

### 8.1 ML Project Lifecycle
> **Covered by Projects:** 4 (full lifecycle for image classifier), 11 (complete ML platform lifecycle)

- Problem framing and feasibility analysis
- Data collection and labeling strategies
- Experimentation and model development
- Model evaluation and fairness auditing
- Deployment and monitoring
- Model maintenance and retraining

### 8.2 Experiment Tracking & Model Registry
> **Covered by Projects:** 4 (MLflow — tracking + registry), 6 (W&B), 11 (MLflow hosted on EC2, staging/prod/archived states)

- MLflow: tracking, projects, registry
- Weights & Biases (W&B)
- Neptune, Comet ML
- DVC for data and model versioning

### 8.3 Model Serving & Deployment
> **Covered by Projects:** 1 (FastAPI REST), 4 (FastAPI + ONNX), 5 (FastAPI + TensorRT), 8 (FastAPI + LangChain), 10 (FastAPI + WebSocket streaming), 13 (FastAPI <10ms latency), 15 (WebSocket real-time ASR)

- REST API serving with FastAPI / Flask
- TorchServe, TensorFlow Serving
- Triton Inference Server (NVIDIA)
- ONNX export and runtime
- gRPC for high-performance inference
- Batch inference vs real-time inference

### 8.4 Containerization & Orchestration
> **Covered by Projects:** 1 (Docker), 4 (Docker + GCP Cloud Run), 8 (Docker Compose), 11 (Docker + Kubernetes basics + ECS), 15 (Docker + HuggingFace Spaces)

- Docker: writing Dockerfiles, multi-stage builds
- Docker Compose for local development
- Kubernetes basics: pods, deployments, services, autoscaling
- Helm charts for ML deployments
- Kubeflow and MLflow on Kubernetes

### 8.5 Cloud Platforms for ML
> **Covered by Projects:** 4 (GCP Cloud Run), 8 (AWS EC2 auto-scaling), 10 (FastAPI + React on cloud), 11 (AWS ECS, ECR, S3), 12 (GCP GPU instance)

- **AWS**: SageMaker, EC2, S3, Lambda, ECR
- **GCP**: Vertex AI, GCS, BigQuery, Cloud Run
- **Azure**: Azure ML, Blob Storage, AKS
- Serverless ML inference
- GPU/TPU cloud instances

### 8.6 CI/CD for ML
> **Covered by Projects:** 6 (A/B testing two model versions), 11 (GitHub Actions → test → Docker → ECR → ECS, canary deployment 10% traffic split)

- GitHub Actions / GitLab CI for ML pipelines
- Automated testing for ML (data validation, model performance)
- Blue-green and canary deployments for models
- Shadow mode deployments

### 8.7 Monitoring & Observability
> **Covered by Projects:** 4 (Evidently AI drift), 6 (W&B model performance), 8 (Prometheus + Grafana), 10 (LangSmith / Arize Phoenix tracing), 11 (Evidently + Prometheus + Grafana + Feast + PagerDuty alerts), 13 (real-time fraud dashboard, SHAP explainability)

- Data drift detection (Evidently AI, NannyML)
- Model performance monitoring
- Feature store concepts (Feast, Tecton, Hopsworks)
- Logging and alerting (Prometheus, Grafana, ELK stack)
- A/B testing for models

### 8.8 Model Optimization
> **Covered by Projects:** 4 (ONNX export, TorchScript), 5 (TensorRT, edge deployment), 9 (GGUF quantisation, llama.cpp, vLLM serving)

- Quantization: INT8, INT4, mixed precision (post-training and quantization-aware)
- Pruning: structured and unstructured
- Knowledge distillation
- Model compilation: TorchScript, `torch.compile`, XLA
- TensorRT and OpenVINO
- Flash Attention and efficient kernels

---

## Phase 9 — Generative AI & LLMs {#phase-9}

### 9.1 LLM Architecture Deep Dive
> **Covered by Projects:** 9 (autoregressive modelling, scaling laws, tokenisation, architecture deep dive before fine-tuning)

- Autoregressive language modeling
- Scaling laws (Chinchilla, compute-optimal training)
- Tokenization at scale
- Context length extensions (RoPE scaling, LongFormer, Mamba)
- Mixture of Experts (MoE)
- State Space Models (SSM): Mamba, RWKV

### 9.2 LLM Training
> **Covered by Projects:** 9 (SFT, instruction tuning, QLoRA), 14 (SFT → RLHF → DPO pipeline, KL divergence monitoring)

- Pre-training pipeline
- Supervised Fine-Tuning (SFT)
- Instruction tuning
- RLHF and DPO for alignment
- Constitutional AI
- Distributed training: FSDP, DeepSpeed, Megatron-LM

### 9.3 Prompt Engineering
> **Covered by Projects:** 8 (query rewriting, structured output prompting), 10 (ReAct framework, Chain-of-Thought, system prompt design)

- Zero-shot, few-shot, and many-shot prompting
- Chain-of-Thought (CoT) and Tree-of-Thought (ToT)
- ReAct framework
- Self-consistency and majority voting
- Structured output prompting
- System prompt design

### 9.4 Retrieval-Augmented Generation (RAG)
> **Covered by Projects:** 8 (Qdrant/pgvector, chunking, dense+BM25 hybrid retrieval, cross-encoder re-ranking, HyDE, multi-hop reasoning, RAGAS evaluation — full RAG depth)

- Vector databases: Pinecone, Weaviate, Qdrant, pgvector, FAISS, Chroma
- Embedding models for retrieval
- Chunking strategies
- Retrieval methods: dense, sparse (BM25), hybrid
- Re-ranking models (cross-encoders)
- Advanced RAG: query rewriting, HyDE, FLARE, multi-hop reasoning
- Evaluation: RAGAS framework

### 9.5 LLM Application Development
> **Covered by Projects:** 8 (LangChain, LlamaIndex, chat interface), 10 (LangGraph / CrewAI agents, tool use, memory systems, streaming, code execution), 12 (voice + vision multimodal app)

- LangChain and LlamaIndex frameworks
- Tool use and function calling
- AI Agents and multi-agent systems (AutoGen, CrewAI, LangGraph)
- Memory systems for agents (short-term, long-term, episodic)
- Code generation and execution (Code Interpreter pattern)
- Streaming responses and async patterns

### 9.6 LLM Evaluation
> **Covered by Projects:** 9 (MT-Bench, LLM-as-Judge, domain benchmarks, base vs fine-tuned comparison), 8 (RAGAS — faithfulness, answer relevancy, context recall)

- Benchmarks: MMLU, HellaSwag, HumanEval, GSM8K, MATH, BIG-Bench
- LLM-as-a-Judge frameworks
- Human evaluation best practices
- Red-teaming and adversarial testing
- Hallucination detection and mitigation

### 9.7 LLM Safety & Alignment
> **Covered by Projects:** 16 (NeMo Guardrails, Llama Guard 3, red-teaming 10 attack types, DP-SGD, privacy-utility tradeoff, toxicity/hallucination/refusal dashboard)

- Types of AI risks (bias, toxicity, hallucination, jailbreaks)
- Guardrails and content filtering (NeMo Guardrails, Llama Guard)
- Responsible AI frameworks
- Differential privacy in ML
- Federated learning concepts

---

## Phase 10 — Specialized Domains & Capstone {#phase-10}

### 10.1 Time Series & Forecasting
> **Covered by Projects:** 3 (ARIMA, SARIMA, Holt-Winters, LightGBM with lag features, LSTM, TFT, multi-horizon 7/30/90-day, MAPE/RMSE/MASE evaluation)

- Time series decomposition (trend, seasonality, residuals)
- Classical methods: ARIMA, SARIMA, Exponential Smoothing
- ML methods: LightGBM for forecasting, feature engineering for time series
- Deep learning: N-BEATS, Temporal Fusion Transformer, PatchTST
- Evaluation: MASE, MAPE, WQL

### 10.2 Recommender Systems
> **Covered by Projects:** 7 (collaborative filtering, ALS, NCF, two-tower, SASRec, cold-start, Feast feature store, NDCG/MAP/Hit Rate, A/B simulation)

- Collaborative filtering (user-based, item-based, matrix factorization)
- Content-based filtering
- Hybrid systems
- Neural collaborative filtering
- Two-tower models
- Sequential recommendation (BERT4Rec, SASRec)
- Cold start problem

### 10.3 Graph Neural Networks (GNN)
> **Covered by Projects:** 13 (GCN for transaction network features, message passing framework, fraud detection application)

- Graph representation and types
- Graph Convolutional Networks (GCN)
- GraphSAGE, GAT (Graph Attention Network)
- Message passing framework
- Knowledge graphs and link prediction
- Applications: fraud detection, molecule property prediction

### 10.4 Audio & Speech
> **Covered by Projects:** 15 (FFT, MFCC, Mel spectrograms via Librosa, Whisper ASR fine-tuning on Indian languages, Wav2Vec 2.0, VITS/Coqui TTS, pyannote speaker diarization, real-time WebSocket streaming)

- Audio signal processing (FFT, MFCC, Mel spectrograms)
- Speech recognition: Wav2Vec 2.0, Whisper
- Text-to-speech: Tacotron, VITS
- Music generation
- Speaker diarization and verification

### 10.5 Tabular Deep Learning
> **Covered by Projects:** 16 (TabNet, FT-Transformer, SAINT benchmarked against XGBoost/LightGBM; SCARF/VIME self-supervised pretraining; analysis of when DL beats tree-based)

- TabNet, SAINT, FT-Transformer
- When to use DL vs tree-based models for tabular data
- Self-supervised pretraining on tabular data

### 10.6 ML System Design
> **Covered by Projects:** 11 (Lambda/Kappa architecture, feature store patterns, scalable ML pipelines, online vs offline evaluation), 13 (real-time streaming architecture, <10ms latency design)

- Designing scalable ML pipelines
- Feature stores and real-time feature serving
- Online vs offline evaluation
- Handling data at scale
- ML infrastructure patterns (Lambda architecture, Kappa architecture)
- Case studies: recommendation systems, search ranking, fraud detection, content moderation

### 10.7 Capstone Projects (Choose 2–3)
> **Covered by Projects:** 8 (RAG chatbot), 9 (LLM fine-tuning), 10 (multi-agent system), 12 (diffusion + multimodal), 14 (RL + RLHF), 15 (speech platform)

- Build an end-to-end RAG chatbot with custom data
- Fine-tune a LLM with LoRA on domain-specific data
- Build a real-time ML inference API with monitoring
- Implement an object detection system deployed on cloud
- Build a recommendation engine with A/B testing
- Train a diffusion model on a custom image dataset
- Build a multi-agent system for a complex task

---

## References

### Books
- *Mathematics for Machine Learning* — Deisenroth, Faisal, Ong
- *Pattern Recognition and Machine Learning* — Bishop
- *Deep Learning* — Goodfellow, Bengio, Courville
- *Designing Machine Learning Systems* — Chip Huyen
- *Building Machine Learning Powered Applications* — Emmanuel Ameisen

### Lectures & Talks
- Fast.ai — Practical Deep Learning
- Stanford CS229 (ML), CS231n (CV), CS224n (NLP)
- Full Stack Deep Learning (FSDL)

### Stay Sharp
- Kaggle competitions
- Papers With Code
- Hugging Face Hub
- GitHub — open-source ML contributions

---

*Last Updated: May 2026*
