# Deep Reinforcement Learning Trading System

**Course:** AAI612: Deep Learning & its Applications  
**Institution:** Lebanese American University  
**Author:** Ali Askar  
**Project Title:** Deep Reinforcement Learning Trading System  
**Submission Date:** 16/3/2025  
**GitHub Repository:** [DLproject](https://github.com/alihaskar/DLproject.git)

## Abstract
This project presents the development of a sophisticated Deep Reinforcement Learning (DRL) Trading System. The system integrates three primary components: Market Regime Detection, Metalabeling, and Reinforcement Learning. The objective is to leverage traditional and modern AI methodologies to enhance the accuracy and efficiency of trading decisions. The project employs advanced machine learning techniques, including transformer-based models, hidden Markov models, and various reinforcement learning algorithms to develop an adaptive and robust trading framework.

## 1. Introduction
In quantitative finance, adapting to dynamic market conditions is essential for strategy robustness. This project integrates market regime detection, metalabeling using the triple-barrier method, and reinforcement learning algorithms to construct an intelligent trading system capable of adjusting to evolving market states.

## 2. System Architecture

### 2.1 Market Regime Detection (`src/regimes/`)
- **Objective:** Identify and classify current market states to tailor trading strategies.
- **Methodologies:**
  - Rule-based detection
  - Hidden Markov Models (using `hmmlearn`)
  - Transformer-based detection for dynamic pattern recognition
- **Files:**
  - `market_regime_detector.py`: Core detection implementation (458 lines)
  - `compare_regime_performance.py`: Performance comparison across detection methods (259 lines)
  - `regime_performance_analysis.py`: Analysis tools for regime effectiveness (221 lines)
  - `run_detector.py`: CLI interface for ease of use
- **Regime Categories:**
  - Uptrend
  - Downtrend
  - Mean Reversion
  - Volatile
  - Neutral

### 2.2 Metalabeling System (`src/metalabel/`)
- **Objective:** Enhance prediction accuracy through signal refinement.
- **Methodologies:**
  - Triple-Barrier Method for dynamic thresholding
  - Machine learning models for signal filtering
- **Files:**
  - `triple_barrier.py`: Main implementation (563 lines)
  - `run_metalabeling.py`: CLI interface for streamlined operations
- **Features:**
  - Dynamic threshold calculation
  - Performance analysis tools

### 2.3 Reinforcement Learning (`src/RL/`)
- **Objective:** Optimize trading decisions through adaptive learning.
- **Implemented Algorithms:**
  - Deep Q-Network (DQN)
  - Proximal Policy Optimization (PPO)
  - Soft Actor-Critic (SAC)
- **Structure:** Each algorithm is encapsulated within its own subdirectory for modularity and clarity.

### 2.4 Integration Layer
- **Main Class:** `DRL` (located in `src/drl.py`)
- **Features:**
  - Lazy loading of system components
  - Unified interface for component interaction and orchestration

## 3. Installation Guide

1. **Pre-requisites:** Python 3.8+
2. **Install Poetry for Dependency Management:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
3. **Clone the Repository and Install Dependencies:**
```bash
git clone https://github.com/alihaskar/DLproject.git
cd DLProject
poetry install
```

## 4. Core Dependencies
- Python 3.12
- PyTorch 2.1.2
- pandas 2.2.0
- scikit-learn 1.4.0
- hmmlearn 0.3.0
- gym 0.26.0
- ta 0.11.0

## 5. Project Directory Structure
```
.
├── data/           # Trading data (gitignored)
├── models/         # Saved models (gitignored)
├── reports/        # Generated reports (gitignored)
├── src/            # Core implementation
│   ├── regimes/    # Market regime detection module
│   ├── metalabel/  # Metalabeling system
│   └── RL/         # Reinforcement learning algorithms
└── tests/          # Unit tests
```

## 6. Running the System

### 6.1 Running All Components Sequentially
```bash
poetry run python run_me.py --mode all
```

### 6.2 Running Individual Components
- **Market Regime Detection:**
```bash
poetry run python run_me.py --mode regimes
```
- **Metalabeling System:**
```bash
poetry run python run_me.py --mode metalabel
```
- **DQN Algorithm:**
```bash
poetry run python run_me.py --mode dqn
```
- **PPO Algorithm:**
```bash
poetry run python run_me.py --mode ppo
```
- **SAC Algorithm:**
```bash
poetry run python run_me.py --mode sac
```

### 6.3 Custom Data Path
```bash
poetry run python run_me.py --mode all --data_path data/custom_data.csv
```

## 7. Output Structure
- **Market Regime Results:** `reports/regimes/`
- **Metalabeling Outputs:** `reports/metalabel/`
- **Saved Models:** `models/`
- **Performance Reports:** `reports/`

## 8. Conclusion
This project successfully integrates market regime detection, metalabeling, and reinforcement learning to create an adaptive trading system. The design emphasizes modularity, performance, and robustness, offering a scalable framework for advanced quantitative trading applications.

## 9. Future Work
- Incorporating online learning for real-time adaptation.
- Enhancing regime detection using unsupervised deep learning techniques.
- Optimization of RL algorithms for faster convergence.

## References
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction.* MIT Press.
- Official documentation of PyTorch, scikit-learn, and hmmlearn.