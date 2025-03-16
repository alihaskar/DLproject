# Deep Reinforcement Learning Trading System

This is a sophisticated trading system that combines three major components:

## 1. Market Regime Detection (`src/regimes/`)
- Implements multiple methods to detect market states:
  - Rule-based detection
  - Hidden Markov Models (using `hmmlearn`)
  - Transformer-based detection
- Files:
  - `market_regime_detector.py`: Core implementation (458 lines)
  - `compare_regime_performance.py`: Performance comparison (259 lines)
  - `regime_performance_analysis.py`: Analysis tools (221 lines)
  - `run_detector.py`: CLI interface

The regime detector classifies market states into:
- Uptrend
- Downtrend
- Mean reversion
- Volatile
- Neutral

## 2. Metalabeling System (`src/metalabel/`)
- Uses the Triple-Barrier Method for labeling
- Main implementation in `triple_barrier.py` (563 lines)
- Features:
  - Dynamic threshold calculation
  - Machine learning-based signal filtering
  - Performance analysis tools
- Includes CLI interface in `run_metalabeling.py`

## 3. Reinforcement Learning (`src/RL/`)
Implements three major RL algorithms:
1. DQN (Deep Q-Network)
2. PPO (Proximal Policy Optimization)
3. SAC (Soft Actor-Critic)

Each algorithm has its own subdirectory with implementation.

## Integration Layer
- Main class `DRL` in `src/drl.py` integrates all components
- Lazy loading of components through properties
- Unified interface for all functionalities

## Installation

1. Make sure you have Python 3.8+ installed
2. Install Poetry (dependency management):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Clone the repository and install dependencies:
```bash
git clone <repository-url>
cd DLProject
poetry install
```

## Dependencies (from pyproject.toml)
Core libraries:
- Python 3.12
- PyTorch 2.1.2
- pandas 2.2.0
- scikit-learn 1.4.0
- hmmlearn 0.3.0
- gym 0.26.0 (for RL environments)
- ta 0.11.0 (technical analysis)

## Project Structure
```
.
├── data/           # Trading data (gitignored)
├── models/         # Saved models (gitignored)
├── reports/        # Generated reports (gitignored)
├── src/           
│   ├── regimes/    # Market regime detection
│   ├── metalabel/  # Metalabeling implementation
│   └── RL/         # Reinforcement learning algorithms
└── tests/          # Unit tests
```

## Running the Code

The system can be run in different modes using the `run_me.py` script:

### All Components
To run all components sequentially:
```bash
poetry run python run_me.py --mode all
```

### Individual Components

1. Market Regime Detection:
```bash
poetry run python run_me.py --mode regimes
```

2. Metalabeling:
```bash
poetry run python run_me.py --mode metalabel
```

3. Deep Q-Network (DQN):
```bash
poetry run python run_me.py --mode dqn
```

4. Proximal Policy Optimization (PPO):
```bash
poetry run python run_me.py --mode ppo
```

5. Soft Actor-Critic (SAC):
```bash
poetry run python run_me.py --mode sac
```

### Custom Data Path
You can specify a custom data path:
```bash
poetry run python run_me.py --mode all --data_path data/custom_data.csv
```

## Output Structure
- Regime detection results → `reports/regimes/`
- Metalabeling results → `reports/metalabel/`
- Trained models → `models/`
- Performance reports → `reports/`

The project is a comprehensive trading system that combines traditional market analysis (regime detection), machine learning (metalabeling), and deep reinforcement learning for optimal trading decisions. The codebase is well-structured with clear separation of concerns and modular design.
