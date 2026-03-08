# NLP Utils

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A collection of machine learning algorithms and NLP utilities implemented from scratch for educational purposes.

## Description

This repository provides Python implementations of classic machine learning and NLP algorithms. While some modules are optimized for performance, others are designed primarily for educational purposes to demonstrate algorithm internals.

## Project Structure

```
NLP_utils/
├── Coding/                    # Algorithm implementations
│   ├── edit_distance.py       # Edit distance (Educational)
│   ├── KMP.py                 # KMP pattern matching (Educational)
│   ├── lda.py                 # LDA topic model
│   └── ...
├── ML/                        # Machine Learning algorithms
│   ├── decision_Tree.py       # ID3 Decision Tree
│   ├── Naive_bayesian.py      # Naive Bayes classifier
│   ├── veterbi.py             # Viterbi algorithm for HMM
│   └── ...
├── Deep_learning/             # Deep learning utilities
├── InfoRetrive/               # Information retrieval
├── benchmark/                 # Performance testing framework
│   ├── README.md
│   ├── run_all.py
│   └── requirements.txt
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.7+
- numpy
- scikit-learn (for benchmark comparisons)

### Setup

```bash
git clone https://github.com/fooSynaptic/NLP_utils.git
cd NLP_utils
pip install -r benchmark/requirements.txt
```

## Usage

### Decision Tree (ID3)

```python
from ML.decision_Tree import ID3Base

# Create and train model
tree = ID3Base(max_depth=10)
tree.fit(X_train, y_train)

# Predict
predictions = [tree.predict(x) for x in X_test]
```

### Viterbi Algorithm

```python
from ML.veterbi import viterbi

# Define HMM parameters
states = ('Rainy', 'Sunny')
observations = ('walk', 'shop', 'clean')
start_prob = {'Rainy': 0.6, 'Sunny': 0.4}
trans_prob = {...}
emit_prob = {...}

# Decode
prob, path = viterbi(observations, states, start_prob, trans_prob, emit_prob)
```

## Performance Benchmark

We provide a benchmark framework to compare our implementations with popular libraries.

### Running Benchmarks

```bash
cd benchmark
python run_all.py
```

### Results

| Algorithm | NLP_utils | Reference | Speedup | Status |
|-----------|-----------|-----------|---------|--------|
| Decision Tree | 0.146s | 0.437s (sklearn) | **3.00x** | WIN |
| Viterbi | 0.034s | 0.051s (hmmlearn) | **1.50x** | WIN |
| Edit Distance | 0.360s | 0.074s (pure Python) | 0.20x | Educational |
| KMP Search | 0.051s | 0.002s (re module) | 0.04x | Educational |

### Performance Notes

**High Performance (Production Ready):**
- **Decision Tree**: 3x faster than sklearn on small datasets
- **Viterbi**: 1.5x faster than hmmlearn

**Educational Implementations:**
- **Edit Distance**: Pure Python DP implementation. For production, use `python-Levenshtein` or `rapidfuzz`
- **KMP**: Demonstrates algorithm internals. For production, use Python's built-in `re` module

## Modules

### Coding/

Algorithm implementations with varying optimization levels:

| File | Purpose | Performance |
|------|---------|-------------|
| `edit_distance.py` | Levenshtein distance | Educational |
| `KMP.py` | Pattern matching | Educational |
| `lda.py` | Topic modeling | Moderate |
| `Dijkstra_Floyd.py` | Shortest path | Moderate |

### ML/

Machine learning algorithms:

| File | Purpose | Performance |
|------|---------|-------------|
| `decision_Tree.py` | ID3 algorithm | High |
| `Naive_bayesian.py` | Text classification | Moderate |
| `veterbi.py` | HMM decoding | High |
| `KD_tree.py` | K-NN search | Moderate |

## Testing

Run the benchmark suite:

```bash
cd benchmark
python run_all.py
```

View detailed results:
```bash
cat benchmark/benchmark_report.md
```

## Contributing

Contributions are welcome! Please ensure:

1. Code follows PEP 8 style guidelines
2. Add docstrings for all public functions
3. Include benchmark tests for new algorithms
4. Update README.md with module descriptions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

Some modules in this repository are implemented for educational purposes and may not be optimized for production use. Please refer to the performance benchmark section and choose appropriate implementations for your use case.

## Acknowledgments

- scikit-learn for providing reference implementations
- hmmlearn for HMM utilities
- NLTK and spaCy communities for NLP tools
