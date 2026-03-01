# NLP Utils

A collection of utility functions and tools for Natural Language Processing tasks.

## Description

This repository provides a set of reusable NLP utilities designed to simplify common text processing tasks. It serves as a personal toolkit for various NLP projects, offering modular, well-documented functions that can be easily integrated into larger applications.

## Features (Planned)

- **Text Preprocessing**: Cleaning, tokenization, normalization
- **Feature Extraction**: TF-IDF, word embeddings, n-grams
- **Text Analysis**: Sentiment analysis, keyword extraction, text statistics
- **Data Utilities**: Data loading, format conversion, batch processing
- **Visualization**: Text visualization tools

## Project Structure

```
NLP_utils/
├── README.md           # This file
├── nlp_utils/          # Main package
│   ├── __init__.py
│   ├── preprocessing.py    # Text cleaning and preprocessing
│   ├── tokenization.py     # Tokenization utilities
│   ├── embeddings.py       # Word embedding tools
│   ├── features.py         # Feature extraction
│   ├── similarity.py       # Text similarity measures
│   └── visualization.py    # Text visualization
├── tests/              # Unit tests
│   ├── __init__.py
│   └── test_*.py
├── examples/           # Usage examples
│   └── example_usage.py
├── docs/               # Documentation
├── requirements.txt    # Dependencies
└── setup.py           # Package setup
```

## Installation

### Prerequisites

- Python 3.6+
- Dependencies: numpy, pandas, scikit-learn, nltk, spaCy

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd NLP_utils
```

2. Install the package:
```bash
pip install -e .
```

Or install dependencies manually:
```bash
pip install -r requirements.txt
```

## Usage

### Text Preprocessing

```python
from nlp_utils.preprocessing import clean_text, normalize_text

# Clean text
text = "Your raw text here! Check out https://example.com"
cleaned = clean_text(text, remove_urls=True, remove_punctuation=True)

# Normalize text
normalized = normalize_text(cleaned, lowercase=True, lemmatize=True)
```

### Tokenization

```python
from nlp_utils.tokenization import word_tokenize, sentence_tokenize

words = word_tokenize("This is a sample sentence.")
sentences = sentence_tokenize("First sentence. Second sentence.")
```

### Feature Extraction

```python
from nlp_utils.features import extract_tfidf, extract_ngrams

# TF-IDF vectorization
vectorizer, vectors = extract_tfidf(documents, max_features=1000)

# N-gram extraction
ngrams = extract_ngrams(text, n=2)
```

### Text Similarity

```python
from nlp_utils.similarity import cosine_similarity, jaccard_similarity

similarity = cosine_similarity(text1, text2)
```

## Module Details

### preprocessing.py

Functions for cleaning and normalizing text:
- `clean_text()`: Remove unwanted characters, URLs, emails
- `normalize_text()`: Lowercase, lemmatize, stem
- `remove_stopwords()`: Filter out common stopwords
- `fix_encoding()`: Handle encoding issues

### tokenization.py

Tokenization utilities:
- `word_tokenize()`: Split text into words
- `sentence_tokenize()`: Split text into sentences
- `character_tokenize()`: Character-level tokenization
- `subword_tokenize()`: BPE or WordPiece tokenization

### embeddings.py

Word embedding tools:
- `load_glove()`: Load GloVe embeddings
- `load_word2vec()`: Load Word2Vec model
- `get_embeddings()`: Extract embeddings for text
- `compute_similarity_matrix()`: Word similarity matrix

### features.py

Feature extraction methods:
- `extract_tfidf()`: TF-IDF features
- `extract_ngrams()`: N-gram features
- `extract_pos_tags()`: Part-of-speech features
- `extract_named_entities()`: NER features

### similarity.py

Text similarity measures:
- `cosine_similarity()`: Cosine similarity
- `jaccard_similarity()`: Jaccard similarity
- `levenshtein_distance()`: Edit distance
- `semantic_similarity()`: Embedding-based similarity

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-utility`)
3. Commit your changes (`git commit -am 'Add new utility'`)
4. Push to the branch (`git push origin feature/new-utility`)
5. Create a Pull Request

## Future Enhancements

- [ ] Add support for multilingual text processing
- [ ] Integrate transformer-based utilities
- [ ] Add benchmarking tools
- [ ] Create comprehensive documentation
- [ ] Add Jupyter notebook tutorials

## Dependencies

Core dependencies:
- numpy
- pandas
- scikit-learn
- nltk
- spacy
- gensim

Optional dependencies:
- transformers (for BERT utilities)
- matplotlib (for visualization)
- jieba (for Chinese text processing)

## License

[License information to be added]

## Contact

For questions or suggestions, please open an issue on this repository.

## Acknowledgments

- NLTK and spaCy communities for excellent NLP tools
- scikit-learn for machine learning utilities
