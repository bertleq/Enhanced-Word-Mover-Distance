# Word Mover's Distance with N-Gram Support

This implementation provides Word Mover's Distance (WMD) calculation between two texts with the following features:

## Features

- **Multi-gram Support**: Uses 1-gram, 2-gram, and 3-gram tokenization
- **Automatic Best Selection**: Automatically selects the best n-gram combination that minimizes distance
- **Asymmetric Penalties**: 
  - Deletion penalty: 3.0x (removing words from text1)
  - Addition penalty: 1.0x (adding words to match text2)
- **Modern Embeddings**: Uses sentence-transformers for high-quality semantic embeddings
- **Fallback Mode**: Works even without pretrained embeddings using hash-based vectors

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from word_mover_distance import WordMoverDistance

# Initialize calculator
wmd = WordMoverDistance(
    deletion_penalty=3.0,
    addition_penalty=1.0,
    use_pretrained_embeddings=True
)

# Compare two texts
text1 = "The quick brown fox jumps over the lazy dog"
text2 = "A fast brown fox leaps over a sleepy dog"

# Get minimum distance across all n-grams
distance = wmd.compute(text1, text2)
print(f"WMD: {distance}")

# Get all n-gram scores
scores = wmd.compute(text1, text2, return_all_scores=True)
print(scores)
# Output: {'1-gram': 0.XX, '2-gram': 0.YY, '3-gram': 0.ZZ, 'best': 0.XX, 'best_ngram': '1-gram'}

# Get similarity score (0-1 scale)
similarity = wmd.compute_similarity(text1, text2)
print(f"Similarity: {similarity}")
```

### Convenience Function

```python
from word_mover_distance import compute_word_mover_distance

distance = compute_word_mover_distance(
    "The cat sat on the mat",
    "The cat sat",
    deletion_penalty=3.0,
    addition_penalty=1.0
)
```

## How It Works

1. **N-gram Extraction**: Extracts 1-grams (words), 2-grams (word pairs), and 3-grams (word triplets) from both texts

2. **Embedding Generation**: Converts each n-gram to a semantic vector using sentence-transformers (MiniLM model by default)

3. **Distance Calculation**: Computes cosine distance between all n-gram pairs

4. **Optimal Transport**: Uses the Hungarian algorithm (linear sum assignment) to find the minimum cost matching between n-grams

5. **Penalty Application**: 
   - Unmatched n-grams from text1 incur deletion penalty (3.0x)
   - Unmatched n-grams from text2 incur addition penalty (1.0x)

6. **Best Selection**: Returns the minimum distance across all n-gram sizes

## Parameters

- `deletion_penalty` (float, default=3.0): Multiplier for the cost of deleting words
- `addition_penalty` (float, default=1.0): Multiplier for the cost of adding words  
- `use_pretrained_embeddings` (bool, default=True): Whether to use sentence-transformers
- `embedding_model` (str, default='all-MiniLM-L6-v2'): Which embedding model to use

## Example Output

```
Text 1: The quick brown fox jumps over the lazy dog
Text 2: A fast brown fox leaps over a sleepy dog

WMD Scores:
  1-gram: 0.3245
  2-gram: 0.2891
  3-gram: 0.3512
  best: 0.2891
  best_ngram: 2-gram

Similarity: 0.7489
```

## Notes

- Higher deletion penalty means the algorithm penalizes removing content more heavily
- The algorithm automatically tries all specified n-gram sizes and picks the best one
- Distance is normalized by the number of n-grams for comparability
- Similarity score uses exponential decay: `similarity = exp(-distance)`
