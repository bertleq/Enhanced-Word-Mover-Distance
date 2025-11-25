"""
Word Mover's Distance implementation with n-gram support and asymmetric penalties.

This module implements Word Mover's Distance (WMD) between two texts using:
- 1-gram, 2-gram, and 3-gram tokenization
- Automatic selection of best n-gram combination
- Asymmetric penalties (higher for deletions than additions)
"""

import numpy as np
from typing import List, Tuple, Dict, Set
from collections import Counter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
import warnings

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn("sentence-transformers not available. Using simple embeddings.")


class WordMoverDistance:
    """
    Calculates Word Mover's Distance between two texts with n-gram support.
    """
    
    def __init__(
        self,
        deletion_penalty: float = 3.0,
        addition_penalty: float = 1.0,
        use_pretrained_embeddings: bool = True,
        embedding_model: str = 'all-MiniLM-L6-v2',
        perturbation_penalty_weight: float = 0.0,
        idf_dict: Dict[str, float] = None
    ):
        """
        Initialize WMD calculator.
        
        Args:
            deletion_penalty: Penalty multiplier for deleting words from text1
            addition_penalty: Penalty multiplier for adding words to match text2
            use_pretrained_embeddings: Whether to use pretrained embeddings
            embedding_model: Name of the sentence-transformers model to use
            perturbation_penalty_weight: Weight for perturbation penalty (0 to disable)
            idf_dict: Dictionary mapping words to IDF weights. If None, uniform weights are used.
        """
        self.deletion_penalty = deletion_penalty
        self.addition_penalty = addition_penalty
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.perturbation_penalty_weight = perturbation_penalty_weight
        self.idf_dict = idf_dict or {}
        
        if use_pretrained_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE:
            self.embedder = SentenceTransformer(embedding_model)
            self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        else:
            self.embedder = None
            self.embedding_dim = 100  # Default dimension for simple embeddings

    def fit(self, corpus: List[str]):
        """
        Compute IDF weights from a corpus.
        
        Args:
            corpus: List of documents (strings)
        """
        doc_count = len(corpus)
        word_doc_counts = Counter()
        
        for doc in corpus:
            # Use set to count each word only once per document
            words = set(doc.lower().split())
            word_doc_counts.update(words)
            
        self.idf_dict = {}
        for word, count in word_doc_counts.items():
            # Standard IDF formula: log(N / (1 + df)) + 1
            self.idf_dict[word] = np.log(doc_count / (1 + count)) + 1.0
            
    def _get_idf_weight(self, ngram: str) -> float:
        """
        Get IDF weight for an n-gram.
        For n-grams (n>1), we take the average IDF of constituent words.
        """
        if not self.idf_dict:
            return 1.0
            
        words = ngram.split()
        weights = [self.idf_dict.get(w, 1.0) for w in words]
        return sum(weights) / len(weights) if weights else 1.0
            
    def _extract_ngrams(self, text: str, n: int) -> List[str]:
        """
        Extract n-grams from text.
        
        Args:
            text: Input text
            n: N-gram size (1, 2, or 3)
            
        Returns:
            List of n-grams
        """
        # Tokenize by splitting on whitespace and converting to lowercase
        tokens = text.lower().split()
        
        if n == 1:
            return tokens
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
        
        return ngrams if ngrams else tokens  # Fallback to unigrams if no n-grams
    
    def _get_embedding(self, ngram: str) -> np.ndarray:
        """
        Get embedding vector for an n-gram.
        
        Args:
            ngram: N-gram string
            
        Returns:
            Embedding vector
        """
        if self.embedder is not None:
            return self.embedder.encode(ngram, convert_to_numpy=True)
        else:
            # Simple hash-based embedding as fallback
            np.random.seed(hash(ngram) % (2**32))
            return np.random.randn(self.embedding_dim)
    
    def _compute_distance_matrix(
        self,
        ngrams1: List[str],
        ngrams2: List[str]
    ) -> np.ndarray:
        """
        Compute pairwise distance matrix between n-grams.
        
        Args:
            ngrams1: N-grams from text 1
            ngrams2: N-grams from text 2
            
        Returns:
            Distance matrix of shape (len(ngrams1), len(ngrams2))
        """
        # Get unique n-grams
        unique_ngrams1 = list(set(ngrams1))
        unique_ngrams2 = list(set(ngrams2))
        
        # Compute embeddings
        embeddings1 = np.array([self._get_embedding(ng) for ng in unique_ngrams1])
        embeddings2 = np.array([self._get_embedding(ng) for ng in unique_ngrams2])
        
        # Compute pairwise cosine distances
        distance_matrix = np.zeros((len(unique_ngrams1), len(unique_ngrams2)))
        
        for i, emb1 in enumerate(embeddings1):
            for j, emb2 in enumerate(embeddings2):
                # Cosine distance
                distance_matrix[i, j] = cosine(emb1, emb2)
        
        # Map back to original ngrams with frequencies
        ngram_counts1 = Counter(ngrams1)
        ngram_counts2 = Counter(ngrams2)
        
        # Create mapping indices
        idx_map1 = {ng: i for i, ng in enumerate(unique_ngrams1)}
        idx_map2 = {ng: i for i, ng in enumerate(unique_ngrams2)}
        
        return distance_matrix, unique_ngrams1, unique_ngrams2, ngram_counts1, ngram_counts2
    
    def _compute_wmd_with_penalties(
        self,
        ngrams1: List[str],
        ngrams2: List[str]
    ) -> float:
        """
        Compute WMD with asymmetric deletion/addition penalties.
        
        Args:
            ngrams1: N-grams from text 1
            ngrams2: N-grams from text 2
            
        Returns:
            Word Mover's Distance
        """
        if not ngrams1 or not ngrams2:
            # Handle empty cases
            if not ngrams1 and not ngrams2:
                return 0.0
            elif not ngrams1:
                return len(ngrams2) * self.addition_penalty
            else:
                return len(ngrams1) * self.deletion_penalty
        
        # Compute distance matrix and get unique n-grams with counts
        dist_matrix, unique_ng1, unique_ng2, counts1, counts2 = \
            self._compute_distance_matrix(ngrams1, ngrams2)
        
        # Map n-grams to indices
        idx_map1 = {ng: i for i, ng in enumerate(unique_ng1)}
        idx_map2 = {ng: i for i, ng in enumerate(unique_ng2)}
        
        # Expand based on frequencies
        expanded_ng1_list = []
        weights1 = []
        for ng1, count1 in counts1.items():
            expanded_ng1_list.extend([ng1] * count1)
            weights1.extend([self._get_idf_weight(ng1)] * count1)
        
        expanded_ng2_list = []
        weights2 = []
        for ng2, count2 in counts2.items():
            expanded_ng2_list.extend([ng2] * count2)
            weights2.extend([self._get_idf_weight(ng2)] * count2)
        
        total_n1 = len(expanded_ng1_list)
        total_n2 = len(expanded_ng2_list)
        
        # Build expanded cost matrix
        # We need a square matrix to use linear_sum_assignment
        max_size = max(total_n1, total_n2)
        cost_matrix = np.zeros((max_size, max_size))
        
        # Fill the actual distances
        for i in range(max_size):
            for j in range(max_size):
                if i < total_n1 and j < total_n2:
                    # Actual matching cost
                    dist = dist_matrix[idx_map1[expanded_ng1_list[i]], idx_map2[expanded_ng2_list[j]]]
                    # Weight the distance by the average importance of the two terms
                    weight = (weights1[i] + weights2[j]) / 2
                    cost_matrix[i, j] = dist * weight
                elif i < total_n1 and j >= total_n2:
                    # Deletion: item from text1 not matched
                    # Cost is penalty * weight of the deleted item
                    cost_matrix[i, j] = self.deletion_penalty * weights1[i]
                elif i >= total_n1 and j < total_n2:
                    # Addition: item from text2 not matched
                    # Cost is penalty * weight of the added item
                    cost_matrix[i, j] = self.addition_penalty * weights2[j]
                # else: both are dummy (i >= total_n1 and j >= total_n2), leave as 0
        
        # Solve the linear assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Calculate total cost
        total_cost = cost_matrix[row_ind, col_ind].sum()
        
        # Normalize by the sum of weights of the larger text (approximation)
        # Ideally we normalize by the total weight of the matched + deleted/added items
        # But since we solve a square assignment, we can sum the weights of all involved items
        # Actually, WMD is usually normalized by the number of words.
        # With weights, we should normalize by the total weight.
        
        # Let's normalize by the max total weight to keep it bounded
        total_weight1 = sum(weights1)
        total_weight2 = sum(weights2)
        normalization_factor = max(total_weight1, total_weight2, 1.0)
        
        normalized_cost = total_cost / normalization_factor
        
        return normalized_cost
    
    def _compute_perturbation_penalty(self, text1: str, text2: str) -> float:
        """
        Compute penalty based on how much differing words perturb global embedding.
        
        This detects cases where texts have high word overlap but differing words
        significantly change the global meaning (e.g., adding "not").
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Perturbation penalty score
        """
        if not text1 or not text2:
            return 0.0
        
        # Get full sentence embeddings
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        
        # Get word sets
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Words unique to each text
        only_in_1 = words1 - words2
        only_in_2 = words2 - words1
        
        penalty = 0.0
        
        # For each word only in text1, measure impact of removing it
        for word in only_in_1:
            # Remove the word and get perturbed embedding
            text1_without = ' '.join([w for w in text1.lower().split() if w != word])
            if not text1_without:  # Skip if removal empties the text
                continue
                
            emb1_perturbed = self._get_embedding(text1_without)
            
            # Impact = how much embedding changes when word is removed
            impact = np.linalg.norm(emb1 - emb1_perturbed)
            
            # Compare to how close this makes us to text2
            cos_dist_original = cosine(emb1, emb2)
            cos_dist_perturbed = cosine(emb1_perturbed, emb2)
            closeness_gain = cos_dist_original - cos_dist_perturbed
            
            # If removing word makes us CLOSER to text2, it's a critical difference
            if closeness_gain > 0:
                penalty += impact * closeness_gain
        
        # Similar for words only in text2
        for word in only_in_2:
            text2_without = ' '.join([w for w in text2.lower().split() if w != word])
            if not text2_without:
                continue
                
            emb2_perturbed = self._get_embedding(text2_without)
            
            impact = np.linalg.norm(emb2 - emb2_perturbed)
            
            cos_dist_original = cosine(emb1, emb2)
            cos_dist_perturbed = cosine(emb1, emb2_perturbed)
            closeness_gain = cos_dist_original - cos_dist_perturbed
            
            if closeness_gain > 0:
                penalty += impact * closeness_gain
        
        return penalty
    
    def compute(
        self,
        text1: str,
        text2: str,
        n_values: List[int] = [1, 2, 3],
        return_all_scores: bool = False
    ) -> float:
        """
        Compute Word Mover's Distance between two texts.
        Automatically selects the best n-gram combination.
        
        Args:
            text1: First text
            text2: Second text
            n_values: List of n-gram sizes to try (default: [1, 2, 3])
            return_all_scores: If True, return dict with all scores
            
        Returns:
            Minimum WMD across all n-gram combinations (or dict if return_all_scores=True)
        """
        if self.idf_dict is None:
            self.idf_dict = self.fit([text1, text2])
        
        scores = {}
        
        for n in n_values:
            ngrams1 = self._extract_ngrams(text1, n)
            ngrams2 = self._extract_ngrams(text2, n)
            
            wmd = self._compute_wmd_with_penalties(ngrams1, ngrams2)
            scores[f'{n}-gram'] = wmd
        
        # Add perturbation penalty if enabled
        perturbation_penalty = 0.0
        if self.perturbation_penalty_weight > 0:
            perturbation_penalty = self._compute_perturbation_penalty(text1, text2)
        
        if return_all_scores:
            scores['perturbation_penalty'] = perturbation_penalty
            scores['best_wmd'] = min([v for k, v in scores.items() if 'gram' in k])
            scores['best_ngram'] = min(
                [(k, v) for k, v in scores.items() if 'gram' in k],
                key=lambda x: x[1]
            )[0]
            # Combined score
            scores['combined'] = scores['best_wmd'] + self.perturbation_penalty_weight * perturbation_penalty
            return scores
        
        # Return the minimum WMD score plus weighted perturbation penalty
        best_wmd = min(scores.values())
        return best_wmd + self.perturbation_penalty_weight * perturbation_penalty
    
    def compute_similarity(
        self,
        text1: str,
        text2: str,
        n_values: List[int] = [1, 2, 3]
    ) -> float:
        """
        Compute similarity score (inverse of distance).
        
        Args:
            text1: First text
            text2: Second text
            n_values: List of n-gram sizes to try
            
        Returns:
            Similarity score in [0, 1] where 1 is most similar
        """
        distance = self.compute(text1, text2, n_values)
        # Convert distance to similarity using exponential decay
        similarity = np.exp(-distance)
        return similarity


def compute_word_mover_distance(
    text1: str,
    text2: str,
    deletion_penalty: float = 3.0,
    addition_penalty: float = 1.0,
    use_pretrained_embeddings: bool = True
) -> float:
    """
    Convenience function to compute WMD between two texts.
    
    Args:
        text1: First text
        text2: Second text
        deletion_penalty: Penalty for deletions (default: 3.0)
        addition_penalty: Penalty for additions (default: 1.0)
        use_pretrained_embeddings: Whether to use pretrained embeddings
        
    Returns:
        Word Mover's Distance
    """
    wmd_calculator = WordMoverDistance(
        deletion_penalty=deletion_penalty,
        addition_penalty=addition_penalty,
        use_pretrained_embeddings=use_pretrained_embeddings
    )
    
    return wmd_calculator.compute(text1, text2)


if __name__ == "__main__":
    # Example usage
    print("Word Mover's Distance - Example Usage\n")
    
    # Initialize calculator
    wmd = WordMoverDistance(
        deletion_penalty=3.0,
        addition_penalty=1.0,
        use_pretrained_embeddings=True
    )
    
    # Example texts
    text1 = "The quick brown fox jumps over the lazy dog"
    text2 = "A fast brown fox leaps over a sleepy dog"
    
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}\n")
    
    # Compute WMD with all n-gram scores
    result = wmd.compute(text1, text2, return_all_scores=True)
    
    print("WMD Scores:")
    for key, value in result.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nSimilarity: {wmd.compute_similarity(text1, text2):.4f}")
    
    # Another example showing deletion penalty
    print("\n" + "="*60)
    text3 = "The cat sat on the mat"
    text4 = "The cat sat"
    
    print(f"\nText 3: {text3}")
    print(f"Text 4: {text4}")
    
    result2 = wmd.compute(text3, text4, return_all_scores=True)
    print("\nWMD Scores (higher due to deletion penalty):")
    for key, value in result2.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
