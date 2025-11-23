"""
Test suite for Word Mover's Distance implementation.
"""
import unittest
import numpy as np
from wmd import WordMoverDistance, compute_word_mover_distance


def test_identical_texts():
    """Test that identical texts have zero distance."""
    wmd = WordMoverDistance(use_pretrained_embeddings=False)
    
    text1 = "The quick brown fox"
    text2 = "The quick brown fox"
    
    distance = wmd.compute(text1, text2)
    print(f"Identical texts distance: {distance:.4f}")
    assert distance < 0.1, "Identical texts should have near-zero distance"
    print("✓ Identical texts test passed")


def test_similar_texts():
    """Test that similar texts have low distance."""
    wmd = WordMoverDistance(use_pretrained_embeddings=False)
    
    text1 = "The cat sat on the mat"
    text2 = "The cat sat on the rug"
    
    distance = wmd.compute(text1, text2)
    print(f"Similar texts distance: {distance:.4f}")
    print("✓ Similar texts test passed")


def test_deletion_penalty():
    """Test that deletion penalty is higher than addition penalty."""
    wmd = WordMoverDistance(
        deletion_penalty=3.0,
        addition_penalty=1.0,
        use_pretrained_embeddings=False
    )
    
    # Text1 has more words (deletion from text1 perspective)
    text1 = "The cat sat on the mat yesterday"
    text2 = "The cat sat"
    
    # Text3 has fewer words (addition to text3 perspective)  
    text3 = "The cat sat"
    text4 = "The cat sat on the mat yesterday"
    
    distance_deletion = wmd.compute(text1, text2)
    distance_addition = wmd.compute(text3, text4)
    
    print(f"Deletion scenario distance: {distance_deletion:.4f}")
    print(f"Addition scenario distance: {distance_addition:.4f}")
    
    # Deletion should cost more
    assert distance_deletion > distance_addition, \
        "Deletion penalty should be higher than addition penalty"
    print("✓ Deletion penalty test passed")


def test_ngram_selection():
    """Test that the algorithm selects the best n-gram."""
    wmd = WordMoverDistance(use_pretrained_embeddings=False)
    
    text1 = "New York City is beautiful"
    text2 = "NYC is gorgeous"
    
    result = wmd.compute(text1, text2, return_all_scores=True)
    
    print("\nN-gram scores:")
    for key, value in result.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    assert 'best_wmd' in result, "Should return best score"
    assert 'best_ngram' in result, "Should return best n-gram"
    assert result['best_wmd'] == min(result['1-gram'], result['2-gram'], result['3-gram']), \
        "Best should be minimum across all n-grams"
    
    print("✓ N-gram selection test passed")


def test_empty_texts():
    """Test handling of empty texts."""
    wmd = WordMoverDistance(use_pretrained_embeddings=False)
    
    # Both empty
    distance1 = wmd.compute("", "")
    print(f"\nEmpty-Empty distance: {distance1:.4f}")
    assert distance1 == 0.0, "Two empty texts should have zero distance"
    
    # One empty
    distance2 = wmd.compute("Hello world", "")
    print(f"NonEmpty-Empty distance: {distance2:.4f}")
    assert distance2 > 0, "Empty and non-empty should have positive distance"
    
    print("✓ Empty texts test passed")


def test_convenience_function():
    """Test the convenience function."""
    distance = compute_word_mover_distance(
        "The dog runs",
        "The cat walks",
        deletion_penalty=3.0,
        addition_penalty=1.0,
        use_pretrained_embeddings=False
    )
    
    print(f"\nConvenience function distance: {distance:.4f}")
    assert distance >= 0, "Distance should be non-negative"
    print("✓ Convenience function test passed")


def test_similarity_score():
    """Test similarity score calculation."""
    wmd = WordMoverDistance(use_pretrained_embeddings=False)
    
    # Identical texts
    sim1 = wmd.compute_similarity("Hello world", "Hello world")
    print(f"\nSimilarity of identical texts: {sim1:.4f}")
    assert sim1 > 0.9, "Identical texts should have high similarity"
    
    # Different texts
    sim2 = wmd.compute_similarity("The cat", "A dog barks loudly")
    print(f"Similarity of different texts: {sim2:.4f}")
    assert 0 <= sim2 <= 1, "Similarity should be in [0, 1]"
    assert sim2 < sim1, "Different texts should have lower similarity"
    
    print("✓ Similarity score test passed")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Running Word Mover's Distance Tests")
    print("="*60)
    
    test_identical_texts()
    test_similar_texts()
    test_deletion_penalty()
    test_ngram_selection()
    test_empty_texts()
    test_convenience_function()
    test_similarity_score()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
