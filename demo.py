"""
Example usage and demonstration of Word Mover's Distance.
"""
import time
from wmd import WordMoverDistance, compute_word_mover_distance

def demo_basic_usage():
    """Demonstrate basic WMD usage."""
    print("="*70)
    print("DEMO 1: Basic Word Mover's Distance")
    print("="*70)
    
    wmd = WordMoverDistance(
        deletion_penalty=3.0,
        addition_penalty=1.0,
        use_pretrained_embeddings=True
    )
    
    # Similar sentences
    text1 = "The cat is sitting on the mat"
    text2 = "A cat is sitting on a rug"
    
    print(f"\nText 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    
    result = wmd.compute(text1, text2, return_all_scores=True)
    
    print("\nDistance scores by n-gram:")
    for key, value in result.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    similarity = wmd.compute_similarity(text1, text2)
    print(f"\nSimilarity score: {similarity:.4f}")


def demo_deletion_penalty():
    """Demonstrate deletion penalty."""
    print("\n" + "="*70)
    print("DEMO 2: Deletion vs Addition Penalties")
    print("="*70)
    
    wmd = WordMoverDistance(
        deletion_penalty=3.0,
        addition_penalty=1.0,
        use_pretrained_embeddings=True
    )
    
    # Scenario 1: Deletion (text1 has more content)
    text1 = "The quick brown fox jumps over the lazy dog in the forest"
    text2 = "The quick brown fox"
    
    print(f"\n--- Deletion Scenario ---")
    print(f"Text 1 (longer): '{text1}'")
    print(f"Text 2 (shorter): '{text2}'")
    
    distance_del = wmd.compute(text1, text2)
    print(f"Distance (deletion from text1): {distance_del:.4f}")
    
    # Scenario 2: Addition (text2 has more content)
    text3 = "The quick brown fox"
    text4 = "The quick brown fox jumps over the lazy dog in the forest"
    
    print(f"\n--- Addition Scenario ---")
    print(f"Text 3 (shorter): '{text3}'")
    print(f"Text 4 (longer): '{text4}'")
    
    distance_add = wmd.compute(text3, text4)
    print(f"Distance (addition to match text4): {distance_add:.4f}")
    
    print(f"\nDeletion penalty is {distance_del / distance_add:.2f}x higher than addition penalty")


def demo_ngram_comparison():
    """Compare performance across different n-grams."""
    print("\n" + "="*70)
    print("DEMO 3: N-gram Comparison")
    print("="*70)
    
    wmd = WordMoverDistance(
        deletion_penalty=3.0,
        addition_penalty=1.0,
        use_pretrained_embeddings=True
    )
    
    # Test with phrases where bigrams/trigrams might help
    text1 = "New York City is a beautiful place"
    text2 = "NYC is a gorgeous location"
    
    print(f"\nText 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    
    result = wmd.compute(text1, text2, return_all_scores=True)
    
    print("\nPerformance by n-gram size:")
    for key, value in result.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nNote: The algorithm automatically selects the best n-gram size")


def demo_semantic_similarity():
    """Demonstrate semantic understanding."""
    print("\n" + "="*70)
    print("DEMO 4: Semantic Similarity")
    print("="*70)
    
    wmd = WordMoverDistance(
        deletion_penalty=3.0,
        addition_penalty=1.0,
        use_pretrained_embeddings=True
    )
    
    reference = "The scientist conducted an experiment"
    
    similar = "The researcher performed a test"
    different = "The chef cooked a delicious meal"
    
    print(f"\nReference: '{reference}'")
    print(f"\nSimilar text: '{similar}'")
    print(f"Different text: '{different}'")
    
    dist_similar = wmd.compute(reference, similar)
    dist_different = wmd.compute(reference, different)
    
    sim_similar = wmd.compute_similarity(reference, similar)
    sim_different = wmd.compute_similarity(reference, different)
    
    print(f"\nDistances:")
    print(f"  Reference ↔ Similar:   {dist_similar:.4f} (similarity: {sim_similar:.4f})")
    print(f"  Reference ↔ Different: {dist_different:.4f} (similarity: {sim_different:.4f})")
    
    print(f"\nThe similar text is {dist_different / dist_similar:.2f}x closer than the different text")


def demo_identical_texts():
    """Test identical texts."""
    print("\n" + "="*70)
    print("DEMO 5: Identical Texts")
    print("="*70)
    
    wmd = WordMoverDistance(use_pretrained_embeddings=True)
    
    text = "Machine learning is fascinating"
    
    print(f"\nText: '{text}'")
    
    distance = wmd.compute(text, text)
    similarity = wmd.compute_similarity(text, text)
    
    print(f"\nDistance with itself: {distance:.6f}")
    print(f"Similarity with itself: {similarity:.6f}")
    print("\nIdentical texts should have near-zero distance and ~1.0 similarity")


if __name__ == "__main__":
    demo_basic_usage()
    demo_deletion_penalty()
    demo_ngram_comparison()
    demo_semantic_similarity()
    demo_identical_texts()
    
    print("\n" + "="*70)
    print("All demos completed!")
    print("="*70)
