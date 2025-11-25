import unittest
import numpy as np
from wmd import WordMoverDistance

class TestIDFWeighting(unittest.TestCase):
    def test_idf_weighting_impact(self):
        """Test that common words have less impact than rare words."""
        wmd = WordMoverDistance(use_pretrained_embeddings=False)
        
        # Create a corpus where "the" is common and "quantum" is rare
        corpus = [
            "the the the the the",
            "the quick brown fox",
            "the lazy dog",
            "quantum physics is hard",
            "quantum mechanics"
        ]
        
        wmd.fit(corpus)
        
        # Check weights
        weight_the = wmd.idf_dict.get("the", 0)
        weight_quantum = wmd.idf_dict.get("quantum", 0)
        
        print(f"\nWeight 'the': {weight_the:.4f}")
        print(f"Weight 'quantum': {weight_quantum:.4f}")
        
        self.assertLess(weight_the, weight_quantum, "Common word should have lower weight")
        
        # Test distance impact
        base_text = "quantum physics"
        
        # Text with "the" removed (low impact expected)
        text_no_the = "quantum physics"
        text_with_the = "the quantum physics"
        
        # Text with "quantum" removed (high impact expected)
        text_no_quantum = "physics"
        
        dist_the = wmd.compute(base_text, text_with_the)
        dist_quantum = wmd.compute(base_text, text_no_quantum)
        
        print(f"Distance adding 'the': {dist_the:.4f}")
        print(f"Distance removing 'quantum': {dist_quantum:.4f}")
        
        # Removing/Adding "quantum" should cost more than "the" because of higher weight
        # Note: dist_the is addition penalty * weight_the
        #       dist_quantum is deletion penalty * weight_quantum
        # Even with default penalties (del=3, add=1), if weights are different enough, we can see effect.
        # Let's use same penalty for clearer test
        wmd.deletion_penalty = 1.0
        wmd.addition_penalty = 1.0
        
        dist_the_equal = wmd.compute(base_text, text_with_the)
        dist_quantum_equal = wmd.compute(base_text, text_no_quantum)
        
        print(f"Distance adding 'the' (equal penalty): {dist_the_equal:.4f}")
        print(f"Distance removing 'quantum' (equal penalty): {dist_quantum_equal:.4f}")
        
        self.assertLess(dist_the_equal, dist_quantum_equal, "Changes in common words should cost less")

if __name__ == "__main__":
    unittest.main()
