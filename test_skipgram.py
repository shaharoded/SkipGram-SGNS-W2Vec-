import unittest
import os
import sys
from skipgram_sgns import SkipGram, normalize_text, normalize_sentence, load_model

class TestSkipGram(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.file_path = cls.corpora_file_path  # Use the passed file path
        cls.base_name = os.path.basename(cls.file_path)
        cls.model_name = os.path.splitext(cls.base_name)[0]
        cls.model_path = f'skipgram_model_{cls.model_name}.pkl'
        cls.normalized_sentences = normalize_text(cls.file_path)

    def test_normalization(self):
        # Normalize the entire text
        normalized_sentences = self.normalized_sentences[:5]
        print("Normalized Text:")
        for i, sentence in enumerate(normalized_sentences):
            print(i + 1, ' - ', sentence)

        # Test normalize_sentence with different options
        test_sentence = "This is a test sentence, with punctuation! And lemmatization: running, runners, ran."
        print("\nOriginal Sentence:")
        print(test_sentence)
        
        print("\nNormalized without lemmatization:")
        normalized = normalize_sentence(test_sentence, lemmatize=False)
        print(normalized)
        
        print("\nNormalized with lemmatization but without POS:")
        normalized = normalize_sentence(test_sentence, lemmatize=True, use_pos=False)
        print(normalized)
        
        print("\nNormalized with lemmatization and POS tagging:")
        normalized = normalize_sentence(test_sentence, lemmatize=True, use_pos=True)
        print(normalized)

    def test_train_skipgram(self):
        sg_model = SkipGram(self.normalized_sentences, print_flag=True)
        
        # Train model
        step_size = 0.0001
        epochs = 50
        early_stopping = 3

        sg_model.learn_embeddings(step_size=step_size, epochs=epochs, early_stopping=early_stopping, model_path=self.model_path)

        # Check if model file is created
        self.assertTrue(os.path.exists(self.model_path))

    def test_model_attr(self):
        model = load_model(self.model_path)
        
        # Check dimensions of embedding matrix T
        self.assertEqual(model.T.shape[1], model.vocab_size)
        
        # Check dimensions of context matrix C
        self.assertEqual(model.C.shape[0], model.vocab_size)
        
        # Check word index mapping
        self.assertEqual(len(model.word_index), model.vocab_size)
        
        # Ensure word indices are within bounds of T
        for word, index in model.word_index.items():
            self.assertLess(index, model.T.shape[1])
            self.assertLess(index, model.C.shape[0])
    
        # Check if all vocabulary words have corresponding embeddings
        missing_embeddings = [word for word, index in model.word_index.items() if index >= model.T.shape[1]]
        self.assertEqual(len(missing_embeddings), 0)
    
        # Check if all vocabulary words have corresponding context vectors
        missing_context_vectors = [word for word, index in model.word_index.items() if index >= model.C.shape[0]]
        self.assertEqual(len(missing_context_vectors), 0)
    
        print("Model attribute checks completed.")

    def test_analogy(self):
        model = load_model(self.model_path)
        tests = [
            ('house', 'mouse', 'box', 'fox'),
            ('game', 'slow', 'socks', 'big'),
            ('luke', 'luck', 'lake', 'like'),
            ('battle', 'beetles', 'puddle', 'paddle'),
            ('house', 'mouse', 'box', 'fox'),
            ('game', 'slow', 'socks', 'big'),
            ('luke', 'luck', 'lake', 'like'),
            ('battle', 'beetles', 'puddle', 'paddle')
            
            # ("harry", "voldemort", "hagrid", "wand"),
            # ("wand", "spell", "car", "drive"),
            # ("harry", "ron", "hermione", "car")
        ]
        for w1, w2, w3, w4 in tests:
            try:
                analogy_result = model.find_analogy(w1, w2, w3)
                analogy_test_result = model.test_analogy(w1, w2, w3, w4, n=5)
                print(f"Analogy Test: '{w1}' to '{w2}' is like '{analogy_result}' to '{w3}'")
                print(f"Test if '{w1}' to '{w2}' is like '{w4}' to '{w3}': {analogy_test_result}")
            except ValueError as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_skipgram.py <corpora_file_path>")
        sys.exit(1)
    
    corpora_file_path = sys.argv[1]

    # Assign the corpora_file_path to the test class
    TestSkipGram.corpora_file_path = corpora_file_path
    
    # Remove the first argument to prevent unittest from processing it
    sys.argv = [sys.argv[0]]
    
    unittest.main()