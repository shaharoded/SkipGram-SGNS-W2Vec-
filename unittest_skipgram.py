import unittest
import os
import sys
from skipgram_sgns import *


class TestSkipGram(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not hasattr(cls, 'corpora_file_path'):
            raise AttributeError("The class attribute 'corpora_file_path' must be set before running tests.")
        
        cls.file_path = cls.corpora_file_path  # Use the passed file path
        cls.base_name = os.path.basename(cls.file_path)
        cls.model_name = os.path.splitext(cls.base_name)[0]
        cls.model_path = os.path.abspath(f'skipgram_model_{cls.model_name}.pkl')  # Ensure absolute path
        cls.normalized_sentences = normalize_text(cls.file_path)

    def test_normalization(self):
        # Normalize the entire text
        normalized_sentences = self.normalized_sentences[:5]
        for i, sentence in enumerate(normalized_sentences):
            print(f"{i + 1} - {sentence}")

        # Test normalize_sentence with different options
        test_sentence = "This is a test sentence, with punctuation! And lemmatization: running, runners, ran."
        normalized = normalize_sentence(test_sentence, lemmatize=False)
        normalized_with_lemma = normalize_sentence(test_sentence, lemmatize=True, use_pos=False)
        normalized_with_pos = normalize_sentence(test_sentence, lemmatize=True, use_pos=True)
        
        # If no exceptions occurred, we assume normalization worked
        print("Normalization tests completed successfully.")

    def test_train_skipgram(self):
        sg_model = SkipGram(self.normalized_sentences, print_flag=True)
        
        # Train model
        step_size = 0.001
        epochs = 50
        early_stopping = 3

        sg_model.learn_embeddings(step_size=step_size, epochs=epochs, early_stopping=early_stopping, model_path=self.model_path)

        # Check if model file is created
        self.assertTrue(os.path.exists(self.model_path), f"Model was not saved at {self.model_path}")
        print("SkipGram model training test passed and model saved successfully.")

    def test_model_attr(self):
        model = load_model(self.model_path)
        
        # Check dimensions of embedding matrix T
        self.assertEqual(model.T.shape[1], model.vocab_size)
        
        # Check dimensions of context matrix C
        self.assertEqual(model.C.shape[0], model.vocab_size)
        
        # Check word index mapping
        self.assertEqual(len(model.word_index), model.vocab_size)
        
        # Ensure word indices are within bounds of T and C
        for word, index in model.word_index.items():
            self.assertLess(index, model.T.shape[1])
            self.assertLess(index, model.C.shape[0])
    
        # Check if all vocabulary words have corresponding embeddings
        missing_embeddings = [word for word, index in model.word_index.items() if index >= model.T.shape[1]]
        self.assertEqual(len(missing_embeddings), 0)
    
        # Check if all vocabulary words have corresponding context vectors
        missing_context_vectors = [word for word, index in model.word_index.items() if index >= model.C.shape[0]]
        self.assertEqual(len(missing_context_vectors), 0)
    
        print("Model attribute checks completed successfully.")

    def test_combine_vectors(self):
        model = load_model(self.model_path)
        
        for combo in range(5):
            combined_V = model.combine_vectors(model.T, model.C, combo=combo)
            expected_shape = (model.d, model.vocab_size) if combo != 4 else (model.d * 2, model.vocab_size)
            self.assertEqual(combined_V.shape, expected_shape)
            print(f"Combined vectors shape for combo {combo}: {combined_V.shape}")

        print("Combine vectors test completed successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Usage: python test_skipgram.py <corpora_file_path>")
    TestSkipGram.corpora_file_path = sys.argv.pop()  # Pass the corpora file path to the class
    unittest.main()
