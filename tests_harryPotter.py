from skipgram_sgns import *

def test_normalization(file_path):
    # Normalize the entire text
    normalized_sentences = normalize_text(file_path)[100:105]
    print("Normalized Text:")
    i = 1
    for sentence in normalized_sentences:
        print(i, ' - ', sentence)
        i += 1

    # Test normalize_sentence with different options
    test_sentence = "This is a test sentence, with punctuation! And lemmatization: running, runners, ran."
    print("\nOriginal Sentence:")
    print(test_sentence)
    
    print("\nNormalized without lemmatization:")
    print(normalize_sentence(test_sentence, lemmatize=False))
    
    print("\nNormalized with lemmatization but without POS:")
    print(normalize_sentence(test_sentence, lemmatize=True, use_pos=False))
    
    print("\nNormalized with lemmatization and POS tagging:")
    print(normalize_sentence(test_sentence, lemmatize=True, use_pos=True))

    
def train_skipgram(file_path, model_name):
    normalized_sentences = normalize_text(file_path)
    sg_model = SkipGram(normalized_sentences, print_flag=True)
    
    # Train model
    step_size = 0.001
    epochs = 50
    early_stopping = 3

    sg_model.learn_embeddings(step_size=step_size, epochs=epochs, early_stopping=early_stopping, model_path=model_name)

    
def test_model_attr(model_path):
    model = load_model(model_path)
    
    # Check dimensions of embedding matrix T
    print(f"T shape: {model.T.shape}")
    
    # Check dimensions of context matrix C
    print(f"C shape: {model.C.shape}")
    
    # Check word index mapping
    print(f"Number of words in vocabulary: {len(model.word_index)}")
    
    
    # Ensure word indices are within bounds of T
    for word, index in model.word_index.items():
        if index >= model.T.shape[1] or index >= model.C.shape[0]:
            print(f"Index out of bounds for word '{word}': {index}")
        else:
            print(f"Word '{word}' has index {index} which is within bounds.")
    
    # Check if all vocabulary words have corresponding embeddings
    missing_embeddings = [word for word, index in model.word_index.items() if index >= model.T.shape[1]]
    if missing_embeddings:
        print(f"Words missing embeddings: {missing_embeddings}")
    else:
        print("All words have corresponding embeddings.")
    
    # Check if all vocabulary words have corresponding context vectors
    missing_context_vectors = [word for word, index in model.word_index.items() if index >= model.C.shape[0]]
    if missing_context_vectors:
        print(f"Words missing context vectors: {missing_context_vectors}")
    else:
        print("All words have corresponding context vectors.")
    
    print("Model attribute checks completed.")


def test_similarity(model_path):
    model = load_model(model_path)
    word1 = 'Dursley'
    word2 = 'Ollivander'
    similarity = model.compute_similarity(word1, word2)
    print(f'The similarity score between {word1} and {word2} is: {similarity}')
    
    word1 = 'wand'
    word2 = 'robe'
    similarity = model.compute_similarity(word1, word2)
    print(f'The similarity score between {word1} and {word2} is: {similarity}')
    
    
def test_get_closest_words(model_path):
    model = load_model(model_path)
    word = 'Dumbledore'
    closest_words = model.get_closest_words(word, n=5)
    print(f"Closest words to '{word}': {closest_words}")
    
    word = 'Harry'
    closest_words = model.get_closest_words(word, n=5)
    print(f"Closest words to '{word}': {closest_words}")
    
    word = 'Hermione'
    closest_words = model.get_closest_words(word, n=5)
    print(f"Closest words to '{word}': {closest_words}")



if __name__ == "__main__":
    # file_path = 'Corpora/drSeuss.txt'
    file_path = 'Corpora/harryPotter1.txt'
    base_name = os.path.basename(file_path)
    model_name = os.path.splitext(base_name)[0]
    model_path = f'skipgram_model_{model_name}.pkl'

    test_normalization(file_path)
    train_skipgram(file_path, model_path)
    test_model_attr(file_path)
    test_get_closest_words(model_path)
    test_similarity(model_path)