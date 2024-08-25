from skipgram_sgns import *


def train_skipgram(file_path, model_path):
    normalized_sentences = normalize_text(file_path)
    sg_model = SkipGram(normalized_sentences, print_flag=True)
    
    # Train model
    step_size = 0.001
    epochs = 50
    early_stopping = 3

    sg_model.learn_embeddings(step_size=step_size, epochs=epochs, early_stopping=early_stopping, model_path=model_path)

def test_similarity(model_path):
    model = load_model(model_path)
    word1 = 'house'
    word2 = 'mouse'
    similarity = model.compute_similarity(word1, word2)
    print(f'The similarity score between {word1} and {word2} is: {similarity}')
    
    word1 = 'eggs'
    word2 = 'ham'
    similarity = model.compute_similarity(word1, word2)
    print(f'The similarity score between {word1} and {word2} is: {similarity}')
    
    
def test_get_closest_words(model_path):
    model = load_model(model_path)
    word = 'fox'
    closest_words = model.get_closest_words(word, n=5)
    print(f"Closest words to '{word}': {closest_words}")
    
    word = 'box'
    closest_words = model.get_closest_words(word, n=5)
    print(f"Closest words to '{word}': {closest_words}")
    
    word = 'Sam-I-am'
    closest_words = model.get_closest_words(word, n=5)
    print(f"Closest words to '{word}': {closest_words}")
    
        
if __name__ == "__main__":
    file_path = 'Corpora/drSeuss.txt'
    base_name = os.path.basename(file_path)
    model_name = os.path.splitext(base_name)[0]
    model_path = f'skipgram_model_{model_name}.pkl' 
    
    # train_skipgram(file_path, model_path)
    test_similarity(model_path)
    test_get_closest_words(model_path)
    