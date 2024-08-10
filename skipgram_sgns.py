'''
TO-DO
- Maybe sparse learning vectors? If vector is binary than no need to update entries with no context?  
- Possible improvement - Dinamically increrase window size if a stop word context word is found.
'''

import numpy as np
import collections
import pickle
import os
import random
import time
import math

import nltk
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.data.clear_cache()
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')


def who_am_i():
    """
    Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
    Make sure you return your own info!
    """
    return {'name': 'Shahar Oded', 'id': '208388918', 'email': 'odedshah@post.bgu.ac.il'}


def expand_contractions(text):
    contractions_dict = {
        "'s": " is",
        "'re": " are",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "can't": "can not",
        "couldn't": "could not",
        "won't": "will not",
        "wouldn't": "would not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "I'm": "I am",
        "we're": "we are",
        "they're": "they are",
        "let's": "let us",
        "it's": "it is",
        "that's": "that is",
        "what's": "what is",
        "here's": "here is"
    }
    for contraction, expansion in contractions_dict.items():
        text = text.replace(contraction, expansion)
    return text


def _get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def normalize_sentence(sentence, lemmatize=False, use_pos=False):
    """
    Normalize a sentence by tokenizing, lemmatizing, and removing punctuation.

    Args:
        sentence: a single sentence as a string
        lemmatize: whether to lemmatize words (default: False)
        use_pos: whether to use POS tagging for lemmatization (default: False)

    Returns:
        A normalized sentence (a single str).
    """
    lemmatizer = WordNetLemmatizer()
    sentence = expand_contractions(sentence.lower().strip())  # Remove constraction abbriviations
    words = sentence.split()
    normalized_words = []
    
    for word in words:
        word = re.sub(r'[^\w\s-]', '', word)    # Remove punctuation / irrelevant chars
        if word == '':
            continue
        elif word:
            if lemmatize:
                if use_pos:
                    pos = _get_wordnet_pos(word)
                    word = lemmatizer.lemmatize(word, pos)
                else:
                    word = lemmatizer.lemmatize(word)
            
            normalized_words.append(word)
    
    return ' '.join(normalized_words)


def normalize_text(fn):
    """
    Loading a text file, normalizing it, and returning a list of sentences.

    Args:
        fn: full path to the text file to process

    Returns:
        A list of normalized sentences.
    """
    sentences = []

    with open(fn, 'r') as f:
        text = f.read()
        raw_sentences = sent_tokenize(text)
        
        for raw_sentence in raw_sentences:
            normalized_sentence = normalize_sentence(raw_sentence)
            sentences.append(normalized_sentence)
    
    return sentences


def sigmoid(x):
    '''
    Args:
        x: float
    Return: Float
    ''' 
    return 1.0 / (1 + np.exp(-x))


def load_model(fn):
    """ 
    Loads a model pickle and return it.

    Args:
        fn: the full path to the model to load.
    """
    if not os.path.exists(fn):
        raise FileNotFoundError(f"The file {fn} does not exist in the expected path.")
    
    try:
        with open(fn, 'rb') as f:
            sg_model = pickle.load(f)
    except IOError as e:
        raise IOError(f"An error occurred while trying to read the file {fn}: {e}")
    
    return sg_model


class SkipGram:
    '''
    An implementation of the skip-gram Word2Vec model using SGNS (negative sampling)
    instead of softmax.
    '''
    def __init__(self, sentences, d=100, neg_samples=4, context=4, word_count_threshold=5, lr_decay = True, decay_rate = 0.01, print_flag = False):
        self.sentences = sentences
        self.d = d  # embedding dimension
        self.neg_samples = neg_samples  # num of negative samples for one positive sample
        self.context = context  # the size of the context window (not counting the target word)
        self.word_count_threshold = word_count_threshold    #ignore low frequency words (appearing under the threshold)
        self.lr_decay = lr_decay    # implement learning rate decay during training
        self.decay_rate = decay_rate    # rate of the decay, if self.lr_decay = True
        self.print_flag = print_flag    # print progress through training (defaults to False).
        
        self.word_counts = collections.Counter([word for sentence in sentences for word in sentence.split()])   # word counter BOW
        self.word_counts = {word: count for word, count in self.word_counts.items() if count >= word_count_threshold}
        self.vocab = {word for word in self.word_counts.keys()}  # V
        self.word_index = {word: i for i, word in enumerate(self.vocab)}    # word -> index mapper
        self.index_word = {i: word for word, i in self.word_index.items()}  # index -> word mapper

        self.vocab_size = len(self.vocab)   # |V|
        # the target word embedding matrix
        self.T = None
        # the context word embedding matrix 
        self.C = None
        
        # each row -> word in the vocabulary
        # each column -> a dimension of the embedding vector.
        self.V = self.combine_vectors(self.T, self.C)   # the combination of T and C, Model's true vector


    def compute_similarity(self, w1, w2):
        """ 
        Returns the cosine similarity (in [0,1]) between the specified words.
        NOTE: Similarity is calculated between the 2 normalized words, so 2 non identical 
        words might have the same score if are mapped to the same word in the vector space.

        Args:
            w1: a word
            w2: a word
        Retunrns: a float in [0,1]; defaults to 0.0 if one of specified words is OOV.
    """
        sim = 0.0  # default
        w1_norm = normalize_sentence(w1)
        w2_norm = normalize_sentence(w2)
        # Check if both words are in our word_index map
        if any([w1_norm not in self.vocab, w2_norm not in self.vocab]):
           self.print_flag and print('''[Warning]: One of the words in your input is not in the vocabulary.
                                     A default value returned''') 
           return 0.0

        # Fetch the embeddings of the words using T&C
        embedding_w1 = self.V[:, self.word_index[w1_norm]]
        embedding_w2 = self.V[:, self.word_index[w2_norm]]
        
        # Compute similarity using cosine similarity between 2 vectors
        dot_product = np.dot(embedding_w1, embedding_w2)
        norm_product = np.linalg.norm(embedding_w1) * np.linalg.norm(embedding_w2)
        sim = dot_product / norm_product

        if sim is None or math.isnan(sim):
            return 0.0
        
        return sim


    def get_closest_words(self, w, n=5):
        """
        Returns a list containing the n words that are the closest to the specified word.

        Args:
            w: the word to find close words to.
            n: the number of words to return. Defaults to 5.
        
        Returns: List() of size n.
        """
        w_norm = normalize_sentence(w)
        if w_norm not in self.vocab:
            raise ValueError(f"Word '{w}' not in vocabulary.")
        
        w_index = self.word_index[w]
        word_emb = self.V[:, w_index]

        # Compute cosine similarities (efficiently!) -> 1D array
        cos_sim = np.dot(self.V.T, word_emb) / (
            np.linalg.norm(self.V.T, axis=1) * np.linalg.norm(word_emb))

        # Sort by similarity -> list of indices 
        max_sorted = list(np.argsort(cos_sim)[::-1][:n + 1])  # sort decending, slice

        # Exclude the target word and get the top n words
        candidates = [self.index_word[i] for i in max_sorted if i != w_index]
        return candidates[:n]

    
    def _create_learning_vectors(self, sentence):
        """
        Creates lists of positive and negative word pairs for a given sentence.
        Positive pairs are actual context pairs within a specified window size, 
        while negative pairs are randomly generated from vcabulary based on frequency,
        Positive samples are from window [...x-2, x-1, t, x+1, x+2...].
        Will compose an appropriate vector using the word indices and occurences.
        
        Args:
            sentence (str): A single sentence.

        Returns:
            list (of tuples):
                sparse learning vectors like [(word, {c_word_i: val})] with positive and negative
                context values for word.
        """
        def generate_skipgrams(words, context):
            '''
            Return list of tuples for all positive context words per sentence.
            Having that context vectors are not aggregated based on occurance, I'll create a word
            vector by word, and not by unique word, to not loose contextual meaning.
            '''
            window_size = context // 2
            pos_l = []

            for i, t in enumerate(words):
                for j in range(1, window_size + 1):
                    context = []
                    if i - j >= 0:
                        context.append(words[i - j])
                    if i + j < len(words):
                        context.append(words[i + j])
                        
                    # Avoid adding t as the context of t (F-drSeuss!)
                    context = [w for w in context if w != t]
                pos_l.append((t, context))

            return pos_l

        
        def generate_negative_samples(words, all_words, all_counts):
            '''
            Will output a list of tuples, by the word, with negative context words.
            Output will have the same order (and outer size) as positive list.
            '''
            neg_l = []
            
            for t in words:
                context = []
                for _ in range(self.neg_samples):
                    neg_word = random.choices(all_words, weights=all_counts, k=1)[0]
                    while neg_word == t:
                        neg_word = random.choices(all_words, weights=all_counts, k=1)[0]
                    context.append(neg_word)
                neg_l.append((t, context))
            
            return neg_l
        
        # Filter only relevant words to vocab (word occurance is significant)
        words = [word for word in sentence.split() if word in self.vocab]           
         
        all_words = list(self.word_counts.keys())
        all_counts = list(self.word_counts.values())
        
        pos = generate_skipgrams(words, self.context)
        neg = generate_negative_samples(words, all_words, all_counts)
        
        # Create context vectors (sparse) containing positive context as 1 and negative as 0
        # list of tuples (token_index, {c_word_index: val, c_word_index: val})
        # val is accumulative 1 for positive index and accumulative -1 for negative index
        learning_vectors = []
        for idx, (t_p, p_context) in enumerate(pos):
            target_idx = self.word_index[t_p]
            context_vector = np.zeros(self.vocab_size)
            for c_word in p_context:
                c_word = self.word_index[c_word]
                context_vector[c_word] = 1
            
            t_n, n_context = neg[idx]
            if t_p != t_n:
                raise ValueError('Your positive and negative lists are scrambeled')
            for c_word in n_context:
                c_word = self.word_index[c_word]
                
                # Avoid randomally updating actual context to be negative
                if not context_vector[c_word] == 1:
                    context_vector[c_word] = -1
                    
            # Make sure a context vector was properly created (both positive and negative)
            if np.any(context_vector > 0) and np.any(context_vector < 0):
                # Normalize the context vector to [0, 1]
                min_val = np.min(context_vector)
                max_val = np.max(context_vector)
                if max_val != min_val:  # Avoid division by zero
                    context_vector = (context_vector - min_val) / (max_val - min_val)                        
                    learning_vectors.append((target_idx, context_vector))        
        return learning_vectors


    def _preprocess_sentences(self):
        """
        Preprocesses all train sentences for the SkipGram model, creating learning vectors using positive and negative sampels.
        Function will pass the vectors as a generator for space efficiency.
        :return:
            learning_vectors: A list of tuples, each with a target word and its corresponding context vector. Each word (key) can appear in
                                a few tuples, as the training process trains word by word, per sentence.
        """
        for sentence in self.sentences:
            # Create the learning context vector
            vectors = self._create_learning_vectors(sentence)
            # In case no proper vectors were created (vectors = [])
            if not vectors:
                continue
            for vec in vectors:     # a tuple like (t, vec)
                yield vec
    
    
    def _forward_pass(self, t_index, c_vector):
        """
        Performs a forward pass of the neural network for the SkipGram model.
        """
        hidden = self.T[:, t_index][:, None]
        y = c_vector    # context vector is the actual y value for supervised task
        output_layer = np.dot(self.C, hidden).flatten()
        y_pred = sigmoid(output_layer)  # Can be replaced with softmax

        return hidden, y_pred, y

    
    def _calculate_loss(self, y, y_pred):
        '''
        Use a Log-Loss calculation in order to estimate the training progress.
        '''
        # Ensure numerical stability
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return loss / len(y)
    

    def _backpropagation(self, t_index, hidden, y_pred, y, step_size):
        '''
        Calculate the error vector and use it to calculate the gradients.
        Updates the matrices T and C according to the gradients.
        '''
        error = y_pred - y  # error shape: (|V|,) where |V| is the vocabulary size
        c_grad = np.dot(error[:, None], hidden.T)  # c_grad shape: (|V|, d)
        t_grad = np.dot(self.C.T, error)  # t_grad shape: (d,)
        self.C -= step_size * c_grad
        self.T[:, t_index] -= step_size * t_grad
        return self.T, self.C
    
    
    def learn_embeddings(self, step_size=0.0001, epochs=50, early_stopping=3, model_path=None, keep_train = False):
        """
        Returns a trained embedding models and saves it in the specified path.
        An intermidiate model is saved to path every epoch.

        Args:
            step_size: step size for  the gradient descent. Defaults to 0.0001
            epochs: number or training epochs. Defaults to 50
            early_stopping: stop training if the Loss was not improved for this number of epochs
            model_path: full path (including file name) to save the model pickle at.
            keep_train: allow the model to continue training on existing model (additional epochs). If used, currently, new text
                        cannot be loaded and trained on with existing model. Will result in collapse.
        """
        s = time.time()
        if not keep_train:
            # Initiate tables with values on the probabilities, to match with context entries direction [0,1]
            # Matrix shape (rows, columns)
            self.T = np.random.rand(self.d, self.vocab_size)  # Embedding matrix of target words
            self.C = np.random.rand(self.vocab_size, self.d)  # Embedding matrix of context words


        self.print_flag and print("Started Model Training")
        best_loss = np.inf  # Initialize the best loss as infinity
        epochs_no_improve = 0  # Initialize epochs without improvement (for stop criteria)

        for i in range(1, epochs + 1):
            epoch_loss = []  # Initialize loss for this epoch
            n_vectors = 0
            for vec in self._preprocess_sentences():
                n_vectors += 1
                t_index, c_vector = vec
                
                # Forward pass
                hidden, y_pred, y = self._forward_pass(t_index, c_vector)

                # Calculate loss
                epoch_loss.append(self._calculate_loss(y, y_pred))
                
                # Backpropagation
                self.T, self.C = self._backpropagation(t_index, hidden, y_pred, y, step_size)
                self.V = self.combine_vectors(self.T, self.C)
                
            epoch_loss = np.mean(epoch_loss)
            if self.lr_decay:
                step_size *= 1 / (1 + self.decay_rate * i)   # Time LR decay
            self.print_flag and print(f"Epoch {i}, Loss: {epoch_loss}, time: {round((time.time()-s)/60,2)}m")

            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0  # Reset the count
            else:
                epochs_no_improve += 1  # Increment the count

            if epochs_no_improve == early_stopping:
                self.print_flag and print("Stopped! Early stopping criteria met.")
                break
       
            if model_path:
                # Backup the last trained model (every epoch)
                with open(model_path, "wb") as file:
                    pickle.dump(self, file)

                self.print_flag and print(f"Model saved to path: '{model_path}'")

        # Return in case you wish to catch them
        return self.T, self.C

    
    def combine_vectors(self, T, C, combo=0, model_path=None):
        """
        Returns a single embedding matrix and saves it to the specified path

        Args:
            T: The learned targets (T) embeddings (as returned from learn_embeddings())
            C: The learned contexts (C) embeddings (as returned from learn_embeddings())
            combo: indicates how wo combine the T and C embeddings (int)
                   0: use only the T embeddings (default)
                   1: use only the C embeddings (transposed)
                   2: return a pointwise average of C and T
                   3: return the sum of C and T
                   4: concat C and T vectors (effectively doubling the dimention of the embedding space)
            model_path: full path (including file name) to save the model pickle at.
        """        
        if combo == 0:
            V = T
        elif combo == 1:
            V = C.T
        elif combo == 2:
            V = np.add(T, C.T) / 2
        elif combo == 3:
            V = np.add(T, C.T)
        elif combo == 4:
            V = np.concatenate((C.T, T), axis=1)
        else:
            raise ValueError("Invalid combo option. Choose a number between 0 and 4.")
        
        self.V = V
        
        if model_path:
            with open(model_path, "wb") as file:
                pickle.dump(self, file)

        return V


    def find_analogy(self, w1, w2, w3):
        """
        Returns a word (string) that matches the analogy test given the three specified words.
        Required analogy: w1 to w2 is like ____ to w3.

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
        """
        w1_norm, w2_norm, w3_norm = normalize_sentence(w1), normalize_sentence(w2), normalize_sentence(w3)
        if any(w not in self.vocab for w in [w1_norm, w2_norm, w3_norm]):
            raise ValueError("At least one of the words is not in the vocabulary.")

        # Get the vector representations of the words & indices
        idx_1 = self.word_index[w1_norm]
        idx_2 = self.word_index[w2_norm]
        idx_3 = self.word_index[w3_norm]
        vec_w1 = self.V[:, idx_1]
        vec_w2 = self.V[:, idx_2]
        vec_w3 = self.V[:, idx_3]

        # Compute the target vector
        # Similarity is based on direction, no need to norm.
        vec_v = vec_w1 - vec_w2 + vec_w3        
        
        # Find the closest word vector (using cosine-similarity)
        # Sort efficiently
        cos_sim = np.dot(self.V.T, vec_v) / (
                np.linalg.norm(self.V.T, axis=1) * np.linalg.norm(vec_v))
        max_sorted = list(np.argsort(cos_sim)[::-1][:4])    # Get top 4
        max_sorted = [idx for idx in max_sorted if idx not in [idx_1, idx_2, idx_3]]
        
        if not max_sorted:
            raise ValueError("Not enough words in the vocabulary to find a suitable analogy.")

        # take the first word that is not w1,w2,w3
        return self.index_word[max_sorted[0]]


    def test_analogy(self, w1, w2, w3, w4, n=1):
        """
        Returns True if sim(w1-w2+w3, w4)@n; Otherwise return False.
        That is, returning True if w4 is one of the n closest words to the vector w1-w2+w3.
        Interpretation: 'w1 to w2 is like w4 to w3'

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
             w4: forth word in the analogy (string)
             n: the distance (work rank) to be accepted as similarity
        """
        w1, w2, w3, w4_norm = normalize_sentence(w1), normalize_sentence(w2), normalize_sentence(w3), normalize_sentence(w4)
        # Check if all words are in the vocabulary
        if any(w not in self.vocab for w in [w1, w2, w3, w4_norm]):
            raise ValueError("At least one of the words is not in the vocabulary.")
        
        # Assuming analogy is found by combining w1 - w2 + w3 =~ w4
        analogy = self.find_analogy(w1, w2, w3)
        closest_words = self.get_closest_words(analogy, n)
        return w4_norm in closest_words or w4_norm == analogy
