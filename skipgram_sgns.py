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
from nltk.corpus import stopwords

nltk.data.clear_cache()
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

lemmatizer = WordNetLemmatizer()

# Load stopwords once globally
STOP_WORDS = set(stopwords.words('english'))

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


def normalize_sentence(sentence, remove_stopwords = True, lemmatize=False, use_pos=False):
    """
    Normalize a sentence by tokenizing, lemmatizing, and removing punctuation.

    Args:
        sentence: a single sentence as a string
        lemmatize: whether to lemmatize words (default: False)
        use_pos: whether to use POS tagging for lemmatization (default: False)

    Returns:
        A normalized sentence (a single str).
    """
    sentence = expand_contractions(sentence.lower().strip())  # Remove constraction abbriviations
    words = sentence.split()
    normalized_words = []
    
    for word in words:
        word = re.sub(r'[^\w\s-]', '', word)    # Remove punctuation / irrelevant chars
        if word == '':
            continue
        if remove_stopwords and word in STOP_WORDS:
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
    def __init__(self, sentences, d=100, neg_samples=4, context=4, word_count_threshold=5, lr_decay = False, decay_rate = 0.01, print_flag = False):
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
        self.sentences = self._init_snip_sentences(sentences)   # Keep only relevant words in training sentences
        
        # the target word embedding matrix
        self.T = None
        # the context word embedding matrix 
        self.C = None
        
        # each row -> word in the vocabulary
        # each column -> a dimension of the embedding vector.
        self.V = self.combine_vectors(self.T, self.C)   # the combination of T and C, Model's true vector


    def _init_snip_sentences(self, sentences):
        '''
        init function to remove non frequent words from sentences in order to optimize 
        vector creation (ensure enough positive entries per word)
        '''
        out_sent = []
        for sentence in sentences:
            words = [w for w in sentence.split() if w in self.vocab]
            if len(words) > 0:
                sentence = ' '.join(words)
                out_sent.append(sentence)
        return out_sent


    def _check_words_in_vocabulary(self, input_words):
        """
        Checks if the input words are in the vocabulary and prints a warning if any word is not recognized.

        Args:
            input_words: A list of words to check.

        Returns:
            bool: True if all words are in the vocabulary, False otherwise.
        """
        unrecognized_words = [word for word in input_words if word not in self.vocab]
        
        if unrecognized_words:
            if self.print_flag:
                print(f"[Warning]: The following word(s) are not in the vocabulary: {', '.join(unrecognized_words)}. "
                    "A default value will be returned from the method.")
            return False
        
        return True
    
    
    def compute_similarity(self, w1, w2):
        """ 
        Returns the cosine similarity (in [0,1]) between the specified words.
        NOTE: Having no external control over the input, function assumes w1 and w2 are normalized.
        Function will check if they are a part of the vocab as is. If not - they will be normalized.
        In general, similarity is calculated between the 2 normalized words, so 2 non identical 
        words might have the same score if are mapped to the same word in the vector space.

        Args:
            w1: a word
            w2: a word
        Retunrns: a float in [0,1]; defaults to 0.0 if one of specified words is OOV (after normalization).
    """
        sim = 0.0  # default
        w1_norm = normalize_sentence(w1) if w1 not in self.vocab else w1
        w2_norm = normalize_sentence(w2) if w2 not in self.vocab else w2
        # Check if both words are in our word_index map
        if not self._check_words_in_vocabulary([w1_norm, w2_norm]):
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
        Word is assumed to be normalized. If not in vocab, function will attemp normalization.
        If w still not in self.vocab -> ValueError.
        Args:
            w: the word to find close words to.
            n: the number of words to return. Defaults to 5.
        
        Returns: List() of size n.
        """
        w_norm = normalize_sentence(w) if w not in self.vocab else w
        if not self._check_words_in_vocabulary([w_norm]):
            return []
        
        w_index = self.word_index[w_norm]
        word_emb = self.V[:, w_index]

        # Compute cosine similarities (efficiently!) -> 1D array
        # Similarity is calculated based on the combination method chose
        cos_sim = np.dot(self.V.T, word_emb) / (
            np.linalg.norm(self.V.T, axis=1) * np.linalg.norm(word_emb))

        # Sort by similarity -> list of indices 
        max_sorted = list(np.argsort(cos_sim)[::-1][:n + 1])  # sort decending, slice

        # Exclude the target word and get the top n words
        candidates = [self.index_word[i] for i in max_sorted if i != w_index]
        return candidates[:n]

    
    def _create_learning_vectors(self, sentence, all_words, all_counts):
        """
        Creates lists of positive and negative word pairs for a given sentence.
        Positive pairs are actual context pairs within a specified window size, 
        while negative pairs are randomly generated from vocabulary based on frequency,
        Positive samples are from window [...x-2, x-1, t, x+1, x+2...].
        Will compose an appropriate vector using the word indices and occurences.
        Each word vector (training vector) is a sparse vector like:
        {target_idx: {pos_context_w: 1, neg_context_w1:0, ..., neg_context_wk:0}}
        for k -> neg_samples.
        
        Args:
            sentence (str): A single sentence.
            all_words(list): A list of all the words based on key order from dict.
            all_counts (list): A list of all words frequency, for random selection (negative sampling)

        Returns:
            dict (of vectors):
                learning vectors like {word_idx: context_vector_from_sentence,...} with positive and negative
                context values for word.
        """
        
        def generate_skipgrams(words, context):
            '''
            Return list of tuples for all positive context words per sentence.
            '''
            window_size = context // 2
            pos_l = []

            for i, t in enumerate(words):
                left = max(0, i - window_size)
                right = min(len(words), i + window_size + 1)
                context = words[left:i] + words[i+1:right]
                
                # Avoid adding t as the context of t (F-drSeuss!)
                context = [w for w in context if w != t]

                pos_l.append((t, context))

            return pos_l
        
        # Filter only relevant words to vocab (word occurance is significant)
        words = sentence.split()
        learning_vectors = []          
        
        skipgrams = generate_skipgrams(words, self.context) # [(t1, [context_words]),...]
        for t, window in skipgrams:
            t_idx = self.word_index[t]
            for c_w in window:
                vector = {}
                # Positive entry:
                c_idx = self.word_index[c_w]
                vector[c_idx] = 1
                # Negative entries:
                for _ in range(self.neg_samples):
                    n_w = random.choices(all_words, weights=all_counts, k=1)[0]
                    while n_w == t or n_w == c_w:  # Negative word will not be t or positive context word
                        n_w = random.choices(all_words, weights=all_counts, k=1)[0]
                    n_idx = self.word_index[n_w]
                    vector[n_idx] = 0
                learning_vectors.append((t_idx, vector))  # Adding training vector of size 1 + self.neg_samples    
        return learning_vectors

    def _preprocess_sentences(self):
        """
        Preprocesses all train sentences for the SkipGram model, 
        Creates learning vectors using positive and negative samples (binary entries).
        Yields a list of tuples [(t_idx, vector),...] for each sentence. t_idx will appear several times
        since it's sampled for |context| times.
        
        :yield:
            A list of tuples for each sentence, where each tuple contains (t_idx, vector).
        """
        all_words = list(self.word_counts.keys())
        all_counts = list(self.word_counts.values())
        
        for sentence in self.sentences:
            # Create the learning context vector
            sentence_vectors = self._create_learning_vectors(sentence, all_words, all_counts)
            
            # Yield each word index and its corresponding vector
            for word_idx, vector in sentence_vectors:
                yield word_idx, vector          
    
    
    def _forward_pass(self, t_index, c_vector):
        """
        Performs a forward pass of the neural network for the SkipGram model.
        c_vector (context vector) is a sparse dictionary with context word indices as keys.
        
        Args:
            t_index: Index of the target word.
            c_vector: A dictionary where keys are context word indices and values are the actual labels (1 or 0).

        Returns:
            hidden: The hidden layer representation of the target word.
            y_pred: The predicted probabilities for the context words (positive and negative samples).
            y: The context words values, the truth vector for this pass.
            context_indices: The indices of the context words used in this pass.
        """
        # Extract the hidden layer vector for the target word (|embedding_size| x 1)
        hidden = self.T[:, t_index][:, None]
        
        # Get the relevant context word indices
        context_indices = list(c_vector.keys())
        
        # Extract the relevant rows of the C matrix (corresponding to context_indices)
        C_subset = self.C[context_indices, :]
        
        # Compute the dot product between the hidden vector and the relevant rows of C
        y_pred = np.dot(C_subset, hidden).flatten()
        
        # No need to convert c_vector to a dense vector; just use its values directly
        y = np.array(list(c_vector.values()))
        
        return hidden, y_pred, y, context_indices

    
    def _calculate_loss(self, y, y_pred):
        '''
        Use a Log-Loss (Cross-Entropy) calculation to estimate the training progress.
        Loss is the sum of the error vector divided by the size (avg).
        Vectors srill contain only the relevant entries of the context calcualtion (1 + self.negative_samples)
        size vector.
        '''
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Avg Cross Entropy Loss Calculation
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return loss
    

    def _backpropagation(self, t_index, hidden, y_pred, y, step_size, context_indices):
        '''
        Calculate the error vector and use it to calculate the gradients.
        Updates the matrices T and C according to the gradients.

        Args:
            t_index: Index of the target word.
            hidden: The hidden layer vector (embedding of the target word).
            y_pred: Predicted probabilities for the context words.
            y: True labels for the context words.
            step_size: The learning rate.
            context_indices: Indices of the context words (both positive and negative).
        '''
        # Calculate the error vector for the relevant context words
        error = y_pred - y  # error shape: (|context_indices|,)

        # Compute the gradient for the C matrix (only for relevant rows)
        c_grad = np.dot(error[:, None], hidden.T)  # c_grad shape: (|context_indices|, d)

        # Update the C matrix only for the relevant context words
        self.C[context_indices, :] -= step_size * c_grad

        # Compute the gradient for the T matrix (only for the target word)
        t_grad = np.dot(self.C[context_indices, :].T, error)  # t_grad shape: (d,)

        # Update the T matrix for the target word
        self.T[:, t_index] -= step_size * t_grad

        return self.T, self.C
    
    
    def learn_embeddings(self, step_size=0.001, epochs=50, early_stopping=3, model_path=None, keep_train = False):
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
            self.T = np.random.uniform(0, 1, size=(self.d, self.vocab_size))  # Embedding matrix of target words
            self.C = np.random.uniform(0, 1, size=(self.vocab_size, self.d))  # Embedding matrix of context words


        self.print_flag and print("[Training Status]: Started Model Training")
        best_loss = np.inf  # Initialize the best loss as infinity
        epochs_no_improve = 0  # Initialize epochs without improvement (for stop criteria)

        for i in range(1, epochs + 1):
            epoch_loss = []  # Initialize loss for this epoch
            n_learning_vectors = 0
            for t_index, c_vector in self._preprocess_sentences():            
                
                n_learning_vectors += 1
                # Forward pass
                hidden, y_pred, y, context_indices = self._forward_pass(t_index, c_vector)

                # Calculate loss
                epoch_loss.append(self._calculate_loss(y, y_pred))
                
                # Backpropagation
                self.T, self.C = self._backpropagation(t_index, hidden, y_pred, y, step_size, context_indices)
            
            # Finalize tmp model and Backup (every epoch)
            self.V = self.combine_vectors(T=self.T, C=self.C, combo=0, model_path=model_path)
                
            epoch_loss = np.mean(epoch_loss)
            if self.lr_decay:
                step_size *= 1 / (1 + self.decay_rate * i)   # Time LR decay
            self.print_flag and print(f"[Training Status]: Epoch {i}, Loss: {epoch_loss}, Time: {round((time.time()-s)/60,2)}m, using {n_learning_vectors} Learning Vectors")

            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0  # Reset the count
            else:
                epochs_no_improve += 1  # Increment the count

            if epochs_no_improve == early_stopping:
                self.print_flag and print("[Training Status]: Stopped! Early stopping criteria met.")
                break

        self.print_flag and print(f"[Train Finished]: Model saved to path: '{model_path}'")

        # Return in case you wish to catch them
        return self.T, self.C

    
    def combine_vectors(self, T, C, combo=0, model_path=None):
        """
        Returns a single embedding matrix and saves it to the specified path

        Args:
            T: The learned targets (T) embeddings (as returned from learn_embeddings())
            C: The learned contexts (C) embeddings (as returned from learn_embeddings())
            combo: indicates how to combine the T and C embeddings (int)
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
            V = np.concatenate((T, C.T), axis=0)
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
        w1_norm = normalize_sentence(w1) if w1 not in self.vocab else w1
        w2_norm = normalize_sentence(w2) if w2 not in self.vocab else w2
        w3_norm = normalize_sentence(w3) if w3 not in self.vocab else w3
        if not self._check_words_in_vocabulary([w1_norm, w2_norm, w3_norm]):
            return 'Cannot find analogy'

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
        max_sorted = [idx for idx in max_sorted if idx not in [idx_1, idx_2, idx_3]] # Take min 1 out of 4 that is not an input
        
        if not max_sorted:
            raise ValueError("[Error]: Not enough words in the vocabulary to find a suitable analogy.")

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
        w1 = normalize_sentence(w1) if w1 not in self.vocab else w1
        w2 = normalize_sentence(w2) if w2 not in self.vocab else w2
        w3 = normalize_sentence(w3) if w3 not in self.vocab else w3
        w4 = normalize_sentence(w4) if w4 not in self.vocab else w4

        # Check if all words are in the vocabulary
        if not self._check_words_in_vocabulary([w1, w2, w3, w4]):
            return False
        
        # Assuming analogy is found by combining w1 - w2 + w3 =~ w4
        analogy = self.find_analogy(w1, w2, w3)
        closest_words = self.get_closest_words(analogy, n)
        return w4 in closest_words or w4 == analogy
