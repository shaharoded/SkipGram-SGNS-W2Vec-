# SkipGram Word2Vec Model with SGNS

This project implements the SkipGram Word2Vec model using SkipGram with Negative Sampling (SGNS). The main components of the project include data preprocessing, training the model, and performing word analogy and similarity tasks. The model can be trained on any text corpora provided by the user.

## Project Structure

- `skipgram_sgns.py`: Contains the SkipGram class and supporting functions for training and evaluating the model.
- `test_skipgram.py`: Contains the unit tests for the SkipGram model and its methods.
- `Corpora/`: Directory to store text corpora for training the model.

## Setup and Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/your-repo/skipgram-sgns.git
    cd skipgram-sgns
    ```

2. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run Unittests**

    ```bash
    python test_skipgram.py {file_path}
    ```

    For example:

    ```bash
    python test_skipgram.py Corpora/drSeuss.txt
    python test_skipgram.py Corpora/harryPotter1.txt
    ```