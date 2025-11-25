"""
Feature engineering and text encoding utilities.

This module builds a vocabulary using Keras Tokenizer,
encodes reviews into padded sequences, and splits the
dataset into training, validation, and test sets.
"""

from __future__ import annotations

import os
import pickle
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# Maximum vocabulary size and sequence length can be tuned later
DEFAULT_MAX_WORDS = 10_000
DEFAULT_MAX_LENGTH = 200


def get_artifacts_dir() -> str:
    """
    Return the absolute path to the artifacts directory,
    used for storing tokenizer and other reusable assets.
    """
    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(src_dir)
    artifacts_dir = os.path.join(project_root, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    return artifacts_dir


class TextEncoder:
    """
    Text encoding utility for sentiment analysis.

    Responsibilities:
      - fit a Keras Tokenizer on preprocessed text;
      - convert reviews to integer sequences;
      - pad / truncate sequences to a fixed length;
      - split data into train / validation / test sets;
      - persist the tokenizer for later reuse (web app, etc.).
    """

    def __init__(
        self,
        max_words: int = DEFAULT_MAX_WORDS,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.max_words = max_words
        self.max_length = max_length
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")

    # ------------------------------------------------------------------ #
    # Tokenizer fitting and persistence
    # ------------------------------------------------------------------ #

    def fit_tokenizer(self, texts: List[str]) -> None:
        """
        Fit the tokenizer on a list of preprocessed texts and
        save it to the artifacts directory.
        """
        self.tokenizer.fit_on_texts(texts)

        artifacts_dir = get_artifacts_dir()
        tokenizer_path = os.path.join(artifacts_dir, "tokenizer.pickle")

        with open(tokenizer_path, "wb") as f:
            pickle.dump(self.tokenizer, f)

        vocab_size = len(self.tokenizer.word_index)
        print(f"Tokenizer fitted. Full vocabulary size: {vocab_size}")
        print(f"Using top {self.max_words} words for modelling.")

    # ------------------------------------------------------------------ #
    # Encoding utilities
    # ------------------------------------------------------------------ #

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert texts to padded integer sequences suitable
        for neural network input.
        """
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding="post",
            truncating="post",
        )
        return padded

    def prepare_data(
        self, reviews: List[str], labels: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode reviews and split the dataset into
        training / validation / test subsets.

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Encode reviews
        X = self.encode_texts(reviews)
        y = np.array(labels)

        # First split: train vs temp (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y,
        )

        # Second split: validation vs test (from temp)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            random_state=42,
            stratify=y_temp,
        )

        print(f"Training set:   {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set:       {X_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test
