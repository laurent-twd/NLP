import os
import json

import tensorflow as tf
import numpy as np
import random

from collections import Counter
import itertools

from encoder import BertEncoder
from utilities.utils import create_padding_mask
from custom_schedule import CustomSchedule
from copy import deepcopy

MAX_VOCAB_SIZE = 250000

class MyELECTRA:

    def __init__(self, parameters, path_model):
        
        """
        Parameters
        ----------
        
        parameters: dict
            Dictionary of parameters to initialize the model.
            
        """
        
        self.path_model = path_model
        self.n_special_tokens = 5 # Padding / [BOS] / [EOS] / [UNKNOWN] / [MASKED]
        self.n_special_characters = 4 # Padding / < / > / [UNKNOWN]
        
        self.d_model = parameters['d_model']
        self.dff = parameters['dff']
        self.num_layers = parameters['num_layers']
        self.pe_input = parameters['pe_input']
  
        self.fitted = False

        #try:
         
        if parameters['fitted'] == True:
            self.fitted = True
            self.word2idx = parameters['word2idx']
            self.char2idx = parameters['char2idx']
            self.idx2word = {v: k for k, v in self.word2idx.items()}
            self.vocabulary = set(parameters['vocabulary'])
            self.list_vocabulary = parameters['vocabulary']
            self.vocab_size = parameters['vocab_size']

        # Model
        self.generator = BertEncoder(vocab_size = MAX_VOCAB_SIZE,
                                    output_dim = MAX_VOCAB_SIZE,
                                    hidden_size = self.d_model,
                                    num_layers = self.num_layers,
                                    num_attention_heads = 4,
                                    max_sequence_length = self.pe_input,
                                    inner_dim = self.dff)

        # Model
        self.discriminator = BertEncoder(vocab_size = MAX_VOCAB_SIZE,
                                    output_dim = 1,
                                    hidden_size = self.d_model,
                                    num_layers = self.num_layers,
                                    num_attention_heads = 4,
                                    max_sequence_length = self.pe_input,
                                    inner_dim = self.dff)

        # Optimizer

        learning_rate_gen = CustomSchedule(self.d_model)
        learning_rate_disc = CustomSchedule(self.d_model)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate_gen, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-9)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate_disc, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-9)
        self.ckpt = tf.train.Checkpoint(generator = self.generator, 
                                        discriminator = self.discriminator,
                                        generator_optimizer = self.generator_optimizer,
                                        discriminator_optimizer = self.discriminator_optimizer)
        checkpoint_path = os.path.join(self.path_model, 'tf_ckpts')
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_path, max_to_keep=5)

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')

    def get_index_word(self, word):

        """
        
        Returns the index of a token. Returns 3 (unknown) if it is an out-of-vocabulary token.

        """
        
        if word in self.vocabulary:
            return self.word2idx[word]
        else:
            if word == '[BOS]':
                return 1
            elif word == '[EOS]':
                return 2
            elif word == '[MASKED]':
                return 4
            else:
                return 3 # index for UNKNOWN
    
    def get_word_index(self, index):

        if index >= self.n_special_tokens and index <= self.vocab_size - 1 + self.n_special_tokens:
            return self.idx2word[index]
        elif index == 1:
            return '[BOS]'
        elif index == 2:
            return '[EOS]'
        elif index == '[MASKED]':
            return 4
        else:
            return '[UNKNOWN]'

        
    def pad_char(self, sentence, max_len, max_len_char):
        
        """
        Parameters
        ----------

        sentence: list of int

        max_len: int
            Maximum sentence length in a batch.

        max_len_char: int
            Maximum word length in a batch

        The output has a shape (max_len, max_len_char).

        Output
        ------

        Returns the padded characters' indexes.

        """

        def get_index_char(x):
            def index2char(char):
                try:
                    return self.char2idx[char]
                except:
                    return 3
            return [index2char(c) for c in x]
        
        indexed_char = list(map(get_index_char, sentence))
        chars = tf.keras.preprocessing.sequence.pad_sequences(indexed_char, maxlen = max_len_char, padding = 'post', truncating = 'post')
        padded_chars = tf.concat([chars, tf.constant(0., shape = (max_len - chars.shape[0], max_len_char))], axis = 0)
        return padded_chars

    def process_sentence(self, tar_sentence, tar_indexes, masking_rate):
        
        inp_sentence = deepcopy(tar_sentence)
        inp_indexes = deepcopy(tar_indexes)
        length = len(tar_sentence) - 2 
        number_of_words_masked = int(masking_rate * length)
        masked_indexes = np.random.choice(range(1, length + 1), number_of_words_masked, replace = False)
        type_mask = np.random.choice([0, 1, 2], size = number_of_words_masked, replace = True, p = [0.8, 0.1, 0.1])
        for i in range(number_of_words_masked):
            if type_mask[i] == 0:
                inp_sentence[masked_indexes[i]] = '[MASKED]'
                inp_indexes[masked_indexes[i]] = 4
            elif type_mask[i] == 1:
                temp = np.random.choice(self.list_vocabulary)
                inp_sentence[masked_indexes[i]] = temp
                inp_indexes[masked_indexes[i]] = self.get_index_word(temp)
        
        return inp_sentence, inp_indexes, masked_indexes
        
    def get_next_batch(self, batch_size, set_index, source_text, indexed_text, masking_rate):

        num_samples = np.minimum(batch_size, len(set_index))
        target_indexes = random.sample(set_index, num_samples)
        set_index.difference_update(set(target_indexes))

        tar_text = list(source_text[target_indexes])
        tar_indexes = list(indexed_text[target_indexes])
        temp = list(map(lambda x: self.process_sentence(tar_text[x], tar_indexes[x], masking_rate), range(num_samples)))
        inp_text, inp_indexes, masked_idx = list(zip(*temp))

        max_len_char = 20 
        max_len = self.pe_input

        language_mask = tf.concat(list(map(lambda x: tf.reduce_sum(tf.one_hot(x, depth = max_len), axis = 0)[tf.newaxis, :], list(masked_idx))), axis = 0)

        inp_chars = list(map(lambda x: self.pad_char(x, max_len, max_len_char)[tf.newaxis, :, :], inp_text))
        inp_chars = tf.concat(inp_chars, axis = 0)

        inp_words = tf.cast(tf.keras.preprocessing.sequence.pad_sequences(inp_indexes, maxlen = max_len, padding = 'post'), tf.int32)
        tar_words = tf.cast(tf.keras.preprocessing.sequence.pad_sequences(tar_indexes, maxlen = max_len, padding = 'post'), tf.int32)

        return inp_words, inp_chars, tar_words, language_mask

    def get_input_discriminator(self, inp_words, gen_words, language_mask):

        enc_padding_mask = create_padding_mask(inp_words)
        gen_words = tf.where(tf.cast(language_mask, tf.bool), gen_words, inp_words)
        get_text = np.vectorize(lambda x: self.get_word_index(x))
        gen_text = list(get_text(gen_words))
        max_len_char = 20 
        max_len = self.pe_input

        gen_chars = list(map(lambda x: self.pad_char(x, max_len, max_len_char)[tf.newaxis, :, :], gen_text))
        gen_chars = tf.concat(gen_chars, axis = 0)

        gen_is_true = tf.cast(inp_words == gen_words, dtype = tf.float32)
        adversarial_mask = tf.maximum(0., language_mask - gen_is_true)

        return gen_words, gen_chars, enc_padding_mask, adversarial_mask

    @tf.function
    def train_step_generator(self, inp_words, inp_chars, tar_words, language_mask):

        enc_padding_mask = create_padding_mask(inp_words)

        with tf.GradientTape() as tape:
            gen_logits = self.generator([inp_words, enc_padding_mask])['logits']

            mask_logits = tf.concat([tf.zeros(self.n_special_tokens + self.vocab_size), tf.ones(MAX_VOCAB_SIZE  - self.n_special_tokens - self.vocab_size)], 0)
            gen_logits += mask_logits[tf.newaxis, tf.newaxis, :] * (-1e9)
            gen_words = tf.compat.v1.distributions.Categorical(logits = gen_logits).sample()
            loss = tf.keras.losses.sparse_categorical_crossentropy(tar_words, gen_logits, from_logits = True)
            loss = tf.reduce_sum(loss * language_mask, axis = 1) / tf.reduce_sum(language_mask, axis = 1)
            batch_loss = tf.reduce_mean(loss)

            variables = self.generator.trainable_variables[:-2]
            gradients = tape.gradient(batch_loss, variables)    
            self.generator_optimizer.apply_gradients(zip(gradients, variables))
        
        return batch_loss, gen_words
               
    @tf.function
    def train_step_discriminator(self, gen_words, gen_chars, enc_padding_mask, adversarial_mask):

        with tf.GradientTape() as tape:
            disc_logits = self.discriminator([gen_words, enc_padding_mask])['logits']
            probs = tf.squeeze(tf.math.sigmoid(disc_logits + 1e-9), axis = 2)
            loss = adversarial_mask * tf.math.log(probs) + (1 - adversarial_mask) * tf.math.log(1. - probs)
            padding_mask = 1. - enc_padding_mask
            loss = tf.reduce_sum(loss * padding_mask, axis = 1) / tf.reduce_sum(padding_mask, axis = 1)
            batch_loss = - tf.reduce_mean(loss)

            variables = self.discriminator.trainable_variables[:-2]
            gradients = tape.gradient(batch_loss, variables)    
            self.discriminator_optimizer.apply_gradients(zip(gradients, variables))
        
        return batch_loss

    def fit(self, corpus, epochs, batch_size, masking_rate = 0.15, min_count = 1):
        
        """
        Fits the model:
            - Initializes or updates the mapping words / indexes and the mapping characters / indexes
            - Processes the text data and arranges it to create batch of sentence with similar lengths
            - Performs the stochastic gradient descent
            
        corpus: list
            List of tokenized sentences.
            
        epochs: int
            Number of epochs.
        
        batch_size: int
            Batch size.
            
        window_size: int
            Window size in the Word2Vec model used to initialize the embedding matrices of the different embedding layers.
            If None, the embedding layers are randomly initialized.

        min_count: int
            Threshold under which the words won't be taken into account to update the vocabulary.
            If a word is not in the vocabulary and has a frequency < min_count, the token will be considered as <UNKNOWN> : 3.
            
        """

        ## Model Definition
        def flatten(l):
            return list(itertools.chain.from_iterable(l))

        frequency = Counter(flatten(corpus))
        vocabulary_corpus = set([x for x in list(frequency.keys()) if frequency[x] >= min_count]) 
        if self.fitted:

            print("Loading Model...")
            new_vocabulary = vocabulary_corpus - self.vocabulary
            vocab_size = len(self.vocabulary)
            add_to_dic_w2i = dict(zip(new_vocabulary, range(vocab_size + self.n_special_tokens, vocab_size + len(new_vocabulary) + self.n_special_tokens)))
            add_to_dic_i2w = dict(zip(range(vocab_size + self.n_special_tokens, vocab_size + len(new_vocabulary) + self.n_special_tokens), new_vocabulary))
            self.word2idx.update(add_to_dic_w2i) 
            self.idx2word.update(add_to_dic_i2w) 
            vocabulary = self.vocabulary.union(vocabulary_corpus)
            assert len(vocabulary) == (self.vocab_size + len(new_vocabulary))
            self.vocabulary = vocabulary
            self.vocab_size = len(self.vocabulary)
            self.list_vocabulary = list(vocabulary)

            # Characters
            characters_corpus = set(Counter(''.join(list(self.vocabulary) + ['[BOS]' , '[EOS]'])).keys())
            new_characters = characters_corpus - set(self.char2idx.keys())
            n_char = len(self.char2idx)
            add_to_dic_c2i = dict(zip(new_characters, range(n_char + 1, n_char + len(new_characters) + 1)))
            self.char2idx.update(add_to_dic_c2i)

        else:

            print("Initializing Model")
            self.vocabulary = vocabulary_corpus 
            self.vocab_size = len(self.vocabulary)

            self.word2idx = {}
            self.idx2word = {}
            for i, word in enumerate(self.vocabulary):
                self.word2idx[word] = i + self.n_special_tokens
                self.idx2word[i + self.n_special_tokens] = word  
            self.list_vocabulary = list(self.vocabulary)

            characters_corpus = set(Counter(''.join(list(self.vocabulary) + ['[BOS]' , '[EOS]'])).keys())
            self.char2idx = {}
            for i, char in enumerate(characters_corpus):
                self.char2idx[char] = i + self.n_special_characters
            self.char2idx['<'] = 1
            self.char2idx['>'] = 2
            self.n_char = len(characters_corpus)

        ## Dataset
        source_text = []
        indexed_text = []
        limit = int(1 / masking_rate) + 1
        for sentence in corpus:
          indexes = list(map(self.get_index_word, sentence))
          if len(sentence) >= limit:
            source_text.append(['[BOS]'] + sentence + ['[EOS]'])
            indexed_text.append([1] + indexes + [2])
        source_text = np.array(source_text)
        indexed_text = np.array(indexed_text)

        ## Training
        print("Training...")
        self.fitted = True
        for _ in range(epochs):
            set_index = set(range(len(source_text)))
            progbar = tf.keras.utils.Progbar(len(source_text))
            while len(set_index) > 0:
                inp_words, inp_chars, tar_words, language_mask = self.get_next_batch(batch_size, set_index, source_text, indexed_text, masking_rate)
                generator_loss, gen_words = self.train_step_generator(inp_words, inp_chars, tar_words, language_mask)
                gen_words, gen_chars, enc_padding_mask, adversarial_mask = self.get_input_discriminator(inp_words, gen_words, language_mask)
                discriminator_loss = self.train_step_discriminator(gen_words, gen_chars, enc_padding_mask, adversarial_mask)
                progbar.add(inp_words.shape[0], values = [('Gen. Loss', generator_loss), ('Disc. Loss', discriminator_loss)])

            self.ckpt_manager.save()

    def save_model(self):
        
        """
        Saves the model.
        
        """
        parameters = {
            'fitted' : self.fitted,
            'word2idx' : self.word2idx,
            'char2idx' : self.char2idx,
            'vocabulary' : list(self.vocabulary),
            'vocab_size' : self.vocab_size,
            'd_model' : self.d_model,
            'dff' : self.dff,
            'num_layers' : self.num_layers,
            'pe_input' : self.pe_input,
        }
        
        with open(os.path.join(self.path_model, 'parameters.json'), 'w') as params:
            json.dump(parameters, params)

        return parameters

