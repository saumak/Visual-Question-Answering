import numpy as np

def text_feature(spacy_lg, question):
    tokenization, word_embed = spacy_lg(question), np.zeros((1, 30, 300))
    for idx in range(len(tokenization)): word_embed[0,idx,:] = tokenization[idx].vector
    return word_embed
