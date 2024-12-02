import numpy as np
import gensim

def smart_procrustes_align_gensim(base_embed, other_embed):
    """
    Aligns other_embed to base_embed using the Procrustes method.
    """
    # Intersection of vocabularies
    in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed)

    # Get the embedding matrices (access via `wv.vectors`)
    base_vecs = in_base_embed.wv.vectors
    other_vecs = in_other_embed.wv.vectors

    # Procrustes transformation
    m = other_vecs.T.dot(base_vecs)
    u, _, v = np.linalg.svd(m)
    ortho = u.dot(v)

    # Apply the transformation
    in_other_embed.wv.vectors = other_embed.wv.vectors.dot(ortho)
    return other_embed

def intersection_align_gensim(m1, m2):
    """
    Intersects the vocabularies of two gensim models and aligns indices.
    """
    # Access vocab through m.wv.key_to_index (for Word2Vec models)
    vocab_m1 = set(m1.wv.key_to_index.keys())
    vocab_m2 = set(m2.wv.key_to_index.keys())
    
    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2

    # Sorting by frequency
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.wv.get_vecattr(w, 'count') + m2.wv.get_vecattr(w, 'count'), reverse=True)

    for m in [m1, m2]:
        # Replace the embedding matrix with the common vocabulary
        indices = [m.wv.key_to_index[w] for w in common_vocab]
        old_arr = m.wv.vectors
        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.vectors = new_arr

        # Update index_to_key and key_to_index
        m.wv.index_to_key = common_vocab
        m.wv.key_to_index = {word: idx for idx, word in enumerate(common_vocab)}

    return m1, m2