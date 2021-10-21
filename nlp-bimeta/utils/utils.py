import json
from dataset.genome import GenomeDataset
import os
import numpy as np
import pandas as pd
import random

from gensim.models.wrappers import LdaMallet
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore

from gensim.models.coherencemodel import CoherenceModel

def load_groups_seeds(bimetaout_dir, dataset_name):
    '''
    Load groups and seeds file (phase 1 of bimeta) according to dataset_name
    Args:
        dataset_name: name of dataset (e.g. S1, L1,...)
        bimetaout_dir: bimeta output dir
    '''
    group_file = os.path.join(bimetaout_dir, dataset_name + '.json')
    try:
        with open(group_file, 'r') as fg:
            group_data = json.load(fg)
        
        return group_data['groups'], group_data['seeds']
    except Exception as e:
        print('Error when loading file {} '.format(group_file))
        print('Cause: ', e)
        return []
    


def load_genomics(dataset_name,
                    kmers, 
                    lmer,
                    maximum_seed_size,
                    num_shared_reads,
                    graph_file=None,
                    is_serialize=False,
                    is_deserialize=False,
                    is_normalize=False,
                    only_seed=False,
                    is_tfidf=False):
    '''
    Loads fna file.
    Args:
        dataset_name: name of dataset (e.g. S1.fna, L1.fna,...)
        kmers: list of kmers.
        lmer: lmer.
        maximum_seed_size.
        num_shared_reads.
        graph_file: computed groups/seeds json file.
        is_serialize: True to serialize computed groups/seeds to json file.
        is_deserialize: True to load computed groups/seeds in json file.
        is_normalize: whether to normalize kmer-features.
        only_seed: True to compute kmer features using seeds only.
    '''
    genomics_dataset = GenomeDataset(
        dataset_name, kmers, lmer,
        graph_file=graph_file,
        only_seed=only_seed,
        maximum_seed_size=maximum_seed_size,
        num_shared_reads=num_shared_reads,
        is_serialize=is_serialize,
        is_deserialize=is_deserialize,
        is_normalize=is_normalize,
        is_tfidf=is_tfidf)

    return genomics_dataset.kmer_features,\
        genomics_dataset.labels,\
        genomics_dataset.groups,\
        genomics_dataset.seeds


def export_clustering_results(raw_reads, groups, n_clusters, y_pred, save_path):
    exported_results = {k+1: [] for k in range(n_clusters)}

    for i, group in enumerate(groups):
        cluster_id = y_pred[i]
        for r in group:
            exported_results[cluster_id + 1].append(r)
    
    with open(save_path, 'w') as f:
        json.dump(exported_results, f)

    print(f'Saved result file at {save_path}')


def do_LDA_Mallet(path_to_mallet_binary, corpus, dictionary, 
                    ldaout_dir, dataset_name, 
                    n_topics=10, n_workers=2, 
                    n_passes=10, max_iters=200):
    """
    A wapper of LDA Mallet
    If the program is run out of memory, 
        you should try to change n_topics, n_worker, n_passes, max_iters
    :param corpus:
    :param dictionary:
    :param n_topics:
    :param n_worker:
    :param n_passes:
    :param max_iters:
    :return: LdaModel
    """

    lda_model = LdaMallet(mallet_path=path_to_mallet_binary, 
                        corpus=corpus, id2word=dictionary, 
                        num_topics=n_topics, workers=n_workers, 
                        optimize_interval=n_passes, iterations=max_iters,
                        prefix=os.path.join(ldaout_dir, dataset_name + '.lda.'))
    return lda_model


def getDocTopicDist(model, corpus):
    """
    LDA transformation, for each doc only returns topics with non-zero weight
    This function makes a matrix transformation of docs in the topic space.
    """
    return model[corpus]


def do_LDA_Multicore(corpus, dictionary, 
                    n_topics=10, n_workers=2, 
                    n_passes=10, max_iters=50):
    """
    Online Latent Dirichlet Allocation (LDA) in Python, 
    using all CPU cores to parallelize and speed up model training.
    :param corpus:
    :param dictionary:
    :param n_topics:
    :param n_workers:
    :param n_passes:
    :param max_iters:
    :return: LdaModel
    """
    lda_model = LdaMulticore(corpus, id2word=dictionary, 
                            num_topics=n_topics, passes=n_passes, 
                            workers=n_workers, iterations=max_iters)
    return lda_model


def getDocTopicDist(model, corpus):
    """
    LDA transformation, for each doc only returns topics with non-zero weight
    This function makes a matrix transformation of docs in the topic space.
    """
    doc_topics = model[corpus]

    doc_topics_df = pd.DataFrame((score for (id, score) in doc_topic) for doc_topic in doc_topics)
    
    return doc_topics_df.fillna(0)
    """
    ndocs = len(doc_topics)
    topics = [None] * ndocs
    for i in range(ndocs):
        topics[i] = np.array([f[1] for f in model[corpus[i]]])
    return pd.DataFrame(topics).fillna(0)
    """

def getLdaDocTopics(lda_topics_file, numtopics):
    """
    This function load lda_topics_file (datasetname.lda.doctopics.txt) 
    This temporary file was created by LdaMallet while training LDA model.
    """

    df = pd.read_csv(lda_topics_file, 
                    header=None, 
                    sep = '\t', 
                    usecols=range(2, numtopics + 2))
    
    return df

def getCoherenceScore(model, corpus, coherence='u_mass'):
    cm = CoherenceModel(model=model, corpus=corpus, coherence='u_mass')
    return cm.get_coherence()



