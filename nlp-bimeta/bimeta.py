import glob, os, time
import json
import numpy as np
from collections import defaultdict
import argparse

from dataset.genome import GenomeDataset
from utils.utils import load_genomics 
from debug.visualize import get_group_label, visualize
from utils.metrics import genome_acc, group_precision_recall

import sys

from sklearn.cluster import KMeans

if __name__ == "__main__":
    
    sys.path.append('.')

    DATASET_DIR = 'data/raw'    # Raw fasta data dir
    DATASET_NAME = 'hmp_test'   # Specifc fasta dataset or all of them
    RESULT_DIR = 'results'      # Result dir

    # Versioning each runs
    ARCH = 'bimeta'
    DATE = '20210903'

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='directory contain fasta file')
    parser.add_argument('--dataset_name', type=str, help='specific name of dataset to run'\
                        '(all is run every found dataset), e.g S1.fna, name is S1')
    parser.add_argument('--result_dir', type=str, default='results', help='directory for saving the resuls')
    parser.add_argument('--verbose', help='Whether to print log in terminal', action='store_true')

    args = parser.parse_args()

    verbose = args.verbose

    DATASET_NAME = args.dataset_name if args.dataset_name else DATASET_NAME
    DATASET_DIR = args.data_dir if args.data_dir else DATASET_DIR
    RESULT_DIR = args.result_dir if args.result_dir else RESULT_DIR

    """
    # Dir for saving training results
    LOG_DIR = os.path.join(RESULT_DIR, 'log', ARCH, DATE)
    # Dir for saving log results: visualization, training logs
    MODEL_DIR = os.path.join(RESULT_DIR, 'model', ARCH, DATE)
	
    if not os.path.exists(RESULT_DIR):
	    os.makedirs(RESULT_DIR)
    """

    # Hyperparameters
    # Follows metaprob
    KMERS = [4]
    LMER = 30
    NUM_SHARED_READS = (5, 45)
    ONLY_SEED = True
    #ONLY_SEED = False
    MAXIMUM_SEED_SIZE = 9000
    #MAXIMUM_SEED_SIZE = 200

    #raw_dir = os.path.join(DATASET_DIR, 'raw')
    #groups_dir = os.path.join(RESULT_DIR, 'groups')
    """
    if not os.path.exists(groups_dir):
        os.makedirs(groups_dir)
    """
    
    if DATASET_NAME == 'all':
        raw_datasets = glob.glob(DATASET_DIR + '/*.fna')
    else:
        raw_datasets = [os.path.join(DATASET_DIR, DATASET_NAME + '.fna')]

    # Mapping of dataset and its corresponding number of clusters
    with open('config/dataset_metadata.json', 'r') as f:
        #n_clusters_mapping = json.load(f)['simulated_short']
        n_clusters_mapping = json.load(f)['datasets']

    raw_datasets.sort()
    for dataset in raw_datasets:
        # Get some parameters
        dataset_name = os.path.basename(dataset).split('.fna')[0]
        
        print("-------------------------------------------------------")
        print('Processing dataset: ', dataset_name)

        if not os.path.exists(dataset):
            continue


        log_file = os.path.join(RESULT_DIR, dataset_name + '.log.txt')
        log = open(log_file, "w")
        log.write('------------------------------------------------------- ')
        log.write('\nProcessing dataset ' + dataset_name)
        
        num_shared_read = NUM_SHARED_READS[1] if 'R' in dataset_name else NUM_SHARED_READS[0]
        is_deserialize = os.path.exists(os.path.join(RESULT_DIR, dataset_name + '.json'))
        n_clusters = n_clusters_mapping[dataset_name]

        print('Prior number of clusters: ', n_clusters)
        print('Prior number of shared reads: ', num_shared_read)

        log.write('\nPrior number of clusters: ' + str(n_clusters))
        log.write('\nPrior number of shared reads: ' + str(num_shared_read))

        t0 = time.time()
        try:
            seed_kmer_features, labels, groups, seeds = load_genomics(
                dataset,
                kmers=KMERS,
                lmer=LMER,
                maximum_seed_size=MAXIMUM_SEED_SIZE,
                num_shared_reads=num_shared_read,
                is_deserialize=is_deserialize,
                is_serialize=~is_deserialize,
                is_normalize=True,
                only_seed=ONLY_SEED,
                graph_file=os.path.join(RESULT_DIR, dataset_name + '.json')
            )
        except:
            seed_kmer_features, labels, groups, seeds = load_genomics(
                dataset,
                kmers=KMERS,
                lmer=LMER,
                maximum_seed_size=MAXIMUM_SEED_SIZE,
                num_shared_reads=num_shared_read,
                is_deserialize=False,
                is_serialize=True,
                is_normalize=True,
                only_seed=ONLY_SEED,
                graph_file=os.path.join(RESULT_DIR, dataset_name + '.json')
            )
        
        print('Total number of reads: ', len(labels))
        print('Total number of groups: ', len(groups))
        print('Bimeta phase 1 time: ', (time.time() - t0))

        log.write('\nTotal number of reads: ' + str(len(labels)))
        log.write('\nTotal number of groups: ' + str(len(groups)))
        log.write('\nBimeta phase 1 time: ' + str(time.time() - t0))

        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        grps_label = get_group_label(groups, labels)
        visualize(seed_kmer_features, 
            grps_label, 
            n_clusters, (-60, 60), 40, 
            is_save = True, 
            save_path = os.path.join(model_dir, f'seed-{dataset_name}.png'))

        print('Visualization results are saved in: ', model_dir) 
        """
    
        t1 = time.time()
        kmeans = KMeans(
            init="random",
            n_clusters=n_clusters,
            n_init=100,
            max_iter=200,
            random_state=20210903)
        kmeans.fit(X=seed_kmer_features, y=labels)
        y_pred_kmeans = kmeans.predict(X=seed_kmer_features)
        #print('length of y_pred_kmeans: ', len(y_pred_kmeans))

        groupPrec = group_precision_recall(labels, groups, n_clusters)[0]
        f1 = genome_acc(groups, y_pred_kmeans, labels, n_clusters)[2]
        
        print('Group precision: ', groupPrec)
        print('F1-score: ', f1)
        print('Clustering time: ', (time.time() - t1))
        print('Total time: ', (time.time() - t0))

        log.write('\nGroup precision: ' + str(groupPrec))
        log.write('\nF1-score: ' + str(f1))
        log.write('\nClustering time: ' + str(time.time() - t1))
        log.write('\nTotal time: ' + str(time.time() - t0))