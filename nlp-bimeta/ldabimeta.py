import glob, os, time, sys
import json
import numpy as np
from collections import defaultdict
import argparse

from dataset.genome import GenomeDataset
from dataset.utils import load_meta_reads, create_document, create_corpus
import utils.utils as utils
from debug.visualize import get_group_label, visualize
from utils.metrics import genome_acc, group_precision_recall

from sklearn.cluster import KMeans

if __name__ == "__main__":
    
    sys.path.append('.')

    DATASET_DIR = 'data/raw'    # Raw fasta data dir
    BIMETAOUT_DIR = 'data/raw'  # bimeta output dir
    DATASET_NAME = 'hmp_test'   # Specifc fasta dataset or all of them
    #DATASET_NAME = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S8', 'S10_S', 'MixKraken_abund1_S', 'MixKraken_abund2_S', ]
    RESULT_DIR = 'results'      # Result dir

    # Versioning each runs
    ARCH = 'ldabimeta'
    DATE = '20211015'

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, \
        help='directory contain fasta file')
    parser.add_argument('--dataset_name', type=str, 
        help='specific name of dataset to run' \
            '(all is run every found dataset), e.g S1.fna, name is S1')
    parser.add_argument('--bimetaout_dir', type=str, \
        help='directory contain group files of bimeta')
    parser.add_argument('--result_dir', type=str, \
        default='results', help='directory for saving the resuls')
    parser.add_argument('--verbose', action='store_true', \
        help='Whether to print log in terminal')

    args = parser.parse_args()

    verbose = args.verbose

    DATASET_NAME = args.dataset_name if args.dataset_name else DATASET_NAME
    DATASET_DIR = args.data_dir if args.data_dir else DATASET_DIR
    BIMETAOUT_DIR = args.bimetaout_dir if args.bimetaout_dir else BIMETAOUT_DIR
    RESULT_DIR = args.result_dir if args.result_dir else RESULT_DIR

    # Hyperparameters
    KMERS = [4]
    IS_TFIDF = False
    MALLET_BINARY = '../mallet/bin/mallet'
    #N_TOPICS = [2, 3] # for test
    N_TOPICS = [5, 8, 10, 12, 15, 20]

    if DATASET_NAME == 'all':
        raw_datasets = glob.glob(DATASET_DIR + '/*.fna')
    else:
        #raw_datasets = [os.path.join(DATASET_DIR, DATASET_NAME + '.fna')]
        if type(DATASET_NAME) == list:
            raw_datasets = [os.path.join(DATASET_DIR, ds_name + '.fna') for ds_name in DATASET_NAME]
        else:    
            raw_datasets = [os.path.join(DATASET_DIR, DATASET_NAME + '.fna')]

    # Mapping of dataset and its corresponding number of clusters
    with open('config/dataset_metadata.json', 'r') as f:
        n_clusters_mapping = json.load(f)['datasets']

    raw_datasets.sort()
    for dataset in raw_datasets:
        # Get some parameters
        dataset_name = os.path.basename(dataset).split('.fna')[0]
        
        print("-------------------------------------------------------")
        print('Processing dataset: ', dataset_name)

        bimetaout_file = os.path.join(BIMETAOUT_DIR, dataset_name + '.json')
        if not os.path.exists(bimetaout_file):
            continue

        log_file = os.path.join(RESULT_DIR, dataset_name + '.log.txt')
        log = open(log_file, "w")
        log.write('------------------------------------------------------- ')
        log.write('\nProcessing dataset ' + dataset_name)
        
        n_clusters = n_clusters_mapping[dataset_name]

        print('Prior number of clusters: ', n_clusters)
        log.write('\nPrior number of clusters: ' + str(n_clusters))

        try:
            t0 = time.time()
            # Load group file (phase 1 of bimeta) according to dataset_name
            print('Loading groups: ...')
            groups, seeds = utils.load_groups_seeds(BIMETAOUT_DIR, dataset_name)
            print('Total number of groups: ', len(groups))
            log.write('\nTotal number of groups: ' + str(len(groups)))
            print('Time to load groups: ', (time.time() - t0))
            log.write('\nTime to load groups: ' + str((time.time() - t0)))

            # Read fasta dataset
            t1 = time.time()
            print('Loading reads ...')
            reads, labels = load_meta_reads(dataset, type='fasta')
            print('Total number of reads: ', len(labels))
            log.write('\nTotal number of reads: ' + str(len(labels)))
            print('Time to load reads: ', (time.time() - t1))
            log.write('\nTime to load reads: ' + str((time.time() - t1)))

            t2 = time.time()
            # Creating document from reads...
            print('Creating documents and corpus ...')
            dictionary, documents = create_document(reads, KMERS)
            
            # Creating corpus...
            corpus = create_corpus(dictionary, documents, is_tfidf=IS_TFIDF)
            print('Time to create corpus from reads: ', (time.time() - t2))
            log.write('\nTime to create corpus from reads: ' + str((time.time() - t2)))

            t3 = time.time()
            print('LDA training ...')
            n_topics_choices = len(N_TOPICS)
            max_coherence = -1
            max_index = -1
            print('Number of topic choices: {}'.format(n_topics_choices))
            for i in range(n_topics_choices):
                print('LDA training for {} topics ...'.format(N_TOPICS[i]))
                lda_model = utils.do_LDA_Mallet(MALLET_BINARY, corpus, dictionary, \
                    ldaout_dir = RESULT_DIR, dataset_name = dataset_name, \
                n_topics=N_TOPICS[i], n_workers=80, n_passes=10, max_iters=200)
                print('LDA model training time for {} topics: {}'.format(N_TOPICS[i], (time.time() - t3)))
                log.write('\nLDA model training time for {} topics: {}'.format(N_TOPICS[i], (time.time() - t3)))

                # get coherence value
                print('Compute coherence value ...')
                coherence = utils.getCoherenceScore(lda_model, corpus)  
                print('Coherence value for {} topics: {}'.format(N_TOPICS[i], coherence))
                log.write('\nCoherence value for {} topics: {}'.format(N_TOPICS[i], coherence))
                
                if (max_coherence < abs(coherence)):
                    max_index = i
                    max_coherence = abs(coherence)
                
                t4 = time.time()
                print('Getting document-topics ...')
                
                # new code
                lda_topics_file = os.path.join(RESULT_DIR, dataset_name + '.lda.doctopics.txt')
                top_dist_df = utils.getLdaDocTopics(lda_topics_file, N_TOPICS[i])
                
                print('Getting document-topics time: {}'.format((time.time() - t4)))
                log.write('\nGetting document-topics time: {}'.format((time.time() - t4)))


                t5 = time.time()
                print('Compute LDA feature ...')
                ngroups = len(groups)
                temp_group_features = [top_dist_df.iloc[groups[i], :].mean() for i in range(ngroups)]
                temp_seed_features = [top_dist_df.iloc[seeds[i], :].mean() for i in range(ngroups)]

                lda_group_features = np.array(temp_group_features)
                lda_seed_features = np.array(temp_seed_features)

                print('Compute LDA feature time: ', (time.time() - t5))
                log.write('\nCompute LDA feature time: ' + str(time.time() - t5))

                # Clustering groups
                t6 = time.time()
                print('Clustering ...')
                gkmeans = KMeans(
                    init="random",
                    n_clusters=n_clusters,
                    n_init=100,
                    max_iter=200,
                    random_state=20211015)
                gkmeans.fit(X=lda_group_features, y=labels)
                y_gpred_kmeans = gkmeans.predict(X=lda_group_features)
                skmeans = KMeans(
                    init="random",
                    n_clusters=n_clusters,
                    n_init=100,
                    max_iter=200,
                    random_state=20211015)
                skmeans.fit(X=lda_seed_features, y=labels)
                y_spred_kmeans = skmeans.predict(X=lda_seed_features)
                print('Clustering time: ', (time.time() - t6))
                log.write('\nClustering time: ' + str(time.time() - t6))

                # Map read to group and compute F-measure
                t7 = time.time()
                print('Compute F-measure ...')
                log.write('\nCompute measures: ')
                groupPrec = group_precision_recall(labels, groups, n_clusters)[0]
                precg, recallg, f1g = genome_acc(groups, y_gpred_kmeans, labels, n_clusters)
                precs, recalls, f1s = genome_acc(groups, y_spred_kmeans, labels, n_clusters)
                print('Group precision: ', groupPrec)
                print('Precision (using group): ', precg)
                print('Recall (using group): ', recallg)
                print('F1-score (using group): ', f1g)
                print('Precision (using seed): ', precs)
                print('Recall (using seed): ', recalls)
                print('F1-score (using seed): ', f1s)
                print('Total time: ', (time.time() - t0))
                log.write('\nGroup precision: ' + str(groupPrec))
                log.write('\nPrecision (using group): ' + str(precg))
                log.write('\nRecall (using group): ' + str(recallg))
                log.write('\nF1-score (using group): ' + str(f1g))
                log.write('\nPrecision (using seed): ' + str(precs))
                log.write('\nRecall (using seed): ' + str(recalls))
                log.write('\nF1-score (using seed): ' + str(f1s))
                log.write('\nTotal time: ' + str(time.time() - t0))
                log.write('\n')

            # del lda_model
            # del top_dist_df
            # del temp_group_features
            # del temp_seed_features
            # del lda_group_features
            # del lda_seed_features
            # Compute group features (using seeds instead of groups)
            print('The best number of topics for {}: {}'.format(dataset_name, \
                N_TOPICS[max_index]))
            log.write('\nThe best number of topics for {}: {}'.format(dataset_name, \
                N_TOPICS[max_index]))
            print('Max coherence score for {}: {}'.format(dataset_name, \
                max_coherence))
            log.write('\nMax coherence score for {}: {}'.format(dataset_name, \
                max_coherence))
            
            
            
        except Exception as e:
            print('Error!!! Cause: ', e)
