from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import time, os
import numpy as np
from collections import defaultdict

def get_group_label(groups, labels):
    # Assign groups labels from reads labels
    # Dominant number of each label is the label of group
    # [0 0 0 0 1 1 0] -> group has label 0 (5 > 2)
    grps_label = []
    for group in groups:
        lb_type = defaultdict(lambda: 0)
        for node in group:
            lb_type[labels[node]] += 1
        max_val = 0
        key = -1
        for kv in lb_type.items():
            if kv[1] > max_val:
                max_val = kv[1]
                key = kv[0]
        if key != -1:
            grps_label.append(key)
            
    return grps_label


def visualize(x, labels, n_clusters, range_lim=(-20, 20), perplexity=40, is_save=False, save_path=None):
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=400, init='pca')
    tsne_results = tsne.fit_transform(x)
    df_subset = pd.DataFrame()
    
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    df_subset['Y'] =  labels
    
    n_comps = len(np.unique(labels).tolist())
    
    plt.figure(figsize=(16,10))
    sns_plot = sns.scatterplot(
        x='tsne-2d-one', y='tsne-2d-two',
        hue='Y',
        palette=sns.color_palette(n_colors=n_comps),
        data=df_subset,
        legend="full",
        alpha=0.3
    ).set(xlim=range_lim,ylim=range_lim)
    
    if is_save:
        save_path = save_path if save_path else ''
        plt.savefig(save_path)
        plt.close('all')


def store_results(groups, seed_feats, latent, labels, y_pred, n_clusters, dataset_name, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    grps_label = get_group_label(groups, labels)
    visualize(seed_feats, grps_label, n_clusters, (-60, 60), 40, is_save=True, save_path=
             os.path.join(save_dir, f'seed-{dataset_name}.png'))
    
    visualize(latent, grps_label, n_clusters, (-60, 60), 40, is_save=True, save_path=
             os.path.join(save_dir, f'gt-seed-latent-{dataset_name}.png'))
    
    visualize(latent, y_pred, n_clusters, (-60, 60), 40, is_save=True, save_path=
             os.path.join(save_dir, f'pred-seed-latent-{dataset_name}.png'))

    print('Visualization results are saved in: ', save_dir)