"""Associate one gene with AD status (ADNI).

For the 246 ADNI participants with WGS and the 1-year MRI collection, participants ever diagnosed with MCI or
dementia in ADNIMERGE are cases and the others controls. The genomic-LLM features of the central Enformer region
are restricted to the 77 AD-related tracks (AD_contexts.txt), reduced by local PCA per bin and averaged over the
two haplotypes; a gradient-boosting classifier is evaluated by 10-fold cross-validated auROC. Run from the repository root.

Input : --llm_path  folder with <chrom>_<subject>_{paternal,maternal}.npy from get_llm_feats.py
Output: the auROC, printed to stdout

Usage:
    python3 get_AD_association.py --gene_name APOE --llm_path llm_feats/APOE --refGene_path refGene_hg19_TSS.bed
"""
import pandas as pd
import numpy as np 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import argparse

def get_subjects_info(merge_info_path='ADNIMERGE_01Jun2023.csv', mri_file='MRI/ADNI1_Complete_1Yr_1.5T_7_18_2023.csv', wgs_file='wgs_subject_id.txt'):
    """
    Extracts subject information and labels.
    
    Parameters:
    - merge_info_path: Path to ADNIMERGE CSV file.
    - mri_file: Path to MRI data file.
    - wgs_file: Path to WGS subject IDs file.
    
    Returns:
    - subjects_selected: List of 246 selected subjects.
    - labels: Diagnosis labels (0: Control, 1: MCI, 2: Dementia).
    - labels_bl, labels_m06, labels_m12: Diagnosis labels at baseline, 6 months, and 12 months.
    """
    merge_info = pd.read_csv(merge_info_path)
    subject_ids_1y = np.unique([item.split(',')[1].strip('"') for item in open(mri_file).readlines()[1:]])
    wgs_subject_ids = [item.rstrip() for item in open(wgs_file).readlines()]
    subjects_selected = [item for item in wgs_subject_ids if item.upper() in subject_ids_1y]

    labels, labels_bl, labels_m06, labels_m12 = [], [], [], []

    for each in subjects_selected:
        each_upper = each.upper()
        if each_upper == "128_S_1407":
            labels_bl.append(merge_info.loc[(merge_info['PTID'] == each_upper) & (merge_info['VISCODE'] == 'bl')]['DX'].to_list()[0])
            labels_m06.append(merge_info.loc[(merge_info['PTID'] == each_upper) & (merge_info['VISCODE'] == 'm06')]['DX'].to_list()[0])
            labels_m12.append('MCI')
        else:
            labels_bl.append(merge_info.loc[(merge_info['PTID'] == each_upper) & (merge_info['VISCODE'] == 'bl')]['DX'].to_list()[0])
            labels_m06.append(merge_info.loc[(merge_info['PTID'] == each_upper) & (merge_info['VISCODE'] == 'm06')]['DX'].to_list()[0])
            labels_m12.append(merge_info.loc[(merge_info['PTID'] == each_upper) & (merge_info['VISCODE'] == 'm12')]['DX'].to_list()[0])

        subject_info = merge_info.loc[merge_info['PTID'] == each_upper][['VISCODE', 'DX']]
        if 'Dementia' in subject_info['DX'].to_list():
            labels.append(2)
        elif 'MCI' in subject_info['DX'].to_list() and 'Dementia' not in subject_info['DX'].to_list():
            labels.append(1)
        elif 'MCI' not in subject_info['DX'].to_list() and 'Dementia' not in subject_info['DX'].to_list():
            labels.append(0)
        else:
            print('Error processing subject:', each_upper)
            
    return subjects_selected, labels, labels_bl, labels_m06, labels_m12


def get_enformer_feats(subjects_selected, llm_path, chrom):
    """
    Loads Enformer features for given subjects.
    
    Parameters:
    - subjects_selected: List of selected subjects.
    - llm_path: Path to Enformer feature files.
    - chrom: Chromosome identifier.
    
    Returns:
    - enformer_feats_p: Features for paternal alleles.
    - enformer_feats_m: Features for maternal alleles.
    """
    enformer_feats_p = np.stack([np.load(f'{llm_path}/{chrom}_{item}_paternal.npy') for item in subjects_selected])
    enformer_feats_m = np.stack([np.load(f'{llm_path}/{chrom}_{item}_maternal.npy') for item in subjects_selected])
    print('Enformer feature shapes:', enformer_feats_p.shape, enformer_feats_m.shape)
    return enformer_feats_p, enformer_feats_m

def reduce_enformer_feats(enformer_feats_p,enformer_feats_m, n_components=4, AD_context_path='AD_contexts.txt'):
    """
    Reduces genomic LLM features using PCA and AD context indices.
    
    Parameters:
    - enformer_feats_p: Paternal features.
    - enformer_feats_m: Maternal features.
    - n_components: Number of PCA components.
    - AD_context_path: Path to file containing AD context indices.
    
    Returns:
    - pca_feats: Reduced features averaged from paternal and maternal.
    """
    AD_context_idx = np.array([int(line.split('\t')[0]) for line in open(AD_context_path).readlines()])
    enformer_feats_p = enformer_feats_p[:,1,:,:][:,:,AD_context_idx]
    enformer_feats_m = enformer_feats_m[:,1,:,:][:,:,AD_context_idx]
    from sklearn.decomposition import PCA
    pca_feats_p, pca_feats_m = [],[]
    for j in range(enformer_feats_p.shape[1]):
        pca_feats_p.append(PCA(n_components=n_components).fit_transform(enformer_feats_p[:,j,:]))
        pca_feats_m.append(PCA(n_components=n_components).fit_transform(enformer_feats_m[:,j,:]))
    #pca_feats = np.concatenate(pca_feats_p+pca_feats_m,axis=1)
    pca_feats_p = np.concatenate(pca_feats_p,axis=1)
    pca_feats_m = np.concatenate(pca_feats_m,axis=1)
    pca_feats = (pca_feats_p+pca_feats_m)/2.0
    return pca_feats

def evaluate_predition(X, y, gene_name = 'APOE', chrom = 'chr19'):
    """
    Evaluates the prediction performance using a classifier and ROC AUC.
    
    Parameters:
    - X: Feature matrix.
    - y: Labels (binary).
    
    Returns:
    - roc_auc: ROC AUC score.
    """
    from sklearn.model_selection import cross_val_predict
    clf = GradientBoostingClassifier(n_estimators=40, learning_rate=0.1)
    y_pred = cross_val_predict(clf, X, y, cv=10, method='predict_proba')[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Associate gene to AD risk")
    parser.add_argument('--gene_name', type=str, required=True, help='Name of the gene.')
    parser.add_argument('--llm_path', type=str, required=True, help='Path to Enformer feature files.')
    parser.add_argument('--refGene_path', type=str, required=True, help='Path to the refGene hg19 TSS bed file.')

    args = parser.parse_args()
    
    gene_name = args.gene_name
    sujects_selected, labels, labels_bl,labels_m06, labels_m12 = get_subjects_info()
    #control: 0, AD or MCI: 1
    y = []
    for each in labels:
        if each==0:
            y.append(0)
        else:
            y.append(1)
    y = np.array(y)
    
    gene2loc = {item.split('\t')[4]: (item.split('\t')[0], int(item.split('\t')[1])) for item in open(args.refGene_path).readlines()}
    chrom, center = gene2loc[gene_name]

    enformer_feats_p, enformer_feats_m = get_enformer_feats(sujects_selected, llm_path = args.llm_path, chrom = chrom)
    pca_feats = reduce_enformer_feats(enformer_feats_p,enformer_feats_m)
    roc_auc = evaluate_predition(pca_feats, y, gene_name = gene_name, chrom = chrom)
    print(f'The auROC for gene {args.gene_name} is {roc_auc:.3f}')
