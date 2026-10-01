"""Extract genomic-LLM (Enformer) features for one gene from one personal haploid genome.

Input : a personal fasta produced by vcf2diploid, e.g. fasta/chr19/chr19_003_S_1057_maternal.fa
        (one sequence named "<chr>_maternal" or "<chr>_paternal").
Output: <output_path>/<fasta basename>.npy, shape (3, 896, 5313)
        = 3 Enformer input regions centred at TSS - 114,688 bp, TSS and TSS + 114,688 bp
        x 896 bins of 128 bp x 5,313 human tracks.

Usage:
    python3 get_llm_feats.py --gene_name APOE --fasta_path fasta/chr19/chr19_003_S_1057_maternal.fa \
        --refGene_path refGene_hg19_TSS.bed --output_path llm_feats/APOE
"""
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import os
import math
from pyfasta import Fasta
import argparse

# Load the Enformer model (TF-Hub, downloaded on first use)
enformer_model = hub.load("https://tfhub.dev/deepmind/enformer/1").model

def seq_to_mat(seq):
    """Convert DNA sequence to a one-hot encoded matrix."""
    d = {'a': 0, 'A': 0, 'c': 1, 'C': 1, 'g': 2, 'G': 2, 't': 3, 'T': 3, 'N': 4, 'n': 4}
    mat = np.zeros((5, len(seq)))
    for i in range(len(seq)):
        mat[d[seq[i]], i] = 1
    mat = mat[:4, :]
    return mat

def main(args):
    """Run Enformer on the regions around the TSS of args.gene_name and save the stacked outputs."""
    # refGene_hg19_TSS.bed columns: chrom, TSS, TSS, transcript ID, gene name, strand
    gene2loc = {item.split('\t')[4]: (item.split('\t')[0], int(item.split('\t')[1])) for item in open(args.refGene_path).readlines()}

    assert args.gene_name in gene2loc, "Gene not found in refGene database"
    chr_id = gene2loc[args.gene_name][0]
    center = gene2loc[args.gene_name][1]
    start = center - 100000
    end = center + 100000

    SEQ_LENGTH = 393216              # Enformer input length (bp)
    interval = 896 * 128             # length covered by the 896 output bins (bp)
    nb_regions = math.ceil((end - start - interval) / (2 * interval))   # = 1 -> 3 regions in total

    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, f"{args.fasta_path.split('/')[-1].split('.')[0]}.npy")

    key = "maternal" if "maternal" in args.fasta_path else "paternal"
    genome = Fasta(args.fasta_path)

    enformer_feats = []
    for coor in range(center - interval * nb_regions, center + interval * (nb_regions + 1), interval):
        seq = genome[f'{chr_id}_{key}'][(coor - SEQ_LENGTH // 2):(coor + SEQ_LENGTH // 2)]
        onehot_mat = seq_to_mat(seq).T
        onehot_mat = np.expand_dims(onehot_mat, 0)
        enformer_feats.append(enformer_model.predict_on_batch(onehot_mat)['human'])

    enformer_feats = np.squeeze(np.stack(enformer_feats))
    np.save(output_file, enformer_feats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Enformer features for a given gene.")
    parser.add_argument('--gene_name', type=str, required=True, default='APOE', help='Name of the gene.')
    parser.add_argument('--fasta_path', type=str, required=True, default='chr19_003_S_1057_maternal.fa', help='Path to the fasta file.')
    parser.add_argument('--refGene_path', type=str, required=True, help='Path to the refGene hg19 TSS bed file.')
    parser.add_argument('--output_path', type=str, required=True, help='Directory to save the output files.')

    args = parser.parse_args()
    main(args)
