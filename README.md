# epiBrain
Associating genotype to imaging and clinical phenotypes of Alzheimer’s disease by leveraging genomic large language model

 ![model](https://github.com/liuq-lab/epiBrain/blob/main/workflow.png)

In this work, we propose a novel computational framework that leverages genomic large language models (LLMs) to enhance the association analysis between genetic variants and Alzheimer's disease (AD)-related phenotypes, including imaging and clinical features.

 # Enviornment
- Python==3.9.0
- java==1.8.0
- TensorFlow==2.12.0
- TensorFlow-hub==0.12.0
- pyfasta==0.5.2
- scikit-learn==1.0.2

Apart from the above softwares/packages, please also make sure the following softwares are installed properly: 1) [vcftools](https://vcftools.github.io/examples.html) for processing WGS .vcf file. 2) [Beagle](https://faculty.washington.edu/browning/beagle/beagle.html) for genotype phasing. 3) [vcf2diploid](http://alleleseq.gersteinlab.org/tools.html) for constructing personal genome. 4) [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) for processing sMRI images. 

# Instructions
We provide detailed step-by-step instructions for running our pipeline.

## Processing genotype data

In our study, we downloaded whole genome sequencing (WGS) data from [ADNI database](https://adni.loni.usc.edu/), which provides .vcf file (gzip compressed) for each chromosome. Here, we take the chr19 and use gene APOE as a demonstration case study. 

**Step 1: remove indels**

```shell
vcftools --gzvcf ADNI.808_indiv.minGQ_21.pass.ADNI_ID.chr19.vcf.gz  --remove-indels --recode --recode-INFO-all --out SNPs_ADNI.808_indiv.minGQ_21.pass.ADNI_ID.chr19
[gzvcf] - input compressed vcf file
[out] - output file name (prefix)
```
**Step 2: genotype to haplotype**

```shell
java -jar preprocess/beagle.22Jul22.46e.jar gt=SNPs_ADNI.808_indiv.minGQ_21.pass.ADNI_ID.chr19.recode.vcf out=SNPs_ADNI.808_indiv.minGQ_21.pass.ADNI_ID.chr19.recode_hap map=plink.GRCh37.map/plink.chr19.GRCh37.map
[jar] - path the the beagle java program
[gt] - input vcf file from Step 1
[out] - output file name (prefix)
```
Note that the beagle .jar file is from [here](https://faculty.washington.edu/browning/beagle/beagle.html) and the plink full map files are from [here](https://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/).

**Step 3: personal genome construction**

```shell
java -jar vcf2diploid_v0.2.6a/vcf2diploid.jar -outDir fasta/chr19  -id  003_S_1057 -chr hg19/chr19.fa -vcf SNPs_ADNI.808_indiv.minGQ_21.pass.ADNI_ID.chr19.recode_hap.vcf.gz
[outDir] - output directory
[id] - personal ID
[chr] - reference genome for a chromosome
[vcf] - input vcf file from Step 2
```

Note that the above command can only construct the personal genome (both maternal and paternal) *per individual per chromosome*. For the construction of multiple individuals, the above command should be iterated over all individuals. 

The `vcf2diploid.jar` was downloaded from [here](http://alleleseq.gersteinlab.org/tools.html). The reference genome for a chromosome can be downloaded from [here](https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/).

Above the above three steps, one should get `chr[CID]_[PID]_maternal.fa` and `chr[CID]_[PID]_paternal.fa` in the `fasta/chr[CID]` folder where `CID`,`PID` denotes chromosome ID and personal ID, respectively. 

In the above example case, it is `chr19_003_S_1057_maternal.fa` and `chr19_003_S_1057_paternal.fa`.

## Extracting genomic LLM features

```shell
python3 get_llm_feats.py --gene_name [gene_name] --fasta_path [fasta_path] --refGene_path [refGene_path] --output_path [output_path]
[gene_name] - gene of interest, e.g., APOE
[fasta_path] - path to the fasta in the last step, e.g., fasta/chr19/chr19_003_S_1057_maternal.fa
[refGene_path] -path to the refGene file, e.g., refGene_hg19_TSS.bed
[output_path] - output path
```
The refGene file records the TSS information for each gene. The `output_path` will be created if not exist. A python .npy file with the same prefix (e.g., `chr19_003_S_1057_maternal.npy`) will be generated under the `output_path` folder with the shape `(3,896,5313)`. It represents the 5313 features in 896 bins for 3 genomic LLM input regions. 

Note that the Python script is designed for per fasta file per gene. For extracting genomic LLM features for a large number of individuals, GPU is recommended to accelerate the process.

## Processing MRI image data

The sMRI images of 246 individuals were downloaded from ADNI database (entry name: ADNI1_Complete_1Yr_1.5T) in .nii format. We put the .nii raw data from each individual each time point in a separate folder. Note that some individuals may have multiple .nii files from the same time point. The image data are organized as the structure below

```
 MRI/
    |-- 003_S_1057_bs/
    |   |   |   |   |--I52821.nii
    |-- 003_S_1057_m06/
    |   |   |   |   |--I81339.nii
    |-- 003_S_1057_m12/
    |   |   |   |   |--I96202.nii
    |-- 007_S_0128_sc_bs/
    |   |   |   |   |--I118683.nii
    |   |   |   |   |--I36640.nii
    |-- 007_S_0128_sc_m06/
    |   |   |   |   |--I121135.nii
    |-- 007_S_0128_sc_m12/
    |   |   |   |   |--I59863.nii
    ...
```

**Step 1: Processing sMRI images**

```shell
export SUBJECTS_DIR=[path-to-project]/img_output
recon-all -s 003_S_1057_bs -i MRI/003_S_1057_bs/I52821.nii -all
```

One can set an environment variable `SUBJECTS_DIR` to specify the output path. Note that the above per individual per time point command needs to be iterated over all individuals and all three time points (baseline, m06, and m12)

**Step 2: Extracting imaging phenotypes**

```shell
aparcstats2table --subjects 003_S_1057_bs ... 007_S_0128_sc_bs  --hemi lh --meas thickness --parc=aparc --tablefile=parcstats_thickness_lh.txt --skip
[subjects] - subject IDs separated by space
[meas] - imaging phenotype (thickness, area, volume etc)
```
We provided the extracted imaging phenotypes (thickness, area, and volume) of 246 individuals across 3 time points in the `MRI` folder.


## Associating genotype to imaging phenotypes

```shell
python3 get_img_association.py --gene_name [gene_name] --llm_path [llm_path] --refGene_path [refGene_path]
[img_feat_type] - MRI image feature type. e.g., 'thickness'
[gene_name] - gene of interest, e.g., APOE
[llm_path] - path to the folder containing genomic LLM feature .npy files
[refGene_path] -path to the refGene file, e.g., refGene_hg19_TSS.bed
[res_path] - path to save the association results
```
`res_path` will contain the association metric (e.g., Pearson's correlation) between a gene and a specific brain region of interest (ROI).

## Associating genotype to clinical AD phenotypes

```shell
python3 get_AD_association.py --img_feat_type [img_feat_type] --gene_name [gene_name] --llm_path [llm_path] --refGene_path [refGene_path] --res_path [res_path]
[gene_name] - gene of interest, e.g., APOE
[llm_path] - path to the folder containing genomic LLM feature .npy files
[refGene_path] -path to the refGene file, e.g., refGene_hg19_TSS.bed
```
The auROC will be calculated for binary AD trait.

## UK Biobank validation

The `ukb/` directory contains the pipeline and figure code for the UK Biobank validation of epiBrainLLM
(23 AD genes, 13,574 participants; AD, non-AD dementia and MCI cohorts; comparison with the raw-SNP baseline
and with PRS-CS, LDpred2 and SDPR). See [`ukb/README.md`](ukb/README.md). The figures can be regenerated from the
aggregate data shipped in `ukb/07_figures/` without access to individual-level data.

# Contact
If you have any questions regarding our code or data, please do not hesitate to open an issue or directly contact me (liuqiao@stanford.edu).

# Cite
If you used our work in your research, please consider citing our paper

Qiao Liu, Wanwen Zeng, Hongtu Zhu, Lexin Li, Wing Hung Wong. [Leveraging Genomic Large Language Models to Enhance Causal Genotype-Brain-Clinical Pathways in Alzheimer’s Disease](https://www.medrxiv.org/content/10.1101/2024.10.03.24314824v2) [J]. medRxiv. 2024.

# License
This project is licensed under the MIT License - see the LICENSE.md file for details.

