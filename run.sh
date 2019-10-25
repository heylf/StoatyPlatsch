#!/bin/bash
#$ -N PENIS
#$ -cwd
#$ -pe smp 20
#$ -l h_vmem=1G
#$ -o /scratch/bi03/heylf/Snakemake_Cluster_Logs/
#$ -e /scratch/bi03/heylf/Snakemake_Cluster_Logs/

source activate stoatyplatsch
python3 StoatyPlatsch.py \
-a peaks_for_deconvolution.bed \
-b RBFOX2_rep1_sorted_truncation_sites.bed \
-c hg38.chrom.sizes.txt \
-t 20 \
-o test_data
source deactivate stoatyplatsch
