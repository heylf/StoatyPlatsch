#!/bin/bash
#$ -N PENIS
#$ -cwd
#$ -pe smp 20
#$ -l h_vmem=1G
#$ -o /scratch/bi03/heylf/Snakemake_Cluster_Logs/
#$ -e /scratch/bi03/heylf/Snakemake_Cluster_Logs/

source activate stoatyplatsch
python3 StoatyPlatsch.py \
-a "/scratch/bi03/heylf/StoatyPlatsch/data_RBFOX2/peaks_for_deconvolution.bed" \
-b "/scratch/bi03/heylf/StoatyPlatsch/data_RBFOX2/RBFOX2_sorted_truncation_sites.bed" \
-c "/scratch/bi03/heylf/StoatyPlatsch/data_RBFOX2/hg38.chrom.sizes.txt" \
-t 20 \
--gene_file "/scratch/bi03/heylf/StoatyPlatsch/data_RBFOX2/Homo_sapiens.GRCh38.98.genes.bed" \
--exon_file "/scratch/bi03/heylf/StoatyPlatsch/data_RBFOX2/Homo_sapiens.GRCh38.98.collapsed.exons.bed" \
-o "/scratch/bi03/heylf/StoatyPlatsch/data_RBFOX2"
source deactivate stoatyplatsch
