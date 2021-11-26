#!/bin/bash

#PBS -l ncpus=24
#PBS -l ngpus=2
#PBS -M raphael.paras@students.mq.edu.au
#PBS -j oe
#PBS -o /scratch/oe7/rp3665/scripts/batch_1/ClinicalBERT/cosine/ClinicalBERT_sigmoid/
#PBS -m e
#PBS -l mem=16GB
#PBS -l jobfs=100GB
#PBS -q gpuvolta
#PBS -P oe7
#PBS -l walltime=10:00:00
#PBS -l storage=scratch/oe7
#PBS -l wd

module load python3/3.9.2
module load tensorflow/2.6.0
export PYTHONPATH=/home/576/rp3665/.local/lib/python3.9/site-packages:/home/576/dma576/lib/python3.7/site-packages:$PYTHONPATH
python3 ClinicalBERT_sigmoid/bert_base_sentence_classification_with_bioasq_preprocessing_finaldraft.py $PBS_NCPUS > /scratch/oe7/rp3665/scripts/batch_1/ClinicalBERT/cosine/ClinicalBERT_sigmoid/$PBS_JOBID.log
