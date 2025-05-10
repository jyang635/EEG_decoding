#!/bin/bash

set -e

for i in $(seq -w 6 10)
do
	python JY_final_reconstruction.py --subject_id "sub-$i" --model ATMS --average_eeg "$@"
	python JY_final_reconstruction.py --subject_id "sub-$i" --model encoder_low_level --average_eeg "$@"
#	python JY_VAE_compare.py --subject_id "sub-$i" --model EEGConformer --average_eeg "$@"
	echo "Metrics computed for the low-level reconstructions of subject-$i"
done


