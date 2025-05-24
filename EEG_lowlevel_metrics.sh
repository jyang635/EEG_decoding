#!/bin/bash

set -e

for i in $(seq -w 1 10)
do
	python EEG_VAE_compare.py --subject_id "sub-$i" --model ATMS --average_eeg "$@"
	python EEG_VAE_compare.py --subject_id "sub-$i" --model encoder_low_level --average_eeg "$@"
	python EEG_VAE_compare.py --subject_id "sub-$i" --model EEGConformer --average_eeg --channels O1,Oz,O2,PO3,POz,PO4,Pz,PO7,PO8 --start_time 0 --end_time 0.6  "$@"
	echo "Metrics computed for the low-level reconstructions of subject-$i"
done


