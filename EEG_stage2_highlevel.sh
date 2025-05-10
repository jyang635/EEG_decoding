#!/bin/bash
set -e

for i in $(seq -w 1 10)
do
	python EEG_high_level_diffusion.py --early_stopping=10 --subject_id "sub-$i" --channels All --loss=mse --start_time=0 --end_time=1  "$@"
	echo "Low-level model done for subject-$i" 
done


