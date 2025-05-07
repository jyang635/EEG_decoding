#!/bin/bash
set -e

for i in $(seq -w 1 10)
do
	python JY_low_level_average.py --early_stopping=10 --subject_id "sub-$i" --channels All --loss=mse --start_time=0 --end_time=1 --project THINGSEEG_Lowlevel  --model encoder_low_level --average_eeg "$@"
	python JY_low_level_average.py --early_stopping=10 --subject_id "sub-$i" --channels All --loss=mse --start_time=0 --end_time=0.5 --project THINGSEEG_Lowlevel --model EEGConformer --average_eeg "$@"
	python JY_low_level_average.py --early_stopping=10 --subject_id "sub-$i" --channels All --loss=mse --start_time=0 --end_time=1 --project THINGSEEG_Lowlevel --model ATMS --average_eeg "$@"
	echo "Low-level model done for subject-$i"
done


