#!/bin/bash

for i in $(seq -w 1 10)
do
	python ATMS_reconstruction.py --insubject True --subjects "sub-$i" --logger True  --output_dir ./outputs/contrast --data_path /home/yjk122/IP_temp/EEG_Image_decode/Preprocessed_data_250Hz
	echo "Train of sub-$i is finished"
done
echo 'Trainings of all samples are finished'
