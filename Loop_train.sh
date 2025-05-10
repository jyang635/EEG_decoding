#!/bin/bash
set -e

python EEG_low_level_average.py --early_stopping=10 --subject_id sub-08 --channels All --loss=mse --start_time=0 --end_time=1 --project ATM_lowlevel  --model encoder_low_level --average_eeg "$@"
python EEG_low_level_average.py --early_stopping=10 --subject_id sub-08 --channels All --loss=mse --start_time=0 --end_time=1 --project ATM_lowlevel  --model encoder_low_level "$@"

python EEG_low_level_average.py --early_stopping=10 --subject_id sub-08 --channels All --loss=mse --start_time=0 --end_time=0.5 --project ATM_lowlevel --model encoder_low_level --average_eeg "$@"
python EEG_low_level_average.py --early_stopping=10 --subject_id sub-08 --channels All --loss=mse --start_time=0 --end_time=0.5 --project ATM_lowlevel --model encoder_low_level  "$@"

python EEG_low_level_average.py --early_stopping=10 --subject_id sub-08 --channels All --loss=mse --start_time=0 --end_time=0.5 --project ATM_lowlevel --model encoder_low_level_channelwise --average_eeg "$@"
python EEG_low_level_average.py --early_stopping=10 --subject_id sub-08 --channels All --loss=mse --start_time=0 --end_time=0.5 --project ATM_lowlevel --model encoder_low_level_channelwise "$@"

python EEG_low_level_average.py --early_stopping=10 --subject_id sub-08 --channels All --loss=mse --start_time=0 --end_time=0.5 --project ATM_lowlevel --model EEGConformer --average_eeg "$@"
python EEG_low_level_average.py --early_stopping=10 --subject_id sub-08 --channels All --loss=mse --start_time=0 --end_time=0.5 --project ATM_lowlevel --model EEGConformer  "$@"
