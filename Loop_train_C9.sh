#!/bin/bash
set -e

python JY_low_level_average.py --early_stopping=10 --subject_id sub-08 --channels 'O1,Oz,O2,PO3,POz,PO4,Pz,PO7,PO8' --loss=mse --start_time=0 --end_time=0.5 --project ATM_lowlevel --model EEGConformer --average_eeg "$@"
python JY_low_level_average.py --early_stopping=10 --subject_id sub-08 --channels 'O1,Oz,O2,PO3,POz,PO4,Pz,PO7,PO8' --loss=mse --start_time=0 --end_time=0.5 --project ATM_lowlevel  --model EEGConformer  "$@"
