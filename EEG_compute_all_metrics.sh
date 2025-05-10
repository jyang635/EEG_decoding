#!/bin/bash

bash EEG_lowlevel_metrics.sh --gpu 0
bash EGG_highlevel_metrics.sh --gpu 0
bash EEG_final_metrics.sh --gpu 0

