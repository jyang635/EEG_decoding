#!/bin/bash

set -e

for i in $(seq -w 1 10)
do
	python JY_DMhighlevel_compare.py --subject_id "sub-$i" --model ATMS --average_eeg "$@"
	python JY_noDMhighlevel_compare.py --subject_id "sub-$i" --model ATMS --average_eeg "$@"
	echo "Metrics computed for the High-level reconstructions of subject-$i"
done


