#!/bin/bash

declare -a StringArray=(
'adult_gender_TF2'
'adult_gender_Torch'
'adult_marital-status_TF2'
'adult_marital-status_Torch'
'german_gender_TF2'
'german_gender_Torch'
'german_foreignworker_TF2'
'german_foreignworker_Torch'
)

for method in "${StringArray[@]}"; do
  echo ${method}
  python3 -u main.py ${method}
done