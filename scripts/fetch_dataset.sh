#!/usr/bin/env bash
set -euo pipefail

script_path=$(dirname "$(readlink -f $0)")
data_dir="$script_path/../data"
mkdir -p $data_dir
cd $data_dir

zenodo_record_id=10358039
folder_name=combo-gesture-joystick-dataset
dataset_url="https://zenodo.org/records/${zenodo_record_id}/files/${folder_name}.zip?download=1"

echo "Downloading dataset from: $dataset_url to: $data_dir"
curl $dataset_url --output ${folder_name}.zip 

echo "Unzip"
unzip ${folder_name}.zip
mv ${folder_name}/* .
rmdir ${folder_name}
rm ${folder_name}.zip
