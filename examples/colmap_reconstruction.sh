#!/bin/bash

if [ -z "$1" ]
	then
		echo "Usage: ./colmap_reconstruction.sh /path/to/colmap/project"
		echo "	/path/to/colmap/project should contain a folder \"images\" containing all images to consider for the reconstruction."
		echo "	Current setup of this script assumes that you are running it from the examples folder."
		echo "	This is just an example script, you should tune colmap parameters for the best reconstruction results!"
		exit 1
fi

ADALAM_PATH=$PWD"/match_colmap_database_example.py"
cd $1
colmap feature_extractor --database_path database_adalam.db --image_path images  # --ImageReader.single_camera 1
python $ADALAM_PATH --database_path="database_adalam.db" --image_pairs_path="image_pairs.txt"
colmap matches_importer --match_type 'pairs' --database_path 'database_adalam.db' --match_list_path "image_pairs.txt"
mkdir sparse_adalam
colmap mapper --database_path database_adalam.db --image_path images --output_path sparse_adalam
