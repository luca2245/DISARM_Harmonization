#!/bin/bash

# CHECK
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <raw_img_folder> <new_folder_path>"
    exit 1
fi

# GET PATH AND FOLDER NAME
raw_img_folder="$1"
new_folder_path="$2"

prep_folder="$new_folder_path"
mkdir -p "$prep_folder"

# OPTIONS FOR FAST COMMAND
common_options="-t 1 -n 3 -H 0.1 -I 4 -l 20.0 --nopve -B"

# LOOP THROUGH INPUT IMAGES
for input_image in "$raw_img_folder"/*.nii.gz; do

    # DEFINE OUTPUT PREFIXES
    output_prefix="$raw_img_folder/output_$(basename ${input_image%.*})"
    restored_output="$raw_img_folder/output_$(basename ${input_image%.*.*})_restore"

    # ORIENT THE IMAGE TO THE STANDARD ORIENTATION
    fslreorient2std "$input_image" "$input_image"

    # APPLY BIAS-FIELD CORRECTION WITH FAST
    $FSLDIR/bin/fast $common_options -o "$output_prefix" "$input_image"

    # REGISTER IMAGE TO COMMON SPACE MNI152_T1_1mm.nii.gz WITH FLIRT
    $FSLDIR/bin/flirt -in "${restored_output}.nii.gz" -ref "$FSLDIR/data/standard/MNI152_T1_1mm.nii.gz" -out "$raw_img_folder/registered_$(basename ${input_image%.*.*}).nii.gz" -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12 -interp trilinear

    mv "$raw_img_folder/registered_$(basename ${input_image%.*.*}).nii.gz" "$prep_folder/"

done