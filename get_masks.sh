cd /data/midi-lab-general/osemis_annotations/osemis_annotation_file_to_masks
python orthanc_annotations.py
matlab -nodesktop -nosplash -r "run('osemis_annotations_to_masks_main.m');exit;"
chmod -R 777 Masks
