targets="pancreatic_cancer celiac_disease lupus heart_attack stroke NAFL"

for target in $targets; do
    echo $target
    python prepare_batches.py "survival_h/${target}_batches" --data_path $P  --dictionary_path ../../gpu_experiments/dictionary --labeled_patients_path /local-scratch/nigam/projects/ethanid/survival_full/subset_labels/${target}.pickle --is_hierarchical
done
