targets="pancreatic_cancer celiac_disease lupus heart_attack stroke NAFL"

rm -r "fixed_batches"
for target in $targets; do
    echo $target
    mkdir -p "fixed_batches"
    python prepare_batches.py "fixed_batches/${target}_batches" --data_path $P  \
        --dictionary_path ../../gpu_experiments/dictionary \
        --labeled_patients_path /local-scratch/nigam/projects/ethanid/survival_full/subset_labels/${target}.pickle --is_hierarchical
    #  python prepare_batches.py "survival_h/${target}_non_hire_batches" --data_path $P  --dictionary_path ../../gpu_experiments/dictionary --labeled_patients_path /local-scratch/nigam/projects/ethanid/survival_full/subset_labels/${target}.pickle
done
