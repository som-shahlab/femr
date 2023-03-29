target_dir="/local-scratch/nigam/projects/ethanid/motor_paper_labels/better_subset_labels"

# targets="pancreatic_cancer celiac_disease lupus heart_attack stroke NAFL"
ls $target_dir | sed -e 's/\.pickle$//'> target_file
targets=$(cat target_file)

for target in $targets; do
    echo $target
    mkdir -p "fixed_batches_text_fixed"
    clmbr_create_batches "fixed_batches_text_fixed/${target}_batches" --data_path $P  \
        --dictionary_path ../../gpu_experiments/dictionary \
	--is_hierarchical \
    	--labeled_patients_path "${target_dir}/${target}.pickle"
        #--labeled_patients_path /local-scratch/nigam/projects/ethanid/survival_full/subset_labels/${target}.pickle --is_hierarchical
    #  python prepare_batches.py "survival_h/${target}_non_hire_batches" --data_path $P  --dictionary_path ../../gpu_experiments/dictionary --labeled_patients_path /local-scratch/nigam/projects/ethanid/survival_full/subset_labels/${target}.pickle
    break
done
