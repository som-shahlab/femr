import pickle
import femr.labelers
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run femr featurization")
    parser.add_argument("--path_to_labeled_patients_pkl", required=True, type=str, help="Path to labeled patients pkl")


    args = parser.parse_args()



    with open(args.path_to_labeled_patients_pkl, 'rb') as f:
        labels = pickle.load(f)

    fixed_labels = {}

    def fix_labels(labels):
        result = []
        for label in labels:
            if label.value == 'Censored':
                continue
            result.append(femr.labelers.Label(time=label.time, value=label.value=='True'))

        return result

    for k, v in labels.items():
        fixed_v = fix_labels(v)
        if len(fixed_v) > 0:
            fixed_labels[k] = fixed_v

    l = femr.labelers.LabeledPatients(fixed_labels, labels.labeler_type)

    path_to_save = args.path_to_labeled_patients_pkl.replace('.pkl', '_fixed.csv')
    l.save(path_to_save)
