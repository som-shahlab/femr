import pickle
import femr.labelers

with open('/local-scratch/nigam/projects/zphuo/data/PE/labels_and_features_censored/12_month_mortality/labeled_patients.pkl', 'rb') as f:
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

l.save('fixed_labels.csv')
