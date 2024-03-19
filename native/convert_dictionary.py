import piton.datasets
import msgpack

old_data = piton.datasets.PatientDatabase('/home/ethan/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_09_05_extract2_lite')
data = piton.datasets.PatientDatabase('/home/ethan/new_extract')

print(dir(data))

dictionary = msgpack.load(open('dictionary', 'rb'))

new_index_map = {}

for code in range(len(old_data.get_code_dictionary())):
    code_str = old_data.get_code_dictionary()[code]
    try:
        new_code = data.get_code_dictionary().index(code_str)
    except:
        new_code = -code
    new_index_map[code] = new_code

new_index_map[14] = 51

new_value_index_map = {}

for code in range(len(old_data.get_shared_value_dictionary())):
    code_str = old_data.get_shared_value_dictionary()[code]
    try:
        new_code = data.get_shared_value_dictionary().index(code_str)
    except:
        new_code = -code
    new_value_index_map[code] = new_code

for entry in dictionary['ontology_rollup']:
    print(entry)
    if new_index_map[entry['code']] < 0:
        entry['type'] = 10
        continue
    else:
        entry['code'] = new_index_map[entry['code']]

    if entry['type'] == 2:
        print(entry)
        break
