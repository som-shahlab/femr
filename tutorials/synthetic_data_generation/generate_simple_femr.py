import datetime
import os
import random
import string

ETL_INPUT = "../etl_input/simple_femr"


def add_random_patient(patient_id, f):
    epoch = datetime.date(1990, 1, 1)
    birth = epoch + datetime.timedelta(days=random.randint(100, 1000))
    current_date = birth

    code_cat = ["Birth", "Gender", "Race", "ICD10CM", "CPT", "Drug", "Vitals"]
    gender_values = ["F", "M"]
    race_values = ["White", "Non-White"]
    index = random.randint(0, 1)
    for code_type in code_cat:
        if code_type == "Birth":
            clarity = "PATIENT"
            visit_id = 1
            value = ""
            dosage = ""
            unit = ""
            f.write(
                f"{patient_id},{birth.isoformat()},{code_type}/{code_type},{value},{dosage},{visit_id},{unit},{clarity}\n"
            )
        elif code_type == "Gender":
            clarity = "PATIENT"
            visit_id = 1
            value = gender_values[index]
            dosage = ""
            unit = ""
            f.write(
                f"{patient_id},{birth.isoformat()},{code_type}/{code_type},{value},{dosage},{visit_id},{unit},{clarity}\n"
            )
        elif code_type == "Race":
            clarity = "PATIENT"
            visit_id = 1
            value = race_values[index]
            dosage = ""
            unit = ""
            f.write(
                f"{patient_id},{birth.isoformat()},{code_type}/{code_type},{value},{dosage},{visit_id},{unit},{clarity}\n"
            )
        else:
            for code in range(random.randint(1, 10)):
                code = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
                current_date = current_date + datetime.timedelta(days=random.randint(0, 100))
                visit_id = random.randint(0, 20)
                if code_type == "ICD10CM":
                    clarity = "DIAGNOSIS"
                    value = ""
                    dosage = ""
                    unit = ""
                elif code_type == "CPT":
                    clarity = "PROCEDURES"
                    value = ""
                    dosage = ""
                    unit = ""
                elif code_type == "Drug":
                    clarity = "MED_ORDER"
                    value = ""
                    dosage = random.randint(10, 50)
                    unit = "mg"
                elif code_type == "Vitals":
                    clarity = "LAB_RESULT"
                    value = random.randint(80, 200)
                    dosage = ""
                    unit = "mmHg"
                f.write(
                    f"{patient_id},{current_date.isoformat()},{code_type}/{code},{value},{dosage},{visit_id},{unit},{clarity}\n"
                )


for file_no in range(1, 3):
    with open(os.path.join(ETL_INPUT, f"many_examples_{file_no}.csv"), "w") as f:
        f.write("patient_id,start,code,value,dosage,visit_ids,lab_units,clarity_source\n")
        for patient_id in range(file_no * 100, (file_no + 1) * 100):
            add_random_patient(patient_id, f)
