import datetime

import meds
import meds_reader
import numpy as np

from femr.model_utils import get_model_vocab

def get_subject_birthdate(subject: meds_reader.Subject) -> datetime.datetime:
    for e in subject.events:
        if e.code == meds.birth_code and e.time is not None:
            return e.time
    raise ValueError("Couldn't find subject birthdate -- Subject has no events " + str(subject))

def get_subject_codes(subjects):
    """
    Get all codes from subjects in a SubjectDatabase, 
    
    Args:
        subjects: An iterable of Subject objects.

    Returns:
        A list of codes from all subjects.

    Example:
        >>> db = subject_database(tmpdir="tmp")
        >>> codes = db.map(get_subject_codes)
    """
    print(f"Getting all codes from subjects")
    codes = []
    for subject in subjects:
        subject_codes = [event.code for event in subject.events if event.code is not None]
        codes.extend(subject_codes)
    
    return codes

def get_all_codes(db: meds_reader.SubjectDatabase,
                  unnest: bool = True):
    """
    Get all codes from all subjects in a SubjectDatabase.
    """
    subject_codes = db.map(get_subject_codes)
    if unnest:
        subject_codes = [code for codes in subject_codes for code in codes]
        print(f"{len(set(subject_codes))} unique subject codes found in SubjectDatabase")
    return subject_codes

def check_vocab_overlap(model_name: str,
                        db: meds_reader.SubjectDatabase):
    """
    Check if the codes in the subject data are present in the model vocabulary.

    Args:
        model_name: The name of the model to check the vocabulary against.
        db: The SubjectDatabase to check the vocabulary against.

    Returns:
        The overlap between the model vocabulary and the subject data.

    Example:
        >>> check_vocab_overlap(model_name="StanfordShahLab/clmbr-t-base",
                                db=subject_db,
                                code_col="code")
    """
    # Get model vocabulary codes
    model_codes = get_model_vocab(model_name=model_name, 
                                  codes_only=True)
    # Get codes for each subject
    subject_codes = get_all_codes(db)

    # First, we should check that at least some of the codes in the model are present in the MEDSV3_DATA
    overlap = set(model_codes).intersection(set(subject_codes))
    if len(overlap) == 0:
        raise ValueError(f"No overlap between model vocabulary and subject data for model {model_name}")
    else:
        print(f"Overlap between model vocabulary and subject data for model {model_name}: {len(overlap)}")
    return overlap
 
def filter_subjects(db: meds_reader.SubjectDatabase,
                     subject_ids: list[int]):
    """
    Filter the subjects in the SubjectDatabase based on the subject ids.

    Args:
        db: The SubjectDatabase to filter.
        subject_ids: The list of subject ids to filter.

    Returns:
        A new SubjectDatabase with the filtered subjects.

    Example:
        >>> filtered_db = filter_subjects(db, [1, 2, 3])
    """
    if isinstance(subject_ids, int):
        subject_ids = [subject_ids]
    return db.filter(lambda subject: subject.subject_id in subject_ids)


def fix_subject_codes(db: meds_reader.SubjectDatabase):
    """
    Fix the subject codes in the SubjectDatabase.
    """
    for subject_id in db:
        for e in db[subject_id].events:
            if 'code' in e and e.code is not None:
                e.code = e.code.replace("//", "/")
    return db   

def count_codes(db: meds_reader.SubjectDatabase):
    """
    Count the number of codes in the SubjectDatabase.
    """
    
    import collections

    code_counts = collections.defaultdict(int)

    for subject_id in db:
        subject = db[subject_id]

        for event in subject.events:
            code_counts[event.code] += 1

    print("Total # of codes", sum(code_counts.values()))

    code_and_counts = list(code_counts.items())
    code_and_counts.sort(key=lambda a: -a[1])

    print("10 most common codes")
    for code, count in code_and_counts[:10]:
        print(code, count)

