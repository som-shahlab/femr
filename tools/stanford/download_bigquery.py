"""
A tool for downloading datasets from BigQuery.

Setup:
```
    pip install --upgrade google-cloud-bigquery
    pip install --upgrade google-cloud-storage
    conda install google-cloud-sdk -c conda-forge
```

Note: After installing above packages, run `gcloud auth application-default login` on your terminal.
You will be prompted with a authorization link that you will need to follow and approve using your
email address. Then you will copy-paste authorization code to the terminal.

How to run:
```
python download_bigquery.py <NAME OF YOUR GCP PROJECT> <GCP BIGQUERY DATASET ID> \
    <PATH TO LOCAL FOLDER WHERE DATASET WHERE DATASET WILL BE DOWNLOADED>
    --excluded_tables <(Optional) NAME OF TABLE 1 TO BE IGNORED> <(Optional) NAME OF TABLE 2 TO BE IGNORED>
```

Example: python download_bigquery.py som-nero-nigam-starr \
    som-rit-phi-starr-prod.starr_omop_cdm5_deid_1pcent_lite_2023_02_08 .
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import threading
from functools import partial

import google
from google.cloud import bigquery, storage

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Google BigQuery dataset")
    parser.add_argument(
        "gcp_project_name",
        type=str,
        help=(
            "The name of *YOUR* GCP project (e.g. 'som-nero-nigam-starr')."
            " Note that this need NOT be the GCP project that contains the dataset."
            " It just needs to be a GCP project where you have Bucket creation + BigQuery creation permissions."
        ),
    )
    parser.add_argument(
        "gcp_dataset_id",
        type=str,
        help=(
            "The Dataset ID of the GCP dataset to download"
            " (e.g. 'som-rit-phi-starr-prod.starr_omop_cdm5_deid_2022_12_03')."
            " Note that this is the full ID of the dataset (project name + dataset name)"
        ),
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help=(
            "Path to output directory. Note: The downloaded files will be saved in a subdirectory of this,"
            " i.e. `output_dir/gcp_dataset_id/...`"
        ),
    )
    parser.add_argument(
        "--excluded_tables",
        type=str,
        nargs="*",  # 0 or more values expected => creates a list
        default=[],
        help=(
            "Optional. Name(s) of tables to exclude. List tables separated by spaces,"
            " i.e. `--excluded_tables observation note_nlp`"
        ),
    )
    parser.add_argument(
        "--scratch_bucket_postfix",
        type=str,
        default="_extract_scratch",
        help="The postfix for the GCP bucket used for storing temporary files while downloading.",
    )
    args = parser.parse_args()

    print('Make sure to run "gcloud auth application-default login" before running this command')

    # Connect to our BigQuery project
    client = bigquery.Client(project=args.gcp_project_name)
    storage_client = storage.Client(project=args.gcp_project_name)

    random_dir = hashlib.md5(random.randbytes(16)).hexdigest()

    scratch_bucket_name = args.gcp_project_name.replace("-", "_") + args.scratch_bucket_postfix

    print(f"Storing temporary files in gs://{scratch_bucket_name}/{random_dir}")

    try:
        bucket = storage_client.get_bucket(scratch_bucket_name)
    except google.api_core.exceptions.NotFound as e:
        print(f"Could not find the requested bucket? gs://{scratch_bucket_name} in project {args.gcp_project_name}")
        raise e

    # Get list of all tables in this GCP dataset
    # NOTE: the `HTTPIterator` can be iterated over like a list, but only once (it's a generator)
    tables: google.api_core.page_iterator.HTTPIterator = client.list_tables(args.gcp_dataset_id)
    print(f"Downloading dataset {args.gcp_dataset_id} using your project {args.gcp_project_name}")

    # Use GZIP compression and export as CVSs
    extract_config = bigquery.job.ExtractJobConfig(
        compression=bigquery.job.Compression.GZIP,
        destination_format=bigquery.job.DestinationFormat.CSV,
        field_delimiter=",",
    )

    sem = threading.Semaphore(value=0)  # needed for keeping track of how many tables have been downloaded

    def download(table_id: str, f):
        """Download the results (a set of .csv.gz's) of the BigQuery extract job to our local filesystem
        Note that a single table will be extracted into possibly dozens of smaller .csv.gz files

        Args:
            table_id (str): Name of table (e.g. "attribute_definition")
        """
        if f.errors is not None:
            print("Could not extract, got errors", f.errors, "for", table_id)
            os.abort()
        print(f"Downloading | table = {table_id}")
        # Setup local directory for storing downloaded .csv.gz's
        target_folder: str = os.path.join(args.output_dir, args.gcp_dataset_id, table_id)
        os.makedirs(target_folder, exist_ok=True)
        # Get all .csv.gz's corresponding to this table
        # NOTE: the `HTTPIterator` can be iterated over like a list, but only once (it's a generator)
        blobs: google.api_core.page_iterator.HTTPIterator = storage_client.list_blobs(
            bucket, prefix=f"{random_dir}/{table_id}/"
        )
        for blob in blobs:
            # Download .csv.gz file to local filesystem
            blob.download_to_filename(os.path.join(target_folder, blob.name.split("/")[-1]))
            # Now that we've downloaded this file, delete it from the Google Cloud Storage bucket
            blob.delete()
        print(f"Download Finished | table = {table_id}")
        sem.release()

    n_tables: int = 0
    for table in tables:
        # Get the full name of the table
        table_name: str = f"{table.project}.{table.dataset_id}.{table.table_id}"
        if table.table_id in args.excluded_tables:
            print(f"Skipping extraction | table = {table.table_id}")
            continue
        print(f"Extracting | table = {table.table_id}")
        # Create Google Cloud Storage bucket to extract this table into
        bucket_target_path: str = f"gs://{scratch_bucket_name}/{random_dir}/{table.table_id}/*.csv.gz"
        extract_job = client.extract_table(table.reference, bucket_target_path, job_config=extract_config)
        # Call the `download()` function asynchronously to download the bucket contents to our local filesystem
        extract_job.add_done_callback(partial(download, table.table_id))
        n_tables += 1

    print(f"\n** Downloading a total of {n_tables} tables**\n")
    for i in range(1, n_tables + 1):
        sem.acquire()
        print(f"====> Finished downloading {i} out of {n_tables} tables")
    print("------\n------")
    print("Successfully downloaded all tables!")
    print("------\n------")

    # Delete the temporary Google Cloud Storage bucket
    print("\nDeleting temporary files...")
    undeleted_blobs: google.api_core.page_iterator.HTTPIterator = storage_client.list_blobs(
        bucket,
        prefix=random_dir + "/",
    )
    for blob in undeleted_blobs:
        blob.delete()
    print("------\n------")
    print("Successfully deleted temporary Google Cloud Storage files!")
    print("------\n------")
