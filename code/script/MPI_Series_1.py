from mpi4py import MPI
import pandas as pd
import csv
import ijson
import os
import time
import gc

# record start time
t1_start = time.process_time()

# Get rank and size
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
NUMBER_OF_BUCKETS = size

# Creating directories
data_path = "./data"
raw_path = "./data/raw"
curated_path = "./data/curated"
csv_path = "./data/curated/csv_buckets"
parquet_path = "./data/curated/parquet_buckets"

if rank == 0:
    dir_list = []
    dir_list.append(data_path)
    dir_list.append(raw_path)
    dir_list.append(curated_path)
    dir_list.append(csv_path)
    dir_list.append(parquet_path)

    for path in dir_list:
        if not os.path.exists(path):
            os.makedirs(path)

    for i in range(NUMBER_OF_BUCKETS):
        with open(f"./data/curated/csv_buckets/bucket_authorID_{i}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(("author_id", "full_name"))

# Syncronize the process
comm.Barrier()

# read and rewrite the tweets data
with open("./data/raw/bigTwitter.json", "rb") as f:
    with open(f"./data/curated/csv_buckets/bucket_authorID_{rank}.csv", "a") as cf:
        writer = csv.writer(cf)
        for record in ijson.items(f, "item"):
            i = int(record["data"]["author_id"]) % NUMBER_OF_BUCKETS
            if i == rank:
                writer.writerow(
                    (
                        record["data"]["author_id"],
                        record["includes"]["places"][0]["full_name"],
                    )
                )



df = pd.read_csv(f"./data/curated/csv_buckets/bucket_authorID_{rank}.csv", dtype = {"author_id":int, "full_name":str})
df.to_parquet(f"./data/curated/parquet_buckets/bucket_authorID_{rank}.parquet")
del df
gc.collect()

# measure end time
t1_end = time.process_time()
print(f"Step1: (Rank{rank}) Elapsed time in seconds:", t1_end - t1_start)
