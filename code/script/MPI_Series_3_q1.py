from mpi4py import MPI
import pandas as pd
import time
import gc

# record start time
t1_start = time.process_time()

# Get rank and size
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Aggregate the number of the tweets by gcc
if rank == 0:
    # Link gcc name with output format
    gcc_names = {
        "1gsyd": "1gsyd (Greater Sydney)",
        "2gmel": "2gmel (Greater Melbourne)",
        "3gbri": "3gbri (Greater Brisbane)",
        "4gade": "4gade (Greater Adelaide)",
        "5gper": "5gper (Greater Perth)",
        "6ghob": "6ghob (Greater Hobart)",
        "7gdar": "7gdar (Greater Darwin)",
        "8acte": "8acte (Australian Capital Territory)",
        "9oter": "9oter (Other Territories)",
    }

    # Count the number of tweets by gcc
    df = pd.read_parquet(f"./data/curated/joint_buckets/bucket_joint_{rank}.parquet")
    df = df["gcc"].value_counts().to_dict()

    # Sum of all tweets numbers by gcc
    for i in range(1, size):
        req = comm.irecv(source=i, tag=i)
        tmp = req.wait()

        for key, item in tmp.items():
            if key in df:
                df[key] += item
            else:
                df[key] = item

    # Rank the values
    df = {gcc_names[key]: item for key, item in df.items() if key in gcc_names}
    df = (
        pd.DataFrame.from_dict(df, orient="index", columns=["Number of Tweets Made"])
        .sort_values(by="Number of Tweets Made", ascending=False)
        .head(10)
    )
    df = df.reset_index()
    df = df.rename(columns={"index":"Greater Capital City"})

    # Print output
    df.to_csv("./q1.csv", index=False)
    print("Q1:  The number of tweets in the various capital cities")
    print(df.to_string(index=False))
    print("\n")
else:
    # Count the number of tweets by gcc
    df = pd.read_parquet(f"./data/curated/joint_buckets/bucket_joint_{rank}.parquet")
    df = df["gcc"].value_counts().to_dict()

    # Send result to Process 0
    req = comm.isend(df, dest=0, tag=rank)
    req.wait()

del df
gc.collect()

# total process time
t1_end = time.process_time()
print(f"Step3_Q1: (Rank{rank}) Elapsed time in seconds:", t1_end - t1_start)