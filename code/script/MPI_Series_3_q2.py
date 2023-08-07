from mpi4py import MPI
import pandas as pd
import time
import gc

# record start time
t1_start = time.process_time()

# Get rank and size
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # rank = process_id 
size = comm.Get_size() # size = number of process

if rank == 0:
    df = pd.read_parquet(
        f"./data/curated/parquet_buckets/bucket_authorID_{rank}.parquet"
    )
    # aggregation by Author ID
    df = df["author_id"].value_counts().head(10).to_dict() # dic of author ID and number of tweets made by author

    # Combine the reusult tables
    for i in range(1, size):
        req = comm.irecv(source=i, tag=i) # receiving from rank 1 to rank n
        tmp = req.wait() # wait for response(dict) from other process

        # Add the result of each process
        for key, item in tmp.items():
            if key in df:
                df[key] += item
            else:
                df[key] = item

    # Top 10 reult
    df = (
        pd.DataFrame.from_dict(df, orient="index", columns=["Number of Tweets Made"])
        .sort_values(by="Number of Tweets Made", ascending=False)
        .head(10)
    )

    # formatting for output
    df = df.reset_index()
    df = df.rename(columns={"index":"Author Id"})
    df["Rank"] = [f"#{i+1}" for i in range(len(df))]
    df = df[["Rank", "Author Id", "Number of Tweets Made"]]

    # Print output
    df.to_csv("./q2.csv", index=False)
    print("Q2:  Top 10 tweeters (in terms of the number of tweets made)")
    print(df.to_string(index=False))
    print("\n")
else:
    # aggregation by Author ID
    df = pd.read_parquet(
        f"./data/curated/parquet_buckets/bucket_authorID_{rank}.parquet"
    )
    df = df["author_id"].value_counts().head(10).to_dict()

    # Send result to process 0
    req = comm.isend(df, dest=0, tag=rank)
    req.wait()

del df
gc.collect()

# total process time
t1_end = time.process_time()
print(f"Step3_Q2: (Rank{rank}) Elapsed time in seconds:", t1_end - t1_start)