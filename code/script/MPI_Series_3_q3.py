from mpi4py import MPI
import pandas as pd
import time
import gc

# Convert to output format
def create_msg(counts, gcc_list, gcc_count, total_count):
    """
    Creates a message string based on the given counts, GCC list, GCC count, and total count.

    Args:
    counts (list): A list of integers representing the number of tweets for each GCC in gcc list.
    gcc_list (list): A list of strings representing the GCCs.
    gcc_count (int): The number of unique GCCs in the dataset.
    total_count (int): The total number of tweets in the dataset.

    Returns:
    A string representing the message created based on the given arguments. The message contains the number of tweets for each
    GCC in gcc_list, the total number of tweets, and the number of unique GCCs.
    """
    msg = f"{gcc_count} (#{total_count} tweets - "
    i = 0
    length = len(gcc_list)
    for c, g in zip(counts, gcc_list):
        i += 1
        msg += str(c) + g[1:]
        if i < length:
            msg += ", "

    msg += ")"
    return msg

# Algorithm to solve question 3
def q3(address):
    """
    Reads a Parquet file from the given address and performs data processing to return a pandas DataFrame with
    information on the top 10 authors who have tweeted from the most number of unique GCC city locations along with the
    number of tweets for each city.

    Args:
        address (str): The path to the Parquet file.

    Returns:
        pandas.DataFrame: A DataFrame with the following columns:
            - Author Id: The ID of the author.
            - gcc_list: A list of GCC city codes for the author.
            - counts: A list of tweet counts for each GCC city.
            - gcc_count: The number of unique GCC city locations tweeted by the author.
            - total_count: The total number of tweets by the author.
            - Number of Unique City Locations and #Tweets: A string indicating the number of unique GCC city locations and
              the number of tweets for each city.
    """
    df = pd.read_parquet(address)
    df = df.loc[
        df["gcc"].isin(
            ["1gsyd", "2gmel", "3gbri", "4gade", "5gper", "6ghob", "7gdar", "8acte", "9oter"]
        )
    ].value_counts()
    df = df.rename("count")
    df = df.reset_index()
    df_gcc = df.groupby("author_id")["gcc"].apply(list)
    df_gcc = df_gcc.rename("gcc_list")
    df_gcc = df_gcc.reset_index()
    df_count = df.groupby("author_id")["count"].apply(list)
    df_count = df_count.rename("counts")
    df_count = df_count.reset_index()

    df_merged = pd.merge(
        df_gcc,
        df_count,
        left_on=["author_id"],
        right_on=["author_id"],
        how="left",
        validate="one_to_one",
    )
    df_merged["gcc_count"] = df_merged["gcc_list"].apply(len)
    df_merged["total_count"] = df_merged["counts"].apply(sum)
    df_merged = df_merged.sort_values(
        ["gcc_count", "total_count"], ascending=False
    ).head(10)
    df_merged = df_merged.reset_index(drop=True)
    df_merged = df_merged.rename(columns={"author_id": "Author Id"})
    df_merged["Number of Unique City Locations and #Tweets"] = [
        create_msg(counts, gcc_list, gcc_count, total_count)
        for counts, gcc_list, gcc_count, total_count in zip(
            df_merged["counts"],
            df_merged["gcc_list"],
            df_merged["gcc_count"],
            df_merged["total_count"],
        )
    ]

    return df_merged

# record start time
t1_start = time.process_time()

# Get rank and size
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Agregate by Author Id and GCC
if rank == 0:
    # aggregation process
    df = q3(f"./data/curated/joint_buckets/bucket_joint_{rank}.parquet")

    # Combine the results
    concat_list = [df]
    for i in range(1, size):
        req = comm.irecv(source=i, tag=i)
        tmp = req.wait()
        tmp = pd.DataFrame.from_dict(tmp)
        concat_list.append(tmp)
    df = pd.concat(concat_list)

    # Top 10 results
    df = df.sort_values(["gcc_count", "total_count"], ascending=False).head(10)
    df = df.reset_index(drop=True)

    # convert to result format
    df["Rank"] = [f"#{i+1}" for i in range(len(df))]
    df = df[["Rank", "Author Id", "Number of Unique City Locations and #Tweets"]]
    df.to_csv("./q3.csv", index=False)
    print("Q3:  Tweeters that have tweeted in the most Greater Capital cities and the number of times they have tweeted from those locations")
    print(df.to_string(index=False))
    print("\n")
else:
    # aggregation process
    df = q3(f"./data/curated/joint_buckets/bucket_joint_{rank}.parquet")
    df = df.to_dict()

    # Send the result to process 0
    req = comm.isend(df, dest=0, tag=rank)
    req.wait()

del df
gc.collect()

# total process time
t1_end = time.process_time()
print(f"Step3_Q3: (Rank{rank}) Elapsed time in seconds:", t1_end - t1_start)