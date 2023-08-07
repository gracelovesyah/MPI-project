from mpi4py import MPI
import pandas as pd
import time, os

# record start time
t1_start = time.process_time()

# Get rank and size
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
NUMBER_OF_BUCKETS = size

# output columns
twi_col = ['author_id', 'state_t', 'city_t']

# Creating directories
joint_path = "./data/curated/joint_buckets"

# Create output directory
if not os.path.exists(joint_path) and rank == 0:
    os.makedirs(joint_path)

# Syncronize
comm.Barrier()

# Preprocess tweets data
def preprocess_twit(twit):
    # split full_name into city and state columns, and convert city to lowercase
    twit['city_t'] = twit['full_name'].str.split(', ').str[0].str.lower()
    twit['state'] = twit['full_name'].str.split(', ').str[1].str.lower()

    # clean city name by dropping suffix and part after " - "
    twit['city_t'] = twit['city_t'].str.split(' - ').str[0].str.lower() # if there is “ - ” in city, only keep the part before “ - ”
    twit['city_t'] = twit['city_t'].str.split('(').str[0].str.strip() # for city, drop state suffix

    # clean state name by dropping prefix and keeping state suffix
    twit['state'] = twit['state'].str.split('(').str[-1].str.strip() # for state, only keep state suffix
    twit['state'] = twit['state'].str.replace(r')', '', regex=False)
    twit = twit[['author_id','city_t','state']]
    state_abbrev = {'victoria': 'vic.', 'new south wales': 'nsw', 'queensland': 'qld', 'western australia': 'wa',
                  'south australia': 'sa', 'tasmania': 'tas.', 'northern territory': 'nt', 'australian capital territory': 'act'}
    twit['state_t'] = twit['state'].replace(state_abbrev)

    # keep only author_id, city_t, and state_t columns, and remove obvious outliers 
    # (eg. victoria, australia; austrlia etc.)
    obvi_outlier = ['australia', 'new south wales', 'victoria', 'queensland','western australia','south australia','northern territory','australian capital territory','tasmania']
    twit = twit[~twit.city_t.isin(obvi_outlier)] # drop obvi outliers 
    twit = twit[~twit.state_t.isin(['ontario', 'denver'])]
    twit = twit[['author_id','city_t','state_t']]
    return twit

# Preprocess sal.json
def preprocess_sal():
    # load raw data from JSON file and transpose
    sal = pd.read_json("./data/raw/sal.json")
    sal = sal.transpose()

    # reset index and rename columns
    sal.reset_index(inplace=True)
    sal.rename(columns={"index": "name"}, inplace=True)

    # extract city and state information
    sal['city'] = sal['name'].str.split(r' \(| - |\) ')
    sal['city'] = sal['city'].apply(lambda x: x[1] if len(x) == 3 else x[0])
    states_dict = {'1': 'nsw', '2': 'vic.', '3': 'qld', '4': 'sa', '5': 'wa', '6': 'tas.', '7': 'nt', '8': 'act', '9': 'oter'}
    sal.loc[(sal['ste'].isin(states_dict.keys())), 'state'] = sal.loc[(sal['ste'].isin(states_dict.keys())), 'ste'].map(states_dict)


    # keep only relevant columns and drop duplicates
    sal = sal[['gcc', 'sal', 'city', 'state']]
    sal.loc[sal[sal['sal'] == '31937'].index[0], ['city']] = 'mount archer'
    sal.loc[sal[sal['sal'] == '32621'].index[0], ['city']] = 'spring creek'
    sal = sal.drop_duplicates(subset=['state','city','gcc'])[['city', 'state','gcc']]
    return sal

# direct match, city - city & state - state
def match_city_state(twit, gcc):
    # print(f'Step2: (Rank{rank}) In total {len(twit)}')
    df1= twit.merge(gcc, how='left', left_on = ['city_t', 'state_t'], right_on = ['city', 'state'],validate = "many_to_one")  # match the not-unique city
    df1_2 = df1[df1.gcc.notnull()]
    df1_unmatch = df1[df1.gcc.isnull()][twi_col]
    # print(f'Step2: (Rank{rank}) {len(df1_2)} ({round(len(df1_2)/len(twit)*100,2)}%) data matched')
    return df1_2, df1_unmatch

# match capital cities (city - city) | (state - city )
def match_capital_city(df, captial_gcc):
    df2 = df.merge(captial_gcc, how='left', left_on="state_t", right_on="city", validate = "many_to_one")  # state(twitter) - city(gcc) match
    df2_2 = df2[df2.gcc.notnull()]
    df2_unmatch = df2[df2.gcc.isnull()][twi_col]
    df3 = df2_unmatch.merge(captial_gcc, how='left', left_on="city_t", right_on="city", validate = "many_to_one") # city(twitter) - city(gcc) match
    df3_2 = df3[df3.gcc.notnull()]
    # print(f'Step2: (Rank{rank}) {len(df2_2)+ len(df3_2)} ({round((len(df2_2)+len(df3_2))/len(twit)*100,2)}%) data matched')
    return df2_2, df3_2

# merge tweets with gcc
def merge(twit, gcc, captial_gcc):
    df1, df1_unmatch = match_city_state(twit, gcc)
    df2, df3 = match_capital_city(df1_unmatch, captial_gcc)
    final = pd.concat([df1, df2, df3], ignore_index=True)
    # print(f'Step2: (Rank{rank}) In total {len(final)} ({round(len(final)/len(twit) *100,2)}%) data merged with gcc code')
    return final

# Process tweet data
twit = pd.read_parquet(f"./data/curated/parquet_buckets/bucket_authorID_{rank}.parquet")
twit = preprocess_twit(twit)

# Process gcc data
capital_city_gcc = ['1gsyd', '2gmel', '3gbri', '4gade', '5gper', '6ghob', '7gdar', '8acte', '9oter']
others = ["christmas island", "home island", "jervis bay", "norfolk island", "west island"]
capital_cities = ['canberra', 'sydney', 'darwin', 'brisbane', 'adelaide', 'hobart', 'melbourne', 'perth'] + others
gcc = preprocess_sal()
# captial_gcc = gcc.loc[(gcc.city.isin(capital_cities))&(gcc.gcc.isin(capital_city_gcc)), ['gcc', 'city','state']]
captial_gcc = gcc[gcc.gcc.isin(capital_city_gcc)][['gcc', 'city', 'state']]
captial_gcc = captial_gcc[captial_gcc.city.isin(capital_cities)][['gcc', 'city', 'state']]

# create merged parquet
final = merge(twit, gcc, captial_gcc)
final = final[["author_id", "gcc"]]
final.to_parquet(f"{joint_path}/bucket_joint_{rank}.parquet")

# total process time
t1_end = time.process_time()
print(f"Step2: (Rank{rank}) Elapsed time in seconds:", t1_end - t1_start)

