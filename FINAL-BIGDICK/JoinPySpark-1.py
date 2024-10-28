import pandas as pd

# Load the two uploaded CSV files
file1_path = 'fb_live_thailand2.csv'
file2_path = 'fb_live_thailand3.csv'

df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# Performing an outer join on the dataframes
# Assuming default join on common columns
outer_joined_df = pd.merge(df1, df2, how='outer') # how='outer,inner,left,right'

# Display the joined data to the user
print(outer_joined_df)

output_path = 'outer_joined_fb_live_thailand.csv' # สร้างไฟล์ใหม่
outer_joined_df.to_csv(output_path, index=False)
