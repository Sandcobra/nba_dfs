import pandas as pd
from collections import defaultdict

# Read the lineups DataFrame
df_lineups = pd.read_csv('lineups.csv')
df_ownership = pd.read_csv('exposures.csv')

# Total number of lineups
total_lineups = len(df_lineups)

# Initialize a dictionary to store total ownership for each player
total_ownership = defaultdict(int)

# Iterate through each lineup in the lineups DataFrame
for _, lineup in df_lineups.iterrows():
    # Iterate through each player in the lineup
    for player in lineup:
        # Extract the player ID from the string
        player_id = player.split(":")[1]
        
        # Add 1 to the ownership count for the player
        total_ownership[player_id] += 1

# Convert the dictionary to a DataFrame
df_total_ownership = pd.DataFrame(list(total_ownership.items()), columns=['Player_ID', 'Total_Ownership'])

# Calculate the percentage of ownership for each player
df_total_ownership['Ownership_Percentage'] = df_total_ownership['Total_Ownership'] / total_lineups * 100

# Display the total ownership DataFrame
print(df_total_ownership)

# Convert 'pOwn%' to a numeric format and multiply by 100 to get percentage
df_ownership['pOwn%'] = pd.to_numeric(df_ownership['pOwn%'].str.rstrip('%'), errors='coerce')

# Merge the DataFrames based on the "Player_ID" and "Player" columns
df_merged = pd.merge(df_total_ownership, df_ownership[['Player', 'pOwn%']], left_on='Player_ID', right_on='Player', how='left')

# Calculate leverage
df_merged['Leverage'] = df_merged['Ownership_Percentage'] - df_merged['pOwn%']
# Check for duplicates in the merged DataFrame
duplicate_rows = df_merged[df_merged.duplicated(subset=['Player_ID'], keep=False)]

# If there are duplicates, display them
if not duplicate_rows.empty:
    print("Duplicate rows:")
    print(duplicate_rows)
    
    # Drop duplicate rows, keeping the first occurrence
    df_merged = df_merged.drop_duplicates(subset=['Player_ID'], keep='first')

# Display the final merged DataFrame without duplicates
print("Final Merged DataFrame:")
print(df_merged)
df_merged.to_csv('lineups_leverage.csv')




