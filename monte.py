import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np



# Load your dataset (assuming you've already processed it)
df = pd.read_csv('modified_all.csv')

def feet_to_inches(height):
    feet, inches = map(int, height.split('-'))
    return feet * 12 + inches

# Apply the function to the 'HEIGHT' column
df['HEIGHT'] = df['HEIGHT'].apply(feet_to_inches)
# Create a mapping of category codes to player names for df
player_mapping = dict(enumerate(df['PLAYER_NAME'].astype('category').cat.categories))
df['PLAYER_NAME'] = df['PLAYER_NAME'].astype('category').cat.codes
df['TEAM_ABBREVIATION_x'] = df['TEAM_ABBREVIATION_x'].astype('category').cat.codes
df['home_team_x'] = df['home_team_x'].astype('category').cat.codes
df['away_team_x'] = df['away_team_x'].astype('category').cat.codes
df['opponent_x'] = df['opponent_x'].astype('category').cat.codes
df['POSITION'] = df['POSITION'].astype('category').cat.codes


features = ['PLAYER_NAME', 'TEAM_ABBREVIATION_x', 'MIN_x', 'PTS_x', 'home_team_x', 'away_team_x', 'opponent_x', 'opponent_id_x', 'MIN_base', 'FGM_base', 'FGA_base',
            'FG_PCT_base', 'FG3M_base', 'FG3A_base', 'FG3_PCT_base', 'FTM_base', 'FTA_base', 'OREB_base', 'DREB_base', 'REB_base', 'AST_base', 'TOV_base', 'STL_base',
            'BLK_base', 'BLKA_base', 'PF_base', 'PFD_base', 'PTS_base', 'PLUS_MINUS_base', 'NBA_FANTASY_PTS_base', 'E_OFF_RATING_adv', 'OFF_RATING_adv', 'sp_work_OFF_RATING_adv',
            'E_DEF_RATING_adv', 'DEF_RATING_adv', 'sp_work_DEF_RATING_adv', 'E_NET_RATING_adv', 'NET_RATING_adv', 'sp_work_NET_RATING_adv', 'AST_PCT_adv', 'AST_TO_adv',
            'AST_RATIO_adv', 'OREB_PCT_adv', 'DREB_PCT_adv', 'REB_PCT_adv', 'TM_TOV_PCT_adv', 'E_TOV_PCT_adv', 'E_TOV_PCT_adv', 'EFG_PCT_adv', 'TS_PCT_adv', 'USG_PCT_adv', 'E_USG_PCT_adv',
            'E_PACE_adv', 'PACE_adv', 'PACE_PER40_adv', 'sp_work_PACE_adv', 'PIE_adv', 'POSS_adv', 'E_OFF_RATING_2nd_Half', 'OFF_RATING_2nd_Half', 'sp_work_OFF_RATING_2nd_Half',
            'E_DEF_RATING_2nd_Half', 'DEF_RATING_2nd_Half', 'sp_work_DEF_RATING_2nd_Half', 'E_NET_RATING_2nd_Half', 'NET_RATING_2nd_Half', 'sp_work_NET_RATING_2nd_Half',
            'AST_PCT_2nd_Half', 'AST_TO_2nd_Half', 'AST_RATIO_2nd_Half', 'OREB_PCT_2nd_Half', 'DREB_PCT_2nd_Half', 'REB_PCT_2nd_Half', 'TM_TOV_PCT_2nd_Half', 'E_TOV_PCT_2nd_Half',
            'EFG_PCT_2nd_Half', 'TS_PCT_2nd_Half', 'USG_PCT_2nd_Half', 'E_USG_PCT_2nd_Half', 'E_PACE_2nd_Half', 'PACE_2nd_Half', 'PACE_PER40_2nd_Half', 'sp_work_PACE_2nd_Half',
            'PIE_2nd_Half', 'Above the Break 3_FG_PCT','Restricted Area_OPP_FG_PCT_L10','In The Paint (Non-RA)_OPP_FG_PCT_L10','Mid-Range_OPP_FG_PCT_L10','Left Corner 3_OPP_FG_PCT_L10',
            'Right Corner 3_OPP_FG_PCT_L10', 'Above the Break 3_OPP_FG_PCT_L10', 'AGE']

target = 'PTS_x'

X = df[features]
y = df[target]

# # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# # Build the neural network model
model = keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output layer with a single neuron for regression
])

# # Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# # Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

train_predictions = model.predict(X_train_scaled)

# # # Make predictions on the test set
predictions = model.predict(X_test_scaled)
# print(predictions)
# # Evaluate the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Extract relevant features for prediction
features_for_prediction = df[features]

# Standardize the features
features_for_prediction_scaled = scaler.transform(features_for_prediction)

# Initialize an empty DataFrame to store results
simulated_results_df = pd.DataFrame()

for i in range(20):
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Introduce more noise during training
    y_train_noisy = y_train + np.random.normal(0, 1.8, len(y_train))
    # y_train_noisy = y_train + np.random.poisson(y_train)

    # Train the model
    model.fit(X_train_scaled, y_train_noisy, epochs=50, batch_size=32, validation_split=0.2)
    
    # Make predictions on the last 10 rows
    last_10_predictions = model.predict(features_for_prediction_scaled[-237:])
    
    # Get the encoded player names for the last 10 rows
    encoded_players = df['PLAYER_NAME'].iloc[-237:]
    
    # Use the player mapping to get the original player names
    original_players = [player_mapping.get(encoded_player) for encoded_player in encoded_players]
    
    # Create a DataFrame for the last 10 predictions
    predictions_df = pd.DataFrame({
        f'Predicted_PTS_x_{i + 1}': last_10_predictions.flatten()  # Use i+1 for column names
    })
    
    # Concatenate the current predictions with the overall results
    simulated_results_df = pd.concat([simulated_results_df, predictions_df], axis=1)
# Assuming df is your DataFrame with the multiple prediction columns
# simulated_results_df['Max_Prediction_Column'] = simulated_results_df.iloc[:, 1:].idxmax(axis=1)
simulated_results_df['Average_Prediction'] = simulated_results_df.iloc[:, 1:].mean(axis=1)

# Extract the actual max predictions
simulated_results_df['Max_Prediction_Column'] = simulated_results_df.iloc[:, 1:].idxmax(axis=1)
simulated_results_df['Max_Predicted_PTS_x'] = simulated_results_df.apply(lambda row: row[row['Max_Prediction_Column']], axis=1)


# Print or use simulated_results_df as needed
print(simulated_results_df)

# # Save the DataFrame to a CSV file
# simulated_results_df.to_csv('predictions/monte_predicted_points.csv', index=False) #### PTS
# simulated_results_df.to_csv('predictions/monte_predicted_fg3.csv', index=False) ### FG3M
# simulated_results_df.to_csv('predictions/monte_predicted_Oreb.csv', index=False) #### OREB
# simulated_results_df.to_csv('predictions/monte_predicted_Dreb.csv', index=False) #### DREB
# simulated_results_df.to_csv('predictions/monte_predicted_ast.csv', index=False) #### AST
# simulated_results_df.to_csv('predictions/monte_predicted_stl.csv', index=False) #### STL
simulated_results_df.to_csv('predictions/monte_predicted_blk.csv', index=False) #### BLK


print(predictions_df)
