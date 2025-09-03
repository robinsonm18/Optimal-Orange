import numpy as np
import pandas as pd

# Number of agents
num_agents = 10

# Generate reservation values
product_1_values = np.random.uniform(0, 1, num_agents)  # Values between 0 and 1
product_2_values = np.random.uniform(1, 2, num_agents)  # Values between 1 and 2

# Create a DataFrame for reservation values
data = {
    f"Agent {i+1}": [product_1_values[i], product_2_values[i]] for i in range(num_agents)
}
df_reservation = pd.DataFrame(data, index=["Product 1", "Product 2"])

# Create a DataFrame for target values
target_values = [0.75, 1.5]  # Updated target value for Product 1
df_target = pd.DataFrame({"Target Value": target_values}, index=["Product 1", "Product 2"])

# Define product costs
product_costs = [0.5, 1.0]
df_costs = pd.DataFrame({"Cost": product_costs}, index=["Product 1", "Product 2"])

# Calculate product margins
df_margins = df_target["Target Value"] - df_costs["Cost"]
df_margins = pd.DataFrame({"Margin": df_margins})

# Create a DataFrame for dummy values
def calculate_dummies(column):
    # Calculate differences between target and reservation values
    diff_product_1 = column["Product 1"] - df_target.loc["Product 1", "Target Value"]
    diff_product_2 = column["Product 2"] - df_target.loc["Product 2", "Target Value"]
    
    # Initialize dummies as 0
    dummy_product_1 = 0
    dummy_product_2 = 0
    
    # Set dummy to 1 if conditions are met
    if diff_product_1 > 0 and diff_product_1 > diff_product_2:
        dummy_product_1 = 1
    if diff_product_2 > 0 and diff_product_2 > diff_product_1:
        dummy_product_2 = 1
    
    return [dummy_product_1, dummy_product_2]

# Apply the logic to each column (agent)
df_dummies = df_reservation.apply(calculate_dummies, axis=0, result_type="expand")
df_dummies.index = ["Product 1", "Product 2"]
df_dummies.columns = df_reservation.columns

# Calculate profit metric: margin * sum of dummies
df_profit = df_margins["Margin"] * df_dummies.sum(axis=1)
df_tprofit = df_profit.sum() * 10
df_profit = pd.DataFrame({"Profit": df_profit})
df_profit["Total Profit"] = df_tprofit

# Display the DataFrames
print("Reservation Values DataFrame:")
print(df_reservation)
print("\nTarget Values DataFrame:")
print(df_target)
print("\nCosts DataFrame:")
print(df_costs)
print("\nMargins DataFrame:")
print(df_margins)
print("\nDummies DataFrame:")
print(df_dummies)
print("\nProfit DataFrame:")
print(df_profit)