import numpy as np
import pandas as pd

class ProfitCalculator:
    def __init__(self, reservation_values, target_values, costs):
        self.df_reservation = reservation_values
        self.df_target = target_values
        self.df_costs = costs
        self.baseline_buy_signals = None
        self.baseline_profit = None
        self.optimal_prices = None
        self.optimal_buy_signals = []
        self.optimal_profit = []

    def calculate_baseline(self):
        # Calculate baseline buy signals
        def calculate_buy_signals(column):
            return [
                1 if column["Product 1"] > self.df_target.loc["Product 1", "Target Value"] else 0,
                1 if column["Product 2"] > self.df_target.loc["Product 2", "Target Value"] else 0,
            ]

        self.baseline_buy_signals = self.df_reservation.apply(
            calculate_buy_signals, axis=0, result_type="expand"
        )
        self.baseline_buy_signals.index = ["Product 1", "Product 2"]
        self.baseline_buy_signals.columns = self.df_reservation.columns

        # Calculate baseline profit
        baseline_margins = self.df_target["Target Value"] - self.df_costs["Cost"]
        self.baseline_profit = baseline_margins * self.baseline_buy_signals.sum(axis=1)

    def calculate_optimal(self, periods=5):
        # Initialize optimal prices with target values for each agent
        self.optimal_prices = pd.DataFrame(
            np.tile(self.df_target["Target Value"].values, (periods, len(self.df_reservation.columns))).reshape(
                periods * len(self.df_reservation.index), len(self.df_reservation.columns)
            ),
            index=pd.MultiIndex.from_product([range(periods), self.df_reservation.index], names=["Period", "Product"]),
            columns=self.df_reservation.columns,
        )

        for period in range(periods):
            # Calculate buy signals for the current period
            def calculate_buy_signals(column):
                diff_product_1 = column["Product 1"] - self.optimal_prices.loc[(period, "Product 1"), column.name]
                diff_product_2 = column["Product 2"] - self.optimal_prices.loc[(period, "Product 2"), column.name]

                buy_signal_product_1 = 1 if diff_product_1 > 0 and diff_product_1 > diff_product_2 else 0
                buy_signal_product_2 = 1 if diff_product_2 > 0 and diff_product_2 > diff_product_1 else 0

                return [buy_signal_product_1, buy_signal_product_2]

            buy_signals = self.df_reservation.apply(
                calculate_buy_signals, axis=0, result_type="expand"
            )
            buy_signals.index = ["Product 1", "Product 2"]
            buy_signals.columns = self.df_reservation.columns
            self.optimal_buy_signals.append(buy_signals)

            # Align the indices of buy_signals to match optimal_margins
            buy_signals = buy_signals.stack().reset_index()
            buy_signals.columns = ["Product", "Agent", "Buy Signal"]
            buy_signals["Period"] = period
            buy_signals = buy_signals.set_index(["Period", "Product", "Agent"])["Buy Signal"]

            # Calculate profit for the current period
            optimal_margins = self.optimal_prices.loc[(period, slice(None)), :] - self.df_costs["Cost"].values[:, None]
            optimal_margins = optimal_margins.stack().reset_index(name="Margin")
            optimal_margins = optimal_margins.rename(columns={"level_2": "Agent"})  # Rename level_2 to Agent
            optimal_margins["Period"] = period
            optimal_margins = optimal_margins.set_index(["Period", "Product", "Agent"])["Margin"]

            # Multiply margins by buy signals
            profit = (optimal_margins * buy_signals).groupby(level=["Period", "Product"]).sum()
            self.optimal_profit.append(profit)

            # Update prices for the next period
            if period < periods - 1:
                for product in ["Product 1", "Product 2"]:
                    for agent in self.df_reservation.columns:
                        current_price = self.optimal_prices.loc[(period, product), agent]
                        if buy_signals.loc[(period, product, agent)] == 0:
                            # Decrease price by half the distance to 1
                            self.optimal_prices.loc[(period + 1, product), agent] = current_price - (current_price - 1) / 2
                        else:
                            # Increase price by half the distance to 2
                            self.optimal_prices.loc[(period + 1, product), agent] = current_price + (2 - current_price) / 2

    def display_results(self):
        print("Baseline Buy Signals:")
        print(self.baseline_buy_signals)
        print("\nBaseline Profit:")
        print(self.baseline_profit)
        print("\nOptimal Prices:")
        print(self.optimal_prices)
        print("\nOptimal Buy Signals (Last Period):")
        print(self.optimal_buy_signals[-1])
        print("\nOptimal Profit (Last Period):")
        print(self.optimal_profit[-1])


# Example Usage
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

# Initialize and calculate profits
calculator = ProfitCalculator(df_reservation, df_target, df_costs)
calculator.calculate_baseline()
calculator.calculate_optimal(periods=5)
calculator.display_results()