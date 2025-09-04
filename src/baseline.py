import numpy as np
import pandas as pd


class ProfitCalculator:
    def __init__(self, reservation_values, target_values, costs):
        self.df_reservation = reservation_values
        self.df_target = target_values
        self.df_costs = costs

    def calculate_buy_signals(self, prices, period):
        """Calculate buy signals for a given period."""
        def calculate_signals(column):
            diff_product_1 = column["Product 1"] - prices.loc[(period, "Product 1"), column.name]
            diff_product_2 = column["Product 2"] - prices.loc[(period, "Product 2"), column.name]

            buy_signal_product_1 = 1 if diff_product_1 > 0 and diff_product_1 > diff_product_2 else 0
            buy_signal_product_2 = 1 if diff_product_2 > 0 and diff_product_2 > diff_product_1 else 0

            return [buy_signal_product_1, buy_signal_product_2]

        buy_signals = self.df_reservation.apply(calculate_signals, axis=0, result_type="expand")
        buy_signals.index = ["Product 1", "Product 2"]
        buy_signals.columns = self.df_reservation.columns

        # Reshape buy signals for easier processing
        buy_signals = buy_signals.stack().reset_index()
        buy_signals.columns = ["Product", "Agent", "Buy Signal"]
        buy_signals["Period"] = period
        return buy_signals.set_index(["Period", "Product", "Agent"])["Buy Signal"]

    def calculate_margins(self, prices, period):
        """Calculate margins for a given period."""
        margins = prices.loc[(period, slice(None)), :] - self.df_costs["Cost"].values[:, None]
        margins = margins.stack().reset_index(name="Margin")
        margins = margins.rename(columns={"level_2": "Agent"})  # Rename level_2 to Agent
        margins["Period"] = period
        return margins.set_index(["Period", "Product", "Agent"])["Margin"]

    def calculate_profits(self, buy_signals, margins):
        """Calculate profits by multiplying buy signals and margins."""
        return (margins * buy_signals).groupby(level=["Period", "Product"]).sum()

    def baseline_price_algorithm(self, periods):
        """Baseline price updating algorithm: prices remain fixed."""
        # Ensure target values are aligned as rows for each product
        target_values = self.df_target["Target Value"].values.reshape(-1, 1)

        # Repeat the target values for all agents and periods
        baseline_prices = pd.DataFrame(
            np.tile(target_values, (periods, len(self.df_reservation.columns))).reshape(
                len(self.df_reservation.index) * periods, len(self.df_reservation.columns)
            ),
            index=pd.MultiIndex.from_product([self.df_reservation.index, range(periods)], names=["Product", "Period"]),
            columns=self.df_reservation.columns,
        )

        # Reorder the index to match the desired structure: Period -> Product
        baseline_prices = baseline_prices.reorder_levels(["Period", "Product"]).sort_index()

        return baseline_prices

    def optimal_price_algorithm(self, periods):
        """Optimal price updating algorithm: prices change dynamically."""
        prices = pd.DataFrame(
            np.tile(self.df_target["Target Value"].values, (periods, len(self.df_reservation.columns))).reshape(
                periods * len(self.df_reservation.index), len(self.df_reservation.columns)
            ),
            index=pd.MultiIndex.from_product([range(periods), self.df_reservation.index], names=["Period", "Product"]),
            columns=self.df_reservation.columns,
        )

        for period in range(periods - 1):
            buy_signals = self.calculate_buy_signals(prices, period)  # Dynamically calculate buy_signals
            for product in ["Product 1", "Product 2"]:
                for agent in self.df_reservation.columns:
                    current_price = prices.loc[(period, product), agent]
                    if buy_signals.loc[(period, product, agent)] == 0:
                        # Decrease price by half the distance to 1
                        prices.loc[(period + 1, product), agent] = current_price - (current_price - 1) / 2
                    else:
                        # Increase price by half the distance to 2
                        prices.loc[(period + 1, product), agent] = current_price + (2 - current_price) / 2
        return prices

    def calculate_profits_over_periods(self, price_algorithm, periods):
        """Calculate profits over all periods using a given price algorithm."""
        prices = price_algorithm(periods)
        all_buy_signals = []
        all_profits = []

        for period in range(periods):
            buy_signals = self.calculate_buy_signals(prices, period)
            margins = self.calculate_margins(prices, period)
            profits = self.calculate_profits(buy_signals, margins)

            all_buy_signals.append(buy_signals)
            all_profits.append(profits)

        return all_buy_signals, all_profits


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
target_values = [0.75, 1.5]
df_target = pd.DataFrame({"Target Value": target_values}, index=["Product 1", "Product 2"])

# Define product costs
product_costs = [0.5, 1.0]
df_costs = pd.DataFrame({"Cost": product_costs}, index=["Product 1", "Product 2"])

# Initialize the calculator
calculator = ProfitCalculator(df_reservation, df_target, df_costs)

# Calculate baseline profits
baseline_buy_signals, baseline_profits = calculator.calculate_profits_over_periods(
    calculator.baseline_price_algorithm, periods=2
)

# Calculate optimal profits
optimal_buy_signals, optimal_profits = calculator.calculate_profits_over_periods(
    calculator.optimal_price_algorithm, periods=2
)

# Display results
print("Baseline Profits:")
print(baseline_profits)
print("\nOptimal Profits:")
print(optimal_profits)

# Display reservation values
print("\nReservation Values:")
print(df_reservation)

# Debugging: Print baseline prices
baseline_prices = calculator.baseline_price_algorithm(periods=2)
print("\nBaseline Prices:")
print(baseline_prices)

# Display buy signals for each agent in each period
print("\nBaseline Buy Signals (for each agent, each period):")
for period, buy_signals in enumerate(baseline_buy_signals):
    print(f"Period {period}:")
    print(buy_signals.unstack(level="Agent"))

print("\nOptimal Buy Signals (for each agent, each period):")
for period, buy_signals in enumerate(optimal_buy_signals):
    print(f"Period {period}:")
    print(buy_signals.unstack(level="Agent"))