import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from sklearn.metrics import mean_squared_error, r2_score
import os
import json
from datetime import datetime


class CoffeeHarvestTester:
    """
    A class for testing and evaluating the coffee harvest prediction model.

    This class provides functionality to:
    1. Run the full prediction pipeline
    2. Test different rainfall scenarios
    3. Evaluate model performance
    4. Generate visualizations
    5. Save and load results
    """

    def __init__(self, api_key=None):
        """
        Initialize the tester with default parameters.

        Args:
            api_key: Alpha Vantage API key for coffee price data
        """
        self.results = {}
        self.config = {
            'drought_threshold': 30,  # Inches
            'optimal_min': 60,  # Inches
            'optimal_max': 90,  # Inches
            'num_samples': 2000,
            'tuning_steps': 1000,
            'random_seed': 42,
            'alpha_vantage_api_key': "G80Z51WM9FY5QSST"  # Store API key in config
        }
        self.output_dir = 'results'
        os.makedirs(self.output_dir, exist_ok=True)

    def run_prediction(self, forecast_rainfall=None, forecast_coffee_price=None, save_results=True):
        """
        Run the full prediction pipeline and store results.

        Args:
            forecast_rainfall: Rainfall forecast value in inches (or None)
            forecast_coffee_price: Coffee price forecast in cents per pound (or None)
            save_results: Whether to save results to disk

        Returns:
            Dictionary of results
        """
        from data_collection import (
            get_brazil_weather_data,
            get_coffee_price_data,
            engineer_features,
            build_mcmc_model,
            predict_harvest,
            prepare_seasonal_features
        )

        print("=" * 80)
        print(f"Running coffee harvest prediction test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # 1. Load data
        print("\n1. Loading weather and coffee price data...")
        weather_data, annual_rainfall = get_brazil_weather_data()

        # Use the Alpha Vantage API key from config if available
        if self.config['alpha_vantage_api_key']:
            coffee_data, coffee_detailed = get_coffee_price_data(self.config['alpha_vantage_api_key'])
            print(f"Using Alpha Vantage API for coffee price data")
        else:
            coffee_data, coffee_detailed = get_coffee_price_data(None)  # Will generate synthetic data
            print(f"No API key provided - using synthetic coffee price data")

        # 2. Engineer features
        print("\n2. Engineering features...")
        features, quarterly_rainfall, coffee_prices = engineer_features(
            weather_data, coffee_data, annual_rainfall
        )

        # 2.5. Add seasonal features
        seasonal_features = prepare_seasonal_features(weather_data)
        features = pd.merge(
            features,
            seasonal_features,
            left_on='year',
            right_index=True,
            how='left'
        )

        # 3. Build and train model
        print("\n3. Building MCMC model...")
        model, trace = build_mcmc_model(features)

        # 4. Make predictions if forecast is provided
        predictions = None
        if forecast_rainfall is not None:
            print(f"\n4. Predicting coffee harvest for {forecast_rainfall} inches of rainfall...")

            # Create a proper climate forecast dictionary
            # Distribute the rainfall across seasons - here's a simple approach
            climate_forecast = {
                'late_grow_rain': forecast_rainfall * 0.3,  # 30% in late growing season
                'flower_rain': forecast_rainfall * 0.2,  # 20% in flowering season
                'harvest_rain': forecast_rainfall * 0.5,  # 50% in harvest season
            }

            # Include temperature if needed
            # climate_forecast['harvest_min_temp'] = 20  # Example value

            # Include coffee price in prediction if provided
            if forecast_coffee_price is not None:
                print(f"   Including coffee price forecast: ${forecast_coffee_price / 100:.2f}/lb")
                # Note: Your predict_harvest doesn't seem to take coffee_price as input
                predictions = predict_harvest(model, trace, climate_forecast)
            else:
                predictions = predict_harvest(model, trace, climate_forecast)

        # 5. Store results
        result = {
            'model': model,
            'trace': trace,
            'features': features,
            'predictions': predictions,
            'forecast_rainfall': forecast_rainfall,
            'forecast_coffee_price': forecast_coffee_price,
            'annual_rainfall': annual_rainfall,
            'coffee_data': coffee_data,
            'coffee_detailed': coffee_detailed,
            'run_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results[run_id] = result

        # 6. Save results if requested
        if save_results:
            self._save_results(run_id)

        return result

    def run_multiple_scenarios(self, rainfall_values, coffee_price=None):
        """
        Run predictions for multiple rainfall scenarios.

        Args:
            rainfall_values: List of rainfall values in inches
            coffee_price: Optional coffee price for all scenarios

        Returns:
            Dictionary mapping rainfall values to predicted yields
        """
        predictions = {}

        for rainfall in rainfall_values:
            print(f"\nTesting rainfall scenario: {rainfall} inches")
            result = self.run_prediction(rainfall, coffee_price, save_results=False)
            if result['predictions'] is not None:
                pred_value = result['predictions']
                if isinstance(pred_value, np.ndarray) and len(pred_value) > 0:
                    predictions[rainfall] = pred_value[0]
                else:
                    predictions[rainfall] = pred_value

        # Generate and save plot
        self._plot_rainfall_scenarios(predictions, coffee_price)

        return predictions

    def run_price_scenarios(self, coffee_prices, rainfall=65):
        """
        Run predictions for multiple coffee price scenarios.

        Args:
            coffee_prices: List of coffee prices in cents per pound
            rainfall: Rainfall value to use for all scenarios (default: 65 inches)

        Returns:
            Dictionary mapping coffee prices to predicted yields
        """
        predictions = {}

        for price in coffee_prices:
            print(f"\nTesting coffee price scenario: ${price / 100:.2f}/lb with {rainfall} inches rainfall")
            result = self.run_prediction(rainfall, price, save_results=False)
            if result['predictions'] is not None:
                pred_value = result['predictions']
                if isinstance(pred_value, np.ndarray) and len(pred_value) > 0:
                    predictions[price] = pred_value[0]
                else:
                    predictions[price] = pred_value

        # Generate and save plot
        self._plot_price_scenarios(predictions, rainfall)

        return predictions

    def evaluate_model(self, run_id=None):
        """
        Evaluate the model performance and generate diagnostic plots.

        Args:
            run_id: ID of the run to evaluate (uses last run if None)

        Returns:
            Dictionary of evaluation metrics
        """
        if run_id is None:
            if not self.results:
                print("No results available. Please run a prediction first.")
                return None
            run_id = list(self.results.keys())[-1]

        if run_id not in self.results:
            print(f"Run ID {run_id} not found in results.")
            return None

        result = self.results[run_id]
        trace = result['trace']
        features = result['features']

        # 1. Model summary
        # Update variable names to match current model structure
        var_names = ['yield_intercept', 'yield_late_rain', 'sigma_yield']

        # Add optional parameters if they exist
        if 'yield_flower_rain' in trace.posterior:
            var_names.append('yield_flower_rain')

        if 'yield_early_rain' in trace.posterior:
            var_names.append('yield_early_rain')
        # Add coffee price coefficient if it exists
        if 'coffee_price_coef' in trace.posterior:
            var_names.append('coffee_price_coef')

        # 2. Evaluate drought impact
        drought_effect = None
        if 'drought_penalty' in trace.posterior:
            drought_effect = trace.posterior['drought_penalty'].values.mean()
            print(f"\nEstimated drought penalty on yield: {drought_effect:.2f} tons per hectare")

        # 3. Evaluate coffee price impact
        coffee_price_effect = None
        if 'coffee_price_coef' in trace.posterior:
            coffee_price_effect = trace.posterior['coffee_price_coef'].values.mean()
            print(f"\nEstimated coffee price effect: {coffee_price_effect:.4f}")

            # Calculate and explain the effect of a 10% price increase
            if 'coffee_price' in features.columns:
                avg_price = features['coffee_price'].mean()
                price_std = features['coffee_price'].std()
                price_change = 0.1 * avg_price  # 10% increase

                # Scale the same way as in the model
                scaled_effect = coffee_price_effect * (price_change / price_std)

                print(f"A 10% increase in coffee prices (${price_change / 100:.2f}/lb) is estimated to")
                print(f"change yield by {scaled_effect:.3f} tons/hectare")

        summary = az.summary(trace, var_names=var_names)
        print("\nModel Summary:")
        print(summary)

        # Update other parameter references
        drought_effect = None
        if 'drought_penalty' in trace.posterior:
            drought_effect = trace.posterior['drought_penalty'].values.mean()
            print(f"\nEstimated drought penalty on yield: {drought_effect:.2f} tons per hectare")

        # Update rainfall effect reference
        rainfall_effect = trace.posterior['yield_late_rain'].values.mean()

        # Update metrics dictionary
        metrics = {
            'model_summary': summary.to_dict(),
            'drought_effect': drought_effect,
            'coffee_price_effect': coffee_price_effect if 'coffee_price_effect' in locals() else None,
            'rainfall_effect': rainfall_effect,
            'run_id': run_id
        }
        return metrics

    def plot_results(self, run_id=None, save_plots=True):
        """
        Create visualizations of the model results.

        Args:
            run_id: ID of the run to visualize (uses last run if None)
            save_plots: Whether to save plots to disk

        Returns:
            Dictionary of figure handles
        """
        if run_id is None:
            if not self.results:
                print("No results available. Please run a prediction first.")
                return None
            run_id = list(self.results.keys())[-1]

        if run_id not in self.results:
            print(f"Run ID {run_id} not found in results.")
            return None

        result = self.results[run_id]
        features = result['features']
        trace = result['trace']
        predictions = result['predictions']
        coffee_detailed = result.get('coffee_detailed', None)

        figures = {}

        # Plot 1: Rainfall vs. Yield scatter with model fit
        fig1, ax1 = plt.subplots(figsize=(12, 8))

        # Check if altitude is in features
        if 'altitude' in features.columns:
            # Add station name as hover text using scatter plot
            scatter = ax1.scatter(
                features['precipitation_inches'],
                features['yield_tons_per_hectare'],
                c=features['altitude'],
                cmap='viridis',
                alpha=0.7,
                s=50
            )

            # Add colorbar for altitude
            cbar = fig1.colorbar(scatter)
            cbar.set_label('Altitude (meters)')
        else:
            # Simple scatter plot without altitude coloring
            ax1.scatter(
                features['precipitation_inches'],
                features['yield_tons_per_hectare'],
                alpha=0.7,
                s=50
            )

        # Generate rainfall range for prediction curve
        rainfall_range = np.linspace(10, 120, 100)

        # Get posterior values with proper variable names
        intercept = trace.posterior['yield_intercept'].values.mean()
        rainfall_linear = trace.posterior['yield_late_rain'].values.mean()

        # Check if the quadratic term exists, otherwise set it to zero
        rainfall_quad = 0.0
        if 'rainfall_quad' in trace.posterior:
            rainfall_quad = trace.posterior['rainfall_quad'].values.mean()

        # Calculate predicted yield across rainfall range - linear model
        predicted_yield = intercept + rainfall_linear * rainfall_range

        # Add quadratic term if it exists
        if 'rainfall_quad' in trace.posterior:
            predicted_yield += rainfall_quad * (rainfall_range - 75) ** 2

        # Add effect from other rainfall periods if they exist
        if 'yield_flower_rain' in trace.posterior:
            flower_rain_coef = trace.posterior['yield_flower_rain'].values.mean()
            # Use the mean flower rain value as a reasonable constant
            if 'flower_rain' in features.columns:
                avg_flower_rain = features['flower_rain'].mean()
                predicted_yield += flower_rain_coef * avg_flower_rain

        # Add early growing season rainfall effect if present
        if 'yield_early_rain' in trace.posterior:
            early_rain_coef = trace.posterior['yield_early_rain'].values.mean()
            if 'early_grow_rain' in features.columns:
                avg_early_rain = features['early_grow_rain'].mean()
                predicted_yield += early_rain_coef * avg_early_rain

        # Check for drought effect with correct variable name
        if 'drought_penalty' in trace.posterior:
            drought_penalty = trace.posterior['drought_penalty'].values.mean()
            predicted_yield = np.where(
                rainfall_range < self.config['drought_threshold'],
                predicted_yield + drought_penalty,
                predicted_yield
            )

        # Plot prediction curve
        ax1.plot(rainfall_range, predicted_yield, 'r-', linewidth=2, label='Model Prediction')

        # Add optimal rainfall band
        ax1.axvspan(self.config['optimal_min'], self.config['optimal_max'],
                    alpha=0.2, color='green',
                    label=f"Optimal Rainfall ({self.config['optimal_min']}-{self.config['optimal_max']} inches)")
        ax1.axvspan(0, self.config['drought_threshold'],
                    alpha=0.2, color='red', label=f"Drought Risk (<{self.config['drought_threshold']} inches)")

        # Add forecast point if available
        if predictions is not None and result['forecast_rainfall'] is not None:
            pred_value = predictions
            if isinstance(pred_value, dict) and 'yield_tons_per_hectare' in pred_value:
                pred_value = pred_value['yield_tons_per_hectare']
            elif isinstance(pred_value, np.ndarray) and len(pred_value) > 0:
                pred_value = pred_value[0]

            ax1.plot(result['forecast_rainfall'], pred_value, 'ro',
                     markersize=10, label=f"Forecast: {result['forecast_rainfall']} inches")

            # Add coffee price info if available
            if result['forecast_coffee_price'] is not None:
                ax1.text(result['forecast_rainfall'], pred_value + 0.1,
                         f"Coffee Price: ${result['forecast_coffee_price'] / 100:.2f}/lb",
                         ha='center', va='bottom',
                         bbox=dict(facecolor='white', alpha=0.5))

        ax1.set_xlabel('Annual Rainfall (inches)', fontsize=12)
        ax1.set_ylabel('Coffee Yield (tons per hectare)', fontsize=12)
        ax1.set_title('Coffee Yield vs. Annual Rainfall in Minas Gerais', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        figures['yield_vs_rainfall'] = fig1

        # Plot 2: Altitude effect on yield (if altitude data is available)
        if 'altitude' in features.columns:
            fig2, ax2 = plt.subplots(figsize=(12, 6))

            # Add rainfall as color
            scatter2 = ax2.scatter(
                features['altitude'],
                features['yield_tons_per_hectare'],
                c=features['precipitation_inches'],
                cmap='Blues',
                alpha=0.7,
                s=50
            )

            # Add colorbar for rainfall
            cbar2 = fig2.colorbar(scatter2)
            cbar2.set_label('Annual Rainfall (inches)')

            # Highlight optimal altitude range
            ax2.axvspan(600, 1200, alpha=0.2, color='green', label="Optimal Altitude (600-1200m)")

            ax2.set_xlabel('Altitude (meters)', fontsize=12)
            ax2.set_ylabel('Coffee Yield (tons per hectare)', fontsize=12)
            ax2.set_title('Coffee Yield vs. Altitude in Minas Gerais', fontsize=14)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

            figures['yield_vs_altitude'] = fig2

        # Plot 3: Station comparison (if station_name is available)
        if 'station_name' in features.columns:
            fig3, ax3 = plt.subplots(figsize=(14, 8))

            # Calculate average yield by station
            station_yields = features.groupby('station_name')['yield_tons_per_hectare'].mean().sort_values(
                ascending=False)
            station_rainfall = features.groupby('station_name')['precipitation_inches'].mean()

            # Create combined dataframe
            station_data = pd.DataFrame({
                'yield': station_yields,
                'rainfall': station_rainfall
            })

            if 'altitude' in features.columns:
                station_altitude = features.groupby('station_name')['altitude'].first()
                station_data['altitude'] = station_altitude

            # Create bar plot
            bars = ax3.bar(station_data.index, station_data['yield'], alpha=0.7)

            # Color bars by rainfall
            rainfall_norm = (station_data['rainfall'] - station_data['rainfall'].min()) / \
                            (station_data['rainfall'].max() - station_data['rainfall'].min())
            for i, bar in enumerate(bars):
                bar.set_color(plt.cm.Blues(0.2 + 0.6 * rainfall_norm.iloc[i]))

            # Add altitude as text if available
            if 'altitude' in station_data.columns:
                for i, station in enumerate(station_data.index):
                    ax3.text(i, station_data.loc[station, 'yield'] + 0.05,
                             f"{int(station_data.loc[station, 'altitude'])}m",
                             ha='center', rotation=90, alpha=0.7)

            ax3.set_xlabel('Station', fontsize=12)
            x_positions = np.arange(len(station_data.index))
            ax3.set_xticks(x_positions)
            ax3.set_xticklabels(station_data.index, rotation=45, ha='right')
            ax3.set_ylabel('Average Coffee Yield (tons per hectare)', fontsize=12)
            ax3.set_title('Average Coffee Yield by Station in Minas Gerais', fontsize=14)
            ax3.set_xticklabels(station_data.index, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3, axis='y')

            # Add a legend for rainfall
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(
                station_data['rainfall'].min(), station_data['rainfall'].max()))
            sm.set_array([])
            cbar3 = fig3.colorbar(sm, ax=ax3)

            cbar3.set_label('Average Annual Rainfall (inches)')

            figures['station_comparison'] = fig3

        # Plot 4: Coffee Price vs. Yield (if coffee_price is available)
        if 'coffee_price' in features.columns and 'coffee_price_coef' in trace.posterior:
            fig4, ax4 = plt.subplots(figsize=(12, 6))

            # Add rainfall as color
            scatter4 = ax4.scatter(
                features['coffee_price'],
                features['yield_tons_per_hectare'],
                c=features['precipitation_inches'],
                cmap='Blues',
                alpha=0.7,
                s=50
            )

            # Add colorbar for rainfall
            cbar4 = fig4.colorbar(scatter4, ax=ax4)
            cbar4.set_label('Annual Rainfall (inches)')

            # Add regression line
            coffee_price_range = np.linspace(
                features['coffee_price'].min() * 0.9,
                features['coffee_price'].max() * 1.1,
                100
            )

            # Use the mean rainfall for the projection
            mean_rainfall = features['precipitation_inches'].mean()
            coffee_price_coef = trace.posterior['coffee_price_coef'].values.mean()

            # Calculate expected yield at mean rainfall - handle with/without quadratic term
            expected_yield_at_mean = intercept + rainfall_linear * mean_rainfall
            if 'rainfall_quad' in trace.posterior:
                expected_yield_at_mean += rainfall_quad * (mean_rainfall - 75) ** 2

            # Calculate coffee price effect
            coffee_mean = features['coffee_price'].mean()
            coffee_std = features['coffee_price'].std()
            price_effect = coffee_price_coef * (coffee_price_range - coffee_mean) / coffee_std

            # Add regression line
            ax4.plot(coffee_price_range, expected_yield_at_mean + price_effect,
                     'r-', linewidth=2, label='Price Effect (at avg rainfall)')

            # Add forecast point if available
            if predictions is not None and result['forecast_coffee_price'] is not None:
                pred_value = predictions
                if isinstance(pred_value, dict) and 'yield_tons_per_hectare' in pred_value:
                    pred_value = pred_value['yield_tons_per_hectare']
                elif isinstance(pred_value, np.ndarray) and len(pred_value) > 0:
                    pred_value = pred_value[0]

                ax4.plot(result['forecast_coffee_price'], pred_value, 'ro',
                         markersize=10, label=f"Forecast: ${result['forecast_coffee_price'] / 100:.2f}/lb")

            ax4.set_xlabel('Coffee Price (cents per pound)', fontsize=12)
            ax4.set_ylabel('Coffee Yield (tons per hectare)', fontsize=12)
            ax4.set_title('Coffee Yield vs. Price', fontsize=14)
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3)

            figures['yield_vs_price'] = fig4

        # Plot 5: Coffee Price Historical Trend (if available)
        if coffee_detailed is not None and not coffee_detailed.empty:
            fig5, ax5 = plt.subplots(figsize=(14, 6))

            # Plot coffee price trend
            ax5.plot(coffee_detailed.index, coffee_detailed['value'], 'g-', linewidth=2)

            # Add markers for drought years
            current_year = pd.Timestamp.now().year
            drought_years = [current_year - 3, current_year - 1]

            for year in drought_years:
                year_data = coffee_detailed[coffee_detailed.index.year == year]
                if not year_data.empty:
                    ax5.axvspan(
                        pd.Timestamp(f"{year}-01-01"),
                        pd.Timestamp(f"{year}-12-31"),
                        alpha=0.2, color='red', label=f"Drought Year ({year})"
                    )

            # Remove duplicate labels
            handles, labels = ax5.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax5.legend(by_label.values(), by_label.keys())

            ax5.set_xlabel('Date', fontsize=12)
            ax5.set_ylabel('Coffee Price (cents per pound)', fontsize=12)
            ax5.set_title('Historical Coffee Price Trend with Drought Years Highlighted', fontsize=14)
            ax5.grid(True, alpha=0.3)

            figures['coffee_price_trend'] = fig5

        # Save figures if requested
        if save_plots:
            for name, fig in figures.items():
                fig.tight_layout()
                fig.savefig(os.path.join(self.output_dir, f"{run_id}_{name}.png"))
                print(f"Saved figure: {run_id}_{name}.png")

        plt.show()
        return figures

    def calculate_financial_impact(self, rainfall_forecast, farm_size_hectares=20, coffee_price_forecast=None,
                                   run_id=None):
        """
        Calculate the financial impact of a rainfall forecast.
        """
        # Run the prediction if not already done
        if run_id is None or run_id not in self.results:
            result = self.run_prediction(rainfall_forecast, coffee_price_forecast)
        else:
            result = self.results[run_id]

        # Get the necessary data
        trace = result['trace']
        features = result['features']
        coffee_data = result['coffee_data']

        # Print debug info
        print(f"Debug - Result predictions type: {type(result['predictions'])}")
        print(f"Debug - Result predictions value: {result['predictions']}")

        # Calculate current coffee price ($/lb)
        if coffee_price_forecast is not None:
            current_price = coffee_price_forecast / 100  # Convert cents to dollars
        else:
            current_price = coffee_data['PX_LAST'].iloc[-1] / 100  # Convert cents to dollars

        # Calculate average yield in normal conditions
        if len(features) > 0 and 'drought' in features.columns and (~features['drought']).any():
            normal_yield = features.loc[~features['drought'], 'yield_tons_per_hectare'].mean()
            if pd.isna(normal_yield):
                normal_yield = 1.8  # Default if mean is NaN
        else:
            # If all data points are drought conditions, use a reasonable default
            normal_yield = 1.8  # Typical yield for coffee

        # Define reference yield - either average of non-drought data or industry standard
        reference_yield = normal_yield if normal_yield > 1.0 else 1.8
        print(f"Using reference yield of {reference_yield:.2f} tons/hectare")
        predicted_yield_data = result['predictions']

        # Extract predicted yield with robust error handling
        if isinstance(predicted_yield_data, dict):
            # Extract the yield value from the dictionary
            predicted_yield = predicted_yield_data.get('yield_tons_per_hectare', 0)
        elif isinstance(predicted_yield_data, np.ndarray) and len(predicted_yield_data) > 0:
            predicted_yield = predicted_yield_data[0]
        else:
            predicted_yield = 0

        # Ensure predicted_yield is a float and handle NaN
        try:
            predicted_yield = float(predicted_yield)
            if np.isnan(predicted_yield):
                print("Warning: Predicted yield is NaN, using 90% of reference yield as fallback")
                predicted_yield = reference_yield * 0.9
        except (ValueError, TypeError):
            print(f"Warning: Could not convert predicted yield to float: {predicted_yield}")
            predicted_yield = reference_yield * 0.9

        print(f"Debug - Final predicted_yield value: {predicted_yield}")

        # Now calculate impact based on difference from reference yield
        yield_diff = reference_yield - predicted_yield

        # Calculate financial impact
        tons_to_pounds = 2204.62  # Conversion factor (1 metric ton = 2,204.62 pounds)
        financial_impact_per_hectare = yield_diff * tons_to_pounds * current_price
        total_financial_impact = financial_impact_per_hectare * farm_size_hectares

        # Calculate yield reduction percentage safely
        if normal_yield > 0:
            yield_reduction_percent = 100 * yield_diff / normal_yield
        else:
            yield_reduction_percent = 0

        # Create impact report
        impact = {
            'normal_yield': normal_yield,
            'predicted_yield': predicted_yield,
            'yield_reduction': yield_diff,
            'yield_reduction_percent': yield_reduction_percent,
            'current_coffee_price': current_price,
            'financial_impact_per_hectare': financial_impact_per_hectare,
            'farm_size_hectares': farm_size_hectares,
            'total_financial_impact': total_financial_impact,
            'rainfall_forecast': rainfall_forecast,
            'coffee_price_forecast': coffee_price_forecast
        }

        # Print report
        print("\n" + "=" * 50)
        print("FINANCIAL IMPACT ANALYSIS")
        print("=" * 50)
        print(f"Rainfall Forecast: {rainfall_forecast} inches")
        print(f"Coffee Price: ${current_price:.2f}/lb")
        print(f"Average Normal Yield: {normal_yield:.2f} tons/hectare")
        print(f"Predicted Yield: {predicted_yield:.2f} tons/hectare")
        print(f"Yield Reduction: {yield_diff:.2f} tons/hectare ({yield_reduction_percent:.1f}%)")
        print(f"Financial Impact: ${financial_impact_per_hectare:,.2f} per hectare")
        print(f"Total Impact for {farm_size_hectares}ha farm: ${total_financial_impact:,.2f}")
        print("=" * 50)
        return impact

    def _save_results(self, run_id):
        """Save results to disk."""
        result = self.results[run_id]

        # Create directory for this run
        run_dir = os.path.join(self.output_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        # Save metadata
        metadata = {
            'run_id': run_id,
            'timestamp': result['run_time'],
            'forecast_rainfall': result['forecast_rainfall'],
            'forecast_coffee_price': result.get('forecast_coffee_price', None),
            'config': self.config
        }

        with open(os.path.join(run_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save trace as netCDF
        result['trace'].to_netcdf(os.path.join(run_dir, 'trace.nc'))

        # Save features
        result['features'].to_csv(os.path.join(run_dir, 'features.csv'))

        # Save predictions
        if result['predictions'] is not None:
            np.save(os.path.join(run_dir, 'predictions.npy'), result['predictions'])

        print(f"Results saved to {run_dir}")

    def _plot_rainfall_scenarios(self, predictions, coffee_price=None):
        """
        Plot the results of multiple rainfall scenarios.

        Args:
            predictions: Dictionary mapping rainfall values to yield predictions
            coffee_price: Optional coffee price used for all scenarios
        """
        if not predictions:
            print("No predictions to plot.")
            return

        # Sort predictions by rainfall
        rainfall_values = sorted(predictions.keys())
        yield_values = [predictions[r] for r in rainfall_values]

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot yield vs rainfall
        ax.plot(rainfall_values, yield_values, 'o-', linewidth=2, markersize=8)

        # Add optimal rainfall band
        ax.axvspan(self.config['optimal_min'], self.config['optimal_max'],
                   alpha=0.2, color='green',
                   label=f"Optimal Rainfall ({self.config['optimal_min']}-{self.config['optimal_max']} inches)")
        ax.axvspan(0, self.config['drought_threshold'],
                   alpha=0.2, color='red', label=f"Drought Risk (<{self.config['drought_threshold']} inches)")

        # Customize plot
        ax.set_xlabel('Annual Rainfall (inches)', fontsize=12)
        ax.set_ylabel('Predicted Coffee Yield (tons per hectare)', fontsize=12)
        title = 'Coffee Yield Predictions for Different Rainfall Scenarios'
        if coffee_price is not None:
            title += f" (Coffee Price: ${coffee_price / 100:.2f}/lb)"
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add value labels
        for i, (rainfall, yield_val) in enumerate(zip(rainfall_values, yield_values)):
            ax.annotate(f"{yield_val:.2f}",
                        xy=(rainfall, yield_val),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center')

        # Save plot
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(os.path.join(self.output_dir, f'scenarios_{timestamp}.png'))

        plt.show()