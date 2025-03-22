import argparse
from coffee_tester import CoffeeHarvestTester


def main():
    parser = argparse.ArgumentParser(description='Run Coffee Harvest Prediction Tests')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Single forecast command
    forecast_parser = subparsers.add_parser('forecast', help='Run a single rainfall forecast')
    forecast_parser.add_argument('--rainfall', type=float, required=True, help='Rainfall forecast in inches')
    forecast_parser.add_argument('--farm_size', type=float, default=20.0, help='Farm size in hectares')

    # Multiple scenarios command
    scenario_parser = subparsers.add_parser('scenarios', help='Run multiple rainfall scenarios')
    scenario_parser.add_argument('--min', type=float, default=15.0, help='Minimum rainfall in inches')
    scenario_parser.add_argument('--max', type=float, default=105.0, help='Maximum rainfall in inches')
    scenario_parser.add_argument('--steps', type=int, default=10, help='Number of steps between min and max')

    # Financial analysis command
    financial_parser = subparsers.add_parser('financial', help='Run financial impact analysis')
    financial_parser.add_argument('--rainfall', type=float, required=True, help='Rainfall forecast in inches')
    financial_parser.add_argument('--farm_size', type=float, default=20.0, help='Farm size in hectares')
    financial_parser.add_argument('--price_adjustment', type=float, default=0.0,
                                  help='Price adjustment in percentage (e.g., 10 for 10% increase)')

    # Model evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('--run_id', type=str, help='Specific run ID to evaluate')

    # Parse args
    args = parser.parse_args()

    # Create tester
    tester = CoffeeHarvestTester()

    # Execute command
    if args.command == 'forecast':
        # Run single forecast
        print(f"Running forecast with {args.rainfall} inches of rainfall...")
        result = tester.run_prediction(args.rainfall)

        # Plot results
        #tester.plot_results()

        # Calculate financial impact
        tester.calculate_financial_impact(args.rainfall, args.farm_size)

    elif args.command == 'scenarios':
        # Run multiple scenarios
        import numpy as np
        rainfall_values = np.linspace(args.min, args.max, args.steps)
        print(f"Running {args.steps} scenarios from {args.min} to {args.max} inches...")
        tester.run_multiple_scenarios(rainfall_values)

    elif args.command == 'financial':
        # Run financial analysis
        print(f"Running financial analysis for {args.rainfall} inches of rainfall...")
        result = tester.run_prediction(args.rainfall)

        # Calculate base financial impact
        base_impact = tester.calculate_financial_impact(args.rainfall, args.farm_size)

        # If price adjustment provided, run additional analysis
        if args.price_adjustment != 0:
            print(f"\nAnalyzing with {args.price_adjustment}% price adjustment...")
            # We can't directly adjust the price in the tester without modifying it,
            # so we'll calculate the adjustment manually
            price_factor = 1 + (args.price_adjustment / 100)
            adjusted_impact = base_impact['total_financial_impact'] * price_factor

            print(f"Original Impact: ${base_impact['total_financial_impact']:,.2f}")
            print(f"Adjusted Impact: ${adjusted_impact:,.2f}")
            print(f"Difference: ${adjusted_impact - base_impact['total_financial_impact']:,.2f}")

    elif args.command == 'evaluate':
        # Evaluate model
        tester.evaluate_model(args.run_id)
        tester.plot_results(args.run_id)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()