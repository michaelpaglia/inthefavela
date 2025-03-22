# 1. Data Collection
import pandas as pd
import numpy as np
import requests
from typing import Tuple


def get_brazil_weather_data(api_url: str = "http://localhost:8000",
                            region: str = "Minas Gerais") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch weather data for Brazil coffee regions from the API.

    Args:
        api_url: Base URL for the Brazil Weather Data API
        region: Brazilian state to filter for (e.g., "Minas Gerais")

    Returns:
        Tuple of (weather_data, annual_rainfall)
    """
    print(f"Fetching Brazil weather data for {region} from API...")

    # Define known Minas Gerais weather stations
    minas_stations = [
        {"IdStationWho": "A505", "StationName": "ARAXA", "Latitude": -19.60583333, "Longitude": -46.94972221,
         "Altitude": 1018.32},
        {"IdStationWho": "A513", "StationName": "OURO BRANCO", "Latitude": -20.55666666, "Longitude": -43.75611111,
         "Altitude": 1048.2},
        {"IdStationWho": "A555", "StationName": "IBIRITE (ROLA MOCA)", "Latitude": -20.03138888,
         "Longitude": -44.01111111, "Altitude": 1198.82},
        {"IdStationWho": "A528", "StationName": "TRES MARIAS", "Latitude": -18.200855, "Longitude": -45.459836,
         "Altitude": 931.01},
        {"IdStationWho": "A567", "StationName": "MACHADO", "Latitude": -21.680722, "Longitude": -45.944389,
         "Altitude": 969},
        {"IdStationWho": "A537", "StationName": "DIAMANTINA", "Latitude": -18.231052, "Longitude": -43.648269,
         "Altitude": 1359.25},
        {"IdStationWho": "A509", "StationName": "MONTE VERDE", "Latitude": -22.86166666, "Longitude": -46.04333333,
         "Altitude": 1544.89},
        {"IdStationWho": "A538", "StationName": "CURVELO", "Latitude": -18.747711, "Longitude": -44.453785,
         "Altitude": 669.48},
        {"IdStationWho": "A536", "StationName": "DORES DO INDAIA", "Latitude": -19.48194443, "Longitude": -45.59388888,
         "Altitude": 721.09},
        {"IdStationWho": "A566", "StationName": "ARACUAI", "Latitude": -16.84888888, "Longitude": -42.03527777,
         "Altitude": 308},
        {"IdStationWho": "A511", "StationName": "TIMOTEO", "Latitude": -19.57388888, "Longitude": -42.62249999,
         "Altitude": 493.42},
        {"IdStationWho": "A520", "StationName": "CONCEICAO DAS ALAGOAS", "Latitude": -19.98586, "Longitude": -48.151574,
         "Altitude": 572.54}
    ]

    # Initialize empty list to store all weather records
    all_weather_records = []

    # Get current year for data range calculation
    current_year = pd.Timestamp.now().year

    try:
        start_date = pd.Timestamp("2023-01-01")
        end_date = pd.Timestamp("2024-01-01")

        # Create chunks of 30 days each
        date_chunks = []
        chunk_start = start_date
        while chunk_start < end_date:
            chunk_end = min(chunk_start + pd.Timedelta(days=30), end_date)
            date_chunks.append((
                chunk_start.strftime('%Y-%m-%d'),  # Format: YYYY-MM-DD
                chunk_end.strftime('%Y-%m-%d')  # Format: YYYY-MM-DD
            ))
            chunk_start = chunk_end + pd.Timedelta(days=1)

        print(
            f"Fetching data in {len(date_chunks)} chunks from {start_date.strftime('%Y-%m-%dd')} to {end_date.strftime('%Y-%m-%dd')}")

        # Fetch data for each station and date chunk
        for station in minas_stations:
            station_id = station["IdStationWho"]
            print(f"Fetching data for station {station_id} ({station['StationName']})...")

            station_records = []

            for chunk_start, chunk_end in date_chunks:
                try:
                    # Construct API URL
                    url = f"{api_url}/weather/{station_id}/{chunk_start}/{chunk_end}/"

                    # Make API request
                    response = requests.get(url)

                    # Check if request was successful
                    if response.status_code == 200:
                        # Parse JSON response
                        data = response.json()

                        # Process each record
                        for record in data:
                            # Convert date strings to datetime
                            date = pd.to_datetime(record['Date'], format='ISO8601')

                            for record in data:
                                # Handle ISO8601 date format with time component
                                date_str = record['Date']
                                if 'T' in date_str:
                                    date_str = date_str.split('T')[0]  # Keep only the date part

                                date = pd.to_datetime(date_str, format='%Y-%m-%d')

                                # Create weather record
                                weather_record = {
                                    'date': date,
                                    'station': station_id,
                                    'station_name': station["StationName"],
                                    'latitude': station["Latitude"],
                                    'longitude': station["Longitude"],
                                    'state': region,
                                    'precipitation': record.get('TotalPrecipitation', 0),  # in mm
                                    'temperature': record.get('DryBulbTemperature', None),  # in °C
                                    'altitude': station["Altitude"],
                                    'relative_humidity': record.get('HourlyRelativeHumidity', None)  # in %
                                }

                                station_records.append(weather_record)
                    else:
                        print(f"  Error fetching data: HTTP {response.status_code}")
                        print(f"  Response: {response.text}")

                except Exception as e:
                    print(f"  Error fetching data for {station_id} from {chunk_start} to {chunk_end}: {e}")

            print(f"  Collected {len(station_records)} records for station {station_id}")
            all_weather_records.extend(station_records)

        print(f"Total records collected: {len(all_weather_records)}")

        # If we couldn't get any data from the API, fall back to synthetic data
        if not all_weather_records:
            print("No data received from API. Falling back to synthetic data generation...")
            return _generate_synthetic_weather_data(minas_stations, region)

        # Convert to DataFrame and set date as index
        weather_data = pd.DataFrame(all_weather_records)
        weather_data.set_index('date', inplace=True)

        # Fill missing values
        weather_data['precipitation'] = weather_data['precipitation'].fillna(0)
        weather_data['temperature'] = weather_data['temperature'].fillna(weather_data['temperature'].mean())
        weather_data['relative_humidity'] = weather_data['relative_humidity'].fillna(
            weather_data['relative_humidity'].mean())

        # Add month and year columns for aggregation
        weather_data['month'] = weather_data.index.month
        weather_data['year'] = weather_data.index.year

        # Group by year and station to calculate annual rainfall
        annual_rainfall = weather_data.groupby(['year', 'station', 'station_name', 'altitude'])[
            'precipitation'].sum().reset_index()

        # Convert mm to inches for consistency with the model
        annual_rainfall['precipitation_inches'] = annual_rainfall['precipitation'] / 25.4

        # Add derived features
        annual_rainfall['drought'] = annual_rainfall['precipitation_inches'] < 30
        annual_rainfall['optimal_rainfall'] = (
                (annual_rainfall['precipitation_inches'] >= 60) &
                (annual_rainfall['precipitation_inches'] <= 90)
        )

        # Generate synthetic yield data based on rainfall patterns and altitude
        annual_rainfall = _add_synthetic_yield(annual_rainfall)

        return weather_data, annual_rainfall

    except Exception as e:
        print(f"Error in API data collection: {e}")
        print("Falling back to synthetic data generation...")
        return _generate_synthetic_weather_data(minas_stations, region)


def _add_synthetic_yield(annual_rainfall: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic yield data to the annual rainfall DataFrame based on real rainfall patterns."""
    # Initialize with base yield
    annual_rainfall['yield_tons_per_hectare'] = 1.5 + np.random.normal(0, 0.1, len(annual_rainfall))

    # Altitude effect (higher altitude often produces better coffee in specific ranges)
    altitude_effect = np.zeros(len(annual_rainfall))
    for idx, row in annual_rainfall.iterrows():
        alt = row['altitude']
        if 600 <= alt <= 1200:
            altitude_effect[idx] = 0.2 + np.random.normal(0, 0.05)  # Positive effect
        elif alt > 1200:
            altitude_effect[idx] = 0.0 + np.random.normal(0, 0.05)  # Neutral effect
        else:
            altitude_effect[idx] = -0.1 + np.random.normal(0, 0.05)  # Negative effect

    annual_rainfall['yield_tons_per_hectare'] += altitude_effect

    # Adjustments for rainfall conditions
    for idx, row in annual_rainfall.iterrows():
        rainfall_inches = row['precipitation_inches']

        if rainfall_inches < 30:  # Drought conditions
            # Severe yield penalty for drought
            penalty = 0.8 + np.random.normal(0, 0.1)
            annual_rainfall.at[idx, 'yield_tons_per_hectare'] -= penalty
        elif rainfall_inches > 90:  # Too much rain
            # Moderate yield penalty for excessive rain
            penalty = 0.3 + np.random.normal(0, 0.1)
            annual_rainfall.at[idx, 'yield_tons_per_hectare'] -= penalty
        elif 60 <= rainfall_inches <= 90:  # Optimal conditions
            # Yield bonus for optimal conditions
            bonus = 0.5 + np.random.normal(0, 0.1)
            annual_rainfall.at[idx, 'yield_tons_per_hectare'] += bonus

    # Ensure no negative yields and realistic values (0.5-3.0 tons/hectare)
    annual_rainfall['yield_tons_per_hectare'] = annual_rainfall['yield_tons_per_hectare'].clip(lower=0.5, upper=3.0)

    return annual_rainfall


def _generate_synthetic_weather_data(stations: list, region: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic weather data when API fails or for testing."""
    print(f"Generating synthetic Brazil weather data for {region}...")

    # Create synthetic dataset with realistic values
    np.random.seed(42)  # For reproducibility

    # Generate date range for 5 years of daily data
    current_year = pd.Timestamp.now().year
    start_date = pd.Timestamp(f"{current_year - 5}-01-01")
    end_date = pd.Timestamp(f"{current_year}-12-31")
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create empty list to store all weather records
    all_weather_records = []

    # Base patterns for Minas Gerais
    # Monthly rainfall means (in mm) - Seasonal pattern: Oct-Mar wet season, Apr-Sep dry season
    monthly_rainfall_means = {
        1: 290, 2: 190, 3: 150, 4: 60,  # Jan-Apr
        5: 35, 6: 15, 7: 10, 8: 15,  # May-Aug (dry season)
        9: 60, 10: 130, 11: 220, 12: 280  # Sep-Dec (wet season starts)
    }

    # Define drought years
    yearly_rainfall_factor = {}
    for year in range(current_year - 5, current_year + 1):
        # Drought years (based on historical drought patterns)
        if year in [current_year - 3, current_year - 1]:
            yearly_rainfall_factor[year] = 0.65 + np.random.rand() * 0.15  # 65-80% of normal rainfall
        else:
            yearly_rainfall_factor[year] = 0.9 + np.random.rand() * 0.3  # 90-120% of normal rainfall

    # Generate data for each station
    for station in stations:
        station_id = station["IdStationWho"]

        # Adjust temperature based on altitude
        altitude_factor = (station["Altitude"] - 500) / 1000  # Temperature adjustment for altitude
        monthly_temp_means = {month: (24 - 3 * altitude_factor if month in [1, 2, 11, 12] else
                                      23 - 3 * altitude_factor if month == 3 else
                                      22 - 2 * altitude_factor if month == 4 else
                                      20 - 2 * altitude_factor if month == 5 else
                                      18 - 2 * altitude_factor if month in [6, 7] else
                                      20 - 2 * altitude_factor if month == 8 else
                                      22 - 2 * altitude_factor if month == 9 else
                                      23 - 2 * altitude_factor)
                              for month in range(1, 13)}

        # Generate data for each day
        for date in date_range:
            year, month = date.year, date.month

            # Apply yearly rainfall factor to monthly mean
            mean_daily_rainfall = monthly_rainfall_means[month] / 30 * yearly_rainfall_factor[year]

            # Daily rainfall is highly variable - use gamma distribution
            # More zeros in dry season
            if month in [5, 6, 7, 8] and np.random.rand() < 0.85:  # 85% chance of no rain in dry season
                daily_rainfall = 0
            elif np.random.rand() < 0.3:  # 30% chance of no rain on any given day
                daily_rainfall = 0
            else:
                # Use gamma distribution for rainy days
                shape, scale = 1.5, mean_daily_rainfall / 1.5
                daily_rainfall = np.random.gamma(shape, scale)

            # Temperature with some daily variation and altitude effect
            daily_temp = monthly_temp_means[month] + np.random.normal(0, 2)  # Daily variation of ±2°C

            # Create record
            record = {
                'date': date,
                'station': station_id,
                'station_name': station["StationName"],
                'latitude': station["Latitude"],
                'longitude': station["Longitude"],
                'state': region,
                'precipitation': daily_rainfall,
                'temperature': daily_temp,
                'altitude': station["Altitude"],
                'relative_humidity': 70 + np.random.normal(0, 10)  # Random humidity around 70%
            }

            all_weather_records.append(record)

    # Convert to DataFrame and set date as index
    weather_data = pd.DataFrame(all_weather_records)
    weather_data.set_index('date', inplace=True)

    # Add month and year columns for aggregation
    weather_data['month'] = weather_data.index.month
    weather_data['year'] = weather_data.index.year

    # Group by year and station to calculate annual rainfall
    annual_rainfall = weather_data.groupby(['year', 'station', 'station_name', 'altitude'])[
        'precipitation'].sum().reset_index()

    # Convert mm to inches for consistency with the model
    annual_rainfall['precipitation_inches'] = annual_rainfall['precipitation'] / 25.4

    # Add drought and optimal rainfall indicators
    annual_rainfall['drought'] = annual_rainfall['precipitation_inches'] < 30
    annual_rainfall['optimal_rainfall'] = (
            (annual_rainfall['precipitation_inches'] >= 60) &
            (annual_rainfall['precipitation_inches'] <= 90)
    )

    # Add synthetic yield data
    annual_rainfall = _add_synthetic_yield(annual_rainfall)

    print(f"Generated synthetic data for {len(stations)} stations over {len(date_range)} days")
    print(f"Total records: {len(weather_data)}")

    return weather_data, annual_rainfall


def get_coffee_price_data(api_key, interval='monthly'):
    """
    Fetch coffee price data from Alpha Vantage API, replacing the Bloomberg Terminal function.

    Args:
        api_key: Your Alpha Vantage API key
        interval: Data interval ('monthly', 'quarterly', or 'annual')

    Returns:
        DataFrame containing coffee price data
    """
    print(f"Fetching coffee price data from Alpha Vantage (interval: {interval})...")

    # This is the specific endpoint for coffee commodity prices
    url = f'https://www.alphavantage.co/query?function=COFFEE&interval={interval}&apikey={api_key}'

    try:
        response = requests.get(url)
        data = response.json()

        # Check if we have valid data
        if 'data' in data:
            # Convert to DataFrame
            df = pd.DataFrame(data['data'])
            df['value'] = pd.to_numeric(df['value'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)

            print(f"Successfully retrieved {len(df)} coffee price records")

            # Format it to match the expected output format from Bloomberg
            coffee_data = pd.DataFrame(df['value'])
            coffee_data.columns = ['PX_LAST']

            return coffee_data, df  # Return in the format expected by the model
        else:
            print("Error: No data in response from Alpha Vantage")
            print(f"Response content: {data}")
            return _generate_synthetic_coffee_data()

    except Exception as e:
        print(f"Error fetching coffee price data: {e}")
        return _generate_synthetic_coffee_data()


def _generate_synthetic_coffee_data():
    """Generate synthetic coffee price data when API fails."""
    print("Generating synthetic coffee price data...")

    # Create date range for 5 years of daily price data
    current_year = pd.Timestamp.now().year
    start = pd.Timestamp(f"{current_year - 5}-01-01")
    end = pd.Timestamp(f"{current_year}-12-31")
    date_range = pd.date_range(start=start, end=end, freq='B')  # Business days

    # Set seed for reproducibility
    np.random.seed(42)

    # Coffee price simulation
    # Starting price around $1.50 per pound (150 cents)
    initial_price = 150.0

    # Add trend, seasonality, and random noise
    trend = np.linspace(0, 50, len(date_range))  # Upward trend
    seasonality = 20 * np.sin(np.linspace(0, 2 * np.pi * 5, len(date_range)))  # Seasonal pattern
    noise = np.random.normal(0, 15, len(date_range))  # Random noise

    # Add drought effects
    drought_years = [current_year - 3, current_year - 1]  # Example drought years
    drought_effect = np.zeros(len(date_range))

    for i, date in enumerate(date_range):
        if date.year in drought_years and date.month in [5, 6, 7, 8]:
            drought_effect[i] = 30 + np.random.normal(0, 10)  # Price spike during drought

    # Combine components
    prices = initial_price + trend + seasonality + noise + drought_effect

    # Create primary dataframe (coffee futures)
    coffee_futures = pd.DataFrame({
        'PX_LAST': prices
    }, index=date_range)

    # Create secondary dataframe (coffee subindex)
    coffee_subindex = pd.DataFrame({
        'value': prices * 0.7  # Just a different scale to simulate a subindex
    }, index=date_range)

    print(f"Generated synthetic coffee price data for {len(date_range)} trading days")
    print(f"Coffee price range: ${min(prices) / 100:.2f} to ${max(prices) / 100:.2f} per pound")

    return coffee_futures, coffee_subindex


def engineer_features(weather_data, coffee_data, annual_rainfall):
    """
    Create features for the coffee harvest prediction model.

    Args:
        weather_data: DataFrame with weather data
        coffee_data: DataFrame with coffee price data
        annual_rainfall: DataFrame with annual rainfall data

    Returns:
        Tuple of (features_df, quarterly_rainfall_df, coffee_prices_df)
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime

    print("Engineering features for model...")

    # Make a copy of annual rainfall data to avoid modifying original
    features = annual_rainfall.copy()

    # 1. Ensure precipitation is in inches (convert if needed)
    if 'precipitation_inches' not in features.columns:
        if weather_data['precipitation'].mean() > 100:  # Likely in mm
            features['precipitation_inches'] = features['precipitation'] / 25.4
        else:  # Assume already in inches
            features['precipitation_inches'] = features['precipitation']

    # 2. Create drought and optimal rainfall indicators
    features['drought'] = features['precipitation_inches'] < 30
    features['optimal_rainfall'] = (
            (features['precipitation_inches'] >= 60) &
            (features['precipitation_inches'] <= 90)
    )

    # 3. Calculate quarterly rainfall distribution
    # Coffee growing season in Brazil is typically Oct-Sep
    seasonal_data = weather_data.copy()
    seasonal_data['growing_season'] = np.where(
        seasonal_data['month'] >= 10,
        seasonal_data['year'] + 1,
        seasonal_data['year']
    )

    # Define quarters
    quarters = {
        'Q1': [10, 11, 12],  # Oct-Dec
        'Q2': [1, 2, 3],  # Jan-Mar
        'Q3': [4, 5, 6],  # Apr-Jun
        'Q4': [7, 8, 9]  # Jul-Sep
    }

    # Calculate rainfall by quarter
    quarterly_rainfall = {}
    for q, months in quarters.items():
        quarterly_data = seasonal_data[seasonal_data['month'].isin(months)]
        quarterly_sum = quarterly_data.groupby(['growing_season', 'station'])['precipitation'].sum()
        quarterly_rainfall[f'rainfall_{q}'] = quarterly_sum

    # Convert to DataFrame
    quarterly_df = pd.DataFrame(quarterly_rainfall)

    # 4. Shift the year to match harvest year
    features['year'] = features['year'] + 1

    # 5. Add coffee price data
    try:
        # Resample coffee prices to yearly averages
        yearly_coffee_prices = coffee_data.resample('YE').mean()
        yearly_coffee_prices.index = yearly_coffee_prices.index.year
        yearly_coffee_prices.columns = ['coffee_price']

        # Merge with features
        features = pd.merge(
            features,
            yearly_coffee_prices,
            left_on='year',
            right_index=True,
            how='left'
        )

        # Add price change as feature
        features_by_station = []
        for station, group in features.groupby('station'):
            group = group.sort_values('year')
            group['coffee_price_change'] = group['coffee_price'].pct_change() * 100
            features_by_station.append(group)

        features = pd.concat(features_by_station)

        # Fill missing values
        price_change_mean = features['coffee_price_change'].mean()
        features['coffee_price_change'] = features['coffee_price_change'].fillna(price_change_mean)

    except Exception as e:
        print(f"Could not add coffee price data: {e}")

        # Add dummy coffee price data
        current_year = datetime.now().year
        features['coffee_price'] = features['year'].map({
            current_year - 4: 140,
            current_year - 3: 155,
            current_year - 2: 165,
            current_year - 1: 190,
            current_year: 210
        }).fillna(150)
        features['coffee_price_change'] = 5.0

    print(f"Engineered {len(features)} feature rows with {len(features.columns)} columns")
    return features, quarterly_df, yearly_coffee_prices


def prepare_seasonal_features(weather_data: pd.DataFrame,
                              coffee_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Prepare seasonal features from weather data for the coffee harvest model.

    Args:
        weather_data: DataFrame with daily/monthly weather data, must include precipitation,
                     temperature, and datetime information
        coffee_data: Optional DataFrame with coffee price data

    Returns:
        DataFrame with seasonal features for the model
    """
    # Ensure we have date index
    if not isinstance(weather_data.index, pd.DatetimeIndex):
        if 'date' in weather_data.columns:
            weather_data = weather_data.set_index('date')
        else:
            raise ValueError("Weather data must have a date column or datetime index")

    # Extract month and year
    weather_data = weather_data.copy()
    weather_data['month'] = weather_data.index.month
    weather_data['year'] = weather_data.index.year

    # Define seasonal periods based on coffee phenology
    seasons = {
        'flower': [1, 2],  # Jan-Feb: Flowering
        'early_grow': [3, 4, 5, 6],  # Mar-Jun: Early growing
        'late_grow': [7, 8, 9],  # Jul-Sep: Late growing
        'harvest': [10, 11, 12]  # Oct-Dec: Harvest
    }

    # Create empty dataframe for seasonal features
    years = weather_data['year'].unique()
    features = pd.DataFrame(index=years)
    features.index.name = 'year'

    # Calculate seasonal rainfall totals
    for season_name, months in seasons.items():
        # Rainfall totals
        season_data = weather_data[weather_data['month'].isin(months)]
        rain_by_year = season_data.groupby('year')['precipitation'].sum()
        features[f'{season_name}_rain'] = rain_by_year

        # Mean temperatures
        if 'temperature' in weather_data.columns:
            temp_by_year = season_data.groupby('year')['temperature'].mean()
            features[f'{season_name}_temp'] = temp_by_year

        # Min temperatures
        if 'min_temperature' in weather_data.columns:
            min_temp_by_year = season_data.groupby('year')['min_temperature'].mean()
            features[f'{season_name}_min_temp'] = min_temp_by_year

        # Max temperatures
        if 'max_temperature' in weather_data.columns:
            max_temp_by_year = season_data.groupby('year')['max_temperature'].mean()
            features[f'{season_name}_max_temp'] = max_temp_by_year

    # Calculate additional features from the paper
    features['drought_late_grow'] = features['late_grow_rain'] < 1600  # mm, threshold from paper
    features['high_harvest_rain'] = features['harvest_rain'] > 750  # mm, threshold from paper
    features['high_harvest_temp'] = features['harvest_min_temp'] > 22  # °C, threshold from paper

    # Add coffee price data if available
    if coffee_data is not None:
        # Assuming coffee_data has a date index and 'price' column
        yearly_coffee_prices = coffee_data.resample('YE').mean()
        yearly_coffee_prices.index = yearly_coffee_prices.index.year

        # Merge with features
        if 'price' in coffee_data.columns:
            features = pd.merge(
                features,
                yearly_coffee_prices['price'],
                left_index=True,
                right_index=True,
                how='left'
            )

    return features


def calculate_price_impact(prob_small_beans: float, prob_defects: float) -> float:
    """
    Calculate expected price impact based on bean quality probabilities.

    According to the paper, farmers can:
    - Lose ~10% of gross returns for poor scores
    - Gain up to 25% for excellent scores
    - Lose up to 33% when both quality metrics are poor

    Args:
        prob_small_beans: Probability of below average bean size (0-1)
        prob_defects: Probability of above average bean defects (0-1)

    Returns:
        Expected percentage impact on price (-0.33 to +0.25)
    """
    # Base impact (percentage change to expected price)
    base_price = 0.0

    # Impact from bean size
    if prob_small_beans > 0.8:  # High probability of small beans
        size_impact = -0.15  # -15% price impact
    elif prob_small_beans < 0.2:  # Low probability of small beans
        size_impact = 0.15  # +15% price impact
    else:
        size_impact = 0.15 - 0.3 * prob_small_beans  # Linear interpolation

    # Impact from defects
    if prob_defects > 0.8:  # High probability of defects
        defect_impact = -0.15  # -15% price impact
    elif prob_defects < 0.2:  # Low probability of defects
        defect_impact = 0.10  # +10% price impact
    else:
        defect_impact = 0.10 - 0.25 * prob_defects  # Linear interpolation

    # Combined impact - includes interaction effect
    combined_impact = size_impact + defect_impact

    # Extra penalty when both problems are likely
    if prob_small_beans > 0.6 and prob_defects > 0.6:
        combined_impact -= 0.05  # Additional -5% for combined issues

    # Ensure the combined impact is within bounds (-33% to +25%)
    combined_impact = max(min(combined_impact, 0.25), -0.33)

    return combined_impact


def build_mcmc_model(features: pd.DataFrame, bean_quality_data: pd.DataFrame = None):
    """
    Build a Bayesian model for coffee harvest prediction incorporating
    seasonal climate effects on both yield quantity and quality.

    Args:
        features: DataFrame with seasonal climate features
        bean_quality_data: Optional DataFrame with bean quality metrics. If not provided,
                          only yield will be modeled.

    Returns:
        Tuple of (model, trace)
    """
    import pymc as pm
    import numpy as np

    # Clean input data
    features_clean = features.dropna(subset=['late_grow_rain', 'yield_tons_per_hectare'])

    # Get available features
    has_quality_data = bean_quality_data is not None
    if has_quality_data:
        # Join quality data with features
        model_data = pd.merge(
            features_clean,
            bean_quality_data,
            left_index=True,
            right_index=True,
            how='inner'
        )
    else:
        model_data = features_clean

    # Extract and scale climate variables by season
    # Flowering period (Jan-Feb)
    X_flower_rain = model_data['flower_rain'].values
    flower_rain_mean, flower_rain_std = np.mean(X_flower_rain), max(np.std(X_flower_rain), 1.0)
    X_flower_rain_scaled = (X_flower_rain - flower_rain_mean) / flower_rain_std

    if 'flower_temp' in model_data.columns:
        X_flower_temp = model_data['flower_temp'].values
        flower_temp_mean, flower_temp_std = np.mean(X_flower_temp), max(np.std(X_flower_temp), 1.0)
        X_flower_temp_scaled = (X_flower_temp - flower_temp_mean) / flower_temp_std
    else:
        X_flower_temp_scaled = np.zeros_like(X_flower_rain)
        flower_temp_mean, flower_temp_std = 0, 1

    # Late growing season (Jul-Sep)
    X_late_grow_rain = model_data['late_grow_rain'].values
    late_grow_rain_mean, late_grow_rain_std = np.mean(X_late_grow_rain), max(np.std(X_late_grow_rain), 1.0)
    X_late_grow_rain_scaled = (X_late_grow_rain - late_grow_rain_mean) / late_grow_rain_std

    if 'late_grow_temp' in model_data.columns:
        X_late_grow_temp = model_data['late_grow_temp'].values
        late_grow_temp_mean, late_grow_temp_std = np.mean(X_late_grow_temp), max(np.std(X_late_grow_temp), 1.0)
        X_late_grow_temp_scaled = (X_late_grow_temp - late_grow_temp_mean) / late_grow_temp_std
    else:
        X_late_grow_temp_scaled = np.zeros_like(X_flower_rain)
        late_grow_temp_mean, late_grow_temp_std = 0, 1

    # Harvest season (Oct-Dec)
    if 'harvest_rain' in model_data.columns:
        X_harvest_rain = model_data['harvest_rain'].values
        harvest_rain_mean, harvest_rain_std = np.mean(X_harvest_rain), max(np.std(X_harvest_rain), 1.0)
        X_harvest_rain_scaled = (X_harvest_rain - harvest_rain_mean) / harvest_rain_std
    else:
        X_harvest_rain_scaled = np.zeros_like(X_flower_rain)
        harvest_rain_mean, harvest_rain_std = 0, 1

    if 'harvest_min_temp' in model_data.columns:
        X_harvest_min_temp = model_data['harvest_min_temp'].values
        harvest_min_temp_mean = np.mean(X_harvest_min_temp)
        harvest_min_temp_std = max(np.std(X_harvest_min_temp), 1.0)
        X_harvest_min_temp_scaled = (X_harvest_min_temp - harvest_min_temp_mean) / harvest_min_temp_std
    else:
        X_harvest_min_temp_scaled = np.zeros_like(X_flower_rain)
        harvest_min_temp_mean, harvest_min_temp_std = 0, 1

    # Target variables
    y_yield = model_data['yield_tons_per_hectare'].values

    if has_quality_data:
        if 'bean_size_score' in model_data.columns:
            y_bean_size = model_data['bean_size_score'].values
        else:
            has_bean_size = False

        if 'bean_defect_score' in model_data.columns:
            y_bean_defects = model_data['bean_defect_score'].values
        else:
            has_bean_defects = False
    else:
        has_bean_size = False
        has_bean_defects = False

    # Build hierarchical model
    with pm.Model() as coffee_model:
        # Yield quantity priors
        beta_yield_intercept = pm.Normal('yield_intercept', mu=1.5, sigma=1.0)
        beta_yield_late_rain = pm.Normal('yield_late_rain', mu=0.4, sigma=0.2)

        # Optional predictors for yield
        if 'flower_rain' in model_data.columns:
            beta_yield_flower_rain = pm.Normal('yield_flower_rain', mu=0.0, sigma=0.2)

        if 'early_grow_rain' in model_data.columns:
            beta_yield_early_rain = pm.Normal('yield_early_rain', mu=0.2, sigma=0.2)

        # Bean size priors (informed by the paper)
        if has_bean_size:
            beta_size_intercept = pm.Normal('size_intercept', mu=0.0, sigma=1.0)
            beta_size_late_rain = pm.Normal('size_late_rain', mu=0.5, sigma=0.2)  # Positive effect (larger beans)
            beta_size_flower_temp = pm.Normal('size_flower_temp', mu=-0.3, sigma=0.2)  # Negative effect

        # Bean defect priors (informed by the paper)
        if has_bean_defects:
            beta_defect_intercept = pm.Normal('defect_intercept', mu=0.0, sigma=1.0)
            beta_defect_harvest_rain = pm.Normal('defect_harvest_rain', mu=0.4, sigma=0.2)  # Positive effect
            beta_defect_harvest_temp = pm.Normal('defect_harvest_temp', mu=0.4, sigma=0.2)  # Positive effect

            # Interaction term for harvest rain + temp
            beta_defect_interaction = pm.Normal('defect_rain_temp_interaction', mu=0.2, sigma=0.1)

        # Model error terms
        sigma_yield = pm.HalfNormal('sigma_yield', sigma=0.5)

        if has_bean_size:
            sigma_size = pm.HalfNormal('sigma_size', sigma=0.5)

        if has_bean_defects:
            sigma_defect = pm.HalfNormal('sigma_defect', sigma=0.5)

        # Expected yield
        mu_yield = beta_yield_intercept + beta_yield_late_rain * X_late_grow_rain_scaled

        # Add optional predictors if available
        if 'flower_rain' in model_data.columns:
            mu_yield += beta_yield_flower_rain * X_flower_rain_scaled

        # Expected bean size score (higher = more large beans)
        if has_bean_size:
            mu_size = (beta_size_intercept +
                       beta_size_late_rain * X_late_grow_rain_scaled +
                       beta_size_flower_temp * X_flower_temp_scaled)

        # Expected bean defect score (higher = more defects)
        if has_bean_defects:
            mu_defect = (beta_defect_intercept +
                         beta_defect_harvest_rain * X_harvest_rain_scaled +
                         beta_defect_harvest_temp * X_harvest_min_temp_scaled +
                         beta_defect_interaction * X_harvest_rain_scaled * X_harvest_min_temp_scaled)

        # Likelihood functions
        yield_obs = pm.Normal('yield_obs', mu=mu_yield, sigma=sigma_yield, observed=y_yield)

        if has_bean_size:
            size_obs = pm.Normal('size_obs', mu=mu_size, sigma=sigma_size, observed=y_bean_size)

        if has_bean_defects:
            defect_obs = pm.Normal('defect_obs', mu=mu_defect, sigma=sigma_defect, observed=y_bean_defects)

        # Sample from the model
        trace = pm.sample(draws=1000, tune=1000, return_inferencedata=True, random_seed=42)

    # Store scaling parameters in the model for prediction
    coffee_model.scaling_params = {
        'flower_rain_mean': flower_rain_mean,
        'flower_rain_std': flower_rain_std,
        'flower_temp_mean': flower_temp_mean,
        'flower_temp_std': flower_temp_std,
        'late_grow_rain_mean': late_grow_rain_mean,
        'late_grow_rain_std': late_grow_rain_std,
        'late_grow_temp_mean': late_grow_temp_mean,
        'late_grow_temp_std': late_grow_temp_std,
        'harvest_rain_mean': harvest_rain_mean,
        'harvest_rain_std': harvest_rain_std,
        'harvest_min_temp_mean': harvest_min_temp_mean,
        'harvest_min_temp_std': harvest_min_temp_std
    }

    # Store flags for available quality metrics
    coffee_model.has_bean_size = has_bean_size
    coffee_model.has_bean_defects = has_bean_defects

    return coffee_model, trace


def predict_harvest(model, trace, climate_forecast):
    """
    Predict coffee harvest quantity and quality based on climate forecast.

    Args:
        model: PyMC model object from build_mcmc_model
        trace: InferenceData object from build_mcmc_model
        climate_forecast: Dictionary with climate forecast values for each season

    Returns:
        Dictionary with predicted yield, bean size score, and defect score
    """
    import numpy as np

    # Get scaling parameters
    scaling = model.scaling_params

    # Scale the forecast data
    scaled_forecast = {}

    # Flowering period (Jan-Feb)
    if 'flower_rain' in climate_forecast:
        scaled_forecast['flower_rain_scaled'] = ((climate_forecast['flower_rain'] -
                                                  scaling['flower_rain_mean']) /
                                                 scaling['flower_rain_std'])

    if 'flower_temp' in climate_forecast:
        scaled_forecast['flower_temp_scaled'] = ((climate_forecast['flower_temp'] -
                                                  scaling['flower_temp_mean']) /
                                                 scaling['flower_temp_std'])

    # Late growing season (Jul-Sep)
    if 'late_grow_rain' in climate_forecast:
        scaled_forecast['late_grow_rain_scaled'] = ((climate_forecast['late_grow_rain'] -
                                                     scaling['late_grow_rain_mean']) /
                                                    scaling['late_grow_rain_std'])

    if 'late_grow_temp' in climate_forecast:
        scaled_forecast['late_grow_temp_scaled'] = ((climate_forecast['late_grow_temp'] -
                                                     scaling['late_grow_temp_mean']) /
                                                    scaling['late_grow_temp_std'])

    # Harvest season (Oct-Dec)
    if 'harvest_rain' in climate_forecast:
        scaled_forecast['harvest_rain_scaled'] = ((climate_forecast['harvest_rain'] -
                                                   scaling['harvest_rain_mean']) /
                                                  scaling['harvest_rain_std'])

    if 'harvest_min_temp' in climate_forecast:
        scaled_forecast['harvest_min_temp_scaled'] = ((climate_forecast['harvest_min_temp'] -
                                                       scaling['harvest_min_temp_mean']) /
                                                      scaling['harvest_min_temp_std'])

    # Extract parameters from trace
    yield_params = {
        'intercept': trace.posterior['yield_intercept'].mean().item(),
        'late_rain': trace.posterior['yield_late_rain'].mean().item()
    }

    # Optional yield parameters
    if 'yield_flower_rain' in trace.posterior:
        yield_params['flower_rain'] = trace.posterior['yield_flower_rain'].mean().item()

    if 'yield_early_rain' in trace.posterior:
        yield_params['early_rain'] = trace.posterior['yield_early_rain'].mean().item()

    # Bean size parameters if available
    if model.has_bean_size:
        size_params = {
            'intercept': trace.posterior['size_intercept'].mean().item(),
            'late_rain': trace.posterior['size_late_rain'].mean().item(),
            'flower_temp': trace.posterior['size_flower_temp'].mean().item()
        }

    # Bean defect parameters if available
    if model.has_bean_defects:
        defect_params = {
            'intercept': trace.posterior['defect_intercept'].mean().item(),
            'harvest_rain': trace.posterior['defect_harvest_rain'].mean().item(),
            'harvest_temp': trace.posterior['defect_harvest_temp'].mean().item(),
            'interaction': trace.posterior['defect_rain_temp_interaction'].mean().item()
        }

    # Calculate yield prediction
    pred_yield = yield_params['intercept']

    if 'late_grow_rain_scaled' in scaled_forecast:
        pred_yield += yield_params['late_rain'] * scaled_forecast['late_grow_rain_scaled']

    if 'flower_rain_scaled' in scaled_forecast and 'flower_rain' in yield_params:
        pred_yield += yield_params['flower_rain'] * scaled_forecast['flower_rain_scaled']

    if 'early_grow_rain_scaled' in scaled_forecast and 'early_rain' in yield_params:
        pred_yield += yield_params['early_rain'] * scaled_forecast['early_grow_rain_scaled']

    # Initialize quality predictions
    pred_size = None
    pred_defects = None
    prob_small_beans = None
    prob_defects = None

    # Calculate bean size prediction if model has it
    if model.has_bean_size:
        pred_size = size_params['intercept']

        if 'late_grow_rain_scaled' in scaled_forecast:
            pred_size += size_params['late_rain'] * scaled_forecast['late_grow_rain_scaled']

        if 'flower_temp_scaled' in scaled_forecast:
            pred_size += size_params['flower_temp'] * scaled_forecast['flower_temp_scaled']

        # Convert to probability of small beans (higher score = larger beans, lower prob of small)
        prob_small_beans = 1 / (1 + np.exp(pred_size))

    # Calculate bean defect prediction if model has it
    if model.has_bean_defects:
        pred_defects = defect_params['intercept']

        if 'harvest_rain_scaled' in scaled_forecast:
            pred_defects += defect_params['harvest_rain'] * scaled_forecast['harvest_rain_scaled']

        if 'harvest_min_temp_scaled' in scaled_forecast:
            pred_defects += defect_params['harvest_temp'] * scaled_forecast['harvest_min_temp_scaled']

        if ('harvest_rain_scaled' in scaled_forecast and
                'harvest_min_temp_scaled' in scaled_forecast):
            pred_defects += (defect_params['interaction'] *
                             scaled_forecast['harvest_rain_scaled'] *
                             scaled_forecast['harvest_min_temp_scaled'])

        # Convert to probability of defects (higher score = more defects)
        prob_defects = 1 / (1 + np.exp(-pred_defects))

    # Create result dictionary
    result = {
        'yield_tons_per_hectare': pred_yield
    }

    # Add quality predictions if available
    if prob_small_beans is not None:
        result['prob_small_beans'] = prob_small_beans

    if prob_defects is not None:
        result['prob_defects'] = prob_defects

    # Calculate price impact if we have both quality metrics
    if prob_small_beans is not None and prob_defects is not None:
        result['expected_price_impact'] = calculate_price_impact(prob_small_beans, prob_defects)

    # Add additional insight from the paper's thresholds
    if 'late_grow_rain' in climate_forecast:
        result['drought_risk'] = climate_forecast['late_grow_rain'] < 1600  # mm

    if 'harvest_rain' in climate_forecast:
        result['high_harvest_rain_risk'] = climate_forecast['harvest_rain'] > 750  # mm

    if 'harvest_min_temp' in climate_forecast:
        result['high_temp_risk'] = climate_forecast['harvest_min_temp'] > 22  # °C

    return result