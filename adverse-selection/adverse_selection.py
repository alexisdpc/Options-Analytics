"""------------------------------------------------------------------------
Author: Alexis D. Plascencia
Date: July 21, 2025
License: MIT

Description:
    Comprehensive machine learning algorithm for detecting informed trading
    in financial markets. Uses ensemble methods to classify trades as 
    informed vs uninformed to help market makers avoid adverse selection.
------------------------------------------------------------------------"""

import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class InformedTradeClassifier:
    """
    Machine learning classifier for detecting informed vs uninformed trades.
    
    This class implements a comprehensive feature engineering pipeline and
    ensemble modeling approach to identify trades that contain superior
    information, which is crucial for market makers to avoid adverse selection.
    
    The algorithm combines multiple signals:
    - Trade size and timing patterns
    - Price impact and market microstructure features
    - Volatility and momentum indicators
    - Order book imbalances (when available)
    
    Attributes:
        lookback_window (int): Number of periods for rolling calculations
        volatility_window (int): Window size for volatility computations
        scaler (StandardScaler): Feature scaling transformer
        models (dict): Dictionary storing trained ML models
        feature_names (list): Names of engineered features used in training
    """
    def __init__(self, lookback_window=50, volatility_window=20):
        """
        Initialize the InformedTradeClassifier.
        
        Args:
            lookback_window (int, optional): Number of historical periods to use
                for rolling statistics and momentum calculations. Defaults to 50.
            volatility_window (int, optional): Specific window for volatility 
                calculations. Defaults to 20.
        """        
        self.lookback_window = lookback_window
        self.volatility_window = volatility_window
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = []
        
    def engineer_features(self, trades_df, quotes_df, orderbook_df=None):
        """
        Comprehensive feature engineering for trade classification.
        
        Transforms raw trade and market data into predictive features that
        capture various aspects of informed trading behavior including:
        - Trade characteristics (size, timing, price impact)
        - Market microstructure (spreads, quotes, trade direction)
        - Market activity patterns (volume, frequency, volatility)
        - Momentum and trend indicators
        - Order book dynamics (when available)
        
        Args:
            trades_df (pd.DataFrame): Trade data with columns:
                - timestamp: Trade execution time
                - price: Trade execution price
                - size: Trade quantity/volume
                - trade_id: Unique trade identifier
            quotes_df (pd.DataFrame): Quote data with columns:
                - timestamp: Quote update time
                - bid: Best bid price
                - ask: Best ask price
            orderbook_df (pd.DataFrame, optional): Order book data with columns:
                - timestamp: Snapshot time
                - bid_depth: Total bid side depth
                - ask_depth: Total ask side depth
                - bid_orders: Number of bid orders
                - ask_orders: Number of ask orders
                
        Returns:
            pd.DataFrame: Engineered features ready for machine learning,
                with NaN values removed and all features computed.
        """
        features = trades_df.copy()
        
        # 1. TRADE-LEVEL FEATURES
        features['trade_size_dollars'] = features['size'] * features['price']
        features['trade_size_log'] = np.log1p(features['size'])
        
        # Trade timing features
        features['hour'] = pd.to_datetime(features['timestamp']).dt.hour
        features['minute'] = pd.to_datetime(features['timestamp']).dt.minute
        features['day_of_week'] = pd.to_datetime(features['timestamp']).dt.dayofweek
        
        # 2. PRICE IMPACT FEATURES
        # Immediate price impact (price change after trade)
        features['price_impact_1'] = features['price'].shift(-1) - features['price']
        features['price_impact_5'] = features['price'].shift(-5) - features['price']
        features['price_impact_10'] = features['price'].shift(-10) - features['price']
        
        # Relative price impact
        features['rel_price_impact_1'] = features['price_impact_1'] / features['price']
        features['rel_price_impact_5'] = features['price_impact_5'] / features['price']
        
        # 3. MARKET MICROSTRUCTURE FEATURES
        # Merge with quotes data
        quotes_df['timestamp'] = pd.to_datetime(quotes_df['timestamp'])
        features['timestamp'] = pd.to_datetime(features['timestamp'])
        
        # Get prevailing quotes at trade time
        features = pd.merge_asof(
            features.sort_values('timestamp'),
            quotes_df.sort_values('timestamp')[['timestamp', 'bid', 'ask']],
            on='timestamp',
            direction='backward'
        )
        
        # Spread and quote-relative features
        features['spread'] = features['ask'] - features['bid']
        features['mid_price'] = (features['bid'] + features['ask']) / 2
        features['relative_spread'] = features['spread'] / features['mid_price']
        
        # Trade direction relative to quotes
        features['trade_direction'] = np.where(
            features['price'] > features['mid_price'], 1,  # Buy
            np.where(features['price'] < features['mid_price'], -1, 0)  # Sell or mid
        )
        
        # Distance from mid-price
        features['distance_from_mid'] = np.abs(features['price'] - features['mid_price'])
        features['relative_distance_from_mid'] = features['distance_from_mid'] / features['mid_price']
        
        # 4. VOLUME AND ACTIVITY FEATURES
        # Rolling volume statistics
        features['volume_ma_10'] = features['size'].rolling(10).mean()
        features['volume_ma_50'] = features['size'].rolling(50).mean()
        features['volume_ratio'] = features['size'] / features['volume_ma_10']
        
        # Trade frequency
        features['time_since_last_trade'] = features['timestamp'].diff().dt.total_seconds()
        features['trades_per_minute'] = 1 / (features['time_since_last_trade'] / 60)
        
        # 5. VOLATILITY FEATURES
        # Price volatility
        features['returns'] = features['price'].pct_change()
        features['volatility_10'] = features['returns'].rolling(10).std()
        features['volatility_50'] = features['returns'].rolling(50).std()
        features['current_vol_ratio'] = features['volatility_10'] / features['volatility_50']
        
        # High-low volatility
        features['price_high_10'] = features['price'].rolling(10).max()
        features['price_low_10'] = features['price'].rolling(10).min()
        features['price_range_10'] = (features['price_high_10'] - features['price_low_10']) / features['price']
        
        # 6. MOMENTUM AND TREND FEATURES
        # Price momentum
        features['momentum_5'] = features['price'] / features['price'].shift(5) - 1
        features['momentum_20'] = features['price'] / features['price'].shift(20) - 1
        
        # Moving averages
        features['ma_5'] = features['price'].rolling(5).mean()
        features['ma_20'] = features['price'].rolling(20).mean()
        features['price_vs_ma_5'] = features['price'] / features['ma_5'] - 1
        features['price_vs_ma_20'] = features['price'] / features['ma_20'] - 1
        
        # 7. ORDER BOOK FEATURES (if available)
        if orderbook_df is not None:
            # Merge order book data
            orderbook_df['timestamp'] = pd.to_datetime(orderbook_df['timestamp'])
            features = pd.merge_asof(
                features,
                orderbook_df[['timestamp', 'bid_depth', 'ask_depth', 'bid_orders', 'ask_orders']],
                on='timestamp',
                direction='backward'
            )
            
            # Order book imbalance
            features['depth_imbalance'] = (features['bid_depth'] - features['ask_depth']) / (features['bid_depth'] + features['ask_depth'])
            features['order_imbalance'] = (features['bid_orders'] - features['ask_orders']) / (features['bid_orders'] + features['ask_orders'])
        
        # 8. INTERACTION FEATURES
        features['size_spread_interaction'] = features['trade_size_log'] * features['relative_spread']
        features['volatility_size_interaction'] = features['current_vol_ratio'] * features['trade_size_log']
        
        # 9. TEMPORAL AGGREGATION FEATURES
        # Features based on recent trade history
        features['avg_trade_size_10'] = features['size'].rolling(10).mean()
        features['std_trade_size_10'] = features['size'].rolling(10).std()
        features['trade_size_zscore'] = (features['size'] - features['avg_trade_size_10']) / features['std_trade_size_10']
        
        # Remove rows with NaN values
        features = features.dropna()
        
        return features
    
    def create_labels(self, features_df, method='price_impact'):
        """
        Create binary labels for supervised learning.
        
        Implements different strategies to identify informed trades based on
        various market signals and post-trade behavior patterns.
        
        Args:
            features_df (pd.DataFrame): Engineered features from engineer_features()
            method (str): Labeling strategy to use:
                - 'price_impact': Label based on significant post-trade price moves
                - 'size_impact': Combine large size with significant impact
                - 'momentum_reversal': Label trades that predict future direction
                
        Returns:
            np.ndarray: Binary labels (0=uninformed, 1=informed)
        """
        if method == 'price_impact':
            # Label based on price impact
            # High price impact trades are likely informed
            impact_threshold = features_df['rel_price_impact_5'].quantile(0.8)
            labels = (np.abs(features_df['rel_price_impact_5']) > impact_threshold).astype(int)
            
        elif method == 'size_impact':
            # Combined size and impact criteria
            size_threshold = features_df['trade_size_log'].quantile(0.9)
            impact_threshold = features_df['rel_price_impact_5'].quantile(0.7)
            
            labels = (
                (features_df['trade_size_log'] > size_threshold) &
                (np.abs(features_df['rel_price_impact_5']) > impact_threshold)
            ).astype(int)
            
        elif method == 'momentum_reversal':
            # Trades that predict momentum vs reversal
            future_return = features_df['price'].shift(-10) / features_df['price'] - 1
            current_momentum = features_df['momentum_5']
            
            # Informed trades should predict continuation
            momentum_continuation = (
                (current_momentum > 0) & (future_return > 0) |
                (current_momentum < 0) & (future_return < 0)
            )
            
            labels = momentum_continuation.astype(int)
        
        return labels
    
    def prepare_features(self, features_df):
        """
        Prepare engineered features for machine learning models.
        
        Selects the most relevant features, handles missing values, and
        prepares the feature matrix for training and prediction.
        
        Args:
            features_df (pd.DataFrame): Output from engineer_features()
            
        Returns:
            pd.DataFrame: Clean feature matrix ready for ML algorithms
        """
        # Select relevant features
        feature_columns = [
            'trade_size_log', 'trade_size_dollars', 'hour', 'minute',
            'spread', 'relative_spread', 'trade_direction',
            'distance_from_mid', 'relative_distance_from_mid',
            'volume_ratio', 'time_since_last_trade',
            'volatility_10', 'volatility_50', 'current_vol_ratio',
            'price_range_10', 'momentum_5', 'momentum_20',
            'price_vs_ma_5', 'price_vs_ma_20',
            'size_spread_interaction', 'volatility_size_interaction',
            'trade_size_zscore'
        ]
        
        # Add order book features if available
        if 'depth_imbalance' in features_df.columns:
            feature_columns.extend(['depth_imbalance', 'order_imbalance'])
        
        self.feature_names = feature_columns
        X = features_df[feature_columns].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(X.median())
        
        return X
    
    def train_models(self, X, y):
        """
        Train ensemble of machine learning models.
        
        Implements a diverse ensemble of algorithms that capture different
        aspects of the trading patterns:
        - Random Forest: Handles non-linear interactions, robust to outliers
        - Gradient Boosting: Sequential error correction, good for complex patterns
        - XGBoost: Optimized boosting with regularization, excellent performance
        
        Args:
            X (pd.DataFrame): Feature matrix from prepare_features()
            y (np.ndarray): Binary labels from create_labels()
        """

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # 1. Random Forest
        self.models['rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        self.models['rf'].fit(X_scaled, y)
        
        # 2. Gradient Boosting
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.models['gb'].fit(X_scaled, y)
        
        # 3. XGBoost
        self.models['xgb'] = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.models['xgb'].fit(X_scaled, y)
        
        print("Models trained successfully!")
        
    def predict_proba(self, X):
        """
        Generate ensemble probability predictions.
        
        Combines predictions from all trained models using simple averaging,
        which often performs as well as more complex ensemble methods while
        being more interpretable and stable.
        
        Args:
            X (pd.DataFrame): Feature matrix for prediction
            
        Returns:
            np.ndarray: Probability of informed trading (0.0 to 1.0)
        """
        X_scaled = self.scaler.transform(X)
        
        predictions = []
        for model_name, model in self.models.items():
            pred = model.predict_proba(X_scaled)[:, 1]  # Probability of informed
            predictions.append(pred)
        
        # Ensemble average
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def predict(self, X, threshold=0.5):
        """
        Generate binary classification predictions.
        
        Converts probability predictions to binary decisions using specified
        threshold. Threshold can be adjusted based on business requirements
        (e.g., higher threshold for conservative classification).
        
        Args:
            X (pd.DataFrame): Feature matrix for prediction
            threshold (float): Decision threshold (default 0.5)
            
        Returns:
            np.ndarray: Binary predictions (0=uninformed, 1=informed)
        """
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)
    
    def evaluate_model(self, X_test, y_test):
        """
        Comprehensive model evaluation on test data.
        
        Provides multiple metrics to assess model performance:
        - Classification report: Precision, recall, F1-score by class
        - ROC-AUC: Overall ranking quality across all thresholds
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (np.ndarray): True test labels
            
        Returns:
            dict: Dictionary containing predictions, probabilities, and AUC score
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
        
        return {
            'predictions': y_pred,
            'probabilities': y_proba,
            'auc': roc_auc_score(y_test, y_proba)
        }
    
    def get_feature_importance(self):
        """
        Extract feature importance from trained models.
        
        Provides insights into which features are most predictive of
        informed trading, helping with model interpretation and feature
        selection for future iterations.
        
        Returns:
            dict: Feature importance scores for each model that supports it
        """
        importance_dict = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[model_name] = dict(zip(
                    self.feature_names, 
                    model.feature_importances_
                ))
        
        return importance_dict
    
    def real_time_classify(self, trade_data, quotes_data, orderbook_data=None):
        """
        Real-time classification of individual trades.
        
        Designed for production deployment where trades need to be classified
        as they occur. Handles the full pipeline from raw data to prediction.
        
        Args:
            trade_data (pd.DataFrame): Single trade or recent trades
            quotes_data (pd.DataFrame): Current market quotes
            orderbook_data (pd.DataFrame, optional): Current order book state
            
        Returns:
            dict: Classification result with probability, decision, and confidence
                - informed_probability: Float between 0.0 and 1.0
                - classification: 'informed' or 'uninformed'
                - confidence: Distance from neutral (0.5) threshold
        """
        # Engineer features for the single trade
        features = self.engineer_features(trade_data, quotes_data, orderbook_data)
        
        if len(features) == 0:
            return None
        
        # Prepare features
        X = self.prepare_features(features)
        
        # Get prediction
        informed_probability = self.predict_proba(X)[-1]  # Last trade
        
        return {
            'informed_probability': informed_probability,
            'classification': 'informed' if informed_probability > 0.5 else 'uninformed',
            'confidence': max(informed_probability, 1 - informed_probability)
        }

# Example usage and testing
def generate_sample_data():
    """
    Generate realistic sample trading data for testing and demonstration.
    
    Creates synthetic but realistic trade and quote data that mimics actual
    market behavior including:
    - Log-normal trade size distribution
    - Geometric Brownian motion for prices
    - Realistic bid-ask spreads
    - Proper timestamp sequences
    
    Returns:
        tuple: (trades_df, quotes_df) containing sample data
    """
    np.random.seed(42)
    n_trades = 1000
    
    # Generate sample trades
    base_price = 100
    timestamps = pd.date_range('2024-01-01 09:30:00', periods=n_trades, freq='10S')
    
    # Simulate price with some trends and noise
    price_changes = np.random.normal(0, 0.001, n_trades)
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    trades_df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'size': np.random.lognormal(3, 1, n_trades),  # Log-normal trade sizes
        'trade_id': range(n_trades)
    })
    
    # Generate sample quotes
    quotes_df = pd.DataFrame({
        'timestamp': timestamps,
        'bid': prices - np.random.uniform(0.01, 0.05, n_trades),
        'ask': prices + np.random.uniform(0.01, 0.05, n_trades)
    })
    
    return trades_df, quotes_df

# Demo the algorithm
if __name__ == "__main__":
    """
    Demonstration of the complete informed trade classification pipeline.
    
    This section shows how to:
    1. Generate or load trading data
    2. Initialize and configure the classifier
    3. Engineer features and create labels
    4. Train the ensemble models
    5. Evaluate performance
    6. Analyze feature importance
    """
        
    print("Informed Trade Classification Algorithm")
    print("=" * 50)
    
    # Generate sample data
    trades_df, quotes_df = generate_sample_data()
    print(f"Generated {len(trades_df)} sample trades")
    
    # Initialize classifier
    classifier = InformedTradeClassifier()
    
    # Engineer features
    print("\nEngineering features...")
    features_df = classifier.engineer_features(trades_df, quotes_df)
    print(f"Features shape: {features_df.shape}")
    
    # Create labels
    print("Creating labels...")
    labels = classifier.create_labels(features_df, method='price_impact')
    print(f"Informed trades: {labels.sum()}/{len(labels)} ({labels.mean():.2%})")
    
    # Prepare features for ML
    X = classifier.prepare_features(features_df)
    y = labels
    
    # Train-test split (time series aware)
    split_point = int(0.8 * len(X))
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    
    # Train models
    print("\nTraining models...")
    classifier.train_models(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating models...")
    results = classifier.evaluate_model(X_test, y_test)
    
    # Feature importance
    print("\nFeature Importance (Random Forest):")
    importance = classifier.get_feature_importance()
    if 'rf' in importance:
        sorted_features = sorted(importance['rf'].items(), key=lambda x: x[1], reverse=True)
        for feature, imp in sorted_features[:10]:
            print(f"{feature}: {imp:.4f}")
    
    print("\nAlgorithm ready for real-time classification!")