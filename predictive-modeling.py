"""
Data-Driven Predictive Modeling
=========================================================
T+1 returns (stationary), rolling correlation, regularization
XGBoost + LightGBM, walk-forward, risk metrics
Bug fixes (dead code, look-ahead bias, scaler, RobustScaler)
Classification + Threshold + Regime Filtering + Auto-Correlation
     + Turkish stocks support + Ensemble + Full reporting
Return-based features (stationary), class weights, fixed scaler leakage,
     fixed walk-forward returns, improved ensemble weighting

Modes:
  1. Regression (original) — R², Sharpe, etc.
  2. Classification — Accuracy, Win Rate, Threshold filtering
  3. Regime-Filtered — VIX + Trend filters on classification
  4. Auto-Mode — Runs all and picks best
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.inspection import permutation_importance
import warnings
import os
import json
import time
from datetime import datetime
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()

# ============================================================
# ASSET UNIVERSE — Expanded with Turkish stocks
# ============================================================
ASSET_CATALOG = {
    # Commodities
    "GC=F":  "Gold Futures",
    "SI=F":  "Silver Futures",
    "PL=F":  "Platinum Futures",
    "CL=F":  "Crude Oil",
    "BZ=F":  "Brent Oil",
    "NG=F":  "Natural Gas",
    "ZC=F":  "Corn Futures",
    "ZW=F":  "Wheat Futures",
    # Crypto
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
    "BNB-USD": "BNB",
    # Forex
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDTRY=X": "USD/TRY",
    "EURTRY=X": "EUR/TRY",
    # US Indices & ETFs
    "^GSPC":  "S&P 500",
    "^IXIC":  "NASDAQ",
    "^VIX":   "VIX (Fear Index)",
    "^DJI":   "Dow Jones",
    "SPY":    "S&P 500 ETF",
    "QQQ":    "NASDAQ ETF",
    "GLD":    "Gold ETF",
    "SLV":    "Silver ETF",
    "USO":    "Oil ETF",
    "TLT":    "Treasury Bond ETF",
    "DX-Y.NYB": "Dollar Index",
    # US Stocks
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "META": "Meta",
    "JPM": "JPMorgan",
    "GS": "Goldman Sachs",
    "XOM": "Exxon",
    # Airlines
    "DAL": "Delta Air Lines",
    "LUV": "Southwest Airlines",
    "AAL": "American Airlines",
    "UAL": "United Airlines",
    "CCL": "Carnival",
    # Turkish Stocks (BIST)
    "THYAO.IS": "Turkish Airlines",
    "XU100.IS": "BIST 100 Index",
    "AKBNK.IS": "Akbank",
    "GARAN.IS": "Garanti BBVA",
    "ISCTR.IS": "Isbank",
    "YKBNK.IS": "Yapi Kredi",
    "HALKB.IS": "Halkbank",
    "KCHOL.IS": "Koc Holding",
    "SAHOL.IS": "Sabanci Holding",
    "SISE.IS": "Sise Cam",
    "TUPRS.IS": "TUPRAS",
    "EREGL.IS": "Eregli Demir Celik",
    "TOASO.IS": "Tofas",
    "FROTO.IS": "Ford Otosan",
    "ARCLK.IS": "Arcelik",
    "BIMAS.IS": "BIM Magazalar",
    "MGROS.IS": "Migros",
    "TCELL.IS": "Turkcell",
    "TTKOM.IS": "Turk Telekom",
    "VESTL.IS": "Vestel",
    "PETKM.IS": "Petkim",
    "SASA.IS": "SASA Polyester",
    "HEKTS.IS": "Hektas",
    "PGSUS.IS": "Pegasus Airlines",
    "CLEBI.IS": "Celebi",
    "DOHOL.IS": "Dogus Holding",
    "ENJSA.IS": "Enerjisa",
    "KRDMD.IS": "Kardemir",
    "ODAS.IS": "Odas Elektrik",
    "OTKAR.IS": "Otokar",
    "TAVHL.IS": "TAV Havalimanlari",
    "ULKER.IS": "Ulker",
    "VAKBN.IS": "Vakifbank",
    "YATAS.IS": "Yatas",
}

# Preset scenarios
PRESETS = {
    "1":  {"name": "Gold (Safe Haven)",        "target": "GC=F",      "features": ["SI=F","CL=F","^VIX","DX-Y.NYB"]},
    "2":  {"name": "Oil (Energy & Macro)",     "target": "CL=F",      "features": ["GC=F","BTC-USD","^GSPC","DX-Y.NYB"]},
    "3":  {"name": "Bitcoin (Crypto & Risk)",  "target": "BTC-USD",   "features": ["GC=F","^GSPC","^IXIC","DX-Y.NYB"]},
    "4":  {"name": "S&P 500 (US Market)",      "target": "SPY",       "features": ["QQQ","^VIX","GC=F","DX-Y.NYB"]},
    "5":  {"name": "USD/TRY (Forex)",          "target": "USDTRY=X",  "features": ["GC=F","CL=F","BTC-USD","^VIX"]},
    "6":  {"name": "NVIDIA (Tech Stock)",       "target": "NVDA",      "features": ["AMD","TSM","^GSPC","^VIX"]},
    "7":  {"name": "Tesla (EV Stock)",         "target": "TSLA",      "features": ["AAPL","NIO","^GSPC","^VIX"]},
    "8":  {"name": "THYAO.IS (Turkish Airlines)","target": "THYAO.IS","features": ["XU100.IS","USDTRY=X","AKBNK.IS","GARAN.IS","KCHOL.IS"]},
    "9":  {"name": "BIST 100 (Turkish Market)","target": "XU100.IS",  "features": ["USDTRY=X","AKBNK.IS","GARAN.IS","^VIX","GC=F"]},
    "10": {"name": "Akbank (Turkish Bank)",    "target": "AKBNK.IS",  "features": ["XU100.IS","GARAN.IS","USDTRY=X","^VIX","GC=F"]},
    "11": {"name": "Turkcell (Turkish Telecom)","target": "TCELL.IS", "features": ["XU100.IS","TTKOM.IS","USDTRY=X","^VIX"]},
    "12": {"name": "Auto-detect correlations",  "target": None,        "features": None},
    "13": {"name": "Custom (manual selection)", "target": None,        "features": None},
}


# ============================================================
# CATALOG DISPLAY
# ============================================================
def show_catalog():
    categories = {
        "COMMODITIES":  ["GC=F","SI=F","PL=F","CL=F","BZ=F","NG=F","ZC=F","ZW=F"],
        "CRYPTO":       ["BTC-USD","ETH-USD","SOL-USD","BNB-USD"],
        "FOREX":        ["EURUSD=X","GBPUSD=X","USDTRY=X","EURTRY=X"],
        "US INDICES":   ["^GSPC","^IXIC","^VIX","^DJI","SPY","QQQ","GLD","SLV","USO","TLT"],
        "US STOCKS":    ["AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","JPM","GS","XOM","DAL","LUV","AAL","UAL"],
        "TURKISH":["THYAO.IS","XU100.IS","AKBNK.IS","GARAN.IS","ISCTR.IS","YKBNK.IS","KCHOL.IS","SAHOL.IS","TCELL.IS","BIMAS.IS","FROTO.IS","ARCLK.IS","TOASO.IS","VAKBN.IS","TAVHL.IS","PGSUS.IS"],
    }
    print(f"\n{'='*60}")
    print("  AVAILABLE ASSETS")
    print(f"{'='*60}")
    for cat, symbols in categories.items():
        print(f"\n  [{cat}]")
        for s in symbols:
            label = ASSET_CATALOG.get(s, s)
            print(f"    {s:<14s} -> {label}")
    print()


def get_user_selection():
    show_catalog()

    print(f"\n{'='*60}")
    print("  PRESET SCENARIOS")
    print(f"{'='*60}")
    for key, preset in PRESETS.items():
        print(f"\n  [{key:>2s}] {preset['name']}")
        if preset['features']:
            feats = ", ".join(preset['features'][:4])
            print(f"       Target: {preset['target']} | Features: {feats}...")
    print()

    choice = input("  Select option (1-13): ").strip()

    if choice in PRESETS and choice in ("12","13"):
        if choice == "12":
            # Auto-detect correlations
            target = input("  Target symbol: ").strip().upper()
            if target not in ASSET_CATALOG:
                print(f"  WARNING: '{target}' not in catalog, trying anyway...")
            features = auto_detect_features(target)
            return target, features
        else:
            # Custom
            target = input("  Target symbol: ").strip().upper()
            feats_input = input("  Predictor symbols (comma-separated): ").strip().upper()
            features = [s.strip() for s in feats_input.split(",") if s.strip()]
            features = [s for s in features if s != target]
            return target, features
    elif choice in PRESETS:
        p = PRESETS[choice]
        print(f"\n  Selected: {p['name']}")
        print(f"  Target:   {p['target']}")
        print(f"  Features: {p['features']}")
        return p['target'], p['features']
    else:
        print("  Invalid selection, using Gold preset.")
        return "GC=F", PRESETS["1"]["features"]


def auto_detect_features(target, period="2y", top_n=5):
    """Automatically find top-N correlated assets for target.

    v9 fix: Uses only the first 80% of data for correlation computation
    to avoid look-ahead bias. Feature selection must never see test data.
    """
    print(f"\n  [AUTO-DETECT] Finding top {top_n} correlated assets for {target}...")

    # Download target + all candidates
    candidates = [s for s in ASSET_CATALOG.keys() if s != target]
    all_symbols = [target] + candidates

    # Download in batches to avoid Yahoo limits
    raw = yf.download(all_symbols, period=period, interval="1d", progress=False)["Close"]
    if hasattr(raw.columns, 'levels'):
        raw.columns = raw.columns.get_level_values(0)
    # v8: Only drop columns that are entirely NaN, then forward-fill remaining gaps
    raw = raw.dropna(axis=1, how="all")
    raw = raw.ffill().dropna()

    if target not in raw.columns:
        print(f"  ERROR: {target} not found in data")
        return ["^GSPC", "^VIX", "GC=F", "DX-Y.NYB"]

    returns = raw.pct_change().dropna()
    if target not in returns.columns:
        print(f"  ERROR: {target} not in returns")
        return ["^GSPC", "^VIX", "GC=F", "DX-Y.NYB"]

    # v9: Use only the first 80% of data for correlation (train portion)
    # to prevent look-ahead bias from leaking test-period information
    # into feature selection
    split_idx = int(len(returns) * 0.8)
    train_returns = returns.iloc[:split_idx]

    # Lag-1 predictive correlation: "today's X predicts tomorrow's target"
    corr = train_returns.shift(1).corrwith(train_returns[target]).drop(target, errors='ignore')
    abs_corr = corr.abs().sort_values(ascending=False)
    top_features = abs_corr.head(top_n).index.tolist()

    print(f"  Top {top_n} correlated (train-only, {split_idx} days):")
    for i, s in enumerate(top_features):
        c = corr.get(s, 0)
        name = ASSET_CATALOG.get(s, s)
        print(f"    {i+1}. {s:<14s} {name:<25s} r = {c:>+.4f}")

    return top_features


# ============================================================
# DATA DOWNLOAD
# ============================================================
def download_data(target_symbol, feature_symbols, period="5y"):
    all_symbols = list(set([target_symbol] + feature_symbols))
    extra = ["DX-Y.NYB", "TLT"]
    for es in extra:
        if es not in all_symbols:
            all_symbols.append(es)

    print(f"\n  Downloading {len(all_symbols)} assets...")
    raw = None
    for attempt in range(1, 4):
        try:
            result = yf.download(all_symbols, period=period, interval="1d", progress=False)
            if result is None or result.empty:
                print(f"  WARNING: yfinance returned no data on attempt {attempt}.")
            else:
                raw = result["Close"] if "Close" in result else result
                break
        except Exception as e:
            msg = str(e)
            print(f"  WARNING: Download attempt {attempt} failed: {msg}")
            if attempt < 3:
                time.sleep(1)
                continue
            return None, None, None

    if raw is None or raw.empty:
        print(f"  ERROR: No data downloaded for symbols: {all_symbols}")
        return None, None, None

    if hasattr(raw.columns, 'levels'):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.dropna()
    if raw.empty:
        print(f"  ERROR: No data remains after dropna() for symbols: {all_symbols}")
        return None, None, None

    for col in raw.columns:
        raw[col] = raw[col].astype(float)

    rename_map = {}
    for s in all_symbols:
        base = s.replace("=X","").replace("-USD","").replace("=F","").replace("^","").replace("-Y.NYB","").replace(".IS","")
        rename_map[s] = base
    raw.rename(columns=rename_map, inplace=True)

    target_col = rename_map.get(target_symbol, target_symbol)
    feature_cols = [rename_map.get(s, s) for s in feature_symbols]
    for extra_name in ["DX", "TLT"]:
        if extra_name in raw.columns and extra_name not in feature_cols and extra_name != target_col:
            feature_cols.append(extra_name)

    print(f"  Dataset: {raw.shape[0]} days | {raw.index[0].date()} -> {raw.index[-1].date()}")
    return raw, target_col, feature_cols


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def engineer_features(data, target_col, feature_cols, mode="both"):
    """
    mode: 'regression', 'classification', or 'both'
    Returns X, y_reg, y_cls, feat_cols, dates, regime_data

    v8 fix: All features are now return/rate-based (stationary) instead of
    raw price levels. This prevents the model from learning spurious
    correlations from non-stationary price series.
    """
    df = data.copy()
    all_cols = [target_col] + feature_cols

    for col in all_cols:
        # === RETURN-BASED FEATURES (stationary) ===
        # Daily returns at various horizons
        df[f"{col}_1d_ret"] = df[col].pct_change(1)
        df[f"{col}_2d_ret"] = df[col].pct_change(2)
        df[f"{col}_3d_ret"] = df[col].pct_change(3)
        df[f"{col}_5d_ret"] = df[col].pct_change(5)
        df[f"{col}_10d_ret"] = df[col].pct_change(10)
        df[f"{col}_14d_ret"] = df[col].pct_change(14)
        df[f"{col}_21d_ret"] = df[col].pct_change(21)

        # Lagged returns (not lagged prices!)
        for lag in [1, 2, 3, 5, 7, 14]:
            df[f"{col}_ret_lag{lag}"] = df[f"{col}_1d_ret"].shift(lag)

        # Rolling return statistics
        for window in [5, 10, 20, 50]:
            df[f"{col}_ret_ma{window}"] = df[f"{col}_1d_ret"].rolling(window).mean()
            df[f"{col}_ret_std{window}"] = df[f"{col}_1d_ret"].rolling(window).std()

        # Price relative to moving averages (ratio, not difference — stationary)
        for window in [5, 10, 20, 50]:
            ma = df[col].rolling(window).mean()
            df[f"{col}_price_to_ma{window}"] = df[col] / ma - 1.0

        # RSI (already stationary — based on price differences)
        delta = df[col].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df[f"{col}_rsi"] = 100 - (100 / (1 + rs))

        # MACD — normalized by price for stationarity
        ema12 = df[col].ewm(span=12, adjust=False).mean()
        ema26 = df[col].ewm(span=26, adjust=False).mean()
        df[f"{col}_macd"] = (ema12 - ema26) / df[col]
        df[f"{col}_macd_signal"] = df[f"{col}_macd"].ewm(span=9, adjust=False).mean()
        df[f"{col}_macd_hist"] = df[f"{col}_macd"] - df[f"{col}_macd_signal"]

        # Bollinger Bands — normalized (stationary)
        ma20 = df[col].rolling(20).mean()
        std20 = df[col].rolling(20).std()
        df[f"{col}_bb_upper"] = (ma20 + 2 * std20) / ma20 - 1.0
        df[f"{col}_bb_lower"] = (ma20 - 2 * std20) / ma20 - 1.0
        df[f"{col}_bb_width"] = std20 / ma20
        df[f"{col}_bb_pct"] = (df[col] - (ma20 - 2 * std20)) / (4 * std20)

        # Volatility (return-based, already stationary)
        df[f"{col}_volatility"] = df[f"{col}_1d_ret"].rolling(20).std()
        df[f"{col}_volatility_50"] = df[f"{col}_1d_ret"].rolling(50).std()

        # Skewness & kurtosis of returns
        df[f"{col}_skew20"] = df[f"{col}_1d_ret"].rolling(20).skew()
        df[f"{col}_kurt20"] = df[f"{col}_1d_ret"].rolling(20).kurt()

    # === CROSS-ASSET FEATURES ===
    for fc in feature_cols:
        if fc in df.columns and fc != target_col:
            ratio = df[target_col] / df[fc]
            df[f"target_to_{fc}_ratio"] = ratio.pct_change(1)
            df[f"target_to_{fc}_ret_diff"] = df[f"{target_col}_1d_ret"] - df[f"{fc}_1d_ret"]

    # Targets
    df["TARGET_RETURN"] = df[target_col].pct_change(1).shift(-1)
    df["TARGET_CLASS"] = (df["TARGET_RETURN"] > 0).astype(int)

    # Regime features (binary — already stationary)
    df[f"{target_col}_ma50"] = df[target_col].rolling(50).mean()
    df[f"{target_col}_ma200"] = df[target_col].rolling(200).mean()
    df["above_ma50"] = (df[target_col] > df[f"{target_col}_ma50"]).astype(int)
    df["above_ma200"] = (df[target_col] > df[f"{target_col}_ma200"]).astype(int)
    df["ma50_above_ma200"] = (df[f"{target_col}_ma50"] > df[f"{target_col}_ma200"]).astype(int)

    if "VIX" in df.columns:
        df["VIX_level"] = df["VIX"]
        df["VIX_ma20"] = df["VIX"].rolling(20).mean()
        df["VIX_to_ma20"] = df["VIX"] / df["VIX_ma20"] - 1.0

    df = df.dropna()

    # Feature columns (exclude targets, raw prices, and regime columns)
    exclude = [target_col, "TARGET_RETURN", "TARGET_CLASS",
               f"{target_col}_ma50", f"{target_col}_ma200",
               "above_ma50", "above_ma200", "ma50_above_ma200",
               "VIX_level", "VIX_ma20", "VIX_to_ma20"]
    # Exclude raw price columns for non-target assets
    for c in all_cols:
        if c != target_col and c not in exclude:
            exclude.append(c)
    feat_cols_final = [c for c in df.columns if c not in exclude]

    X = df[feat_cols_final].values
    y_reg = df["TARGET_RETURN"].values
    y_cls = df["TARGET_CLASS"].values
    dates = df.index
    prices = df[target_col].values

    regime_data = {
        "vix_level": df["VIX_level"].values if "VIX_level" in df.columns else None,
        "above_ma50": df["above_ma50"].values,
        "above_ma200": df["above_ma200"].values,
        "prices": prices,
    }

    cls_mean = y_cls.mean()
    balance_warning = ""
    if cls_mean > 0.60 or cls_mean < 0.40:
        balance_warning = " ⚠️ IMBALANCED — class weights applied!"
    print(f"  Features: {len(feat_cols_final)} | Samples: {len(df)} | Class balance: {cls_mean:.1%} up{balance_warning}")
    return X, y_reg, y_cls, feat_cols_final, dates, regime_data




# ============================================================
# FEATURE SELECTION
# ============================================================
def select_features(X_train, y_train, feat_cols, max_features=25, task="classification"):
    if len(feat_cols) <= max_features:
        return X_train, feat_cols, list(range(len(feat_cols))), None, None

    print(f"  Selecting top {max_features} features (permutation importance)...")
    # v8: Scale only for feature selection, return indices only
    fs_scaler = RobustScaler()
    X_fs = fs_scaler.fit_transform(X_train)

    if task == "classification":
        fs_model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42, n_jobs=-1)
    else:
        fs_model = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42, n_jobs=-1)

    fs_model.fit(X_fs, y_train)
    perm = permutation_importance(fs_model, X_fs, y_train, n_repeats=10, random_state=42, n_jobs=-1)
    imp_order = perm.importances_mean.argsort()[::-1]
    top_features = [feat_cols[i] for i in imp_order[:max_features]]
    selected_idx = [feat_cols.index(f) for f in top_features]
    print(f"  Top 5: {', '.join(top_features[:5])}")
    return X_train[:, selected_idx], top_features, selected_idx, perm, imp_order


# ============================================================
# REGRESSION MODELS
# ============================================================
def get_regressors():
    return {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=-1),
        "ANN": MLPRegressor(hidden_layer_sizes=(16,), activation="relu", solver="adam",
                            max_iter=2000, random_state=42, early_stopping=True,
                            validation_fraction=0.2, learning_rate="adaptive", alpha=1.0),
        "XGBoost": XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.3,
                                 reg_lambda=1.5, min_child_weight=5, random_state=42, n_jobs=-1, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, reg_alpha=0.3,
                                   reg_lambda=1.5, min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1),
    }


def train_regressors(X_train, y_train, X_test, y_test):
    tscv = TimeSeriesSplit(n_splits=3)
    models = get_regressors()
    results = {}
    best_name = ""
    best_r2 = -float("inf")

    print(f"\n  {'Model':<18s} {'Train R²':>10s} {'Test R²':>10s} {'CV R²':>10s} {'Dir Acc':>9s}")
    print(f"  {'-'*60}")

    for name, model in models.items():
        cv_scores = []
        for tr_idx, val_idx in tscv.split(X_train):
            m = type(model)(**model.get_params())
            m.fit(X_train[tr_idx], y_train[tr_idx])
            cv_scores.append(r2_score(y_train[val_idx], m.predict(X_train[val_idx])))

        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        dir_acc = np.mean(np.sign(y_pred_test) == np.sign(y_test))
        cv_mean = np.mean(cv_scores)

        # v9 fix: Compute risk metrics per model so plots can display them
        risk = compute_risk_metrics(y_pred_test, y_test)

        results[name] = {
            "train_r2": train_r2, "test_r2": test_r2,
            "cv_mean": cv_mean, "dir_acc": dir_acc,
            # Risk metrics for plotting
            "win_rate": risk["win_rate"],
            "sharpe": risk["sharpe"],
            "total_return": risk["total_return"],
            "bh_return": risk["bh_return"],
            "max_drawdown": risk["max_drawdown"],
            "profit_factor": risk["profit_factor"],
        }
        marker = ""
        if test_r2 > best_r2:
            best_r2 = test_r2
            best_name = name
            marker = " ★"
        print(f"  {name:<18s} {train_r2:>10.4f} {test_r2:>10.4f} {cv_mean:>10.4f} {dir_acc:>8.1%}{marker}")

    print(f"  Best regressor: {best_name} (Test R²={best_r2:.4f})")
    return results, best_name


# ============================================================
# CLASSIFICATION MODELS
# ============================================================
def get_classifiers():
    return {
        "LogisticRegression": LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=-1, class_weight="balanced"),
        "XGBoost": XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, reg_alpha=0.3,
                                  reg_lambda=1.5, min_child_weight=5, random_state=42, n_jobs=-1,
                                  verbosity=0, eval_metric="logloss", scale_pos_weight=1),
        "LightGBM": LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.3,
                                    reg_lambda=1.5, min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1, is_unbalance=True),
    }


def train_classifiers(X_train, y_train, X_test, y_test):
    tscv = TimeSeriesSplit(n_splits=3)
    models = get_classifiers()
    results = {}
    trained = {}
    best_name = ""
    # v9 fix: Use AUC instead of accuracy for model selection
    # Accuracy is misleading with imbalanced class distributions
    best_auc = 0.0

    print(f"\n  {'Model':<20s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'AUC':>7s} {'CV':>7s}")
    print(f"  {'-'*60}")

    for name, model in models.items():
        cv_scores = []
        cv_aucs = []
        for tr_idx, val_idx in tscv.split(X_train):
            m = type(model)(**model.get_params())
            m.fit(X_train[tr_idx], y_train[tr_idx])
            val_pred = m.predict(X_train[val_idx])
            cv_scores.append(accuracy_score(y_train[val_idx], val_pred))
            try:
                val_prob = m.predict_proba(X_train[val_idx])[:, 1]
                cv_aucs.append(roc_auc_score(y_train[val_idx], val_prob))
            except (ValueError, AttributeError):
                cv_aucs.append(0.5)

        model.fit(X_train, y_train)
        trained[name] = model

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.0
        cv_mean = np.mean(cv_scores)
        cv_auc_mean = np.mean(cv_aucs)

        results[name] = {
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "auc": auc, "cv_mean": cv_mean, "cv_auc_mean": cv_auc_mean,
            "y_pred": y_pred, "y_prob": y_prob,
        }
        marker = ""
        # v9: Select best model by AUC (more robust than accuracy for imbalanced data)
        if auc > best_auc:
            best_auc = auc
            best_name = name
            marker = " ★"
        print(f"  {name:<20s} {acc:>6.1%} {prec:>6.1%} {rec:>6.1%} {f1:>6.3f} {auc:>6.3f} {cv_mean:>6.1%}{marker}")

    print(f"  Best classifier: {best_name} (AUC={best_auc:.3f})")
    return results, trained, best_name


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================
def threshold_analysis(y_prob, y_test, model_name, prob_thresholds=None):
    if prob_thresholds is None:
        prob_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

    print(f"\n  --- Threshold Analysis ({model_name}) ---")
    print(f"  {'Thresh':>8s} {'Trades':>7s} {'Cover%':>8s} {'Win Rate':>9s} {'Sharpe':>8s}")
    print(f"  {'-'*45}")

    results = {}
    for thresh in prob_thresholds:
        long_mask = y_prob > thresh
        short_mask = y_prob < (1 - thresh)
        trade_mask = long_mask | short_mask
        n = trade_mask.sum()
        if n > 0:
            lc = (y_test[long_mask] == 1).sum() if long_mask.any() else 0
            sc = (y_test[short_mask] == 0).sum() if short_mask.any() else 0
            wr = (lc + sc) / n
            cov = n / len(y_test)
            rets = np.where(trade_mask, 2 * (y_test == (y_prob > 0.5).astype(int)).astype(float) - 1, np.nan)
            valid = rets[trade_mask]
            sr = (valid.mean() / valid.std() * np.sqrt(252)) if valid.std() > 1e-10 else 0
            print(f"  {thresh:>8.2f} {n:>7d} {cov:>7.1%} {wr:>8.1%} {sr:>8.3f}")
            results[thresh] = {"n_trades": int(n), "win_rate": float(wr), "coverage": float(cov), "sharpe": float(sr)}
        else:
            print(f"  {thresh:>8.2f} {0:>7d} {'0.0%':>8s} {'N/A':>9s} {'N/A':>8s}")
            results[thresh] = {"n_trades": 0, "win_rate": 0, "coverage": 0, "sharpe": 0}

    return results


# ============================================================
# REGIME ANALYSIS
# ============================================================
def regime_analysis(y_prob, y_test, vix_test, ma200_test, model_name, prob_threshold=0.55):
    print(f"\n  --- Regime Analysis ({model_name}, thresh={prob_threshold}) ---")

    # VIX regimes
    if vix_test is not None:
        print(f"\n  VIX Regimes:")
        for vt in [15, 18, 20, 22, 25]:
            low = vix_test < vt
            high = ~low
            n_low, n_high = low.sum(), high.sum()
            if n_low > 0:
                lp = (y_prob[low] > 0.5).astype(int)
                lw = (lp == y_test[low]).mean()
            else:
                lw = 0
            if n_high > 0:
                hp = (y_prob[high] > 0.5).astype(int)
                hw = (hp == y_test[high]).mean()
            else:
                hw = 0
            print(f"    VIX < {vt:>2d}: Low={lw:.1%} ({n_low}d) | High={hw:.1%} ({n_high}d)")

    # Trend regimes
    if ma200_test is not None:
        up = ma200_test.astype(bool)
        down = ~up
        n_up, n_down = up.sum(), down.sum()
        if n_up > 0:
            uw = ((y_prob[up] > 0.5).astype(int) == y_test[up]).mean()
        else:
            uw = 0
        if n_down > 0:
            dw = ((y_prob[down] > 0.5).astype(int) == y_test[down]).mean()
        else:
            dw = 0
        print(f"\n  Trend Regimes:")
        print(f"    Uptrend (MA200):  {uw:.1%} ({n_up}d)")
        print(f"    Downtrend (MA200): {dw:.1%} ({n_down}d)")

    # Combined: VIX + Trend grid
    print(f"\n  --- Combined Regime + Threshold Grid ---")
    best_wr = 0
    best_cfg = None

    for pt in [0.50, 0.55, 0.60, 0.65]:
        # No filter
        tm = (y_prob > pt) | (y_prob < (1 - pt))
        n = tm.sum()
        if n >= 10:
            wr = ((y_prob[tm] > 0.5).astype(int) == y_test[tm]).mean()
            if wr > best_wr:
                best_wr = wr
                best_cfg = {"type": "none", "prob_threshold": pt, "win_rate": wr, "n_trades": n}

        # VIX filter
        if vix_test is not None:
            for vt in [18, 20, 22, 25]:
                vm = vix_test < vt
                tm2 = tm & vm
                n2 = tm2.sum()
                if n2 >= 10:
                    wr2 = ((y_prob[tm2] > 0.5).astype(int) == y_test[tm2]).mean()
                    if wr2 > best_wr:
                        best_wr = wr2
                        best_cfg = {"type": "vix", "vix_threshold": vt, "prob_threshold": pt, "win_rate": wr2, "n_trades": n2}

        # MA200 filter
        if ma200_test is not None:
            tm3 = tm & ma200_test.astype(bool)
            n3 = tm3.sum()
            if n3 >= 10:
                wr3 = ((y_prob[tm3] > 0.5).astype(int) == y_test[tm3]).mean()
                if wr3 > best_wr:
                    best_wr = wr3
                    best_cfg = {"type": "ma200", "prob_threshold": pt, "win_rate": wr3, "n_trades": n3}

        # VIX + MA200
        if vix_test is not None and ma200_test is not None:
            for vt in [20, 22, 25]:
                vm = vix_test < vt
                tm4 = tm & vm & ma200_test.astype(bool)
                n4 = tm4.sum()
                if n4 >= 10:
                    wr4 = ((y_prob[tm4] > 0.5).astype(int) == y_test[tm4]).mean()
                    if wr4 > best_wr:
                        best_wr = wr4
                        best_cfg = {"type": "vix+ma200", "vix_threshold": vt, "prob_threshold": pt, "win_rate": wr4, "n_trades": n4}

    if best_cfg:
        print(f"  ★ BEST: Win Rate={best_cfg['win_rate']:.1%} | Trades={best_cfg['n_trades']}")
        print(f"    Config: {best_cfg}")

    return best_cfg


# ============================================================
# ENSEMBLE
# ============================================================
def build_ensemble(results, y_test):
    """Soft voting ensemble weighted by CV AUC score.

    v9 fix: Use CV-AUC for weighting instead of test-set AUC.
    Using test-set AUC for ensemble weights is data leakage —
    in a real system, weights must be determined on validation data only.
    """
    print(f"\n  --- Ensemble (Soft Voting) ---")
    ens_prob = np.zeros(len(y_test))
    w_total = 0
    for name, res in results.items():
        # v9: Use CV-AUC for weighting (no test-set leakage)
        cv_auc = res.get("cv_auc_mean", 0.5)
        w = max(0.01, (cv_auc - 0.5) ** 2)  # Squared to emphasize better models
        ens_prob += w * res["y_prob"]
        w_total += w
        print(f"    {name}: CV-AUC={cv_auc:.3f}, weight={w:.4f}")
    if w_total > 0:
        ens_prob /= w_total

    ens_pred = (ens_prob > 0.5).astype(int)
    ens_acc = accuracy_score(y_test, ens_pred)
    ens_f1 = f1_score(y_test, ens_pred, zero_division=0)
    try:
        ens_auc = roc_auc_score(y_test, ens_prob)
    except ValueError:
        ens_auc = 0.0
    print(f"    Ensemble: Acc={ens_acc:.1%} | F1={ens_f1:.3f} | AUC={ens_auc:.3f}")
    return ens_prob, ens_acc


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================
def walk_forward_validation(X, y, task="classification", n_splits=5):
    """Walk-forward validation.

    v9 fix: Evaluates ALL models (not just LightGBM) for fair comparison.
    Returns per-model scores so results are comparable with training phase.
    """
    n_samples = len(X)
    min_train = int(n_samples * 0.5)
    test_sz = n_samples - min_train
    fold_sz = test_sz // n_splits

    if task == "classification":
        all_models = {
            "LogisticRegression": LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1, class_weight="balanced"),
            "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=-1, class_weight="balanced"),
            "XGBoost": XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8, reg_alpha=0.3,
                                      reg_lambda=1.5, min_child_weight=5, random_state=42, n_jobs=-1,
                                      verbosity=0, eval_metric="logloss", scale_pos_weight=1),
            "LightGBM": LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                                        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.3,
                                        reg_lambda=1.5, min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1, is_unbalance=True),
        }
    else:
        all_models = {
            "Ridge": Ridge(alpha=1.0),
            "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=-1),
            "ANN": MLPRegressor(hidden_layer_sizes=(16,), activation="relu", solver="adam",
                                max_iter=2000, random_state=42, early_stopping=True,
                                validation_fraction=0.2, learning_rate="adaptive", alpha=1.0),
            "XGBoost": XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8, reg_alpha=0.3,
                                     reg_lambda=1.5, min_child_weight=5, random_state=42, n_jobs=-1, verbosity=0),
            "LightGBM": LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8, reg_alpha=0.3,
                                       reg_lambda=1.5, min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1),
        }

    # Per-model scores
    model_scores = {name: {"acc": [], "f1": []} for name in all_models}

    for i in range(n_splits):
        tr_end = min_train + i * fold_sz
        te_end = n_samples if i == n_splits - 1 else min(tr_end + fold_sz, n_samples)
        if te_end <= tr_end:
            continue

        X_tr, y_tr = X[:tr_end], y[:tr_end]
        X_te, y_te = X[tr_end:te_end], y[tr_end:te_end]

        s = RobustScaler()
        X_tr_s = s.fit_transform(X_tr)
        X_te_s = s.transform(X_te)

        for name, model_template in all_models.items():
            m = type(model_template)(**model_template.get_params())
            m.fit(X_tr_s, y_tr)
            p = m.predict(X_te_s)

            if task == "classification":
                model_scores[name]["acc"].append(accuracy_score(y_te, p))
                model_scores[name]["f1"].append(f1_score(y_te, p, zero_division=0))
            else:
                model_scores[name]["acc"].append(r2_score(y_te, p))

    # Print results for all models
    for name, scores in model_scores.items():
        if scores["acc"]:
            if task == "classification":
                print(f"  Walk-Forward {name}: Acc={np.mean(scores['acc']):.1%} ± {np.std(scores['acc']):.1%} | "
                      f"F1={np.mean(scores['f1']):.3f} ± {np.std(scores['f1']):.3f}")
            else:
                print(f"  Walk-Forward {name}: R²={np.mean(scores['acc']):.4f} ± {np.std(scores['acc']):.4f}")

    return model_scores


# ============================================================
# RISK METRICS
# ============================================================
def compute_risk_metrics(y_pred, y_actual):
    positions = np.sign(y_pred)
    strat_returns = positions * y_actual.ravel()
    bh_returns = y_actual.ravel()

    cum_strat = np.cumprod(1 + strat_returns) - 1
    cum_bh = np.cumprod(1 + bh_returns) - 1

    total_return = cum_strat[-1] if len(cum_strat) > 0 else 0.0
    bh_return = cum_bh[-1] if len(cum_bh) > 0 else 0.0

    mean_ret = np.mean(strat_returns)
    std_ret = np.std(strat_returns)
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 1e-10 else 0.0

    running_max = np.maximum.accumulate(1 + cum_strat)
    drawdowns = (1 + cum_strat) / running_max - 1
    max_dd = np.min(drawdowns)

    wins = np.sum(strat_returns > 0)
    total_trades = np.sum(strat_returns != 0)
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    gp = np.sum(strat_returns[strat_returns > 0])
    gl = np.abs(np.sum(strat_returns[strat_returns < 0]))
    pf = gp / gl if gl > 1e-10 else float("inf")

    return {
        "sharpe": sharpe, "max_drawdown": max_dd, "win_rate": win_rate,
        "profit_factor": pf, "total_return": total_return, "bh_return": bh_return,
    }


# ============================================================
# PLOTTING
# ============================================================
def create_plots(reg_results, clf_results, best_clf_name, ens_prob, y_test, dates_test, prices_test,
                 target_name, regime_data, output_dir):
    """Create comprehensive analysis plots."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))

    # 1. Price chart
    ax = axes[0, 0]
    n = min(60, len(prices_test))
    ax.plot(range(n), prices_test[-n:], "b-", lw=1.5)
    ax.set_title(f"{target_name} Price (last {n} days)", fontweight="bold")
    ax.grid(alpha=0.3)

    # 2. Regression R² comparison
    ax = axes[0, 1]
    if reg_results:
        names = list(reg_results.keys())
        test_r2 = [reg_results[nm]["test_r2"] for nm in names]
        colors = ["#4CAF50" if r >= 0 else "#F44336" for r in test_r2]
        ax.barh(names, test_r2, color=colors, edgecolor="black", alpha=0.8)
        ax.axvline(x=0, color="gray", ls="--", lw=1)
        ax.set_title("Regression Test R²", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    # 3. Classification accuracy comparison
    ax = axes[0, 2]
    if clf_results:
        names = list(clf_results.keys())
        accs = [clf_results[nm]["accuracy"] for nm in names]
        f1s = [clf_results[nm]["f1"] for nm in names]
        x = np.arange(len(names))
        ax.bar(x - 0.2, accs, 0.35, label="Accuracy", color="#4CAF50", alpha=0.8)
        ax.bar(x + 0.2, f1s, 0.35, label="F1", color="#2196F3", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)
        ax.axhline(y=0.5, color="red", ls="--", lw=1)
        ax.set_title("Classification Performance", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # 4. Threshold win rate
    ax = axes[1, 0]
    if clf_results and best_clf_name:
        prob = clf_results[best_clf_name]["y_prob"]
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
        wrs = []
        for t in thresholds:
            tm = (prob > t) | (prob < (1 - t))
            n = tm.sum()
            if n > 0:
                wrs.append(((prob[tm] > 0.5).astype(int) == y_test[tm]).mean())
            else:
                wrs.append(0)
        ax.plot(thresholds, wrs, "-o", color="#4CAF50", lw=2, markersize=8)
        ax.axhline(y=0.5, color="red", ls="--", lw=1)
        ax.axhline(y=0.7, color="green", ls=":", lw=1, alpha=0.5, label="Target 70%")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Win Rate")
        ax.set_title(f"Win Rate vs Threshold ({best_clf_name})", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # 5. Probability over time
    ax = axes[1, 1]
    if ens_prob is not None and len(ens_prob) > 0:
        n = min(60, len(ens_prob))
        ax.plot(range(n), ens_prob[-n:], "b-", lw=1, alpha=0.7)
        ax.axhline(y=0.5, color="red", ls="--", lw=1)
        ax.fill_between(range(n), 0.5, ens_prob[-n:], where=ens_prob[-n:] > 0.5, alpha=0.2, color="green")
        ax.fill_between(range(n), 0.5, ens_prob[-n:], where=ens_prob[-n:] < 0.5, alpha=0.2, color="red")
        ax.set_ylim(0.2, 0.8)
    ax.set_title("Ensemble P(UP) over time", fontweight="bold")
    ax.grid(alpha=0.3)

    # 6. Actual vs Predicted (last 30 days)
    ax = axes[1, 2]
    # v9 fix: actual/pred must be initialized before use regardless of ens_prob
    actual = np.array([])
    pred = np.array([])
    if ens_prob is not None and len(ens_prob) > 0:
        n = min(30, len(y_test))
        actual = y_test[-n:]
        pred = (ens_prob[-n:] > 0.5).astype(int)
    if len(actual) > 0 and len(pred) > 0:
        cols_a = ["green" if a == 1 else "red" for a in actual]
        cols_p = ["green" if p == 1 else "red" for p in pred]
        n = len(actual)
        ax.scatter(range(n), actual, c=cols_a, s=40, marker="o", label="Actual", alpha=0.7)
        ax.scatter(range(n), pred, c=cols_p, s=25, marker="x", label="Predicted", alpha=0.7)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Down", "Up"])
        ax.set_title(f"Actual vs Predicted (last {n}d)", fontweight="bold")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No ensemble predictions\navailable", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
        ax.set_title("Actual vs Predicted", fontweight="bold")
    ax.grid(alpha=0.3)

    # 7. Risk metrics (regression)
    ax = axes[2, 0]
    if reg_results:
        metrics = ["win_rate", "sharpe", "total_return"]
        model_names = list(reg_results.keys())
        x = np.arange(len(model_names))
        for i, m in enumerate(metrics):
            vals = [reg_results[nm].get(m, 0) for nm in model_names]
            ax.bar(x + i * 0.25, vals, 0.25, label=m, alpha=0.8)
        ax.set_xticks(x + 0.25)
        ax.set_xticklabels(model_names, fontsize=7)
        ax.set_title("Regression Risk Metrics", fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    # 8. Cumulative returns
    ax = axes[2, 1]
    if reg_results and ens_prob is not None and len(ens_prob) > 0:
        best_reg = max(reg_results, key=lambda k: reg_results[k]["test_r2"])
        # Use best regression model's direction
        # We don't have predictions stored, so use ensemble
        # v8: Fix return calculation — use proper pct_change
        daily_returns = np.diff(prices_test) / prices_test[:-1]
        daily_returns = np.insert(daily_returns, 0, 0)  # First day = 0 return
        ens_returns = np.where(ens_prob > 0.5, 1, -1) * daily_returns
        cum = np.cumprod(1 + ens_returns) - 1
        bh = np.cumprod(1 + daily_returns) - 1
        ax.plot(range(len(cum)), cum * 100, "b-", lw=1.5, label="Strategy")
        ax.plot(range(len(bh)), bh * 100, "r--", lw=1, label="Buy & Hold")
        ax.set_title("Cumulative Returns %", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # 9. Summary text
    ax = axes[2, 2]
    ax.axis("off")
    summary_lines = [
        f"Target: {target_name}",
        f"Best Classifier: {best_clf_name}",
        f"Best Regressor: {max(reg_results, key=lambda k: reg_results[k]['test_r2']) if reg_results else 'N/A'}",
        f"Ensemble Acc: {accuracy_score(y_test, (ens_prob > 0.5).astype(int)):.1%}" if ens_prob is not None and len(ens_prob) > 0 else "Ensemble Acc: N/A",
        f"Walk-Forward: See console",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ]
    ax.text(0.1, 0.5, "\n".join(summary_lines), transform=ax.transAxes,
            fontsize=12, verticalalignment="center", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_title("Summary", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: analysis.png")


# ============================================================
# NEXT-DAY PREDICTION
# ============================================================
def predict_next_day(clf_results, ens_prob, dates, prices, regime_data, target_name, y_test):
    """Generate next-day prediction."""
    print(f"\n{'='*60}")
    print(f"  🔮 NEXT-DAY PREDICTION FOR {target_name}")
    print(f"{'='*60}")

    current_price = prices[-1]
    current_date = dates[-1]

    print(f"\n  Current ({current_date.date()}): {current_price:.2f}")

    # Last 5 days
    print(f"\n  Last 5 days:")
    for i in range(-5, 0):
        r = (prices[i] / prices[i-1] - 1) * 100
        print(f"    {dates[i].date()}: {prices[i]:.2f} ({r:+.2f}%)")

    # Model predictions
    print(f"\n  Model predictions:")
    for name, res in clf_results.items():
        p = res["y_prob"][-1]
        d = "UP ↑" if p > 0.5 else "DOWN ↓"
        c = max(p, 1 - p)
        print(f"    {name:<20s}: {d} (P={p:.3f}, conf={c:.1%})")

    ens_p = ens_prob[-1]
    ens_d = "UP ↑" if ens_p > 0.5 else "DOWN ↓"
    ens_c = max(ens_p, 1 - ens_p)
    print(f"    {'ENSEMBLE':<20s}: {ens_d} (P={ens_p:.3f}, conf={ens_c:.1%})")

    # Recent accuracy
    print(f"\n  Recent accuracy (last 20 days):")
    for name, res in clf_results.items():
        acc = accuracy_score(y_test[-20:], res["y_pred"][-20:])
        print(f"    {name:<20s}: {acc:.1%}")

    # Price target
    recent_returns = np.diff(prices[-21:]) / prices[-21:-1]
    avg_up = recent_returns[recent_returns > 0].mean() if (recent_returns > 0).any() else 0
    avg_down = recent_returns[recent_returns < 0].mean() if (recent_returns < 0).any() else 0

    if ens_p > 0.5:
        target = current_price * (1 + avg_up)
        print(f"\n  Direction: UP ↑")
        print(f"    Target: {target:.2f} (avg up: {avg_up:.2%})")
    else:
        target = current_price * (1 + avg_down)
        print(f"\n  Direction: DOWN ↓")
        print(f"    Target: {target:.2f} (avg down: {avg_down:.2%})")

    print(f"\n  ⚠️  NOT: Bu bir model tahminidir, yatırım tavsiyesi değildir.")
    print(f"{'='*60}")

    return {"direction": ens_d, "confidence": ens_c, "p_up": ens_p, "target": target}


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n{'='*60}")
    print("  DATA-DRIVEN PREDICTIVE MODELING v7")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Mode selection
    print("\n  MODES:")
    print("    1. Regression (R², Sharpe, etc.)")
    print("    2. Classification (Accuracy, Win Rate, Threshold)")
    print("    3. Full Analysis (Regression + Classification + Regime)")
    print("    4. Quick Classification (fast, no plots)")
    mode = input("\n  Select mode (1-4) [3]: ").strip() or "3"

    # Asset selection
    target_symbol, feature_symbols = get_user_selection()
    if not target_symbol:
        return

    # Download
    data, target_col, feature_cols = download_data(target_symbol, feature_symbols)
    if data is None or len(data) < 200:
        print("  ERROR: Insufficient data")
        return

    # Feature engineering
    X, y_reg, y_cls, feat_cols, dates, regime_data = engineer_features(data, target_col, feature_cols)

    # v8: Replace inf/-inf with NaN, then drop rows with NaN
    X = np.where(np.isinf(X), np.nan, X)
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    y_reg = y_reg[valid_mask]
    y_cls = y_cls[valid_mask]
    dates = dates[valid_mask]
    regime_data["prices"] = regime_data["prices"][valid_mask]
    regime_data["above_ma50"] = regime_data["above_ma50"][valid_mask]
    regime_data["above_ma200"] = regime_data["above_ma200"][valid_mask]
    if regime_data["vix_level"] is not None:
        regime_data["vix_level"] = regime_data["vix_level"][valid_mask]
    print(f"  After NaN/inf cleanup: {len(X)} samples")

    # Split
    split_idx = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_reg_train, y_reg_test = y_reg[:split_idx], y_reg[split_idx:]
    y_cls_train, y_cls_test = y_cls[:split_idx], y_cls[split_idx:]

    # Regime data for test
    vix_test = regime_data["vix_level"][split_idx:] if regime_data["vix_level"] is not None else None
    ma200_test = regime_data["above_ma200"][split_idx:]
    dates_test = dates[split_idx:]
    prices_test = regime_data["prices"][split_idx:]

    # Feature selection
    task = "classification" if mode in ("2", "4") else "both"
    X_train_sel, sel_feat, sel_idx, perm, imp_order = select_features(
        X_train_raw, y_cls_train, feat_cols, max_features=25, task=task
    )
    X_test_sel = X_test_raw[:, sel_idx]

    # Scale
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train_sel)
    X_test = scaler.transform(X_test_sel)

    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(sel_feat)}")

    # ==================== REGRESSION ====================
    reg_results = {}
    if mode in ("1", "3"):
        print(f"\n{'='*50}")
        print("  REGRESSION ANALYSIS")
        print(f"{'='*50}")
        reg_results, best_reg = train_regressors(X_train, y_reg_train, X_test, y_reg_test)

        # Risk metrics for best regressor
        # v9 fix: Risk metrics are now computed inside train_regressors
        if best_reg and best_reg in reg_results:
            r = reg_results[best_reg]
            print(f"\n  Risk Metrics ({best_reg}):")
            print(f"    Sharpe: {r['sharpe']:.3f} | Win Rate: {r['win_rate']:.1%} | "
                  f"Return: {r['total_return']:.1%} | B&H: {r['bh_return']:.1%} | "
                  f"Max DD: {r['max_drawdown']:.1%}")

        # Walk-forward (raw/unscaled data — walk_forward_validation scales internally per fold)
        print(f"\n  Walk-Forward (Regression):")
        wf_reg = walk_forward_validation(
            np.vstack([X_train_sel, X_test_sel]),  # raw features, scaled inside walk_forward_validation
            np.concatenate([y_reg_train, y_reg_test]),
            task="regression"
        )

    # ==================== CLASSIFICATION ====================
    clf_results = {}
    best_clf_name = ""
    ens_prob = None

    if mode in ("2", "3", "4"):
        print(f"\n{'='*50}")
        print("  CLASSIFICATION ANALYSIS")
        print(f"{'='*50}")
        clf_results, trained_clf, best_clf_name = train_classifiers(X_train, y_cls_train, X_test, y_cls_test)

        # Threshold analysis
        best_prob = clf_results[best_clf_name]["y_prob"]
        thresh_results = threshold_analysis(best_prob, y_cls_test, best_clf_name)

        # Ensemble
        ens_prob, ens_acc = build_ensemble(clf_results, y_cls_test)

        # Ensemble threshold
        print(f"\n  --- Ensemble Threshold ---")
        threshold_analysis(ens_prob, y_cls_test, "Ensemble")

        # Regime analysis
        best_regime_cfg = None
        if mode == "3":
            best_regime_cfg = regime_analysis(best_prob, y_cls_test, vix_test, ma200_test, best_clf_name)

        # Walk-forward (raw/unscaled data — walk_forward_validation scales internally per fold)
        print(f"\n  Walk-Forward (Classification):")
        wf_clf = walk_forward_validation(
            np.vstack([X_train_sel, X_test_sel]),  # raw features, scaled inside walk_forward_validation
            np.concatenate([y_cls_train, y_cls_test]),
            task="classification"
        )

        # Next-day prediction
        predict_next_day(clf_results, ens_prob, dates, regime_data["prices"], regime_data, target_symbol, y_cls_test)

    # ==================== PLOTS ====================
    if mode != "4":
        print(f"\n  Creating plots...")
        create_plots(
            reg_results, clf_results, best_clf_name, ens_prob,
            y_cls_test, dates_test, prices_test,
            target_symbol, regime_data, OUTPUT_DIR
        )

    # ==================== SAVE RESULTS ====================
    output = {
        "target": target_symbol,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mode": mode,
        "n_samples": len(X),
        "n_features": len(sel_feat),
        "best_classifier": best_clf_name,
        "classifier_results": {k: {kk: float(vv) for kk, vv in v.items() if kk not in ("y_pred", "y_prob")}
                               for k, v in clf_results.items()},
        "regression_results": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in reg_results.items()},
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: results.json")
    print(f"\n  Analysis complete!")


if __name__ == "__main__":
    main()
