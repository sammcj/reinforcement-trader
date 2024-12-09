import argparse
import asyncio
import uuid
import ccxt.async_support as ccxt_async
import copy
import gymnasium as gym
import json
import logging
import numpy as np
import optuna
import os
import pandas as pd
import signal
import sys
import torch
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from ta import trend, volatility
from ta.momentum import RSIIndicator, StochasticOscillator, UltimateOscillator
from ta.trend import IchimokuIndicator, PSARIndicator, CCIIndicator, TRIXIndicator, MACD
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from typing import Optional


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

load_dotenv()

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")


exchange_config = {
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True,
    "options": {
        "defaultType": "future",
        "adjustForTimeDifference": True,
        "recvWindow": 10000,
    },
    "timeout": 30000,
}


def setup_argparse():
    parser = argparse.ArgumentParser(description="Crypto Trading AI System")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "train", "trade", "simulate"],
        help="Operation mode: full (default), train, trade, or simulate (paper trading)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="DOGE/USDT:USDT",
        help="Trading symbol (default: DOGE/USDT:USDT)",
    )
    parser.add_argument(
        "--timeframe", type=str, default="5m", help="Trading timeframe (default: 5m)"
    )
    parser.add_argument(
        "--models-dir", type=str, default="models", help="Directory for model storage"
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=10.0,
        help="Initial balance for training (default: 10.0)",
    )
    parser.add_argument(
        "--risk-percentage",
        type=float,
        default=0.7,
        help="Risk percentage (default: 0.7)",
    )
    return parser


async def train_model(
    async_exchange: ccxt_async.Exchange,
    symbol: str,
    timeframe: str,
    models_dir: str,
    n_trials: int = 15,
) -> tuple[Optional[PPO], Optional[dict]]:
    """Training mode functionality with proper symbol handling"""
    try:
        if not await verify_symbol(async_exchange, symbol):
            logging.error(f"Symbol {symbol} unavailable")
            return None, None

        # Fetch and prepare data
        df = await get_full_data(async_exchange, symbol, timeframe=timeframe)
        if df is None or df.empty:
            logging.error("Failed to load data or data is empty")
            return None, None

        df = add_technical_indicators(df)
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size].reset_index(drop=True)
        test_df = df.iloc[train_size:].reset_index(drop=True)

        # Optimize and train
        study = optuna.create_study(
            direction="maximize", pruner=optuna.pruners.MedianPruner()
        )
        executor = ThreadPoolExecutor()
        try:
            await run_optuna(
                study, train_df, test_df, n_trials, symbol
            )  # Pass symbol here
            best_params = study.best_params
            logging.info(f"Best optimization parameters: {best_params}")

            model, norm_params = await asyncio.get_event_loop().run_in_executor(
                executor,
                get_or_train_model_sync,
                symbol,
                train_df,
                models_dir,
                best_params,
            )

            # Run backtest
            await asyncio.get_event_loop().run_in_executor(
                executor, backtest_model_sync, model, test_df, symbol, norm_params
            )

            return model, norm_params

        finally:
            executor.shutdown(wait=True)

    except Exception as e:
        logging.error(f"Error in training mode: {e}")
        return None, None


def signal_handler(signum, frame):
    """Handle interrupt signals"""
    print("\nReceived interrupt signal. Cleaning up...")
    # Force exit after cleanup
    sys.exit(0)


async def shutdown_handler(sig):
    """Async handler for shutdown"""
    logging.info(f"Received signal {sig}. Initiating cleanup...")

    # Cancel all running tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()

    logging.info(f"Cancelled {len(tasks)} tasks")

    # Wait for all tasks to complete with a timeout
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        pass

    # Stop the event loop
    loop = asyncio.get_event_loop()
    loop.stop()

    # Force exit if we're still here
    sys.exit(0)


async def trade_model(
    async_exchange: ccxt_async.Exchange,
    model: PPO,
    symbol: str,
    norm_params: dict,
    window_size: int = 20,
    simulate: bool = False,
) -> None:
    """Trading mode functionality"""
    try:
        # Initialize trading state
        state = LiveTradingState(window_size=window_size)

        # Get initial balance
        initial_balance = await get_real_balance_async(async_exchange)
        if initial_balance is None:
            initial_balance = 100.0 if simulate else 10.0
            if simulate:
                logging.info(
                    f"[SIMULATION] Using initial balance: {initial_balance} USDT"
                )

        # Get initial data
        ohlcv = await async_exchange.fetch_ohlcv(
            symbol, timeframe="5m", limit=window_size
        )
        if not ohlcv:
            logging.error("Failed to get initial OHLCV data")
            return

        # Initialize state with historical data
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        for _, row in df.iterrows():
            state.update(row, initial_balance, row["timestamp"])

        # Start live trading
        await live_trading(async_exchange, model, symbol, norm_params, state, simulate)

    except Exception as e:
        logging.error(f"Error in trading mode: {e}")


class TradingEnvironment(gym.Env):
    def __init__(
        self,
        data,
        norm_params=None,
        initial_balance=10,
        risk_percentage=0.7,
        short_term_threshold=10,
        long_term_threshold=50,
        history_size=100,
        window_size=20,
    ):
        super(TradingEnvironment, self).__init__()
        logging.debug("Initializing TradingEnvironment")

        # Ensure we have enough data
        if len(data) < window_size:
            # Replicate the last row to fill up to window_size
            last_row = data.iloc[-1:]
            data = pd.concat(
                [data] + [last_row] * (window_size - len(data)), ignore_index=True
            )

        self.timestamps = data["timestamp"].reset_index(drop=True)
        self.data = data.drop(columns=["timestamp"]).reset_index(drop=True)
        self.initial_balance = initial_balance
        self.risk_percentage = risk_percentage
        self.short_term_threshold = short_term_threshold
        self.long_term_threshold = long_term_threshold
        self.window_size = window_size

        # Rest of the initialization code remains the same
        if norm_params is None:
            self.means = self.data.mean()
            self.stds = self.data.std().replace(0, 1e-8)
        else:
            self.means = pd.Series(norm_params["means"])
            self.stds = pd.Series(norm_params["stds"])
        self.normalized_data = (self.data - self.means) / self.stds
        low = self.normalized_data.min().values - 1
        high = self.normalized_data.max().values + 1
        num_features = self.data.shape[1]
        self.observation_space = spaces.Box(
            low=np.tile(low, self.window_size).astype(np.float32),
            high=np.tile(high, self.window_size).astype(np.float32),
            shape=(self.window_size * num_features,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)  # Changed from 4 to 3
        self.obs_window = deque(maxlen=self.window_size)
        self.history = deque(maxlen=history_size)
        self.reset()
        self.save_state()

    def reset(self, *, seed=None, options=None):
        logging.debug("Resetting environment")
        self.balance = self.initial_balance
        self.previous_balance = self.initial_balance
        self.position = None
        self.entry_price = 0
        self.entry_step = 0
        self.current_step = 0
        self.done = False
        self.total_profit = 0
        self.positions = []
        self.position_size = 0
        self.units = 0
        self.balance_history = [self.balance]
        self.history.clear()
        self.obs_window.clear()
        initial_window = self.normalized_data.iloc[
            self.current_step : self.current_step + self.window_size
        ]
        for _, row in initial_window.iterrows():
            self.obs_window.append(row.values.astype(np.float32))
        self.current_step += self.window_size
        self.save_state()
        return self._get_observation(), {}

    def _get_observation(self):
        if len(self.obs_window) < self.window_size:
            padding = [np.zeros(self.normalized_data.shape[1], dtype=np.float32)] * (
                self.window_size - len(self.obs_window)
            )
            window = list(padding) + list(self.obs_window)
        else:
            window = list(self.obs_window)
        obs = np.concatenate(window)
        return obs.astype(np.float32)

    def save_state(self):
        state = {
            "balance": self.balance,
            "position": self.position,
            "entry_price": self.entry_price,
            "entry_step": self.entry_step,
            "current_step": self.current_step,
            "done": self.done,
            "total_profit": self.total_profit,
            "positions": copy.deepcopy(self.positions),
            "position_size": self.position_size,
            "units": self.units,
            "balance_history": copy.deepcopy(self.balance_history),
            "obs_window": copy.deepcopy(self.obs_window),
            "previous_balance": self.previous_balance,
        }
        self.history.append(state)

    def load_state(self, steps_back=2):
        if len(self.history) >= steps_back:
            state = self.history[-steps_back]
            self.balance = state["balance"]
            self.position = state["position"]
            self.entry_price = state["entry_price"]
            self.entry_step = state["entry_step"]
            self.current_step = state["current_step"]
            self.done = state["done"]
            self.total_profit = state["total_profit"]
            self.positions = copy.deepcopy(state["positions"])
            self.position_size = state["position_size"]
            self.units = state["units"]
            self.balance_history = copy.deepcopy(state["balance_history"])
            self.obs_window = copy.deepcopy(state["obs_window"])
            self.previous_balance = state["previous_balance"]
            logging.debug("State loaded successfully")
        else:
            logging.warning("Not enough history to rollback")

    def detect_error(self):
        if self.balance < self.initial_balance * 0.5:
            logging.error("Balance dropped below half of the initial value")
            return True
        return False

    def handle_error(self):
        logging.info("Handling error by rolling back state")
        self.load_state(steps_back=2)

    def step(self, action):
        self.save_state()
        reward = 0
        info = {}
        if self.current_step >= len(self.data):
            self.done = True
            profit = self.balance - self.previous_balance
            volatility = self.data["atr"].iloc[self.current_step - 1]
            reward = profit / (volatility + 1e-8)
            logging.debug(
                f"Episode ended. Profit: {profit}, Volatility: {volatility}, Reward: {reward}"
            )
            return self._get_observation(), reward, self.done, False, info
        price = self.data["close"].iloc[self.current_step]
        timestamp = self.timestamps[self.current_step]
        atr = self.data["atr"].iloc[self.current_step]
        logging.debug(
            f"Current step: {self.current_step}, Price: {price}, Time: {timestamp}, ATR: {atr}"
        )

        if action == 0:
            logging.debug("Action: Hold position")
            pass
        elif action == 1:
            if self.position == "short":
                logging.info("Action: Switch from short to long")
                reward += self._close_position(price, timestamp)
            if self.position != "long":
                logging.info("Action: Open long position")
                self._open_position("long", price, timestamp, atr)
        elif action == 2:
            if self.position == "long":
                logging.info("Action: Switch from long to short")
                reward += self._close_position(price, timestamp)
            if self.position != "short":
                logging.info("Action: Open short position")
                self._open_position("short", price, timestamp, atr)

        # Remove take-profit and stop-loss logic

        profit = self.balance - self.previous_balance
        volatility = self.data["atr"].iloc[self.current_step - 1]
        reward += profit / (volatility + 1e-8)
        if profit > 0:
            reward += 0.1
        elif profit < 0:
            reward -= 0.1
        reward += 0.01
        logging.debug(f"Profit: {profit}, Volatility: {volatility}, Reward: {reward}")
        self.previous_balance = self.balance
        obs = self.normalized_data.iloc[self.current_step]
        self.obs_window.append(obs.values.astype(np.float32))
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True
            logging.debug("End of data reached")
        self.balance_history.append(self.balance)
        if self.detect_error():
            self.handle_error()
            reward -= 10
            self.done = False
        return self._get_observation(), reward, self.done, False, info

    def _open_position(self, position_type, price, timestamp, atr):
        self.position = position_type
        self.entry_price = price
        self.entry_step = self.current_step
        self.position_size = self.balance * self.risk_percentage
        self.units = self.position_size / price
        # Remove take-profit and stop-loss variables
        # self.take_profit_multiplier = 2
        # self.stop_loss_multiplier = 2
        self.positions.append(
            {
                "entry_time": timestamp,
                "entry_price": price,
                "entry_step": self.current_step,
                "atr": atr,
            }
        )
        logging.info(f"Position opened: {position_type} at price {price}")

    def _close_position(self, price, timestamp):
        if self.entry_price == 0:
            logging.warning("Attempt to close position without entry price")
            return 0
        fee_rate = 0.001
        slippage = 0.001
        duration = self.current_step - self.entry_step
        atr = self.data["atr"].iloc[self.entry_step]
        if self.position == "long":
            effective_price = price * (1 - slippage)
            profit = (effective_price - self.entry_price) * self.units
        else:
            effective_price = price * (1 + slippage)
            profit = (self.entry_price - effective_price) * self.units
        fee = self.position_size * fee_rate * 2
        profit -= fee
        self.balance += profit
        self.total_profit += profit
        reward = profit / self.position_size
        # Remove the impact of take-profit and stop-loss on reward
        # if self.position == 'long' and profit < 0:
        #     reward -= 0.1
        # if duration <= self.short_term_threshold and profit > 0:
        #     reward += 0.05
        # if profit > self.take_profit_multiplier * atr:
        #     reward += 0.1
        # elif profit < -self.stop_loss_multiplier * atr:
        #     reward -= 0.1
        # if duration <= self.short_term_threshold and profit > 0:
        #     reward += 0.05
        # if duration > self.long_term_threshold and profit < 0:
        #     reward -= 0.05
        self.positions[-1].update(
            {
                "exit_time": timestamp,
                "exit_price": price,
                "duration": duration,
                "profit": profit,
                "atr": atr,
            }
        )
        logging.info(
            f"Position closed: {self.position} at price {price}, Profit: {profit}"
        )
        self.position = None
        self.entry_price = 0
        self.position_size = 0
        self.units = 0
        return reward


def calculate_rvi(df, window=10):
    close_open = df["close"] - df["open"]
    high_low = df["high"] - df["low"]
    rvi = close_open / high_low
    rvi = rvi.rolling(window=window).mean()
    logging.debug("RVI calculated")
    return rvi


def add_technical_indicators(df):
    logging.debug("Adding technical indicators")
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    df["ema20"] = trend.EMAIndicator(df["close"], window=20).ema_indicator()
    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    bollinger = volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bollinger_hband"] = bollinger.bollinger_hband()
    df["bollinger_lband"] = bollinger.bollinger_lband()
    df["stoch"] = StochasticOscillator(
        df["high"], df["low"], df["close"], window=14
    ).stoch()
    cumulative_volume = df["volume"].cumsum()
    cumulative_volume[cumulative_volume == 0] = 1
    df["vwap"] = (
        df["volume"] * (df["high"] + df["low"] + df["close"]) / 3
    ).cumsum() / cumulative_volume
    df["atr"] = AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()
    ichimoku = IchimokuIndicator(
        df["high"], df["low"], window1=9, window2=26, window3=52
    )
    df["ichimoku_a"] = ichimoku.ichimoku_a()
    df["ichimoku_b"] = ichimoku.ichimoku_b()
    df["psar"] = PSARIndicator(
        df["high"], df["low"], df["close"], step=0.02, max_step=0.2
    ).psar()
    df["cci"] = CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()
    df["trix"] = TRIXIndicator(df["close"], window=15).trix()
    df["ultimate_osc"] = UltimateOscillator(
        df["high"], df["low"], df["close"], window1=7, window2=14, window3=28
    ).ultimate_oscillator()
    df["rvi"] = calculate_rvi(df, window=10)
    df["obv"] = OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    df["chaikin_money_flow"] = ChaikinMoneyFlowIndicator(
        df["high"], df["low"], df["close"], df["volume"], window=20
    ).chaikin_money_flow()
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)
    logging.debug("Technical indicators added")
    return df


async def get_full_data(
    exchange, symbol, timeframe="5m", since=None, limit=2000
):  # Changed from '1m' to '5m'
    all_ohlcv = []
    logging.info(f"Start getting data for symbol {symbol}")
    while True:
        try:
            ohlcv = await exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since, limit=limit
            )
            if not ohlcv:
                logging.debug("No new data to load")
                break
            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            since = (
                last_timestamp + 5 * 60 * 1000
            )  # Changed from 60 seconds to 5 minutes
            if last_timestamp >= exchange.milliseconds():
                logging.debug("Current timestamp reached")
                break
        except Exception as e:
            logging.error(f"Error getting data: {e}")
            break
    df = pd.DataFrame(
        all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    logging.info(f"Received {len(df)} data records for symbol {symbol}")
    return df


async def list_available_symbols(exchange):
    try:
        await exchange.load_markets()
        logging.debug("Markets loaded")
        return exchange.symbols
    except Exception as e:
        logging.error(f"Error loading markets: {e}")
        return []


async def verify_symbol(exchange, symbol):
    try:
        await exchange.load_markets()
        is_valid = symbol in exchange.symbols
        logging.debug(
            f"Symbol verification {symbol}: {'available' if is_valid else 'unavailable'}"
        )
        return is_valid
    except Exception as e:
        logging.error(f"Error verifying symbol: {e}")
        return False


def get_or_train_model_sync(symbol, train_df, models_dir, best_params=None):
    logging.info(f"Getting or training model for symbol {symbol}")
    model_path = f'{models_dir}/{symbol.replace("/", "_").replace(":", "_")}_ppo'
    norm_path = f"{model_path}_norm.json"
    env = TradingEnvironment(train_df)
    norm_params = {"means": env.means.to_dict(), "stds": env.stds.to_dict()}

    if os.path.exists(f"{model_path}.zip"):
        logging.info("Finding existing model, loading model")
        if os.path.exists(norm_path):
            with open(norm_path, "r") as f:
                norm_params = json.load(f)
        env = TradingEnvironment(train_df, norm_params=norm_params)
        env = DummyVecEnv([lambda: env])
        model = PPO.load(model_path, env=env)
        logging.info("Model loaded successfully")
    else:
        logging.info("Model not found, starting training")
        env = TradingEnvironment(train_df)
        means = env.means.to_dict()
        stds = env.stds.to_dict()
        env = DummyVecEnv([lambda: env])

        # Generate a UUID for the run
        run_id = str(uuid.uuid4())

        # Create cryptocurrency-specific tensorboard directory
        crypto_symbol = symbol.split("/")[
            0
        ]  # Extract base currency (e.g., 'DOGE' from 'DOGE/USDT:USDT')
        tensorboard_log = os.path.join(
            "./ppo_tensorboard", crypto_symbol, f"run_{run_id}"
        )
        os.makedirs(os.path.dirname(tensorboard_log), exist_ok=True)

        if best_params:
            # Rest of the existing parameters code remains the same
            net_arch = []
            n_layers = best_params.get("n_layers", 1)
            for i in range(n_layers):
                layer_size = best_params.get(f"n_units_l{i}", 64)
                net_arch.append(layer_size)
            activation = best_params.get("activation", "tanh")
            activation_mapping = {
                "relu": torch.nn.ReLU,
                "tanh": torch.nn.Tanh,
                "elu": torch.nn.ELU,
            }
            activation_fn = activation_mapping.get(activation, torch.nn.Tanh)
            policy_kwargs = dict(
                net_arch=dict(pi=net_arch, vf=net_arch), activation_fn=activation_fn
            )
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=best_params["learning_rate"],
                n_steps=best_params["n_steps"],
                gamma=best_params["gamma"],
                ent_coef=best_params["ent_coef"],
                vf_coef=best_params["vf_coef"],
                max_grad_norm=best_params["max_grad_norm"],
                policy_kwargs=policy_kwargs,
                tensorboard_log=tensorboard_log,
                verbose=1,
            )
        else:
            model = PPO(
                "MlpPolicy",
                env,
                tensorboard_log=tensorboard_log,
                verbose=1,
            )

        model.learn(total_timesteps=500000)
        model.save(model_path)
        with open(norm_path, "w") as f:
            json.dump(norm_params, f)
        logging.info("Model trained and saved")
    return model, norm_params


def backtest_model_sync(model, test_df, symbol, norm_params):
    logging.info(f"Starting backtest for symbol {symbol}")
    test_env = TradingEnvironment(test_df, norm_params=norm_params)
    obs, _ = test_env.reset()
    while not test_env.done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = test_env.step(action)
    logging.info("Backtest completed")


def objective_sync(trial, train_df, test_df, symbol: str):
    # Force UUID-based logging instead of hostname
    os.environ["HOSTNAME"] = str(uuid.uuid4())
    try:
        logging.debug(f"Starting optimisation trial {trial.number} for {symbol}")
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        n_steps = trial.suggest_categorical("n_steps", [128, 256, 512])
        gamma = trial.suggest_float("gamma", 0.9, 0.9999)
        ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True)
        vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0)
        n_layers = trial.suggest_int("n_layers", 1, 3)
        net_arch = []
        for i in range(n_layers):
            layer_size = trial.suggest_int(f"n_units_l{i}", 64, 512)
            net_arch.append(layer_size)
        activation = trial.suggest_categorical("activation", ["tanh", "relu", "elu"])
        activation_mapping = {
            "relu": torch.nn.ReLU,
            "tanh": torch.nn.Tanh,
            "elu": torch.nn.ELU,
        }
        activation_fn = activation_mapping.get(activation, torch.nn.Tanh)
        policy_kwargs = dict(
            net_arch=dict(pi=net_arch, vf=net_arch), activation_fn=activation_fn
        )

        # Create environment for training
        env = TradingEnvironment(train_df)
        env = DummyVecEnv([lambda: env])

        # Create unique log directory for this trial with crypto symbol
        trial_id = str(trial.number)
        crypto_symbol = symbol.split("/")[
            0
        ]  # Extract base currency (e.g., 'XRP' from 'XRP/USDT')
        tensorboard_log = os.path.join(
            "./ppo_tensorboard", crypto_symbol, f"trial_{trial_id}"
        )
        os.makedirs(os.path.dirname(tensorboard_log), exist_ok=True)

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=0,
        )
        model.learn(total_timesteps=100000)

        # Test environment using same normalization parameters as training
        test_env = TradingEnvironment(
            test_df,
            norm_params={
                "means": env.get_attr("means")[0],
                "stds": env.get_attr("stds")[0],
            },
        )
        obs, _ = test_env.reset()
        total_reward = 0
        while not test_env.done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = test_env.step(action)
            total_reward += reward
        env.close()
        test_env.close()
        logging.debug(f"Trial {trial.number} completed with reward {total_reward}")
        return total_reward
    except Exception as e:
        logging.error(f"Error in trial {trial.number}: {e}")
        return float("-inf")


async def run_optuna(study, train_df, test_df, n_trials: int, symbol: str):
    """Run Optuna optimization with proper symbol passing"""
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor()
    try:
        for _ in range(n_trials):
            trial = study.ask()
            score = await loop.run_in_executor(
                executor, objective_sync, trial, train_df, test_df, symbol
            )
            study.tell(trial, score)
    finally:
        executor.shutdown(wait=True)


def get_real_balance_sync(exchange):
    try:
        balance = asyncio.run(exchange.fetch_balance())
        real_balance = balance["total"].get("USDT", 0)
        logging.debug(f"Current balance: {real_balance} USDT")
        return real_balance
    except Exception as e:
        logging.error(f"Error getting balance: {e}")
        return None


class LiveTradingState:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.data = deque(maxlen=window_size)
        self.balance_history = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.balance = None
        self.simulated_positions = []  # For tracking simulation trades
        self.simulation_total_profit = 0.0  # For tracking simulation profit

    def update(self, new_row, current_balance, timestamp):
        self.data.append(new_row)
        self.balance_history.append(current_balance)
        self.timestamps.append(timestamp)
        self.balance = current_balance
        logging.debug("Live trading state updated")

    def get_dataframe(self):
        if len(self.data) == self.window_size:
            df = pd.DataFrame(list(self.data))
            df["balance"] = list(self.balance_history)
            df["timestamp"] = pd.to_datetime(list(self.timestamps))
            logging.debug("DataFrame for live trading ready")
            return df
        else:
            logging.debug("Not enough data to form DataFrame")
            return None


async def get_real_balance_async(exchange):
    try:
        balance = await exchange.fetch_balance()
        real_balance = balance["total"].get("USDT", 0)
        logging.debug(f"Current balance (async): {real_balance} USDT")
        return real_balance
    except Exception as e:
        logging.error(f"Error getting balance: {e}")
        return None


async def live_trading(
    async_exchange, model, symbol, norm_params, state, simulate=False
):
    trading_interval = 300  # 5 minutes
    logging.info(
        "Starting paper trading simulation" if simulate else "Starting live trading"
    )
    while True:
        try:
            real_balance = await get_real_balance_async(async_exchange)
            if real_balance is None:
                real_balance = 100.0 if simulate else 10.0
                if simulate:
                    logging.info(
                        f"[SIMULATION] Using simulated balance: {real_balance} USDT"
                    )
                else:
                    logging.warning("Failed to get balance, using default value")

            # Fetch current market data
            ohlcv = await async_exchange.fetch_ohlcv(symbol, timeframe="5m", limit=1)
            if not ohlcv:
                logging.warning("Failed to get new OHLCV data")
                await asyncio.sleep(trading_interval)
                continue

            # Create DataFrame with current data
            df_new = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms")
            timestamp = df_new["timestamp"].iloc[0]

            # Update state with new data
            state.update(df_new.iloc[0], real_balance, timestamp)
            df = state.get_dataframe()
            if df is None:
                if simulate:
                    logging.info(
                        "[SIMULATION] Waiting for enough data to form initial window"
                    )
                await asyncio.sleep(trading_interval)
                continue

            # Add technical indicators
            df = add_technical_indicators(df)

            # Verify all required features are present
            missing_features = [
                feat for feat in norm_params["means"].keys() if feat not in df.columns
            ]
            if missing_features:
                logging.warning(f"Missing features: {missing_features}")
                await asyncio.sleep(trading_interval)
                continue

            # Normalize data
            try:
                normalized_df = (
                    df.drop(columns=["timestamp", "balance"])
                    - pd.Series(norm_params["means"])
                ) / pd.Series(norm_params["stds"])
                obs = normalized_df.values.flatten().astype(np.float32)
                if simulate:
                    logging.info(f"[SIMULATION] Observation shape: {obs.shape}")
            except Exception as e:
                logging.error(f"Error normalizing data: {e}")
                await asyncio.sleep(trading_interval)
                continue

            # Get model prediction
            action, _states = model.predict(obs, deterministic=True)
            action_text = {0: "HOLD", 1: "LONG", 2: "SHORT"}[action]
            if simulate:
                logging.info(f"[SIMULATION] Model action: {action_text}")

            # Get current positions
            positions = await async_exchange.fetch_positions(symbol)
            has_position = False
            current_contracts = 0
            current_side = None
            entry_price = 0

            if positions and isinstance(positions, list):
                for position in positions:
                    if (
                        position
                        and "contracts" in position
                        and float(position.get("contracts", 0)) > 0
                    ):
                        has_position = True
                        current_contracts = float(position["contracts"])
                        current_side = position.get("side", "").lower()
                        entry_price = float(position.get("entryPrice", 0))
                        if simulate:
                            logging.info(
                                f"[SIMULATION] Current position: {current_side.upper()} {current_contracts} contracts at {entry_price}"
                            )
                        break

            current_price = float(df["close"].iloc[-1])
            amount = real_balance * 1.0 / current_price
            atr = df["atr"].iloc[-1]

            if simulate:
                logging.info(f"[SIMULATION] Current price: {current_price}")
                logging.info(f"[SIMULATION] Current ATR: {atr}")
                logging.info(
                    f"[SIMULATION] Potential position size: {amount:.4f} contracts (${real_balance:.2f} worth)"
                )

            if action == 1:  # LONG
                if has_position and current_side in ["sell", "short"]:
                    if simulate:
                        logging.info(
                            f"[SIMULATION] Would close short position: {current_contracts} contracts at ${current_price}"
                        )
                    else:
                        logging.info("Closing short position")
                        order = await async_exchange.create_order(
                            symbol=symbol,
                            type="market",
                            side="buy",
                            amount=current_contracts,
                        )
                if not has_position or current_side != "long":
                    if simulate:
                        logging.info(
                            f"[SIMULATION] Would open long position: {amount:.4f} contracts at ${current_price}"
                        )
                    else:
                        logging.info("Opening long position")
                        order = await async_exchange.create_order(
                            symbol=symbol, type="market", side="buy", amount=amount
                        )

            elif action == 2:  # SHORT
                if has_position and current_side in ["buy", "long"]:
                    if simulate:
                        logging.info(
                            f"[SIMULATION] Would close long position: {current_contracts} contracts at ${current_price}"
                        )
                    else:
                        logging.info("Closing long position")
                        order = await async_exchange.create_order(
                            symbol=symbol,
                            type="market",
                            side="sell",
                            amount=current_contracts,
                        )
                if not has_position or current_side != "short":
                    if simulate:
                        logging.info(
                            f"[SIMULATION] Would open short position: {amount:.4f} contracts at ${current_price}"
                        )
                    else:
                        logging.info("Opening short position")
                        order = await async_exchange.create_order(
                            symbol=symbol, type="market", side="sell", amount=amount
                        )

            else:  # HOLD
                if simulate:
                    logging.info("[SIMULATION] Holding current position")

            if simulate:
                logging.info("-" * 50)

        except Exception as e:
            logging.error(
                f"Error in {'simulation' if simulate else 'live trading'}: {e}"
            )
        await asyncio.sleep(trading_interval)


async def main():
    # Parse command line arguments
    parser = setup_argparse()
    args = parser.parse_args()

    # Setup
    os.makedirs(args.models_dir, exist_ok=True)
    load_dotenv()

    # Configure event loop for Windows
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Setup signal handlers for both Windows and Unix
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize exchange
    async_exchange = ccxt_async.bybit(exchange_config)

    try:
        # Additional setup for Unix systems
        if not sys.platform.startswith("win"):
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(
                    sig, lambda s=sig: asyncio.create_task(shutdown_handler(s))
                )

        model = None
        norm_params = None

        # Training phase or model loading
        if args.mode in ["full", "train"]:
            # Training phase
            model, norm_params = await train_model(
                async_exchange, args.symbol, args.timeframe, args.models_dir
            )
            if model is None or norm_params is None:
                logging.error("Training failed")
                return

            if args.mode == "train":
                logging.info("Training completed successfully")
                return

        elif args.mode in ["trade", "simulate"]:
            # Load existing model for trading-only or simulation mode
            model_path = f'{args.models_dir}/{args.symbol.replace("/", "_").replace(":", "_")}_ppo'
            norm_path = f"{model_path}_norm.json"

            if not os.path.exists(f"{model_path}.zip") or not os.path.exists(norm_path):
                logging.error(f"No existing model found for trading at {model_path}")
                logging.error("Please train a model first using --mode train")
                return

            logging.info(f"Loading existing model from {model_path}")
            with open(norm_path, "r") as f:
                norm_params = json.load(f)

            # Create environment for loading model
            dummy_df = pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp.now()] * 20,  # Window size worth of data
                    "open": [0.0] * 20,
                    "high": [0.0] * 20,
                    "low": [0.0] * 20,
                    "close": [0.0] * 20,
                    "volume": [0.0] * 20,
                }
            )
            # Add technical indicators to ensure all required columns exist
            dummy_df = add_technical_indicators(dummy_df)
            env = TradingEnvironment(dummy_df, norm_params=norm_params)
            env = DummyVecEnv([lambda: env])
            model = PPO.load(model_path, env=env)
            logging.info("Model loaded successfully")

        # Check if we have a valid model before proceeding
        if model is None or norm_params is None:
            logging.error("No valid model available for trading/simulation")
            return

        # Trading/Simulation phase
        if args.mode in ["full", "trade", "simulate"]:
            # Set up logging for simulation mode
            if args.mode == "simulate":
                logging.getLogger().setLevel(logging.INFO)
                logging.info("[SIMULATION MODE] Starting paper trading simulation")

            await trade_model(
                async_exchange,
                model,
                args.symbol,
                norm_params,
                simulate=(args.mode == "simulate"),
            )

    except asyncio.CancelledError:
        logging.info("Tasks were cancelled")
    except Exception as e:
        logging.error(f"Error in main: {e}")
    finally:
        await async_exchange.close()
        logging.info("Exchange closed")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Program interrupted by user")
