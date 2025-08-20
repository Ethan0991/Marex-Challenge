import websocket
import json
import threading
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class OrderBookLevel:
    """Represents a single level in the order book"""
    price: float
    size: float

@dataclass
class Trade:
    """Represents an executed trade"""
    timestamp: datetime
    price: float
    size: float
    side: str  # 'buy' or 'sell'

@dataclass
class Position:
    """Represents current trading position"""
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    exposure_usd: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

class OrderBook:
    """Manages the Level 2 order book data"""

    def __init__(self):
        self.bids = {}  # price -> size
        self.asks = {}  # price -> size
        self.last_update = None
        self.lock = threading.Lock()

    def update(self, data):
        """Update order book with new data"""
        with self.lock:
            changes = data.get('changes', [])
            for change in changes:
                side, price_str, size_str = change
                price = float(price_str)
                size = float(size_str)

                if side == 'buy':
                    if size == 0:
                        self.bids.pop(price, None)
                    else:
                        self.bids[price] = size
                elif side == 'sell':
                    if size == 0:
                        self.asks.pop(price, None)
                    else:
                        self.asks[price] = size

            self.last_update = datetime.now()

    def get_best_bid_ask(self) -> Tuple[Optional[float], Optional[float]]:
        """Get best bid and ask prices"""
        with self.lock:
            best_bid = max(self.bids.keys()) if self.bids else None
            best_ask = min(self.asks.keys()) if self.asks else None
            return best_bid, best_ask

    def get_top_levels(self, n=5) -> Tuple[List[OrderBookLevel], List[OrderBookLevel]]:
        """Get top N levels of bids and asks"""
        with self.lock:
            top_bids = []
            top_asks = []

            # Sort bids descending (highest first)
            for price in sorted(self.bids.keys(), reverse=True)[:n]:
                top_bids.append(OrderBookLevel(price, self.bids[price]))

            # Sort asks ascending (lowest first)
            for price in sorted(self.asks.keys())[:n]:
                top_asks.append(OrderBookLevel(price, self.asks[price]))

            return top_bids, top_asks

    def calculate_spread_for_size(self, target_size: float) -> Optional[float]:
        """Calculate spread needed to trade a specific size"""
        with self.lock:
            if not self.bids or not self.asks:
                return None

            # Calculate weighted average price for buying target_size
            sorted_asks = sorted(self.asks.items())
            buy_cost = 0.0
            remaining_size = target_size

            for price, size in sorted_asks:
                if remaining_size <= 0:
                    break
                trade_size = min(remaining_size, size)
                buy_cost += trade_size * price
                remaining_size -= trade_size

            if remaining_size > 0:
                return None  # Not enough liquidity

            avg_buy_price = buy_cost / target_size

            # Calculate weighted average price for selling target_size
            sorted_bids = sorted(self.bids.items(), reverse=True)
            sell_revenue = 0.0
            remaining_size = target_size

            for price, size in sorted_bids:
                if remaining_size <= 0:
                    break
                trade_size = min(remaining_size, size)
                sell_revenue += trade_size * price
                remaining_size -= trade_size

            if remaining_size > 0:
                return None  # Not enough liquidity

            avg_sell_price = sell_revenue / target_size

            return avg_buy_price - avg_sell_price

class SpreadAnalyzer:
    """Analyzes spread statistics over time"""

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.spread_data = {
            0.1: deque(maxlen=window_size),
            1.0: deque(maxlen=window_size),
            5.0: deque(maxlen=window_size),
            10.0: deque(maxlen=window_size)
        }
        self.timestamps = deque(maxlen=window_size)

    def update(self, order_book: OrderBook):
        """Update spread analysis with current order book"""
        timestamp = datetime.now()
        self.timestamps.append(timestamp)

        for size in [0.1, 1.0, 5.0, 10.0]:
            spread = order_book.calculate_spread_for_size(size)
            if spread is not None:
                self.spread_data[size].append(spread)

    def get_statistics(self) -> Dict:
        """Get spread statistics for all sizes"""
        stats = {}
        for size, spreads in self.spread_data.items():
            if spreads:
                stats[size] = {
                    'avg': np.mean(spreads),
                    'median': np.median(spreads),
                    'min': min(spreads),
                    'max': max(spreads),
                    'std': np.std(spreads),
                    'count': len(spreads)
                }
            else:
                stats[size] = None
        return stats

class MarketMaker:
    """Implements basic market making strategy"""

    def __init__(self, initial_capital=1000000, max_loss=100000):
        self.initial_capital = initial_capital
        self.max_loss = max_loss
        self.position = Position()
        self.trades = []
        self.active_orders = {}  # price -> {'side', 'size', 'timestamp'}
        self.base_spread = 0.01  # 1% base spread
        self.order_size = 0.1  # BTC

        # Risk management
        self.max_exposure = initial_capital
        self.max_position = 10.0  # Max 10 BTC position

    def calculate_quotes(self, mid_price: float, current_spread: float) -> Tuple[float, float]:
        """Calculate bid and ask quotes with position skewing"""
        # Adjust spread based on current market conditions
        adaptive_spread = max(self.base_spread, current_spread * 1.2)

        # Position-based skewing
        position_skew = 0.0
        if abs(self.position.quantity) > 0:
            # If long, skew quotes lower to encourage selling
            # If short, skew quotes higher to encourage buying
            max_skew = 0.005  # 0.5% max skew
            skew_factor = self.position.quantity / self.max_position
            position_skew = -skew_factor * max_skew

        half_spread = adaptive_spread / 2
        bid = mid_price * (1 - half_spread + position_skew)
        ask = mid_price * (1 + half_spread + position_skew)

        return bid, ask

    def check_risk_limits(self) -> bool:
        """Check if risk limits are breached"""
        # Check max loss
        total_pnl = self.position.realized_pnl + self.position.unrealized_pnl
        if total_pnl < -self.max_loss:
            logger.warning(f"Max loss limit breached: {total_pnl:.2f}")
            return False

        # Check max exposure
        if abs(self.position.exposure_usd) > self.max_exposure:
            logger.warning(f"Max exposure limit breached: {abs(self.position.exposure_usd):.2f}")
            return False

        # Check max position
        if abs(self.position.quantity) > self.max_position:
            logger.warning(f"Max position limit breached: {abs(self.position.quantity):.4f}")
            return False

        return True

    def simulate_fill(self, trade: Trade, mid_price: float):
        """Simulate order fills based on trade feed"""
        fills = []

        # Check if any active orders would be filled by this trade
        for price, order in list(self.active_orders.items()):
            filled = False

            if order['side'] == 'buy' and trade.side == 'sell' and trade.price <= price:
                # Our buy order gets filled
                filled = True
                fill_side = 'buy'
            elif order['side'] == 'sell' and trade.side == 'buy' and trade.price >= price:
                # Our sell order gets filled
                filled = True
                fill_side = 'sell'

            if filled:
                # Execute the fill
                fill_size = min(order['size'], trade.size)
                fill_price = price

                # Update position
                old_quantity = self.position.quantity
                old_avg_price = self.position.avg_entry_price

                if fill_side == 'buy':
                    new_quantity = old_quantity + fill_size
                    if old_quantity * new_quantity >= 0:  # Same direction or starting
                        self.position.avg_entry_price = ((old_quantity * old_avg_price) +
                                                         (fill_size * fill_price)) / new_quantity
                    else:  # Closing position partially
                        # Calculate realized PnL
                        close_size = min(abs(old_quantity), fill_size)
                        self.position.realized_pnl += close_size * (fill_price - old_avg_price) * (-1 if old_quantity < 0 else 1)
                else:  # sell
                    new_quantity = old_quantity - fill_size
                    if old_quantity * new_quantity >= 0:  # Same direction or starting
                        if new_quantity != 0:
                            self.position.avg_entry_price = ((old_quantity * old_avg_price) -
                                                             (fill_size * fill_price)) / new_quantity
                        else:
                            self.position.avg_entry_price = 0.0
                    else:  # Closing position partially
                        close_size = min(abs(old_quantity), fill_size)
                        self.position.realized_pnl += close_size * (fill_price - old_avg_price) * (1 if old_quantity > 0 else -1)

                self.position.quantity = new_quantity
                self.position.exposure_usd = self.position.quantity * mid_price

                # Calculate unrealized PnL
                if self.position.quantity != 0:
                    self.position.unrealized_pnl = self.position.quantity * (mid_price - self.position.avg_entry_price)
                else:
                    self.position.unrealized_pnl = 0.0

                # Record the trade
                trade_record = {
                    'timestamp': datetime.now(),
                    'side': fill_side,
                    'price': fill_price,
                    'size': fill_size,
                    'position_after': self.position.quantity,
                    'realized_pnl': self.position.realized_pnl,
                    'unrealized_pnl': self.position.unrealized_pnl
                }
                self.trades.append(trade_record)
                fills.append(trade_record)

                # Remove or update the order
                if order['size'] <= fill_size:
                    del self.active_orders[price]
                else:
                    self.active_orders[price]['size'] -= fill_size

                logger.info(f"Fill: {fill_side.upper()} {fill_size:.4f} BTC @ ${fill_price:.2f}")

        return fills

    def update_quotes(self, mid_price: float, current_spread: float):
        """Update market making quotes"""
        if not self.check_risk_limits():
            # Clear all orders if risk limits breached
            self.active_orders.clear()
            return

        bid, ask = self.calculate_quotes(mid_price, current_spread)

        # Clear old orders
        self.active_orders.clear()

        # Place new orders
        if abs(self.position.quantity) < self.max_position:
            self.active_orders[bid] = {
                'side': 'buy',
                'size': self.order_size,
                'timestamp': datetime.now()
            }

            self.active_orders[ask] = {
                'side': 'sell',
                'size': self.order_size,
                'timestamp': datetime.now()
            }

class BinanceWebSocketClient:
    """WebSocket client for Binance API"""

    def __init__(self, symbol='btcusdt'):
        self.symbol = symbol.lower()
        self.ws = None
        self.order_book = OrderBook()
        self.trades = deque(maxlen=1000)
        self.spread_analyzer = SpreadAnalyzer()
        self.market_maker = MarketMaker()
        self.running = False

        # Display update frequency
        self.last_display_update = 0
        self.display_interval = 2  # seconds

    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)

            # Handle Binance WebSocket message formats
            if 'e' in data:
                # Direct event message (single stream)
                event_type = data['e']

                if event_type == 'depthUpdate':
                    logger.debug(f"Received depth update: {len(data.get('b', []))} bids, {len(data.get('a', []))} asks")
                    self.process_depth_update(data)
                    self.update_analysis()
                elif event_type == 'trade':
                    logger.debug(f"Received trade: {data.get('p')} @ {data.get('q')}")
                    self.process_trade_update(data)
                else:
                    logger.debug(f"Received event: {event_type}")

            elif 'stream' in data:
                # Combined stream message format
                stream_data = data['data']
                stream_name = data['stream']

                if 'depth' in stream_name:
                    logger.debug(f"Received depth update for {stream_name}")
                    self.process_depth_update(stream_data)
                    self.update_analysis()
                elif 'trade' in stream_name:
                    logger.debug(f"Received trade: {stream_data.get('p')} @ {stream_data.get('q')}")
                    self.process_trade_update(stream_data)

            elif 'result' in data:
                # Subscription confirmation
                if data['result'] is None:
                    logger.info("Successfully subscribed to streams")
                else:
                    logger.info(f"Subscription result: {data}")
            elif 'id' in data and 'result' in data:
                logger.info(f"Subscription response: {data}")
            else:
                logger.debug(f"Unknown message format: {list(data.keys())}")

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            logger.error(f"Message content: {message[:200]}...")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def process_depth_update(self, data):
        """Process Binance depth (order book) update"""
        try:
            # Binance sends incremental updates
            with self.order_book.lock:
                # Process bids (b array)
                for bid_data in data.get('b', []):
                    price, size = float(bid_data[0]), float(bid_data[1])
                    if size == 0.0:
                        self.order_book.bids.pop(price, None)
                    else:
                        self.order_book.bids[price] = size

                # Process asks (a array)
                for ask_data in data.get('a', []):
                    price, size = float(ask_data[0]), float(ask_data[1])
                    if size == 0.0:
                        self.order_book.asks.pop(price, None)
                    else:
                        self.order_book.asks[price] = size

                self.order_book.last_update = datetime.now()

        except Exception as e:
            logger.error(f"Error processing depth update: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def process_trade_update(self, data):
        """Process Binance trade update"""
        try:
            # Handle both formats: direct trade events and aggTrade events
            if 'p' in data and 'q' in data:
                # Regular trade format
                price = float(data['p'])
                quantity = float(data['q'])
                timestamp_ms = data.get('T', data.get('E', int(time.time() * 1000)))
                is_buyer_maker = data.get('m', False)
            elif 'price' in data and 'quantity' in data:
                # Alternative format
                price = float(data['price'])
                quantity = float(data['quantity'])
                timestamp_ms = data.get('time', int(time.time() * 1000))
                is_buyer_maker = data.get('is_buyer_maker', False)
            else:
                logger.warning(f"Unknown trade format: {data}")
                return

            trade = Trade(
                timestamp=datetime.fromtimestamp(timestamp_ms / 1000),
                price=price,
                size=quantity,
                side='sell' if is_buyer_maker else 'buy'  # If buyer is maker, trade direction is sell
            )
            self.trades.append(trade)

            # Simulate market maker fills
            best_bid, best_ask = self.order_book.get_best_bid_ask()
            if best_bid and best_ask:
                mid_price = (best_bid + best_ask) / 2
                self.market_maker.simulate_fill(trade, mid_price)

        except Exception as e:
            logger.error(f"Error processing trade update: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def initialize_order_book_snapshot(self):
        """Initialize order book by fetching snapshot from REST API"""
        try:
            import urllib.request
            import urllib.parse

            # Get order book snapshot from Binance REST API
            url = f"https://api.binance.com/api/v3/depth?symbol={self.symbol.upper()}&limit=1000"

            with urllib.request.urlopen(url) as response:
                snapshot_data = json.loads(response.read())

            with self.order_book.lock:
                self.order_book.bids.clear()
                self.order_book.asks.clear()

                # Load bids
                for bid_data in snapshot_data.get('bids', []):
                    price, size = float(bid_data[0]), float(bid_data[1])
                    if size > 0:
                        self.order_book.bids[price] = size

                # Load asks
                for ask_data in snapshot_data.get('asks', []):
                    price, size = float(ask_data[0]), float(ask_data[1])
                    if size > 0:
                        self.order_book.asks[price] = size

                self.order_book.last_update = datetime.now()
                logger.info(f"Order book initialized with {len(self.order_book.bids)} bids and {len(self.order_book.asks)} asks")

        except Exception as e:
            logger.error(f"Error initializing order book snapshot: {e}")

    def update_analysis(self):
        """Update spread analysis and market making"""
        self.spread_analyzer.update(self.order_book)

        # Update market maker quotes
        best_bid, best_ask = self.order_book.get_best_bid_ask()
        if best_bid and best_ask:
            mid_price = (best_bid + best_ask) / 2
            current_spread = (best_ask - best_bid) / mid_price
            self.market_maker.update_quotes(mid_price, current_spread)

        # Display updates at regular intervals
        current_time = time.time()
        if current_time - self.last_display_update > self.display_interval:
            self.display_live_data()
            self.last_display_update = current_time

    def display_live_data(self):
        """Display live market data and trading status"""
        print("\n" + "="*80)
        print(f"CRYPTO TRADING SYSTEM - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

        # Order book
        top_bids, top_asks = self.order_book.get_top_levels(5)
        best_bid, best_ask = self.order_book.get_best_bid_ask()

        if best_bid and best_ask:
            mid_price = (best_bid + best_ask) / 2
            spread_bps = ((best_ask - best_bid) / mid_price) * 10000

            print(f"MID PRICE: ${mid_price:.2f} | SPREAD: {spread_bps:.1f} bps")
            print("\nORDER BOOK (Top 5 levels):")
            print("-" * 40)
            print("ASKS (Size @ Price)")
            for ask in reversed(top_asks):
                print(f"  {ask.size:8.4f} @ ${ask.price:8.2f}")
            print("-" * 40)
            for bid in top_bids:
                print(f"  {bid.size:8.4f} @ ${bid.price:8.2f}")
            print("BIDS (Size @ Price)")

        # Recent trades
        if self.trades:
            print(f"\nRECENT TRADES (Last 5):")
            for trade in list(self.trades)[-5:]:
                print(f"  {trade.timestamp.strftime('%H:%M:%S')} | "
                      f"{trade.side.upper():4s} | ${trade.price:8.2f} | {trade.size:8.4f} BTC")

        # Spread analysis
        stats = self.spread_analyzer.get_statistics()
        print(f"\nSPREAD ANALYSIS:")
        print("Size (BTC) | Avg Spread | Median | Min | Max")
        print("-" * 50)
        for size in [0.1, 1.0, 5.0, 10.0]:
            if stats[size]:
                s = stats[size]
                print(f"  {size:6.1f}   | ${s['avg']:8.4f} | ${s['median']:6.4f} | "
                      f"${s['min']:6.4f} | ${s['max']:6.4f}")
            else:
                print(f"  {size:6.1f}   | No data available")

        # Trading position and P&L
        pos = self.market_maker.position
        print(f"\nTRADING POSITION:")
        print(f"  Position: {pos.quantity:10.4f} BTC")
        print(f"  Avg Entry: ${pos.avg_entry_price:9.2f}")
        print(f"  Exposure: ${pos.exposure_usd:12.2f}")
        print(f"  Realized P&L: ${pos.realized_pnl:9.2f}")
        print(f"  Unrealized P&L: ${pos.unrealized_pnl:7.2f}")
        print(f"  Total P&L: ${pos.realized_pnl + pos.unrealized_pnl:11.2f}")

        # Active orders
        if self.market_maker.active_orders:
            print(f"\nACTIVE ORDERS:")
            for price, order in self.market_maker.active_orders.items():
                print(f"  {order['side'].upper():4s} {order['size']:6.4f} BTC @ ${price:.2f}")

        # Recent fills
        recent_fills = [t for t in self.market_maker.trades if
                        datetime.now() - t['timestamp'] < timedelta(minutes=5)]
        if recent_fills:
            print(f"\nRECENT FILLS (Last 5):")
            for fill in recent_fills[-5:]:
                print(f"  {fill['timestamp'].strftime('%H:%M:%S')} | "
                      f"{fill['side'].upper():4s} | ${fill['price']:8.2f} | "
                      f"{fill['size']:6.4f} BTC | P&L: ${fill['realized_pnl']:.2f}")

    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        logger.info("WebSocket connection closed")
        self.running = False

    def on_open(self, ws):
        """Handle WebSocket open"""
        logger.info("WebSocket connection opened")
        logger.info("Initializing order book snapshot...")

        # Initialize order book with REST API snapshot
        self.initialize_order_book_snapshot()

        logger.info("WebSocket ready - market data will stream automatically")
        self.running = True

    def start(self):
        """Start the WebSocket client"""
        websocket.enableTrace(False)  # Disable verbose tracing

        # Binance single stream WebSocket URL - this automatically sends depthUpdate events
        ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol}@depth"

        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )

        print("Starting Crypto Algorithm Trading System...")
        print("Connecting to Binance WebSocket (Public Feed)...")
        print(f"Trading Pair: {self.symbol.upper()}")
        print("Press Ctrl+C to stop")
        print("-" * 50)

        try:
            self.ws.run_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.stop()

    def stop(self):
        """Stop the WebSocket client"""
        self.running = False
        if self.ws:
            self.ws.close()

        # Export results
        self.export_results()

    def export_results(self):
        """Export trading results to CSV files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Export trades
        if self.market_maker.trades:
            trades_df = pd.DataFrame(self.market_maker.trades)
            trades_file = f'trades_{timestamp}.csv'
            trades_df.to_csv(trades_file, index=False)
            print(f"Trades exported to {trades_file}")

        # Export spread analysis
        stats = self.spread_analyzer.get_statistics()
        spread_data = []
        for size, stat in stats.items():
            if stat:
                spread_data.append({
                    'size_btc': size,
                    'avg_spread': stat['avg'],
                    'median_spread': stat['median'],
                    'min_spread': stat['min'],
                    'max_spread': stat['max'],
                    'std_spread': stat['std'],
                    'sample_count': stat['count']
                })

        if spread_data:
            spread_df = pd.DataFrame(spread_data)
            spread_file = f'spread_analysis_{timestamp}.csv'
            spread_df.to_csv(spread_file, index=False)
            print(f"Spread analysis exported to {spread_file}")

        # Print final summary
        pos = self.market_maker.position
        print(f"\nFINAL TRADING SUMMARY:")
        print(f"Total Trades: {len(self.market_maker.trades)}")
        print(f"Final Position: {pos.quantity:.4f} BTC")
        print(f"Total P&L: ${pos.realized_pnl + pos.unrealized_pnl:.2f}")
        print(f"Realized P&L: ${pos.realized_pnl:.2f}")
        print(f"Unrealized P&L: ${pos.unrealized_pnl:.2f}")

def main():
    """Main function to run the trading system"""
    client = BinanceWebSocketClient('BTCUSDT')

    try:
        client.start()
    except KeyboardInterrupt:
        print("\nStopping trading system...")
        client.stop()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        client.stop()

if __name__ == "__main__":
    main()