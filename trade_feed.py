from collections import deque

class TradeFeed:
    """
    Stocke et gère le flux des trades.
    """
    def __init__(self, max_trades=100):
        self.trades = deque(maxlen=max_trades)

    def add_trade(self, trade_data):
        """
        Ajoute un nouveau trade au flux à partir d'un message de trade de Binance.
        """
        trade = {
            'time': trade_data.get('T'), # Timestamp en millisecondes
            'trade_id': trade_data.get('t'),
            'price': float(trade_data.get('p')),
            'size': float(trade_data.get('q')),
            'side': 'buy' if trade_data.get('m') else 'sell' # 'm' is True if the buyer is the maker
        }
        self.trades.append(trade)

    def get_recent_trades(self, n=10):
        return list(self.trades)[-n:]

if __name__ == "__main__":
    trade_feed = TradeFeed(max_trades=5)
    # Exemple de message trade de Binance
    sample_trade = {'e': 'trade', 'E': 1672531200000, 's': 'BTCUSDT', 't': 12345, 'p': '16500.00', 'q': '0.001', 'b': 88, 'a': 92, 'T': 1672531200000, 'm': False, 'M': True}
    trade_feed.add_trade(sample_trade)
    print(trade_feed.get_recent_trades())
