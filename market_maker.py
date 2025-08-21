import time
from collections import deque

class MarketMaker:
    """
    Implémente la stratégie de market making .
    - Calculs des quotes.
    - Execution des fills.
    - Gestion Position et PnL.
    """
    def __init__(self, order_book, trade_feed, quote_spread=0.01, order_size=0.01):
        """
        Initialise le market maker.

        Args:
            order_book (OrderBook): L'instance du carnet d'ordres.
            trade_feed (TradeFeed): L'instance du flux de trades.
            quote_spread (float): Le spread en pourcentage .
            order_size (float): La taille de chaque ordre à coter (en BTC).
        """
        self.order_book = order_book
        self.trade_feed = trade_feed
        self.quote_spread = quote_spread / 100  # Convertir en décimal
        self.order_size = order_size

        # Nos cotations actuelles
        self.quote_bid = None
        self.quote_ask = None

        # État de la stratégie
        self.position = 0.0  # en BTC
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.cash = 1_000_000 # Capital de départ notionnel en USD
        self.exposure = 0.0 # en USD

        self.executed_trades = deque(maxlen=50)

    def update_quotes(self):
        """
        Calcule et met à jour les prix bid ask).
        """
        best_bid, best_ask, _ = self.order_book.get_spread()

        if best_bid is None or best_ask is None:
            self.quote_bid = None
            self.quote_ask = None
            return

        mid_price = (best_bid + best_ask) / 2
        skew_adjustment = -self.position * 0.1
        skewed_mid_price = mid_price + skew_adjustment
        self.quote_bid = skewed_mid_price * (1 - self.quote_spread / 2)
        self.quote_ask = skewed_mid_price * (1 + self.quote_spread / 2)
        # self.quote_bid = best_bid
        # self.quote_ask = best_ask

    def check_fills(self):
        """
        Vérifie si nos cotations auraient été exécutées par les récents trades du marché.
        """
        if self.quote_bid is None or self.quote_ask is None:
            return

        # On ne vérifie que le dernier trade pour la simplicité
        recent_trades = self.trade_feed.get_recent_trades(1)
        if not recent_trades:
            return

        last_trade = recent_trades[0]
        trade_price = last_trade['price']

        # Simulation d'un fill sur  bid
        if trade_price <= self.quote_bid:
            self._execute_trade('buy', self.order_size, trade_price)

        # Simulation d'un fill sur ask
        elif trade_price >= self.quote_ask:
            self._execute_trade('sell', self.order_size, trade_price)

    def _execute_trade(self, side, size, price):
        """
        Exécute un trade simulé, met à jour la position et le PnL.

        Args:
            side (str): 'buy' ou 'sell'
            size (float): taille de l'ordre
            price (float): prix d'exécution
        """
        timestamp = time.time()
        executed_size = 0

        if side == 'buy':
            # --- Couverture si short ---
            if self.position < 0:
                cover_size = min(size, abs(self.position))
                self.realized_pnl += (self.avg_entry_price - price) * cover_size
                self.position += cover_size
                size -= cover_size
                executed_size += cover_size

            # --- Ouverture d'une position long si reste du size ---
            if size > 0:
                new_total_cost = (self.position * self.avg_entry_price) + (size * price)
                self.position += size
                self.avg_entry_price = new_total_cost / self.position if self.position != 0 else 0
                executed_size += size

        elif side == 'sell':
            # --- Vente si long ---
            if self.position > 0:
                sell_size = min(size, self.position)
                self.realized_pnl += (price - self.avg_entry_price) * sell_size
                self.position -= sell_size
                size -= sell_size
                executed_size += sell_size

            # --- Ouverture d'une position short si reste du size ---
            if size > 0:
                new_total_cost = (self.position * self.avg_entry_price) - (size * price)
                self.position -= size
                self.avg_entry_price = new_total_cost / self.position if self.position != 0 else 0
                executed_size += size

        # Enregistrement du trade
        trade_record = {
            'timestamp': timestamp,
            'side': side,
            'size': executed_size,
            'price': price
        }
        self.executed_trades.append(trade_record)

        print(f"--- FILL SIMULÉ --- {side.upper()} {executed_size:.4f} @ {price:.2f}")



    def update_pnl(self):
        """
        Met à jour l'exposition et le P&L unrealized.
        """
        best_bid, best_ask, _ = self.order_book.get_spread()
        if best_bid is None or best_ask is None:
            return

        # Choisir le prix de liquidation correct selon le signe de la position
        if self.position > 0:
            market_price = best_bid   # vendre au bid si on est long
        elif self.position < 0:
            market_price = best_ask   # racheter à l'ask si on est short
        else:
            market_price = 0

        # Exposition notionnelle
        self.exposure = self.position * market_price
        self.unrealized_pnl = self.position * (market_price - self.avg_entry_price)


if __name__ == "__main__":
    class MockOrderBook:
        def get_spread(self): return 100.0, 101.0, 1.0

    class MockTradeFeed:
        def __init__(self): self.trades = []
        def get_recent_trades(self, n): return self.trades[-n:] if self.trades else []

    mock_book = MockOrderBook()
    mock_feed = MockTradeFeed()

    mm = MarketMaker(mock_book, mock_feed, quote_spread=0.2, order_size=0.1)

    print("--- Test initial ---")
    mm.update_quotes()
    print(f"Position: {mm.position}, Quote Bid: {mm.quote_bid:.2f}, Quote Ask: {mm.quote_ask:.2f}")

    print("\n--- Test de fill à l'achat ---")
    mock_feed.trades.append({'price': 99.0}) # Prix sous notre bid
    mm.check_fills()
    print(f"Position: {mm.position}, Avg Entry: {mm.avg_entry_price:.2f}")

    print("\n--- Test de skewing ---")
    mm.update_quotes()
    print(f"Position: {mm.position}, Quote Bid: {mm.quote_bid:.2f}, Quote Ask: {mm.quote_ask:.2f}")

    print("\n--- Test de fill à la vente ---")
    mock_feed.trades.append({'price': 102.0}) # Prix au-dessus de notre ask
    mm.check_fills()
    print(f"Position: {mm.position}, Realized PNL: {mm.realized_pnl:.2f}")
