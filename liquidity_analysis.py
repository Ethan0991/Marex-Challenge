import numpy as np

class LiquidityAnalyzer:
    """
    Calcule et affiche les statistiques de liquidity
    """
    def __init__(self, order_book, sizes_to_track=None):
        if sizes_to_track is None:
            sizes_to_track = [0.1, 1, 5, 10]
        self.order_book = order_book
        self.sizes_to_track = sizes_to_track
        self.spread_history = {size: [] for size in self.sizes_to_track}
        self.stats = {}

    def _calculate_vwap(self, side, required_size):
        book_levels = self.order_book.get_top_levels(n=1000) # Obtenir une profondeur suffisante
        levels = book_levels[0] if side == 'bid' else book_levels[1]

        filled_size = 0
        total_cost = 0

        for price, size in levels:
            size_to_fill = min(required_size - filled_size, size)
            total_cost += size_to_fill * price
            filled_size += size_to_fill

            if filled_size >= required_size:
                return total_cost / filled_size
        return None

    def update_stats(self):
        for size in self.sizes_to_track:
            buy_vwap = self._calculate_vwap('ask', size)
            sell_vwap = self._calculate_vwap('bid', size)

            if buy_vwap is not None and sell_vwap is not None:
                effective_spread = buy_vwap - sell_vwap
                self.spread_history[size].append(effective_spread)

                history = self.spread_history[size]
                self.stats[size] = {
                    'current': effective_spread,
                    'avg': np.mean(history),
                    'median': np.median(history),
                    'min': np.min(history),
                    'max': np.max(history)
                }

# --- TEST ---
if __name__ == "__main__":

    class MockOrderBook:
        def get_top_levels(self, n):
            bids = [
                (100, 1), (99, 2), (98, 5) # Total 8
            ]
            asks = [
                (101, 2), (102, 3), (103, 4) # Total 9
            ]
            return bids, asks

    mock_book = MockOrderBook()
    analyzer = LiquidityAnalyzer(mock_book, sizes_to_track=[1, 5, 10])

    # Test du VWAP
    print("--- Test du calcul VWAP ---")
    buy_vwap_1 = analyzer._calculate_vwap('ask', 1)
    print(f"VWAP pour acheter 1 BTC: {buy_vwap_1:.2f}")

    buy_vwap_5 = analyzer._calculate_vwap('ask', 5)
    # (2 * 101 + 3 * 102) / 5 = 101.6
    print(f"VWAP pour acheter 5 BTC: {buy_vwap_5:.2f}")

    sell_vwap_3 = analyzer._calculate_vwap('bid', 3)
    # (1 * 100 + 2 * 99) / 3 = 99.33
    print(f"VWAP pour vendre 3 BTC: {sell_vwap_3:.2f}")

    buy_vwap_10 = analyzer._calculate_vwap('ask', 10)
    print(f"VWAP pour acheter 10 BTC: {buy_vwap_10}")

    print("\n--- Test de la mise à jour des stats ---")
    analyzer.update_stats()
    print(analyzer.stats)
    analyzer.update_stats()
    print("\nStats après une deuxième mise à jour:")
    print(analyzer.stats)
