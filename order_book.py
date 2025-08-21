import time
import requests
from collections import deque

class OrderBook:
    """
    gere le carnet d'ordres  Binance avec synchronisations.
    """
    def __init__(self, symbol="btcusdt"):
        self.symbol = symbol.upper()
        self.rest_url = f"https://api.binance.com/api/v3/depth?symbol={self.symbol}&limit=1000"

        self._bids = {}
        self._asks = {}

        self.last_update_id = None
        self.last_update = None

        self.update_buffer = deque()
        self.is_initialized = False

    def reset(self):
        """Réinitialise l'état du carnet d'ordres pour une nouvelle synchronisation."""
        self.is_initialized = False
        self._bids.clear()
        self._asks.clear()
        self.update_buffer.clear()
        self.last_update_id = None
        print("Carnet d'ordres et tampon réinitialisés.")

    def initialize_snapshot(self):
        """Récupère le snapshot et traite le buffer pour synchroniser le carnet."""
        try:
            print("Récupération du snapshot du carnet d'ordres de Binance...")
            response = requests.get(self.rest_url)
            response.raise_for_status()
            data = response.json()

            snapshot_last_update_id = data['lastUpdateId']

            # Attendre que le tampon contienne des données
            while self.update_buffer and self.update_buffer[-1]['u'] < snapshot_last_update_id:
                time.sleep(0.1)

            self.last_update_id = snapshot_last_update_id

            self._bids = {float(price): float(size) for price, size in data['bids']}
            self._asks = {float(price): float(size) for price, size in data['asks']}

            print("Traitement du buffer de mises à jour pour la synchronisation...")
            while self.update_buffer:
                update = self.update_buffer.popleft()
                if update['u'] <= self.last_update_id:
                    continue

                if update['U'] <= self.last_update_id + 1 and update['u'] >= self.last_update_id + 1:
                    print("Synchronisation réussie")
                    self._apply_update(update)
                    self.is_initialized = True
                    break

            if self.is_initialized:
                while self.update_buffer:
                    self._apply_update(self.update_buffer.popleft())

        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la récupération du snapshot: {e}")
            self.is_initialized = False

    def handle_depth_update(self, update_data):
        """Gère les mises à jour entrantes."""
        if not self.is_initialized:
            self.update_buffer.append(update_data)
        else:
            if update_data['U'] == self.last_update_id + 1:
                self._apply_update(update_data)
            else:
                print("Désynchronisation, relance de la synchronisation")
                self.is_initialized = False

    def _apply_update(self, update_data):
        """Applique une mise à jour au carnet d'ordres."""
        for price_str, size_str in update_data['b']:
            price, size = float(price_str), float(size_str)
            if size == 0:
                self._bids.pop(price, None)
            else:
                self._bids[price] = size

        for price_str, size_str in update_data['a']:
            price, size = float(price_str), float(size_str)
            if size == 0:
                self._asks.pop(price, None)
            else:
                self._asks[price] = size

        self.last_update_id = update_data['u']
        self.last_update = time.time()

    def get_top_levels(self, n=5):
        sorted_bids = sorted(self._bids.items(), key=lambda item: item[0], reverse=True)
        sorted_asks = sorted(self._asks.items(), key=lambda item: item[0])
        return sorted_bids[:n], sorted_asks[:n]

    def get_spread(self):
        if not self._bids or not self._asks:
            return None, None, None
        best_bid = max(self._bids.keys())
        best_ask = min(self._asks.keys())
        return best_bid, best_ask, best_ask - best_bid if best_ask > best_bid else None
