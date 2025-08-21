import websocket
import json
import threading
import time

class WebSocketClient:
    """
    Handle websocket connection to Binance .
    """
    def __init__(self, symbol="btcusdt"):
        self.base_url = "wss://stream.binance.com:9443/stream?streams="
        self.symbol = symbol.lower()

        self.streams = [f"{self.symbol}@depth", f"{self.symbol}@trade"]
        self.url = self.base_url + '/'.join(self.streams)

        self.ws = None
        self.thread = None

        self.on_depth_update = None
        self.on_trade = None

    def connect(self):
        self.ws = websocket.WebSocketApp(self.url,
                                         on_open=self._on_open,
                                         on_message=self._on_message,
                                         on_error=self._on_error,
                                         on_close=self._on_close)

        self.thread = threading.Thread(target=self.ws.run_forever)
        self.thread.daemon = True
        self.thread.start()
        print("Connexion au WebSocket de Binance en cours...")

    def disconnect(self):
        if self.ws:
            self.ws.close()

    def _on_open(self, ws):
        print("Connexion WebSocket ouverte.")

    def _on_close(self, ws, close_status_code, close_msg):
        print(f"Connexion WebSocket fermée. Code: {close_status_code}")

    def _on_error(self, ws, error):
        print(f"Erreur WebSocket: {error}")

    def _on_message(self, ws, message):
        data = json.loads(message)
        stream_name = data.get('stream')
        event_data = data.get('data')

        if not event_data:
            return

        if stream_name == f"{self.symbol}@depth":
            if self.on_depth_update:
                self.on_depth_update(event_data)
        elif stream_name == f"{self.symbol}@trade":
            if self.on_trade:
                self.on_trade(event_data)

# --- TEST ---
if __name__ == "__main__":
    def test_depth_handler(data):
        print(f"--- Depth Update --- Bids: {len(data['b'])}, Asks: {len(data['a'])}")

    def test_trade_handler(data):
        print(f"--- New Trade --- Price: {data['p']}, Qty: {data['q']}")

    ws_client = WebSocketClient(symbol="btcusdt")
    ws_client.on_depth_update = test_depth_handler
    ws_client.on_trade = test_trade_handler

    ws_client.connect()

    try:
        time.sleep(20)
    finally:
        ws_client.disconnect()
