import time
import os
from datetime import datetime
import matplotlib.pyplot as plt  
from connector import WebSocketClient
from order_book import OrderBook
from trade_feed import TradeFeed
from liquidity_analysis import LiquidityAnalyzer
from market_maker import MarketMaker
from risk_manager import RiskManager

def clear_console():
    """Efface l'affichage de la console."""
    os.system('cls' if os.name == 'nt' else 'clear 2>/dev/null')

def format_book(bids, asks, max_levels=10):
    """Met en forme le carnet d'ordres """
    print(f"{'BIDS':>12} {'SIZE':>12} | {'PRICE':>12} | {'SIZE':>12} {'ASKS':<12}")
    print("-" * 60)
    for i in range(max_levels):
        bid_price, bid_size = bids[i] if i < len(bids) else ('-', '-')
        ask_price, ask_size = asks[i] if i < len(asks) else ('-', '-')
        bid_price_str = f"{bid_price:,.2f}" if isinstance(bid_price, (int, float)) else bid_price
        bid_size_str = f"{bid_size:.4f}" if isinstance(bid_size, (int, float)) else bid_size
        ask_price_str = f"{ask_price:,.2f}" if isinstance(ask_price, (int, float)) else ask_price
        ask_size_str = f"{ask_size:.4f}" if isinstance(ask_size, (int, float)) else ask_size
        print(f"{bid_price_str:>12} {bid_size_str:>12} | {ask_price_str:>12} | {ask_size_str:>12}")

def format_trades(trades):
    """Met en forme les trades récents."""
    print("\n" + "-" * 60)
    print("TRADES RÉCENTS")
    print("-" * 60)
    print(f"{'TIME':<15} {'SIDE':<6} {'PRICE':>12} {'SIZE':>15}")
    for trade in reversed(trades):
        trade_time = datetime.fromtimestamp(trade['time'] / 1000).strftime('%H:%M:%S.%f')[:-3]
        side = trade['side'].upper()
        color_code = '\033[92m' if side == 'BUY' else '\033[91m'
        reset_code = '\033[0m'
        print(f"{trade_time:<15} {color_code}{side:<6}{reset_code} "
              f"{trade['price']:>12,.2f} {trade['size']:>15,.4f}")

def format_liquidity_stats(analyzer):
    """Met en forme les statistiques de liquidité."""
    print("\n" + "-" * 60)
    print("ANALYSE DU SPREAD")
    print("-" * 60)
    print(f"{'SIZE (BTC)':<12} {'CURRENT':>10} {'AVG':>10} {'MEDIAN':>10} {'MIN':>8} {'MAX':>8}")
    for size, stats in analyzer.stats.items():
        print(f"{size:<12.1f} {stats['current']:>10.2f} {stats['avg']:>10.2f} "
              f"{stats['median']:>10.2f} {stats['min']:>8.2f} {stats['max']:>8.2f}")

def format_strategy_status(mm, rm):
    """Met en forme le statut de la stratégie et du risque."""
    print("\n" + "-" * 60)
    print("ETAT DE LA STRATÉGIE")
    print("-" * 60)
    status_color = '\033[92m' if rm.is_active else '\033[91m'
    status_text = "ACTIVE" if rm.is_active else "INACTIVE (RISK LIMIT)"
    print(f"Statut         : {status_color}{status_text}\033[0m")
    bid_str = f"{mm.quote_bid:,.2f}" if mm.quote_bid else "N/A"
    ask_str = f"{mm.quote_ask:,.2f}" if mm.quote_ask else "N/A"
    print(f"Nos Cotations  : Bid @ {bid_str} | Ask @ {ask_str}")
    pnl_color = '\033[92m' if (mm.realized_pnl + mm.unrealized_pnl) >= 0 else '\033[91m'
    reset_color = '\033[0m'
    print(f"Position       : {mm.position:,.4f} BTC (Avg Entry: {mm.avg_entry_price:,.2f})")
    print(f"Exposition     : {mm.exposure:,.2f} / {rm.MAX_EXPOSURE:,.0f} USD")
    print(f"PnL Realized    : {mm.realized_pnl:,.2f} USD")
    print(f"PnL Unrealized: {mm.unrealized_pnl:,.2f} USD")
    print(f"PnL (Total)    : {pnl_color}{mm.realized_pnl + mm.unrealized_pnl:,.2f} USD{reset_color} (Max Loss: {rm.MAX_LOSS:,.0f} USD)")

def main():
    symbol = "btcusdt"
    spread_history_for_plot = []

    ws_client = WebSocketClient(symbol=symbol)
    order_book = OrderBook(symbol=symbol)
    trade_feed = TradeFeed(max_trades=100)
    liquidity_analyzer = LiquidityAnalyzer(order_book)
    market_maker = MarketMaker(order_book, trade_feed, quote_spread=0.01, order_size=1)
    risk_manager = RiskManager(market_maker)

    # Connecter le client WebSocket
    ws_client.on_depth_update = order_book.handle_depth_update
    ws_client.on_trade = trade_feed.add_trade
    sync_attempts = 0
    MAX_ATTEMPTS = 10 # Nombre maximum de tentatives de synchronisation

    while not order_book.is_initialized and sync_attempts < MAX_ATTEMPTS:
        sync_attempts += 1
        print(f"--- Tentative de synchronisation {sync_attempts}/{MAX_ATTEMPTS} ---")
        if ws_client.ws:
            ws_client.disconnect()
        order_book.reset()
        ws_client.connect()
        print("Connexion WebSocket en cours, attente de la mise en tampon des données...")
        time.sleep(5)
        order_book.initialize_snapshot()
        if not order_book.is_initialized:
            print(f"Échec de la synchronisation, nouvelle tentative dans 2 secondes...")
            ws_client.disconnect()
            time.sleep(2)

    if not order_book.is_initialized:
        print(f"Impossible de synchroniser le carnet d'ordres après {MAX_ATTEMPTS} tentatives. Arrêt du programme. !!!")
        exit()

    print("\n*** Synchronisation du carnet d'ordres réussie ***\n")
    time.sleep(2)
    try:
        while True:
            market_maker.update_pnl()
            risk_manager.check_risk()

            if risk_manager.is_active:
                market_maker.update_quotes()
                if market_maker.quote_bid is not None or market_maker.quote_ask is not None:
                    last_trade = trade_feed.get_recent_trades(1)
                    if last_trade:
                        trade_price = last_trade[0]['price']
                        if market_maker.quote_bid is not None and trade_price <= market_maker.quote_bid:
                            if risk_manager.can_place_trade('buy', market_maker.order_size, trade_price):
                                market_maker._execute_trade('buy', market_maker.order_size, trade_price)
                        elif market_maker.quote_ask is not None and trade_price >= market_maker.quote_ask:
                            if risk_manager.can_place_trade('sell', market_maker.order_size, trade_price):
                                market_maker._execute_trade('sell', market_maker.order_size, trade_price)

            liquidity_analyzer.update_stats()
            clear_console()

            top_bids, top_asks = order_book.get_top_levels(5)
            recent_trades = trade_feed.get_recent_trades(10)
            best_bid, best_ask, spread = order_book.get_spread()

            if spread is not None and spread > 0:
                spread_history_for_plot.append((datetime.now(), spread))

            print(f"\n--- Marché: {symbol.upper()} --- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

            if best_bid and best_ask:
                spread_str = f"{spread:,.2f}" if spread is not None else "CROSSED"
                print(f"Spread: {spread_str} (Bid: {best_bid:,.2f} / Ask: {best_ask:,.2f})\n")
                if spread_str == "CROSSED":
                    print("Carnet crossed détecté, réinitialisation...")
                    order_book.initialize_snapshot()
                    time.sleep(1)
                    continue
            format_strategy_status(market_maker, risk_manager)
            if top_bids and top_asks:
                format_book(top_bids, top_asks)
            if liquidity_analyzer.stats:
                format_liquidity_stats(liquidity_analyzer)
            if recent_trades:
                format_trades(recent_trades)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nArrêt")
    finally:
        ws_client.disconnect()
        market_maker.update_pnl()
        total_pnl = market_maker.realized_pnl + market_maker.unrealized_pnl
        print(f"\n--- PnL Final ---")
        print(f"PnL Réalisé    : {market_maker.realized_pnl:.2f} USD")
        print(f"PnL Non-Réalisé: {market_maker.unrealized_pnl:.2f} USD (sur {market_maker.position:.4f} BTC)")
        print(f"PnL Total Final: {total_pnl:.2f} USD")

        if spread_history_for_plot:
            print("\nGénération du graphique de l'historique du spread...")
            timestamps, spreads = zip(*spread_history_for_plot)

            plt.figure(figsize=(15, 7))
            plt.plot(timestamps, spreads, label=f'Spread sur {symbol.upper()}')
            plt.title(f'Historique du Spread sur {symbol.upper()}')
            plt.xlabel('Temps')
            plt.ylabel('Spread')
            plt.grid(True)
            plt.legend()
            plt.gcf().autofmt_xdate()
            plt.show()

if __name__ == "__main__":
    main()