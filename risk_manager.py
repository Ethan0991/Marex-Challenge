class RiskManager:
    """
    Intègre les contraintes de risque à la stratégie .
    - Surveille l'exposition notionnelle.
    - Stop Loss.
    """
    def __init__(self, market_maker, max_exposure=1_000_000, max_loss=-100_000):
        self.market_maker = market_maker
        self.MAX_EXPOSURE = max_exposure
        self.MAX_LOSS = max_loss
        self.is_active = True

    def check_risk(self):
        """
        Vérifie toutes les limites de risque. Si une limite est dépassée,
        la stratégie est désactivée.
        """
        if not self.is_active:
            return

        # 1. Vérifier la perte maximale
        total_pnl = self.market_maker.realized_pnl + self.market_maker.unrealized_pnl
        if total_pnl <= self.MAX_LOSS:
            print(f" Alerte: Perte maximale de {self.MAX_LOSS} USD atteinte, Arrêt de la stratégie")
            self.is_active = False
            # Idéalement, ici on enverrait aussi un ordre pour fermer la position restante.

    def can_place_trade(self, side, size, price):
        """
        Vérifie si le nouveau trade respecterait la limite d'exposition.

        Args:
            side (str): 'buy' or 'sell'.
            size (float): La taille du trade.
            price (float): Le prix du trade.

        Returns:
            bool: True si le trade est autorisé, False sinon.
        """
        if not self.is_active:
            return False

        current_exposure = self.market_maker.exposure
        trade_value = size * price

        if side == 'buy':
            potential_exposure = abs(current_exposure + trade_value)
        else: # sell
            potential_exposure = abs(current_exposure - trade_value)

        if potential_exposure > self.MAX_EXPOSURE:
            return False

        return True

if __name__ == "__main__":
    class MockMarketMaker:
        def __init__(self):
            self.realized_pnl = 0
            self.unrealized_pnl = 0
            self.exposure = 0

    mock_mm = MockMarketMaker()
    risk_manager = RiskManager(mock_mm, max_exposure=10000, max_loss=-1000)

    print("--- Test d'exposition ---")
    print(f"Trade autorisé ? {risk_manager.can_place_trade('buy', 1, 50000)}") # Devrait être False
    mock_mm.exposure = 5000
    print(f"Trade autorisé ? {risk_manager.can_place_trade('buy', 0.1, 60000)}") # Devrait être False
    print(f"Trade autorisé ? {risk_manager.can_place_trade('buy', 0.05, 60000)}") # Devrait être True

    print("\n--- Test de perte maximale ---")
    mock_mm.realized_pnl = -1001
    risk_manager.check_risk()
    print(f"Stratégie active ? {risk_manager.is_active}") # Devrait être False
