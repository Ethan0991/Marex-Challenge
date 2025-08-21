# Simulation d'un Market Maker Crypto

Ce projet simule en Python une stratégie market making sur le marché BTC/USDT. L'application se connecte en temps réel aux flux de données de l'échange Binance pour gérer un carnet d'ordres, analyser la liquidité et simuler une stratégie de trading tout en appliquant des contraintes de risque.

## Fonctionnalités Principales

-   **Connexion Temps Réel** : Utilise les WebSockets de Binance pour s'abonner aux flux de données du carnet d'ordres (Level 2, `@depth`) et des transactions du marché (`@trade`).
-   **Gestion Robuste du Carnet d'Ordres** : Implémente la méthode de synchronisation recommandée par Binance, combinant un snapshot initial via l'API REST avec la mise en tampon et l'application des mises à jour du flux WebSocket pour garantir un carnet d'ordres local précis et sans désynchronisation.
-   **Analyse de la Liquidité** : Calcule le **spread effectif** en temps réel pour différentes tailles d'ordres (0.1, 1, 5, 10 BTC) en se basant sur le coût VWAP (Volume-Weighted Average Price) pour traverser le carnet.
-   **Stratégie de Market Making** :
    -   Place des bids et asks simulés autour du midprice.
    -   Ajuste les cotations en fonction de la position actuelle par skewing pour gérer l'inventaire.
    -   (Bonus) Adapte le spread des cotations en fonction de la volatilité récente du marché.
-   **Simulation d'Exécution** : Simule l'exécution des fills en se basant sur le flux des transactions réelles du marché.
-   **Gestion du Risque** : Ajoute des limites pour désactiver la stratégie en cas de dépassement :
    -   Exposition notionnelle maximale.
    -   Perte maximale autorisée.
-   **Tableau de Bord en Console** : Affiche en continu de logs incluant :
    -   L'état du carnet d'ordres (top levels).
    -   Les dernières transactions du marché.
    -   Les statistiques de liquidité (spread effectif).
    -   Le statut de la stratégie (Position, P&L réalisé et non-réalisé, Exposition).
-   **Visualisation de Données** : À l'arrêt du programme, génère et affiche un graphique de l'historique du spread du marché sur la durée de la simulation.

## Architecture du Projet

Le code est structuré de manière OOP pour séparer chaque composant :

-   `main.py`: Le point d'entrée de l'application. Il initialise tous les modules, gère la boucle principale de la simulation et l'affichage du tableau de bord.
-   `connector.py`: Gère la connexion au WebSocket de Binance, l'abonnement aux flux et la distribution des messages aux modules appropriés.
-   `order_book.py`: Maintient l'état du carnet d'ordres local. Contient la logique complexe de synchronisation initiale (snapshot + buffer).
-   `trade_feed.py`: Stocke et gère la liste des transactions les plus récentes du marché.
-   `liquidity_analysis.py`: Responsable du calcul des statistiques de liquidité et du spread effectif.
-   `market_maker.py`: Contient la logique de la stratégie de trading (comment et où coter, gestion de la position et du P&L).
-   `risk_manager.py`: Surveille la stratégie du `MarketMaker` et s'assure que les limites de risque ne sont pas violées.

## Prérequis et Installation

Le projet est développé en Python 3. Les librairies nécessaires sont listées dans le fichier `requirements.txt`.

1.  **Clonez le dépôt :**
    ```bash
    git clone [URL_DU_DEPOT]
    cd [NOM_DU_DOSSIER]
    ```

2.  **Installez les dépendances :**
    Il est recommandé d'utiliser un environnement virtuel.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    Installez les librairies nécessaires :
    ```bash
    pip install -r requirements.txt
    ```

## Comment Lancer la Simulation

Une fois les dépendances installées, lancez le programme avec la commande suivante :

```bash
python3 main.py
```

-   Le programme va d'abord tenter de se synchroniser avec le carnet d'ordres de Binance. Ce processus peut prendre quelques secondes et plusieurs tentatives.
-   Une fois la synchronisation réussie, le tableau de bord s'affichera et se mettra à jour en temps réel.
-   Pour arrêter la simulation, appuyez sur `Ctrl+C`.
-   Après l'arrêt, le performances s'afficheront, suivi du graphique de l'historique du spread.

## Configuration

Les principaux paramètres de la stratégie peuvent être ajustés directement dans le fichier `main.py` lors de l'initialisation de la classe `MarketMaker` :

```python
# Dans main.py, fonction main()

market_maker = MarketMaker(
    order_book,
    trade_feed,
    order_size=0.01, # Taille de chaque ordre coté en BTC
    base_spread=0.08, # Spread de base en pourcentage (0.08%)
    volatility_factor=10.0 # Sensibilité du spread à la volatilité
)
```

Le symbole (ex: `btcusdt`) est également défini au début de la fonction `main()`.
