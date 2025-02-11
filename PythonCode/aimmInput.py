

import pandas as pd
import numpy as np
from google.colab import data_table
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display
from pandas.plotting import table
from IPython.display import Image, display
from tabulate import tabulate
import random


data_table.enable_dataframe_formatter()
data_table.DataTable.max_columns = 25



dataset = pd.read_csv("cod.csv")

dataset


nan_mask = dataset.isna()
nan_count = nan_mask.sum()

nan_count



dataset.describe()


pos = dataset.columns.get_loc("kills")
col = dataset.pop("deaths")
dataset.insert(pos + 1, "deaths", col)

pos = dataset.columns.get_loc("wins")
col = dataset.pop("losses")
dataset.insert(pos + 1, "losses", col)

pos = dataset.columns.get_loc("losses")
col = dataset.pop("gamesPlayed")
dataset.insert(pos + 1, "wlRatio",np.nan)
dataset.insert(pos + 2, "gamesPlayed", col)

pos = dataset.columns.get_loc("prestige")
col = dataset.pop("shots")
dataset.insert(pos + 1, "shots", col)

pos = dataset.columns.get_loc("hits")
col = dataset.pop("misses")
dataset.insert(pos + 1, "misses", col)
col = dataset.pop("headshots")
dataset.insert(pos + 2, "headshots", col)
dataset.insert(pos + 3, "precisionHead", np.nan)
dataset.insert(pos + 4, "precisionAim", np.nan)
col = dataset.pop("timePlayed")
dataset.insert(pos + 5, "timePlayed", col)

dataset


dataset.drop(['averageTime'], axis = 1, inplace = True)
dataset.drop(['assists'], axis = 1, inplace = True)
dataset.drop(['xp'], axis = 1, inplace = True)
dataset.drop(['scorePerMinute'], axis = 1, inplace = True)

dataset['gamertag'] = dataset['name'].apply(lambda x: x.split('#')[0] if '#' in x else x)
dataset['tag'] = dataset['name'].apply(lambda x: x.split('#')[1] if '#' in x else '')


dataset

dataset['gamertag']

dataset['tag']



dataset.drop(['name'], axis = 1, inplace = True)
col = dataset.pop("gamertag")
dataset.insert(0, "gamertag", col)
dataset.drop(['tag'], axis = 1, inplace = True)
dataset



dataset["wlRatio"] = dataset["wins"] / dataset["losses"]
dataset["precisionHead"] = dataset["headshots"] / dataset["hits"]
dataset["precisionAim"] = dataset["hits"] / dataset["shots"]

dataset


dataset["wlRatio"] = dataset["wlRatio"].round(2)
dataset["kdRatio"] = dataset["kdRatio"].round(2)
dataset["precisionHead"] = dataset["precisionHead"].round(2)
dataset["precisionAim"] = dataset["precisionAim"].round(2)

dataset["wlRatio"] = dataset["wlRatio"].fillna(0)
dataset["precisionHead"] = dataset["precisionHead"].fillna(0)
dataset["precisionAim"] = dataset["precisionAim"].fillna(0)

dataset



infinity_rows = dataset[dataset.isin([float('inf'), float('-inf')]).any(axis=1)]
infinity_rows

dataset["wlRatio"] = np.where(dataset["losses"] == 0, dataset["wins"], dataset["wins"] / dataset["losses"])

infinity_rows = dataset[dataset.isin([float('inf'), float('-inf')]).any(axis=1)]
infinity_rows



# Rimuovi righe con shots < headshots
dataset = dataset[dataset["shots"] >= dataset["headshots"]]

# Rimuovi righe con shots = 0 e level > 1
dataset = dataset[~((dataset["shots"] == 0) & (dataset["level"] > 1))]

# Applicare il rounding PERMANENTE dopo il filtraggio
dataset["wlRatio"] = dataset["wlRatio"].round(2)

dataset



dataset["wlRatio"] = np.where(dataset["losses"] == 0, dataset["wins"], dataset["wins"] / dataset["losses"])

infinity_rows = dataset[dataset.isin([float('inf'), float('-inf')]).any(axis=1)]
infinity_rows


dataset = dataset[dataset["headshots"] <= (dataset["shots"] - dataset["misses"])]

infinity_rows = dataset[dataset.isin([float('inf'), float('-inf')]).any(axis=1)]


infinity_rows

dataset["wlRatio"] = dataset["wlRatio"].round(2)
dataset



dataset = dataset[~((dataset["prestige"] > 0) & (dataset["gamesPlayed"] < 1))]

dataset = dataset[~((dataset["kills"] > 0) | (dataset["deaths"] > 0)) | (dataset["gamesPlayed"] > 0)]

dataset["wlRatio"] = dataset["wlRatio"].round(2)
dataset

dataset



# Modifica i valori di prestigio superiori a 10, impostandoli a 11
dataset['prestige'] = dataset['prestige'].apply(lambda x: 11 if x > 10 else x)

def adjust_level_and_prestige(row):
    if row['prestige'] < 11 and row['level'] > 55:
        # Calcolare il numero di prestigio guadagnato
        prestige_gained = int(row['level'] // 55) - 1

        # Nuovo livello: livello attuale meno 55 per ogni prestigio guadagnato
        new_level = row['level'] - (prestige_gained * 55)

        # Nuovo prestigio: prestigio attuale più quello guadagnato
        new_prestige = row['prestige'] + prestige_gained

        return pd.Series([new_level, new_prestige])
    else:
        return pd.Series([row['level'], row['prestige']])

# Applicare la funzione al dataset
dataset[['level', 'prestige']] = dataset.apply(adjust_level_and_prestige, axis=1)
dataset["wlRatio"] = dataset["wlRatio"].round(2)

dataset



dataset["gamesPlayed"] = dataset["wins"] + dataset["losses"]
dataset["shots"] = dataset["hits"] + dataset["misses"]

dataset



dataset = dataset[~((dataset['kills'] > 0) & (dataset['hits'] == 0))]
dataset = dataset.assign(killstreak=dataset['killstreak'].where(dataset['killstreak'] <= 30, 30))



dataset["wlRatio"] = dataset["wlRatio"].round(2)
dataset

dataset.describe()


dataset['totalLevelAccount'] = dataset['level'] + 55 * dataset['prestige']
dataset.drop(['level'], axis = 1, inplace = True)
dataset.drop(['prestige'], axis = 1, inplace = True)

pos = dataset.columns.get_loc("killstreak")
col = dataset.pop("totalLevelAccount")
dataset.insert(pos + 1, "totalLevelAccount", col)

dataset


dataset = dataset[~(dataset["wlRatio"] > 100)]
dataset



X = dataset.iloc[:, [3, 4, 7, 8, 9, 14, 15, 16]]
X.head()



# Creiamo un array di pesi basato sulla divisione 70-30

weights = np.array([
    2.0,  # Feature 3  -> Gruppo 70%
    1.0,  # Feature 4  -> Gruppo 30%
    2.0,  # Feature 7  -> Gruppo 70%
    2.0,  # Feature 8  -> Gruppo 70%
    1.0,  # Feature 9  -> Gruppo 30%
    2.0,  # Feature 14 -> Gruppo 70%
    2.0,  # Feature 15 -> Gruppo 70%
    1.0   # Feature 16 -> Gruppo 30%
])


# Applichiamo i pesi moltiplicando le feature
X_weighted = X * weights

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_weighted) #

wcss = []
max_clusters = 10

for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Grafico Elbow
plt.figure(figsize=(8,6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method - Ottimizzazione numero di cluster")
plt.xlabel("Numero di cluster")
plt.ylabel("Somma degli errori quadrati (SSE)")
plt.show()


# Imposta il numero di cluster scelto (6)
optimal_k = 6
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Aggiungi le etichette dei cluster al dataset originale
dataset['Cluster'] = cluster_labels

# Riduci le 8 dimensioni a 2 componenti
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# Crea lo scatter plot, colorando i punti in base ai cluster
plt.figure(figsize=(8, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=dataset['Cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Componente Principale 1')
plt.ylabel('Componente Principale 2')
plt.title('Visualizzazione dei Cluster (PCA)')
plt.colorbar(label='Cluster')
plt.show()


# Riduci le 8 dimensioni a 3 componenti
pca = PCA(n_components=3)
principal_components = pca.fit_transform(X_scaled)

# Crea la figura e il grafico 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot dei cluster in 3D
scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2],
                     c=dataset['Cluster'], cmap='viridis', alpha=0.6)

# Etichette e titolo
ax.set_xlabel('Componente Principale 1')
ax.set_ylabel('Componente Principale 2')
ax.set_zlabel('Componente Principale 3')
ax.set_title('Visualizzazione dei Cluster (PCA 3D)')

# Barra dei colori
cbar = plt.colorbar(scatter)
cbar.set_label('Cluster')

# Mostra il grafico
plt.show()



for cluster_id in range(optimal_k):  # 'optimal_k' è il numero di cluster
    cluster_players = dataset[dataset['Cluster'] == cluster_id]  # Seleziona i giocatori nel cluster
    print(f"Cluster {cluster_id}:")
    display(cluster_players[['gamertag', 'wlRatio', 'gamesPlayed', 'kdRatio', 'killstreak', 'totalLevelAccount', 'precisionHead', 'precisionAim', 'timePlayed']])  # Personalizza le colonne



player_debutante = {
    'gamertag': input("Inserisci il Gamertag: "),
    'wlRatio': float(input("Inserisci il Win/Loss Ratio: ")),
    'gamesPlayed': int(input("Inserisci il numero di partite giocate: ")),
    'kdRatio': float(input("Inserisci il Kill/Death Ratio: ")),
    'killstreak': int(input("Inserisci la killstreak più alta: ")),
    'totalLevelAccount': int(input("Inserisci il livello totale dell'account: ")),
    'precisionHead': float(input("Inserisci la precisione alla testa (es. 0.1 per 10%): ")),
    'precisionAim': float(input("Inserisci la precisione generale (es. 0.05 per 5%): ")),
    'timePlayed': float(input("Inserisci il tempo giocato (in ore): "))
}

# Creiamo un DataFrame con i nuovi giocatori
new_players = pd.DataFrame([player_debutante])

# Rimuoviamo il 'gamertag' dal DataFrame prima di normalizzarlo
new_players_features = new_players.drop(columns=['gamertag'])

# Normalizziamo le caratteristiche dei nuovi giocatori
new_players_scaled = scaler.transform(new_players_features)

# Predizione del cluster per ciascun nuovo giocatore
predicted_clusters = kmeans.predict(new_players_scaled)

# Assegniamo i cluster ai nuovi giocatori
new_players['Predicted Cluster'] = predicted_clusters

# Visualizziamo il risultato con il gamertag mantenuto
print(new_players[['gamertag', 'Predicted Cluster']])


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Creiamo le lobby per i due nuovi giocatori
lobbies = {}
features = ['wlRatio', 'gamesPlayed', 'kdRatio', 'killstreak',
            'totalLevelAccount', 'precisionHead', 'precisionAim', 'timePlayed']

for i, player in new_players.iterrows():
    cluster_id = player['Predicted Cluster']  # Cluster assegnato tramite KMeans
    player_data = player[features].values.reshape(1, -1)  # Caratteristiche numeriche per il calcolo

    # Selezioniamo tutti i giocatori che appartengono al cluster del giocatore corrente
    cluster_players = dataset[dataset['Cluster'] == cluster_id]

    # Rimuoviamo il giocatore principale dalla selezione
    cluster_players = cluster_players[cluster_players['gamertag'] != player['gamertag']]

    if cluster_players.empty:
        print(f"Cluster {cluster_id} vuoto o già processato, salto il giocatore {player['gamertag']}")
        continue

    # Normalizziamo i dati del giocatore e dei giocatori del cluster
    player_data_scaled = scaler.transform(player_data)
    cluster_players_scaled = scaler.transform(cluster_players[features])

    # Adattiamo il numero di vicini in base ai giocatori disponibili
    n_neighbors = min(len(cluster_players), 11)  # Max 11 vicini da aggiungere
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(cluster_players_scaled)
    distances, indices = knn.kneighbors(player_data_scaled)

    # Selezioniamo i giocatori più vicini
    lobby_players = cluster_players.iloc[indices[0]].copy()

    # Aggiungiamo il nuovo giocatore in cima alla lista
    lobby_players = pd.concat([pd.DataFrame(player).T, lobby_players])

    # Salviamo la lobby per questo giocatore
    lobbies[f'Lobby_Player_{i}'] = lobby_players

# Stampiamo le lobby per ogni giocatore
for i, (lobby_name, lobby) in enumerate(lobbies.items(), start=1):
    print(f"\n===== {lobby_name} =====")

    # Otteniamo il giocatore principale
    main_player = lobby.iloc[0][['gamertag'] + features]

    # Stampa il giocatore principale con tabulate
    print("Giocatore principale:")
    print(tabulate([main_player.values], headers=['Gamertag'] + features, tablefmt='grid'))

    # Stampiamo gli altri membri della lobby
    other_players = lobby.iloc[1:][['gamertag'] + features]
    print("\nAltri giocatori nella lobby:")
    print(tabulate(other_players.values, headers=['Gamertag'] + features, tablefmt='grid'))

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def plot_all_clusters():
    unique_clusters = dataset['Cluster'].unique()

    for cluster_n in unique_clusters:
        cluster_players = dataset[dataset['Cluster'] == cluster_n]
        if cluster_players.empty:
            continue

        # Applichiamo PCA per ridurre a 2D
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaler.transform(cluster_players[features]))

        plt.figure(figsize=(8, 6))
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c='blue', label=f'Cluster {cluster_n}')

        # Etichette con i gamertag
        for i, txt in enumerate(cluster_players['gamertag']):
            plt.annotate(txt, (reduced_data[i, 0], reduced_data[i, 1]), fontsize=8, alpha=0.7)

        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title(f"Visualizzazione dei giocatori nel Cluster {cluster_n}")
        plt.legend()
        plt.show()
plot_all_clusters()

# Funzione per salvare la tabella come immagine
def save_table_as_image(df, filename='table_image.png'):
    fig, ax = plt.subplots(figsize=(8, 4))  # Impostiamo la dimensione della figura
    ax.axis('tight')  # Disabilita gli assi
    ax.axis('off')  # Disabilita gli assi

    # Crea la tabella
    tab = table(ax, df, loc='center', colWidths=[0.2]*len(df.columns))

    # Personalizza la tabella
    tab.auto_set_font_size(False)  # Disabilita il ridimensionamento automatico del font
    tab.set_fontsize(10)  # Impostiamo la dimensione del font
    tab.scale(1.2, 1.2)  # Aumentiamo la scala della tabella per renderla più visibile

    # Salva l'immagine
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # Chiudiamo la figura per evitare che venga visualizzata due volte

# Iteriamo su tutte le lobby e salviamo ogni tabella come immagine
for i, (lobby_name, lobby) in enumerate(lobbies.items(), start=1):
    lobby_data = lobby[['gamertag'] + features]
    filename = f'lobby_{i}_table.png'  # Definiamo il nome del file per ogni lobby
    save_table_as_image(lobby_data, filename)  # Salviamo l'immagine della lobby

    # Mostriamo l'immagine nel notebook
    display(Image(filename=filename))



# Funzione per separare i giocatori in squadre da 2
def crea_squadre(lobby, input_player):
    # Creiamo una lista di giocatori nella lobby
    players = lobby[features].values.tolist()

    # Mescoliamo i giocatori in modo casuale
    random.shuffle(players)

    # Creiamo le squadre
    squadre = {
        "Rogue Black Ops": [],
        "Crimson One": []
    }

    # Aggiungiamo i giocatori alle squadre
    for i in range(0, len(players), 2):
        # Alterniamo tra le due squadre
        squadra = "Rogue Black Ops" if i % 4 == 0 else "Crimson One"
        squadre[squadra].append(players[i:i+2])

    # Stampa delle squadre
    for squad_name, members in squadre.items():
        print(f"\n{squad_name}\n")
        for pair in members:
            for player in pair:
                # Troviamo il gamertag corrispondente
                gamertag = lobby.loc[lobby[features].apply(tuple, axis=1) == tuple(player), 'gamertag'].values[0]
                # Verifica se il giocatore è quello di input
                if player == input_player:
                    print(f"*{gamertag}* (Giocatore input)")  # Evidenziato
                else:
                    print(gamertag)  # Player normale

# Esempio: Per ogni lobby separa i giocatori in squadre
for i, (lobby_name, lobby) in enumerate(lobbies.items(), start=1):
    print(f"\n===== Lobby {i} =====")

    # Supponiamo che il primo giocatore della lobby sia quello di input
    input_player = lobby.iloc[0][features].values.tolist()  # O prendi il tuo giocatore specifico

    # Crea le squadre e stampa
    crea_squadre(lobby, input_player)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Definiamo le feature chiave per valutare la skill
skill_features = ['wlRatio', 'kdRatio', 'precisionAim', 'totalLevelAccount', 'timePlayed']

# Creiamo un dizionario per salvare i risultati
cluster_coherence = {}

# Calcoliamo la varianza media per ogni cluster
for cluster_id in range(optimal_k):
    cluster_players = dataset[dataset['Cluster'] == cluster_id][skill_features]

    if cluster_players.empty:
        print(f"⚠️ Cluster {cluster_id} è vuoto, salto il calcolo.")
        continue

    # Calcoliamo la deviazione standard media tra i giocatori del cluster
    cluster_std = cluster_players.std().mean()

    # Normalizziamo la giocabilità: minore è la varianza, maggiore è la coesione
    # Usiamo una scala percentuale inversa (più bassa la std, più alta la percentuale di giocabilità)
    max_std = dataset[skill_features].std().mean()
    giocabilita_percent = max(0, 100 - (cluster_std / max_std) * 100)

    cluster_coherence[cluster_id] = giocabilita_percent

# Stampa i risultati
print("\n===== Giocabilità nei Cluster =====")
for cluster_id, giocabilita in cluster_coherence.items():
    print(f"🎯 Cluster {cluster_id}: {giocabilita:.2f}% coerenza di skill")

# Creiamo un grafico a barre per visualizzare la giocabilità nei cluster
plt.figure(figsize=(8,5))
plt.bar(cluster_coherence.keys(), cluster_coherence.values(), color='royalblue')
plt.axhline(y=40, color='r', linestyle='--', label="Soglia sbilanciamento")
plt.xlabel("Cluster")
plt.ylabel("Giocabilità (%)")
plt.title("Distribuzione della Giocabilità per Cluster")
plt.ylim(0, 100)
plt.xticks(range(optimal_k))
plt.show()

import random

# Funzione per calcolare la fairness di una partita
def calcola_fairness(team1, team2):
    # Calcoliamo la media skill per team (usando le stesse feature di skill)
    team1_skill = team1[skill_features].mean().mean()
    team2_skill = team2[skill_features].mean().mean()

    # Calcoliamo il fairness score (rapporto tra il team più forte e quello più debole)
    fairness_score = max(team1_skill, team2_skill) / min(team1_skill, team2_skill)

    return fairness_score

# Funzione per generare le squadre e calcolare la fairness per ogni partita
def valuta_partite(lobbies):
    fairness_scores = []

    for i, (lobby_name, lobby) in enumerate(lobbies.items(), start=1):
        print(f"\n===== Valutazione Fairness - {lobby_name} =====")

        # Mischiamo casualmente i giocatori e dividiamoli in due squadre
        shuffled_players = lobby.sample(frac=1).reset_index(drop=True)
        metà = len(shuffled_players) // 2
        team1 = shuffled_players.iloc[:metà]
        team2 = shuffled_players.iloc[metà:]

        # Calcoliamo la fairness della partita
        fairness_score = calcola_fairness(team1, team2)
        fairness_scores.append(fairness_score)

        # Stampiamo il risultato
        print(f"🔹 Fairness Score: {fairness_score:.2f} (Vicino a 1.4 è meglio)")
        if fairness_score > 1.4:
            print("⚠️ Attenzione: Partita potenzialmente sbilanciata!")

    # Grafico della fairness di tutte le partite
    plt.figure(figsize=(8,5))
    plt.bar(range(len(fairness_scores)), fairness_scores, color=['green' if x <= 1.4 else 'red' for x in fairness_scores])
    plt.axhline(y=1.4, color='r', linestyle='--', label="Soglia sbilanciamento")
    plt.xlabel("Partita")
    plt.ylabel("Fairness Score")
    plt.title("Distribuzione della Fairness delle Partite")
    plt.legend()
    plt.show()

# Eseguiamo la valutazione
valuta_partite(lobbies)

