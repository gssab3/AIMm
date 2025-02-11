# AiMm

## Introduzione
AIMm è un progetto che mira a bilanciare in maniera esaustiva le partite dei videogiochi online, garantendo un'esperienza **propositiva**, **evolutiva** e **divertente** per ogni tipo di utenza, da quella principiante a quella esperta. L'idea di questo progetto è nata dall'**inefficacia** di molti sistemi di bilanciamento presenti in rete o per il bilanciamento **eccessivo** che, con grandi dati, tende a far giocare le persone con il *"proprio specchio"*, avendo partite statiche, prive di evoluzione, poichè piene di utenti con le stesse e identiche abilità. AIMm prevede un bilancio nè eccessivamente scarso, nè eccessivamente alto, garantendo la **dinamicità** delle partite.

## Come Funziona?
AIMm analizza le varie esperienze presenti all'interno di un dataset e, in base allo studio dei dati riguardanti le abilità di ogni giocatore ed il tempo di gioco, crea delle lobby che presentano Fairness. Gli algoritmi utilizzati sono **unsupervisioned** basati sul clustering: Elbow-Method per capire il numero di cluster in base alla somma degli errori quadratici; K-Means per suddividere il dataset in cluster basati sull'abilità; KNN per aggiungere i giocatori in coda a delle lobby; PCA per la visualizzazione bidimensionale o tridimensionale del dataset con la riduzione della dimensionalità.
