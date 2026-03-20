# Praktisches Experiment: Toy World Model / Reliability Paradox

Dieses Projekt demonstriert in einer kontrollierten **GridWorld** den Unterschied zwischen realer Dynamik (Realität) und einem gelernten Modell (**DreamEnv**). Aus Offline-Daten wird ein Weltmodell trainiert, in dem anschließend ein RL-Agent (PPO) lernt. Der Agent wirkt im DreamEnv leistungsfähig, scheitert aber teilweise beim Transfer zurück in die echte GridWorld. Dieser Sim-to-Real-Bruch zeigt das Reliability Paradox bzw. Reward-Hacking in kompakter Form.

## Projektstruktur

- `reliability_paradox_toy.py`: zentraler Startpunkt für das komplette Experiment.
- `toywm/cli.py`: Orchestrierung der Pipeline (Datensammlung, Weltmodell-Training, PPO, Evaluation, Plot).
- `toywm/envs.py`: Definition von `GridWorldEnv` (Realität) und `DreamEnv` (Weltmodell-Simulation).
- `toywm/models.py`, `toywm/train.py`, `toywm/data.py`: Modellarchitektur, Training und Offline-Datengenerierung.
- `toywm/eval_plot.py`: Evaluationslogik und Visualisierung der Trajektorien.
- `Figure_1.png`: Standard-Ausgabepfad für die erzeugte Vergleichsabbildung.
- `artifacts/lean/` (bei Ausführung erzeugt): Modell- und Datensatzartefakte.

## Installation

1. Python **3.10+**
2. Optional: virtuelle Umgebung erstellen und aktivieren:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
3. Abhängigkeiten installieren:
   - `pip install numpy torch gymnasium matplotlib stable-baselines3`
   oder
   - `pip3 install numpy torch gymnasium matplotlib stable-baselines3`

## Ausführung

Zentrales Startkommando:

```bash
python3 reliability_paradox_toy.py
```

Nützliche Optionen (optional):

```bash
python3 reliability_paradox_toy.py --ppo-steps 1000000 --eval-episodes 20 --workdir artifacts/lean
```

## Ergebnis

- `artifacts/lean/offline_data.npz`: gesammelte Offline-Transitionsdaten.
- `artifacts/lean/world_model.pt`: trainiertes Weltmodell.
- `artifacts/lean/ppo_dream.zip`: im DreamEnv trainierte PPO-Policy.
- Konsole mit Dream-vs-Real-Metriken (u. a. Return, Success-Rate, Schrittzahl) sowie `Figure_1.png` als Vergleich der Laufwege.
