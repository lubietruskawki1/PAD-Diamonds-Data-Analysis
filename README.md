# Diamonds Data Analysis (PAD 2023/2024)

Project written for the Programming for Data Analytics (pol. Programowanie dla Analityki Danych, PAD) course offered by the Faculty of Computer Science at the Polish-Japanese Academy of Information Technology 2023/2024.

Interactive data analysis dashboard showcasing the process of cleaning messy diamonds-related data, visualizing it, building a regression model and creating customizable visualizations.

## Set up

#### First time

```bash
python3 -m venv .venv
source ".venv/bin/activate"
pip install -r requirements.txt
jupyter notebook
```

#### Every other time

```bash
source ".venv/bin/activate"
jupyter notebook
```

#### Dashboard

```bash
streamlit run dashboard/dashboard.py
```

