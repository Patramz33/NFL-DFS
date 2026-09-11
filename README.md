# NFL DFS Dashboard — Version 2

Local Streamlit dashboard for DraftKings NFL Classic main-slate analysis.

## 2026 Week 1 slate included
V2 is preconfigured for the 12 Sunday afternoon games on September 13, 2026:

- CHI @ CAR — 1:00 PM ET
- BAL @ IND — 1:00 PM ET
- ATL @ PIT — 1:00 PM ET
- CLE @ JAC — 1:00 PM ET
- TB @ CIN — 1:00 PM ET
- NYJ @ TEN — 1:00 PM ET
- NO @ DET — 1:00 PM ET
- BUF @ HOU — 1:00 PM ET
- ARI @ LAC — 4:25 PM ET
- GB @ MIN — 4:25 PM ET
- MIA @ LV — 4:25 PM ET
- WAS @ PHI — 4:25 PM ET

The Sunday night DAL @ NYG game is intentionally excluded from the standard main slate.

## V2 features
- Full 12-game slate view
- CSV player-pool upload
- Team/game/kickoff filters
- Projection, ceiling, ownership, value and leverage metrics
- DraftKings-style $50,000 optimizer
- QB stacking and optional bring-back
- Minimum salary and team/game player caps
- Combined ownership cap
- Multi-lineup portfolio generation (up to 50)
- Minimum lineup uniqueness
- Default and per-player max exposure controls
- Portfolio exposure report
- Single-lineup and portfolio CSV export

## Important data note
The included `sample_week1_main_slate.csv` is **synthetic test data**. It contains placeholder player names and test salaries/projections/ownership so you can verify the software. It is not a current DraftKings salary or projection feed.

For real play, manually download your DraftKings salary CSV, merge in your chosen projections/ownership, and upload the completed file.

## Run it
Requires Python 3.10+.

### Windows
Double-click `run_windows.bat`, or run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### macOS / Linux

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

Streamlit will normally open at `http://localhost:8501`.

## Required CSV columns

```text
name,position,team,opponent,salary,projection,ceiling,ownership,status,game
```

For the built-in Week 1 teams, V2 can infer `opponent`, `game`, and `kickoff` when blank.

## Responsible use
DFS outcomes are uncertain. This tool is decision support, not a guarantee of profit, and it does not scrape or submit lineups to DraftKings.
