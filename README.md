# Fantasy Draft Lab · NBA Analytics (2026–27)

Two Streamlit applications for fantasy basketball analysis. **Fantasy Draft Lab** is the primary interactive application; **NBA Analytics** is a companion read-only player projections dashboard.

## Run locally

Install Python 3.11 or 3.12, then from the repository root:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

For the companion dashboard:

```bash
streamlit run public_app.py
```

## Deploy both apps with Streamlit Community Cloud

1. Push the contents of this folder (not the original development ZIP) to a new GitHub repository.
2. At https://share.streamlit.io/ create an app using this repository, branch `main`, entry point `app.py`.
3. Create a second app from the same repository with entry point `public_app.py`.
4. Select Python 3.11 or 3.12 where available. The root `requirements.txt` serves both apps.
5. Open each public URL in a private browser window to test access.

## Features

- Fantasy Draft Lab: preset and custom ranking analysis, player comparison, draft tracking, snake-draft position tools, and downloadable pick logs.
- NBA Analytics: preseason player projection table, plus live/final box-score exploration when an appropriate local database is available.
- Draft selections and configuration are isolated per browser session in Streamlit session state; they are **not permanently saved** and can be lost on refresh, session expiry, or app restart. Export your pick log before leaving.
- The companion dashboard will normally display preseason projections on Community Cloud; the local SQLite live-game database is deliberately excluded. This is **not** an automatically updating live-score service.

## Data and limitations

The repository includes prepared player data and model outputs to support immediate app startup. Projections and team assignments are provisional, and experimental forecasts should not be presented as validated draft advantage. The repository excludes development archives, the local virtual environment, personal draft state, and local live-score storage.

The dataset is a snapshot. Verify source data usage rights and attribution before public release. Do not commit API keys or private information.
