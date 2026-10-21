
# NBA Fantasy Draft Lab

### Interactive Fantasy Basketball Analytics & Draft Strategy

**Python | Streamlit | Sports Analytics | Player Valuation | Data Visualization**

[Launch the Live Fantasy Draft Lab](https://nba-fantasy-draft-lab.streamlit.app/)

## Project Overview

NBA Fantasy Draft Lab is an interactive fantasy basketball analytics application built with Python and Streamlit.

The application brings together NBA player data, fantasy valuation methods, rankings, and draft-tracking tools to help users evaluate players and make informed fantasy basketball draft decisions.

Rather than presenting a static list of player rankings, the application allows users to interact with player information, compare options, and track draft selections.

The project combines software development, data analysis, quantitative modeling, and interactive visualization.

## Live Application

**Fantasy Draft Lab:**

https://nba-fantasy-draft-lab.streamlit.app/

The application is publicly accessible. No local installation is required to explore the draft tools.

## Key Features

### 1. Player Rankings & Valuation

- Explore NBA fantasy player rankings.
- Evaluate players using prepared fantasy valuation data.
- Compare players and assess potential draft selections.
- Examine preset and custom ranking approaches.

### 2. Interactive Draft Tracking

- Record draft selections through the Streamlit interface.
- Track selected players and draft progress.
- Use snake-draft position tools to support draft planning.
- Download pick logs for later reference.

Draft selections are stored in each visitor's Streamlit session rather than a shared file.

### 3. Draft Strategy Analysis

- Explore player value relative to available draft options.
- Use rankings and position-based valuation to inform decisions.
- Evaluate draft scenarios through interactive controls.

These tools provide decision support rather than guarantees of fantasy performance.

### 4. Projection Research & Model Evaluation

The broader project includes Python scripts for:

- Player valuation and position-based analysis.
- Minutes, productivity, and role-change projection experiments.
- Historical backtesting and model comparison.
- Calibration and analytical audits.

Experimental projections are maintained separately from established ranking outputs until their performance is sufficiently evaluated.

## Technical Implementation

| Component | Technology / Approach |
| --- | --- |
| Application framework | Streamlit |
| Programming language | Python |
| Data analysis | Player datasets and analytical data processing |
| Player evaluation | Fantasy valuation and ranking models |
| Draft functionality | Interactive selections and session-based tracking |
| Research | Projection experiments, backtesting, and calibration |
| Deployment | Streamlit Community Cloud |
| Version control | GitHub |

### Application Structure

The repository includes:

- `app.py` — Main Fantasy Draft Lab application.
- `public_app.py` — Companion read-only analytics dashboard, retained in the codebase.
- `draft_engine.py` — Draft-related analytical logic.
- `build_value_model.py` — Player valuation model development.
- `build_position_value.py` — Position-based valuation.
- `build_competitive_edge.py` — Supporting draft strategy research.
- `requirements.txt` — Python dependencies.
- `data/` — Prepared application datasets and model outputs.

Additional scripts support projections, historical evaluation, and testing.

## Research & Evaluation

The project includes experimental work investigating whether additional player information can improve projection accuracy.

One documented experiment reported:

| Evaluation | Mean Absolute Error |
| --- | ---: |
| Experimental projection | 5.068 |
| Previous-season baseline | 5.225 |

The comparison covered 935 returning-player observations.

These results describe a specific research experiment. They do **not** establish that the experimental approach consistently outperforms other forecasting methods or produces better fantasy draft outcomes.

The main application retains its established draft-value outputs while the newer projection approaches remain experimental.

## Run Locally

Install Python 3.11 or 3.12.

Clone the repository:

```bash
git clone https://github.com/lpigues-clt/nba-fantasy-draft-lab.git
cd nba-fantasy-draft-lab
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the Fantasy Draft Lab:

```bash
streamlit run app.py
```

Streamlit will provide a local URL for accessing the application in your browser.

### Companion Analytics Application

The repository also contains a read-only NBA analytics interface:

```bash
streamlit run public_app.py
```

This companion application is not currently maintained as a separate public deployment.

## Data Sources & Limitations

The repository contains prepared player datasets and model outputs used by the application.

Important limitations include:

- **Data freshness:** Player information and projections reflect prepared data snapshots and may not represent current NBA rosters, injuries, or performance.
- **Projection uncertainty:** Fantasy outcomes depend on factors that cannot be predicted reliably from historical data alone.
- **Experimental modeling:** Research results should not be interpreted as proof of a competitive draft advantage.
- **Session persistence:** Draft selections are temporary and may be lost after a refresh, session expiration, or application restart. Users should export their pick logs when needed.
- **Live statistics:** The publicly deployed Fantasy Draft Lab should not be interpreted as a real-time NBA statistics service.
- **Data rights:** Underlying datasets and their permitted uses should be verified before redistributing or extending the project.

The application is intended for analytics, exploration, and fantasy draft decision support.

## Future Improvements

Potential enhancements include:

- Integrating updated NBA player statistics.
- Expanding historical backtesting across multiple seasons.
- Improving projection calibration and uncertainty reporting.
- Adding more detailed player comparison visualizations.
- Supporting persistent, user-controlled draft sessions.
- Expanding automated testing and data validation.

## About the Developer

**Levi Pigues**

B.S. Data Science — University of North Carolina at Charlotte  
Expected Graduation: Spring 2028

Interests: Sports Analytics, Machine Learning, Business Intelligence, and Applied Data Science.

[Portfolio](https://lpigues-clt.github.io/data-science-portfolio/) | [GitHub](https://github.com/lpigues-clt) | [Live Fantasy Draft Lab](https://nba-fantasy-draft-lab.streamlit.app/)
