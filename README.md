# Banking Customer Intelligence & Next-Best-Action Platform

A local, decision-support portfolio project using the customer and transaction
CSVs in `data/raw/`. It provides data cleaning, customer-level transaction
features, descriptive segmentation, an exploratory engagement-review
indicator, deterministic human-reviewed actions, a FastAPI service, a
Streamlit dashboard, and an optional aggregate-only LLM assistant.

## Current project status

| Area | Status |
| --- | --- |
| Raw data audit | Implemented; source files were inspected and a reproducible audit is available |
| Cleaning and customer features | Implemented; rerunnable from raw CSVs |
| Segmentation | Rebuilt; K=2 selected by silhouette (0.1622), with 2 descriptive groups |
| Churn | No valid target; an exploratory relative engagement indicator replaces the unsupported classifier |
| Product propensity | No valid uptake labels; current product ownership is descriptive only |
| Next-best-action | Deterministic review rules; no product offer or automated contact |
| Dashboard | Executive, customer, engagement, product/segment, and NBA pages |
| API | Customer, analytics, risk-indicator, product-ownership, NBA, and assistant endpoints |
| PostgreSQL | Schema and loading scripts exist; a live database was not available for validation |
| Tests | Focused unit tests are provided; run them in the configured environment |
| Licence/provenance | Not established by files in this workspace; verify before redistributing data |

The workspace audit found 48,723 customer rows and 3,159,157 raw transaction
rows. The cleaning step removed 102 duplicate transactions, leaving 3,159,055.
Customer IDs are unique in the customer file; all transaction customer IDs
matched a customer in the inspected snapshot. Transactions span 2023-01-04
through 2023-12-29. The available files do not identify currency.
These observations describe the local files at audit time and are not a claim
about other COFINFAD releases.

The API and dashboard currently read the validated processed Parquet snapshot.
PostgreSQL schema/loading scripts are present as a separate integration path,
but connecting the application to a live database remains unverified.

## Important interpretation limits

- `source_churn_probability` is an undocumented precomputed field, not a
  verified churn label. The application does not train a churn classifier or
  present an indicator as a calibrated probability.
- Product fields describe current ownership only. There are no campaign
  responses, future uptake labels, needs, eligibility, or consent fields, so
  product-propensity scores and sales leads are not produced.
- Transaction amount is not customer income, spending, profitability, or
  wealth. Currency is not specified by the source file.
- Segments are descriptive clusters, not immutable customer types. The legacy
  K=4 assignment has been replaced by a corrected K=2 fit using one-hot
  categorical encoding. Silhouette is modest and does not establish stability
  or business value; validate the segment profiles before using them.
- The next-best-action engine only suggests a human review of observed
  engagement, recency, or satisfaction signals. It does not approve products,
  contact customers, or establish business impact.
- LLM calls use aggregate tools only. Customer-level requests and messages
  containing likely identifiers/contact details are blocked from the external
  LLM. Direct customer records remain available only through local API/dashboard
  paths.

## Requirements and setup (Windows / conda)

Use the project environment:

```powershell
conda activate agentic_env
python -m pip install -r requirements.txt
```

Keep the raw source files at `data/raw/customer_data.csv` and
`data/raw/transactions_data.csv`, or set `RAW_DATA_DIR` in a local `.env`.
Raw data, processed Parquet files, model binaries, and `.env` are ignored by
Git. Copy `.env.example` to `.env` and set local database/provider variables
only when using those optional integrations. Never commit credentials.
Legacy local churn/product model pickle files may remain from earlier runs;
the current application does not load or use them.

Before sharing this project or dataset, confirm the original dataset source,
licence, and permitted use. Those details are not documented in the supplied
workspace.

## Run

1. Inspect the data and write `reports/data_quality_report.json` and
   `reports/data_dictionary.csv`:

   ```powershell
   python scripts\phase1_inspect.py
   ```

2. Rebuild cleaned data, features, segments, indicators, ownership summaries,
   and next-best-actions:

   ```powershell
   python scripts\run_pipeline.py
   ```

   This pipeline does not contact an LLM or database. It writes derived data
   under `data/processed/` and reports under `reports/`.

3. Start the API:

   ```powershell
   uvicorn api.main:app --reload
   ```

   Open `http://127.0.0.1:8000/docs` for the OpenAPI interface.

4. Start the dashboard in a second terminal:

   ```powershell
   streamlit run dashboard/app.py
   ```

5. Optional PostgreSQL setup (requires a running local PostgreSQL instance
   and correct local `.env` settings):

   ```powershell
   python scripts\create_db.py
   python scripts\load_data.py
   ```

   Database commands have not been verified against a live database in this
   workspace.

6. Run automated tests:

   ```powershell
   python -m pytest -q
   ```

## API endpoints

- `GET /health`
- `GET /api/v1/customers/{customer_id}`
- `GET /api/v1/customers/{customer_id}/segment`
- `GET /api/v1/customers/{customer_id}/risk`
- `GET /api/v1/customers/{customer_id}/ownership` (ownership facts only)
- `GET /api/v1/customers/{customer_id}/nba`
- `GET /api/v1/kpis/executive`
- `GET /api/v1/segments`
- `GET /api/v1/risk/indicators/top?n=20`
- `POST /api/v1/chat`

The old `/api/v1/customers/{customer_id}/propensity` path remains as a hidden
compatibility alias for ownership facts. The `/api/v1/churn/top` route remains as a compatibility alias for the
exploratory indicator; its response explicitly states that the score is not a
churn probability.

## LLM assistant

Set `OPENAI_API_KEY` and `OPENAI_MODEL` in `.env` to enable OpenAI tool calling.
The assistant can retrieve aggregate KPIs, segment summaries, product
ownership rates, and model-status reports through a fixed allowlist of
read-only functions. It cannot execute SQL or access individual customer
records. Without a key, the endpoint reports that the assistant is not
configured; local analytics endpoints continue to work.

## Architecture

```text
data/raw CSVs
  -> scripts/phase1_inspect.py
  -> src/data/cleaning.py
  -> src/features/engineering.py
  -> src/models/{segmentation,churn,propensity}.py
  -> src/recommendations/nba_engine.py
  -> data/processed Parquet + reports/
       -> api/main.py (FastAPI)
       -> dashboard/app.py (Streamlit)
       -> src/llm/assistant.py (aggregate-only tools)
```

## Technical interview summary

> I built a local banking analytics decision-support application that audits
> customer and transaction data, derives customer-level activity features,
> segments customers, and exposes results through a dashboard and API. I
> separated descriptive facts from predictive claims: the supplied churn
> score is undocumented and product fields are current ownership, so I did
> not treat either as a supervised outcome. Instead, the application reports
> a transparent relative engagement indicator and uses conservative,
> human-reviewed action rules.

Likely discussion topics:

1. Why is a precomputed churn score not a valid training target?
2. What event definition and observation window would be needed for churn
   prediction, and how would you use temporal validation?
3. Why does predicting current product ownership not estimate future uptake?
4. What does a silhouette score establish—and what does it fail to establish?
5. How do transaction-level and customer-level features differ?
6. How would you validate the effect of a retention intervention without
   making a causal claim from offline model metrics?
7. Which eligibility, suitability, and consent data would be required before
   making a product recommendation?
8. How does the assistant restrict tools and prevent arbitrary SQL execution?
