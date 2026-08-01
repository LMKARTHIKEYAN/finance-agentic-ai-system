# Snowflake practice setup

Snowflake is a cloud service; no local Snowflake server is required.

1. Create a Snowflake trial account and open a Snowsight SQL worksheet.
2. Run `01_setup.sql`.
3. In Snowsight, upload the five files under `data/` to the named internal stage
   `FINANCE_AI.RAW.FINANCE_CSV_STAGE`, keeping the original filenames.
4. Run `02_load_and_validate.sql`.
5. Confirm the expected row counts before starting the Python connection module.

Expected counts:

| Table | Rows |
|---|---:|
| `ORDERS` | 500,000 |
| `CORPORATE_EXPENSES` | 12 |
| `BUDGET` | 132 |
| `BUDGET_CORPORATE_EXPENSES` | 12 |
| `BUSINESS_ASSUMPTIONS` | 10 |

Use an X-Small warehouse with 60-second auto-suspend to limit trial-credit use.
Never store Snowflake passwords, private keys, or tokens in this repository.
