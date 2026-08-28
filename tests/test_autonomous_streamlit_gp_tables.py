from src.ui.autonomous_app import _build_gp_product_table


def test_gp_product_table_contains_vehicle_rows_and_reconciled_total() -> None:
    products = [
        {
            "vehicle_category": "2W",
            "actual_volume": 10,
            "actual_price_per_unit": 30,
            "actual_cost_per_unit": 5,
            "actual_revenue": 300,
            "actual_direct_cost": 50,
            "actual_gross_profit": 250,
            "actual_gp_percentage": 83.3333,
            "actual_mix_percentage": 100,
            "budget_volume": 8,
            "budget_price_per_unit": 25,
            "budget_cost_per_unit": 10,
            "budget_revenue": 200,
            "budget_direct_cost": 80,
            "budget_gross_profit": 120,
            "budget_gp_percentage": 60,
            "budget_mix_percentage": 100,
            "price_effect_percentage_points": 6.6667,
            "cost_effect_percentage_points": 16.6666,
            "check_percentage_points": 0,
            "mix_indicator": 1,
        }
    ]
    portfolio = {
        "actual_revenue": 300,
        "actual_gross_profit": 250,
        "actual_gp_percentage": 83.3333,
        "base_revenue": 200,
        "base_gross_profit": 120,
        "budget_gp_percentage": 60,
        "price_effect_percentage_points": 6.6667,
        "cost_effect_percentage_points": 16.6666,
        "reconciliation_difference": 0,
        "reconciliation_status": "PASS",
    }

    table = _build_gp_product_table(products, portfolio)

    assert table.iloc[0]["Vehicle Category"] == "2W"
    assert table.iloc[0]["Mix Indicator"] == "Favorable"
    assert table.iloc[-1]["Vehicle Category"] == "TOTAL"
    assert table.iloc[-1]["Actual Price/Order"] == 30
    assert table.iloc[-1]["Budget Direct Cost"] == 80
    assert table.iloc[-1]["Mix Indicator"] == "PASS"
