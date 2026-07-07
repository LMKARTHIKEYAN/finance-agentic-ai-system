"""
Variance Agent for the Finance Agentic AI System.

This module calculates revenue variance decomposition between
actual operational performance and budget performance.

Current class:
- RevenueVarianceAgent

Future classes:
- GPProductVarianceAgent
- GPPortfolioVarianceAgent
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RevenueVarianceResult:
    actual_orders: int
    budget_orders: int
    actual_revenue: float
    budget_revenue: float
    actual_aov: float
    budget_aov: float
    order_variance: int
    revenue_variance: float
    aov_variance: float
    price_effect: float
    volume_effect: float
    new_discontinued_effect: float
    variance_check: float
    vehicle_variance_summary: list[dict[str, Any]] = field(default_factory=list)


class RevenueVarianceAgent:
    """
    Calculates revenue variance between actual and budget.

    Formula:
        Revenue = Quantity × Price

        Total Variance = Actual Revenue - Budget Revenue

        Price Effect =
        (Actual AOV - Budget AOV) × Actual Orders

        Volume Effect =
        (Actual Orders - Budget Orders) × Budget AOV

        New / Discontinued Effect:
        Used when a vehicle category exists only in actual or only in budget.

        Variance Check =
        Total Variance - Price Effect - Volume Effect - New/Discontinued Effect
    """

    def analyze(self, actual_result: Any, budget_result: Any) -> RevenueVarianceResult:
        """
        Compare actual performance with budget performance.

        Args:
            actual_result: Output from OperationsAnalysisAgent.
            budget_result: Output from BudgetAgent.

        Returns:
            RevenueVarianceResult.
        """

        actual_orders = int(actual_result.completed_orders)
        budget_orders = int(budget_result.total_budget_orders)

        actual_revenue = float(actual_result.total_revenue)
        budget_revenue = float(budget_result.total_budget_revenue)

        actual_aov = self._safe_divide(actual_revenue, actual_orders)
        budget_aov = self._safe_divide(budget_revenue, budget_orders)

        revenue_variance = actual_revenue - budget_revenue
        order_variance = actual_orders - budget_orders
        aov_variance = actual_aov - budget_aov

        price_effect = (actual_aov - budget_aov) * actual_orders
        volume_effect = (actual_orders - budget_orders) * budget_aov

        new_discontinued_effect = 0.0

        variance_check = (
            revenue_variance
            - price_effect
            - volume_effect
            - new_discontinued_effect
        )

        vehicle_variance_summary = self._calculate_vehicle_variance(
            actual_vehicle_summary=actual_result.vehicle_summary,
            budget_vehicle_summary=budget_result.vehicle_summary,
        )

        return RevenueVarianceResult(
            actual_orders=actual_orders,
            budget_orders=budget_orders,
            actual_revenue=round(actual_revenue, 2),
            budget_revenue=round(budget_revenue, 2),
            actual_aov=round(actual_aov, 2),
            budget_aov=round(budget_aov, 2),
            order_variance=order_variance,
            revenue_variance=round(revenue_variance, 2),
            aov_variance=round(aov_variance, 2),
            price_effect=round(price_effect, 2),
            volume_effect=round(volume_effect, 2),
            new_discontinued_effect=round(new_discontinued_effect, 2),
            variance_check=round(variance_check, 2),
            vehicle_variance_summary=vehicle_variance_summary,
        )

    def _calculate_vehicle_variance(
        self,
        actual_vehicle_summary: list[dict[str, Any]],
        budget_vehicle_summary: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Calculate vehicle-category-level revenue variance.

        Match actual and budget using vehicle_category.
        """

        actual_map = {
            row["vehicle_category"]: row for row in actual_vehicle_summary
        }

        budget_map = {
            row["vehicle_category"]: row for row in budget_vehicle_summary
        }

        all_vehicle_categories = sorted(set(actual_map) | set(budget_map))

        variance_rows: list[dict[str, Any]] = []

        for vehicle_category in all_vehicle_categories:
            actual_row = actual_map.get(vehicle_category)
            budget_row = budget_map.get(vehicle_category)

            actual_orders = (
                int(actual_row["completed_orders"]) if actual_row else 0
            )
            budget_orders = (
                int(budget_row["budget_orders"]) if budget_row else 0
            )

            actual_revenue = (
                float(actual_row["total_revenue"]) if actual_row else 0.0
            )
            budget_revenue = (
                float(budget_row["budget_revenue"]) if budget_row else 0.0
            )

            actual_aov = self._safe_divide(actual_revenue, actual_orders)
            budget_aov = self._safe_divide(budget_revenue, budget_orders)

            total_variance = actual_revenue - budget_revenue

            is_existing_vehicle = actual_row is not None and budget_row is not None

            if is_existing_vehicle:
                price_effect = (actual_aov - budget_aov) * actual_orders
                volume_effect = (actual_orders - budget_orders) * budget_aov
                new_discontinued_effect = 0.0
                status = "Existing Vehicle Category"
            else:
                price_effect = 0.0
                volume_effect = 0.0
                new_discontinued_effect = total_variance

                if actual_row and not budget_row:
                    status = "New Vehicle Category"
                elif budget_row and not actual_row:
                    status = "Discontinued Vehicle Category"
                else:
                    status = "Unknown"

            variance_check = (
                total_variance
                - price_effect
                - volume_effect
                - new_discontinued_effect
            )

            variance_rows.append(
                {
                    "vehicle_category": vehicle_category,
                    "status": status,
                    "actual_orders": actual_orders,
                    "budget_orders": budget_orders,
                    "actual_revenue": round(actual_revenue, 2),
                    "budget_revenue": round(budget_revenue, 2),
                    "actual_aov": round(actual_aov, 2),
                    "budget_aov": round(budget_aov, 2),
                    "order_variance": actual_orders - budget_orders,
                    "revenue_variance": round(total_variance, 2),
                    "price_effect": round(price_effect, 2),
                    "volume_effect": round(volume_effect, 2),
                    "new_discontinued_effect": round(
                        new_discontinued_effect,
                        2,
                    ),
                    "variance_check": round(variance_check, 2),
                    "commentary": self._generate_commentary(
                        price_effect=price_effect,
                        volume_effect=volume_effect,
                        new_discontinued_effect=new_discontinued_effect,
                        status=status,
                    ),
                }
            )

        return variance_rows

    def _generate_commentary(
        self,
        price_effect: float,
        volume_effect: float,
        new_discontinued_effect: float,
        status: str,
    ) -> str:
        """
        Generate simple business commentary.
        """

        if status == "New Vehicle Category":
            return "New vehicle category created additional revenue impact."

        if status == "Discontinued Vehicle Category":
            return "Discontinued vehicle category reduced revenue."

        comments: list[str] = []

        if price_effect > 0:
            comments.append("Price/AOV increased and improved revenue.")
        elif price_effect < 0:
            comments.append("Price/AOV decreased and reduced revenue.")
        else:
            comments.append("Price/AOV had no major impact.")

        if volume_effect > 0:
            comments.append("Completed orders increased and improved revenue.")
        elif volume_effect < 0:
            comments.append("Completed orders decreased and reduced revenue.")
        else:
            comments.append("Volume had no major impact.")

        if new_discontinued_effect != 0:
            comments.append("New/discontinued category impacted revenue.")

        return " ".join(comments)

    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """
        Safely divide numbers.
        """

        if denominator == 0:
            return 0.0

        return float(numerator / denominator)
    

@dataclass
class GPProductVarianceResult:
    product_analysis: list[dict[str, Any]] = field(default_factory=list)


class GPProductVarianceAgent:
    """
    Calculates product-level GP% variance.

    GP% = Profit / Revenue
    GP% = (Revenue - COGS) / Revenue

    Product Price Effect:
        ((Actual Price - Base Price) / Actual Price) - Base GP%

    Product Cost Effect:
        Actual GP% - Base GP% - Price Effect
    """

    def analyze(self, data: list[dict[str, Any]]) -> GPProductVarianceResult:
        """
        Analyze product-level GP% variance.

        Args:
            data: List of product dictionaries containing:
                category
                product
                base_volume
                actual_volume
                base_price
                actual_price
                base_cogs_per_unit
                actual_cogs_per_unit

        Returns:
            GPProductVarianceResult.
        """

        result_rows: list[dict[str, Any]] = []

        for row in data:
            base_revenue = row["base_volume"] * row["base_price"]
            actual_revenue = row["actual_volume"] * row["actual_price"]

            base_profit = row["base_volume"] * (
                row["base_price"] - row["base_cogs_per_unit"]
            )

            actual_profit = row["actual_volume"] * (
                row["actual_price"] - row["actual_cogs_per_unit"]
            )

            base_gp_percentage = self._safe_divide(base_profit, base_revenue)
            actual_gp_percentage = self._safe_divide(actual_profit, actual_revenue)

            price_effect = self._calculate_price_effect(
                base_gp_percentage=base_gp_percentage,
                base_price=row["base_price"],
                actual_price=row["actual_price"],
            )

            cost_effect = (
                actual_gp_percentage
                - base_gp_percentage
                - price_effect
            )

            total_gp_change = actual_gp_percentage - base_gp_percentage

            check = total_gp_change - price_effect - cost_effect

            result_rows.append(
                {
                    "category": row["category"],
                    "product": row["product"],
                    "base_volume": row["base_volume"],
                    "actual_volume": row["actual_volume"],
                    "base_price": row["base_price"],
                    "actual_price": row["actual_price"],
                    "base_cogs_per_unit": row["base_cogs_per_unit"],
                    "actual_cogs_per_unit": row["actual_cogs_per_unit"],
                    "base_revenue": round(base_revenue, 2),
                    "actual_revenue": round(actual_revenue, 2),
                    "base_profit": round(base_profit, 2),
                    "actual_profit": round(actual_profit, 2),
                    "base_gp_percentage": round(base_gp_percentage, 6),
                    "actual_gp_percentage": round(actual_gp_percentage, 6),
                    "total_gp_change": round(total_gp_change, 6),
                    "price_effect": round(price_effect, 6),
                    "cost_effect": round(cost_effect, 6),
                    "check": round(check, 6),
                    "commentary": self._generate_commentary(
                        price_effect=price_effect,
                        cost_effect=cost_effect,
                    ),
                }
            )

        return GPProductVarianceResult(product_analysis=result_rows)

    def _calculate_price_effect(
        self,
        base_gp_percentage: float,
        base_price: float,
        actual_price: float,
    ) -> float:
        """
        Product-level price effect formula:

        ((Actual Price - Base Price) / Actual Price) - Base GP%
        """

        if actual_price == 0:
            return 0.0

        return ((actual_price - base_price) / actual_price) - base_gp_percentage

    def _generate_commentary(
        self,
        price_effect: float,
        cost_effect: float,
    ) -> str:
        """
        Generate simple product-level GP commentary.
        """

        comments: list[str] = []

        if price_effect > 0:
            comments.append("Price movement improved GP%.")
        elif price_effect < 0:
            comments.append("Price movement reduced GP%.")
        else:
            comments.append("Price had no GP% impact.")

        if cost_effect > 0:
            comments.append("Cost movement improved GP%.")
        elif cost_effect < 0:
            comments.append("Cost movement reduced GP%.")
        else:
            comments.append("Cost had no GP% impact.")

        return " ".join(comments)

    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """
        Safely divide numbers.
        """

        if denominator == 0:
            return 0.0

        return float(numerator / denominator)
    
@dataclass
class GPPortfolioBridgeResult:
    base_gp_percentage: float
    mix_gp_percentage: float
    price_gp_percentage: float
    actual_gp_percentage: float
    mix_effect: float
    price_effect: float
    cost_effect: float
    total_gp_change: float
    check: float
    bridge_summary: list[dict[str, Any]] = field(default_factory=list)


class GPPortfolioVarianceAgent:
    """
    Calculates portfolio-level GP% variance bridge.

    This agent treats all products together as one portfolio.

    Formula flow:
        Base GP%
            ↓
        Mix Effect
            ↓
        Price Effect
            ↓
        Cost Effect
            ↓
        Actual GP%

    Definitions:
        V0 = Base Volume
        P0 = Base Price
        C0 = Base COGS per unit

        V1 = Actual Volume
        P1 = Actual Price
        C1 = Actual COGS per unit
    """

    def analyze(self, data: list[dict[str, Any]]) -> GPPortfolioBridgeResult:
        """
        Calculate portfolio-level GP% bridge.

        Args:
            data: List of product dictionaries containing:
                category
                product
                base_volume
                actual_volume
                base_price
                actual_price
                base_cogs_per_unit
                actual_cogs_per_unit

        Returns:
            GPPortfolioBridgeResult.
        """

        if not data:
            raise ValueError("Input data cannot be empty.")

        base_revenue = 0.0
        base_gp = 0.0

        mix_revenue = 0.0
        mix_gp = 0.0

        price_revenue = 0.0
        price_gp = 0.0

        actual_revenue = 0.0
        actual_gp = 0.0

        for row in data:
            base_volume = float(row["base_volume"])
            actual_volume = float(row["actual_volume"])
            base_price = float(row["base_price"])
            actual_price = float(row["actual_price"])
            base_cost = float(row["base_cogs_per_unit"])
            actual_cost = float(row["actual_cogs_per_unit"])

            # Base Revenue = SUM(V0 × P0)
            base_revenue += base_volume * base_price

            # Base GP$ = SUM(V0 × (P0 - C0))
            base_gp += base_volume * (base_price - base_cost)

            # Mix Revenue = SUM(V1 × P0)
            # Uses actual volume but base price.
            # This isolates the impact of volume mix change.
            mix_revenue += actual_volume * base_price

            # Mix GP$ = SUM(V1 × (P0 - C0))
            mix_gp += actual_volume * (base_price - base_cost)

            # Price Revenue = SUM(V1 × P1)
            # Uses actual volume and actual price.
            price_revenue += actual_volume * actual_price

            # Price GP$ = SUM(V1 × (P1 - C0))
            # Uses actual price but base cost.
            # This isolates price impact before cost movement.
            price_gp += actual_volume * (actual_price - base_cost)

            # Actual Revenue = SUM(V1 × P1)
            actual_revenue += actual_volume * actual_price

            # Actual GP$ = SUM(V1 × (P1 - C1))
            actual_gp += actual_volume * (actual_price - actual_cost)

        base_gp_percentage = self._safe_divide(base_gp, base_revenue)
        mix_gp_percentage = self._safe_divide(mix_gp, mix_revenue)
        price_gp_percentage = self._safe_divide(price_gp, price_revenue)
        actual_gp_percentage = self._safe_divide(actual_gp, actual_revenue)

        # Mix Effect = Mix GP% - Base GP%
        mix_effect = mix_gp_percentage - base_gp_percentage

        # Price Effect = Price GP% - Mix GP%
        price_effect = price_gp_percentage - mix_gp_percentage

        # Cost Effect = Actual GP% - Price GP%
        cost_effect = actual_gp_percentage - price_gp_percentage

        # Total GP% Change = Actual GP% - Base GP%
        total_gp_change = actual_gp_percentage - base_gp_percentage

        # Check should be zero or very close to zero.
        check = total_gp_change - mix_effect - price_effect - cost_effect

        bridge_summary = [
            {
                "metric": "Base GP%",
                "value": round(base_gp_percentage, 6),
                "value_percentage": round(base_gp_percentage * 100, 2),
                "explanation": "Starting GP% using base volume, base price, and base cost.",
            },
            {
                "metric": "Mix Effect",
                "value": round(mix_effect, 6),
                "value_percentage": round(mix_effect * 100, 2),
                "explanation": "Impact of actual volume mix using base price and base cost.",
            },
            {
                "metric": "Price Effect",
                "value": round(price_effect, 6),
                "value_percentage": round(price_effect * 100, 2),
                "explanation": "Impact of actual price while keeping base cost.",
            },
            {
                "metric": "Cost Effect",
                "value": round(cost_effect, 6),
                "value_percentage": round(cost_effect * 100, 2),
                "explanation": "Impact of actual cost movement.",
            },
            {
                "metric": "Actual GP%",
                "value": round(actual_gp_percentage, 6),
                "value_percentage": round(actual_gp_percentage * 100, 2),
                "explanation": "Ending GP% using actual volume, actual price, and actual cost.",
            },
            {
                "metric": "Total GP% Change",
                "value": round(total_gp_change, 6),
                "value_percentage": round(total_gp_change * 100, 2),
                "explanation": "Actual GP% minus Base GP%.",
            },
            {
                "metric": "Check",
                "value": round(check, 6),
                "value_percentage": round(check * 100, 2),
                "explanation": "Should be zero. Confirms bridge is mathematically correct.",
            },
        ]

        return GPPortfolioBridgeResult(
            base_gp_percentage=round(base_gp_percentage, 6),
            mix_gp_percentage=round(mix_gp_percentage, 6),
            price_gp_percentage=round(price_gp_percentage, 6),
            actual_gp_percentage=round(actual_gp_percentage, 6),
            mix_effect=round(mix_effect, 6),
            price_effect=round(price_effect, 6),
            cost_effect=round(cost_effect, 6),
            total_gp_change=round(total_gp_change, 6),
            check=round(check, 6),
            bridge_summary=bridge_summary,
        )

    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """
        Safely divide numbers.
        """

        if denominator == 0:
            return 0.0

        return float(numerator / denominator)