"""Streamlit chat for the Autonomous Finance Agentic AI System."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import altair as alt

from src.ui.api_client import (
    FinanceApiClientError,
    ask_autonomous_question,
    download_monthly_report,
)


EXAMPLES = (
    "Show P&L for August 2026",
    "Show revenue variance for August 2026",
    "Show GP% decomposition for August 2026",
    "Which day had the lowest orders in July 2026?",
    "Show P&L, revenue variance, GP decomposition and forecast for August 2026",
    "Show P&L 2026",
)


def main() -> None:
    st.set_page_config(page_title="Autonomous Finance AI", page_icon="📊", layout="wide")
    st.title("Autonomous Finance Agentic AI")
    st.caption(
        "CFO goal → autonomous tool selection → Snowflake → finance calculations "
        "→ validation → cited answer"
    )

    if "autonomous_messages" not in st.session_state:
        st.session_state.autonomous_messages = []

    with st.sidebar:
        st.header("Test questions")
        st.info("Revenue definition: commission_amount only")
        for index, example in enumerate(EXAMPLES):
            if st.button(example, key=f"autonomous_example_{index}", use_container_width=True):
                _submit(example)
        if st.button("Clear chat", use_container_width=True):
            st.session_state.autonomous_messages = []
            st.rerun()

    for message_index, message in enumerate(st.session_state.autonomous_messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("evidence"):
                _render_pnl_table(message["evidence"])
                _render_kpi_table(message["evidence"])
                _render_daily_kpi_extreme(message["evidence"])
                _render_revenue_variance_table(message["evidence"])
                _render_gp_decomposition_tables(message["evidence"])
                _render_forecast_table(message["evidence"])
                _render_advanced_analysis(message["evidence"])
                with st.expander(f"Verified evidence ({len(message['evidence'])})"):
                    for evidence in message["evidence"]:
                        source = evidence.get("source_tool", "unknown tool")
                        period = evidence.get("period") or "period not supplied"
                        category = evidence.get("category") or "all vehicle categories"
                        status = (
                            "verified and reconciled"
                            if evidence.get("verified") and evidence.get("reconciled")
                            else "review required"
                        )
                        st.markdown(
                            f"**[{evidence.get('evidence_id')}] {source}**  \n"
                            f"Period: {period}  \n"
                            f"Scope: {category}  \n"
                            f"Status: {status}"
                        )
                with st.expander("Technical evidence payload"):
                    st.json(message["evidence"])
            if message.get("metrics"):
                st.caption(f"Execution metrics: {message['metrics']}")
            if message.get("langgraph_shadow"):
                _render_langgraph_shadow(message["langgraph_shadow"])
            if message.get("management_commentary"):
                _render_management_commentary(message["management_commentary"])
            if message.get("evidence"):
                _render_pdf_download(message["evidence"], widget_suffix=f"message_{message_index}")

    question = st.chat_input("Ask a finance question, for example: Show P&L for August 2026")
    if question:
        _submit(question)


def _submit(question: str) -> None:
    st.session_state.autonomous_messages.append({"role": "user", "content": question})
    with st.spinner("The autonomous agent is deciding and checking finance evidence..."):
        try:
            response = ask_autonomous_question(question)
            content = (
                response.get("answer")
                or response.get("question")
                or "The autonomous workflow stopped without an answer."
            )
            st.session_state.autonomous_messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "evidence": response.get("evidence") or [],
                    "metrics": response.get("metrics") or {},
                    "management_commentary": response.get("management_commentary") or {},
                    "langgraph_shadow": response.get("langgraph_shadow") or {},
                }
            )
        except FinanceApiClientError as exc:
            st.session_state.autonomous_messages.append(
                {"role": "assistant", "content": f"API error: {exc}"}
            )
    st.rerun()


def _render_langgraph_shadow(shadow: dict) -> None:
    if not shadow.get("executed"):
        return
    succeeded = bool(shadow.get("succeeded"))
    summary = shadow.get("summary") or {}
    paused = bool(summary.get("pending_question") or summary.get("approval_request"))
    with st.expander(
        "LangGraph shadow execution — " + (
            "successful" if succeeded else "paused" if paused else "incomplete"
        ),
        expanded=False,
    ):
        if not succeeded:
            st.warning(
                summary.get("pending_question")
                or summary.get("approval_request")
                or shadow.get("error")
                or "Shadow execution ended without a validated final answer."
            )
            return
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Runtime", "LangGraph")
        col2.metric("Graph steps", summary.get("step_count", 0))
        col3.metric("Evidence", summary.get("evidence_count", 0))
        col4.metric("Validation", "Passed" if summary.get("validation_passed") else "Review")
        st.caption(
            "Reviewer: " + str(summary.get("reviewer_decision") or "not completed")
        )
        tools = summary.get("selected_tools") or []
        if tools:
            st.markdown("**LLM-selected tools:** " + " → ".join(tools))
        trace = summary.get("execution_trace") or []
        if trace:
            st.markdown("**Execution trace**")
            st.dataframe(pd.DataFrame(trace), use_container_width=True, hide_index=True)
        if summary.get("final_answer"):
            st.markdown("**Shadow answer**")
            st.info(summary["final_answer"])


def _render_pnl_table(evidence_records: list[dict]) -> None:
    pnl = next(
        (item for item in evidence_records if item.get("result_type") == "pnl"),
        None,
    )
    if not pnl:
        return
    payload = pnl.get("compact_payload") or {}
    summary = payload.get("pnl_summary") or {}
    actual = summary.get("actual") or {}
    budget = summary.get("budget") or {}
    if not actual or not budget:
        return

    definitions = (
        ("Revenue (commission_amount)", "revenue", 1, True),
        ("Total Direct Cost", "direct_cost", -1, True),
        ("Gross Profit", "gross_profit", 1, True),
        ("Gross Margin %", "gross_margin_percentage", 1, False),
        ("Sales & Marketing", "sales_marketing", -1, True),
        ("Other OPEX", "other_opex", -1, True),
        ("EBITDA", "ebitda", 1, True),
        ("Depreciation", "depreciation", -1, True),
        ("EBIT", "ebit", 1, True),
        ("Interest", "interest", -1, True),
        ("EBT", "ebt", 1, True),
        ("Income Tax", "income_tax", -1, True),
        ("Net Profit", "net_profit", 1, True),
    )
    rows = []
    for account, key, sign, currency in definitions:
        actual_value = actual.get(key)
        budget_value = budget.get(key)
        signed_actual = _signed(actual_value, sign)
        signed_budget = _signed(budget_value, sign)
        variance = (
            signed_actual - signed_budget
            if signed_actual is not None and signed_budget is not None
            else None
        )
        variance_pct = (
            variance / abs(signed_budget) * 100
            if variance is not None and signed_budget not in (None, 0)
            else None
        )
        rows.append(
            {
                "Account": account,
                "Actual": _display(signed_actual, currency),
                "Budget": _display(signed_budget, currency),
                "Variance": _display(variance, currency),
                "Variance %": _display(variance_pct, False),
            }
        )
    st.markdown("#### Actual vs Budget P&L")
    st.caption(
        "Total Direct Cost is calculated internally as Incentive + Goodwill + "
        "Dry-run + Surge. Budget uses total BUDGET_COGS."
    )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    _render_pnl_visual(actual, budget)


def _render_management_commentary(commentary: dict) -> None:
    st.markdown("#### CFO Management Commentary")
    st.info(commentary.get("executive_summary") or "Validated finance analysis completed.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Key risks**")
        for item in commentary.get("risks") or []:
            st.markdown(f"- {item}")
    with right:
        st.markdown("**Recommended actions**")
        for item in commentary.get("recommendations") or []:
            st.markdown(f"- {item}")


def _render_pdf_download(evidence_records: list[dict], *, widget_suffix: str) -> None:
    period = next((item.get("period") for item in evidence_records if item.get("period")), None)
    if not period:
        return
    try:
        year, month = (int(value) for value in period[:7].split("-"))
    except (TypeError, ValueError):
        return
    key = f"monthly_pdf_{year}_{month}"
    if st.button("Generate complete monthly PDF", key=f"generate_{key}_{widget_suffix}"):
        with st.spinner("Generating validated monthly management report..."):
            try:
                st.session_state[key] = download_monthly_report(year, month)
            except FinanceApiClientError as exc:
                st.error(f"PDF generation failed: {exc}")
    if key in st.session_state:
        st.download_button(
            "Download monthly finance PDF",
            data=st.session_state[key],
            file_name=f"monthly_finance_report_{year:04d}-{month:02d}.pdf",
            mime="application/pdf",
            key=f"download_{key}_{widget_suffix}",
        )
def _render_daily_kpi_extreme(evidence_records: list[dict]) -> None:
    evidence = next(
        (item for item in evidence_records if item.get("result_type") == "daily_kpi_extreme"),
        None,
    )
    if not evidence:
        return
    payload = evidence.get("compact_payload") or {}
    rows = payload.get("daily_kpis") or []
    if not rows:
        return
    st.subheader("Daily Order KPI Analysis")
    left, middle, right = st.columns(3)
    if payload.get("analysis_mode") == "high_revenue_low_orders":
        high_revenue = payload.get("highest_revenue_day") or {}
        low_orders = payload.get("lowest_order_day") or {}
        high_aov = payload.get("highest_aov_day") or {}
        left.metric(
            "Highest-revenue date",
            high_revenue.get("date", "N/A"),
            f"INR {high_revenue.get('revenue', 0):,.2f}",
        )
        middle.metric(
            "Lowest-order date",
            low_orders.get("date", "N/A"),
            f"{low_orders.get('total_orders', 0):,.0f} orders",
        )
        right.metric(
            "Highest-AOV date",
            high_aov.get("date", "N/A"),
            f"INR {high_aov.get('aov', 0):,.2f} per order",
        )
    else:
        left.metric("Lowest-order date", payload.get("selected_day", "N/A"))
        middle.metric("Orders on that date", f"{payload.get('selected_total_orders', 0):,.0f}")
        right.metric(
            "Versus daily average",
            f"{payload.get('difference_from_daily_average_percentage', 0):+.2f}%",
        )
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    chart = (
        alt.Chart(frame)
        .mark_line(point=True, color="#1F77B4")
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("total_orders:Q", title="Total orders", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("total_orders:Q", title="Orders"),
                alt.Tooltip("completed_orders:Q", title="Completed"),
                alt.Tooltip("cancelled_orders:Q", title="Cancelled"),
            ],
        )
        .properties(height=300, title="Daily order trend")
    )
    st.altair_chart(chart, use_container_width=True)
    display = frame.rename(columns={
        "date": "Date",
        "total_orders": "Total Orders",
        "completed_orders": "Completed Orders",
        "cancelled_orders": "Cancelled Orders",
        "fulfillment_percentage": "Fulfillment %",
        "cancellation_percentage": "Cancellation %",
        "revenue": "Commission Revenue",
        "aov": "AOV",
    })
    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_advanced_analysis(evidence_records: list[dict]) -> None:
    for evidence in evidence_records:
        result_type = evidence.get("result_type")
        payload = evidence.get("compact_payload") or {}
        if result_type == "trend" and payload.get("trend"):
            st.subheader(f"{str(payload.get('frequency', 'daily')).title()} KPI Trend")
            frame = pd.DataFrame(payload["trend"])
            frame["period"] = pd.to_datetime(frame["period"])
            orders_col, revenue_col = st.columns(2)
            with orders_col:
                st.altair_chart(
                    alt.Chart(frame).mark_line(point=True, color="#1F77B4").encode(
                        x=alt.X("period:T", title="Period"),
                        y=alt.Y("total_orders:Q", title="Total orders", scale=alt.Scale(zero=False)),
                        tooltip=[alt.Tooltip("period:T"), alt.Tooltip("total_orders:Q", format=",")],
                    ).properties(height=300, title="Order trend"),
                    use_container_width=True,
                )
            with revenue_col:
                st.altair_chart(
                    alt.Chart(frame).mark_line(point=True, color="#18A999").encode(
                        x=alt.X("period:T", title="Period"),
                        y=alt.Y("revenue:Q", title="Commission revenue (INR)", scale=alt.Scale(zero=False)),
                        tooltip=[alt.Tooltip("period:T"), alt.Tooltip("revenue:Q", format=",.2f")],
                    ).properties(height=300, title="Revenue trend"),
                    use_container_width=True,
                )
            st.dataframe(frame, use_container_width=True, hide_index=True)
        elif result_type == "period_comparison" and payload.get("metrics"):
            st.subheader("Period Comparison")
            st.caption(f"{payload.get('current_period')} versus {payload.get('comparison_period')}")
            st.dataframe(pd.DataFrame(payload["metrics"]), use_container_width=True, hide_index=True)
        elif result_type == "drilldown" and payload.get("rows"):
            st.subheader(f"Drill-down by {payload.get('dimension')}")
            st.dataframe(pd.DataFrame(payload["rows"]), use_container_width=True, hide_index=True)
        elif result_type == "anomaly":
            st.subheader("KPI Anomaly Detection")
            anomalies = payload.get("anomalies") or []
            if anomalies:
                st.dataframe(pd.DataFrame(anomalies), use_container_width=True, hide_index=True)
            else:
                st.success("No KPI observations exceeded the configured robust anomaly threshold.")
        elif result_type == "category_profitability" and payload.get("rows"):
            st.subheader("Category Profitability and Unit Economics")
            frame = pd.DataFrame(payload["rows"])
            st.dataframe(frame, use_container_width=True, hide_index=True)
            if {"vehicle_category", "net_profit"} <= set(frame.columns):
                st.altair_chart(
                    alt.Chart(frame).mark_bar().encode(
                        x=alt.X("vehicle_category:N", sort="-y", title="Vehicle category"),
                        y=alt.Y("net_profit:Q", title="Net profit (INR)"),
                        tooltip=["vehicle_category", "net_profit", "gp_percentage", "profit_per_completed_order"],
                    ).properties(height=320),
                    use_container_width=True,
                )
        elif result_type == "customer_route":
            st.subheader("Customer and Route Analysis")
            if payload.get("analysis_scope") in {"customer", "both"} and not payload.get("customer_analysis_available"):
                st.warning(payload.get("customer_data_requirement"))
            routes = payload.get("routes") or []
            if routes:
                summary = payload.get("route_summary") or {}
                top = summary.get("highest_gross_profit_route") or {}
                bottom = summary.get("lowest_gross_profit_route") or {}
                left, middle, right = st.columns(3)
                left.metric("Routes analysed", summary.get("route_count", len(routes)))
                middle.metric("Highest route GP", f"INR {top.get('gross_profit', 0):,.2f}")
                right.metric("Lowest route GP", f"INR {bottom.get('gross_profit', 0):,.2f}")
                st.caption(payload.get("profitability_basis"))
                route_frame = pd.DataFrame(routes)
                route_frame.insert(
                    0,
                    "route",
                    route_frame["pickup_cluster"].astype(str)
                    + " → "
                    + route_frame["drop_cluster"].astype(str),
                )
                top_routes = route_frame.nlargest(10, "gross_profit")
                bottom_routes = route_frame.nsmallest(10, "gross_profit")
                top_col, bottom_col = st.columns(2)
                with top_col:
                    st.markdown("**Top 10 routes by gross profit**")
                    st.dataframe(
                        top_routes[["route", "total_orders", "revenue", "gross_profit", "gp_percentage"]],
                        use_container_width=True,
                        hide_index=True,
                    )
                with bottom_col:
                    st.markdown("**Bottom 10 routes by gross profit**")
                    st.dataframe(
                        bottom_routes[["route", "total_orders", "revenue", "gross_profit", "gp_percentage"]],
                        use_container_width=True,
                        hide_index=True,
                    )
                with st.expander(f"View all {len(route_frame)} directional routes"):
                    st.dataframe(route_frame, use_container_width=True, hide_index=True)
        elif result_type == "forecast_accuracy":
            st.subheader("Forecast Accuracy")
            if not payload.get("available"):
                st.warning(payload.get("reason"))
            else:
                st.json(payload.get("summary") or {})
                st.dataframe(pd.DataFrame(payload.get("rows") or []), use_container_width=True, hide_index=True)
        elif result_type == "scenario" and payload.get("scenario"):
            st.subheader("What-if Scenario")
            st.dataframe(pd.DataFrame([payload["scenario"]]), use_container_width=True, hide_index=True)
            with st.expander("Scenario assumptions"):
                st.json(payload.get("assumptions") or {})
        elif result_type == "driver_forecast" and payload.get("forecast"):
            st.subheader("Driver-based Forecast")
            frame = pd.DataFrame(payload["forecast"])
            st.dataframe(frame, use_container_width=True, hide_index=True)
            st.altair_chart(
                alt.Chart(frame).mark_line(point=True).encode(
                    x=alt.X("period:N", title="Forecast period"),
                    y=alt.Y("revenue:Q", title="Commission revenue (INR)"),
                    tooltip=["period", "completed_orders", "aov", "revenue", "gp_percentage"],
                ).properties(height=300),
                use_container_width=True,
            )
        elif result_type == "profitability_alert":
            st.subheader("Profitability Alerts")
            summary = payload.get("monitoring_summary") or {}
            thresholds = payload.get("thresholds") or {}
            left, middle, right = st.columns(3)
            left.metric("Categories evaluated", summary.get("categories_evaluated", 0))
            middle.metric(
                "Lowest GP%",
                f"{summary.get('lowest_gp_percentage', 0):.2f}%",
                f"Threshold ≥ {thresholds.get('minimum_gp_percentage', 0):.2f}%",
            )
            right.metric(
                "Highest cancellation",
                f"{summary.get('highest_cancellation_percentage', 0):.2f}%",
                f"Threshold ≤ {thresholds.get('maximum_cancellation_percentage', 0):.2f}%",
            )
            alerts = payload.get("alerts") or []
            if alerts:
                st.dataframe(pd.DataFrame(alerts), use_container_width=True, hide_index=True)
            else:
                st.success("No configured profitability thresholds were breached.")
        elif result_type == "operational_drivers" and payload.get("drivers"):
            st.subheader("Operational Driver Evidence")
            st.caption(payload.get("cause_status"))
            st.json(payload.get("drivers"))


def _render_kpi_table(evidence_records: list[dict]) -> None:
    evidence = next(
        (item for item in evidence_records if item.get("result_type") == "kpi"),
        None,
    )
    if not evidence:
        return
    payload = evidence.get("compact_payload") or {}
    category_kpis = payload.get("category_kpis") or []
    if category_kpis:
        _render_category_kpi_table(category_kpis, payload.get("selected_kpis") or [])
        return
    selected = payload.get("selected_kpis") or []
    if not selected:
        return
    rows = []
    for item in selected:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "KPI": item.get("display_name") or item.get("kpi"),
                "Value": _format_kpi_display(item.get("value"), item.get("unit")),
                "Unit": str(item.get("unit") or "").replace("_", " ").title(),
                "Source": item.get("source"),
            }
        )
    st.markdown("#### Actual, Budget and Variance KPI Summary")
    st.caption("Revenue and AOV use commission_amount only; fare is excluded.")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_category_kpi_table(category_kpis: list[dict], selected_kpis: list[dict]) -> None:
    rows = [
        {
            "Vehicle Category": item.get("vehicle_category"),
            "Total Orders": item.get("total_orders", 0),
            "Completed Orders": item.get("completed_orders", 0),
            "Cancelled Orders": item.get("cancelled_orders", 0),
            "Fulfillment %": item.get("fulfillment_percentage", 0),
            "Cancellation %": item.get("cancellation_percentage", 0),
            "Revenue": item.get("total_revenue", 0),
            "AOV": item.get("average_order_value", 0),
        }
        for item in category_kpis
    ]
    overall = {
        item.get("kpi"): item.get("value")
        for item in selected_kpis
        if isinstance(item, dict)
    }
    rows.append(
        {
            "Vehicle Category": "TOTAL",
            "Total Orders": overall.get("total_orders", 0),
            "Completed Orders": overall.get("completed_orders", 0),
            "Cancelled Orders": overall.get("cancelled_orders", 0),
            "Fulfillment %": overall.get("fulfillment_percentage", 0),
            "Cancellation %": overall.get("cancellation_percentage", 0),
            "Revenue": overall.get("actual_revenue", 0),
            "AOV": overall.get("actual_aov", 0),
        }
    )
    dataframe = pd.DataFrame(rows)
    count_columns = [
        "Total Orders", "Completed Orders", "Cancelled Orders",
    ]
    percentage_columns = ["Fulfillment %", "Cancellation %"]
    currency_columns = ["Revenue", "AOV"]
    styler = (
        dataframe.style
        .format({column: "{:,.0f}" for column in count_columns}, na_rep="-")
        .format({column: "{:,.2f}%" for column in percentage_columns}, na_rep="-")
        .format({column: "{:,.2f}" for column in currency_columns}, na_rep="-")
        .apply(
            lambda row: [
                "font-weight: bold; border-top: 2px solid #555;"
                if row["Vehicle Category"] == "TOTAL" else ""
                for _ in row
            ],
            axis=1,
        )
    )
    st.markdown("#### Category-wise KPI Summary")
    st.caption(
        "Actual operational KPIs only. Each row is a vehicle category; revenue and "
        "AOV use commission_amount only. Budget and variance are excluded."
    )
    st.dataframe(styler, use_container_width=True, hide_index=True)
    _render_kpi_visual(dataframe.iloc[:-1].copy())


def _format_kpi_display(value, unit) -> str:
    if value is None:
        return "N/A"
    if unit == "count":
        return f"{float(value):,.0f}"
    if unit in {"percentage", "decimal_percentage", "percentage_points"}:
        return f"{float(value):,.2f}%"
    if unit in {"currency", "currency_per_order"}:
        return f"INR {float(value):,.2f}"
    return str(value)


def _render_revenue_variance_table(evidence_records: list[dict]) -> None:
    evidence = next(
        (
            item
            for item in evidence_records
            if item.get("result_type") == "revenue_variance"
        ),
        None,
    )
    if not evidence:
        return
    payload = evidence.get("compact_payload") or {}
    source_rows = payload.get("vehicle_variance_summary") or []
    if not source_rows:
        return

    rows = []
    for item in sorted(
        source_rows,
        key=lambda row: abs(float(row.get("revenue_variance", 0))),
        reverse=True,
    ):
        rows.append(
            {
                "Vehicle Category": item.get("vehicle_category"),
                "Budget Orders": item.get("budget_orders", 0),
                "Actual Orders": item.get("actual_orders", 0),
                "Budget Price/Order": item.get("budget_aov", 0),
                "Actual Price/Order": item.get("actual_aov", 0),
                "Budget Revenue": item.get("budget_revenue", 0),
                "Actual Revenue": item.get("actual_revenue", 0),
                "Price/AOV Effect": item.get("price_effect", 0),
                "Volume Effect": item.get("volume_effect", 0),
                "New/Discontinued Effect": item.get("new_discontinued_effect", 0),
                "Revenue Variance": item.get("revenue_variance", 0),
                "Variance Check": item.get("variance_check", 0),
            }
        )
    rows.append(
        {
            "Vehicle Category": "TOTAL",
            "Budget Orders": payload.get("budget_orders", 0),
            "Actual Orders": payload.get("actual_orders", 0),
            "Budget Price/Order": payload.get("budget_aov", 0),
            "Actual Price/Order": payload.get("actual_aov", 0),
            "Budget Revenue": payload.get("budget_revenue", 0),
            "Actual Revenue": payload.get("actual_revenue", 0),
            "Price/AOV Effect": payload.get("price_effect", 0),
            "Volume Effect": payload.get("volume_effect", 0),
            "New/Discontinued Effect": payload.get("new_discontinued_effect", 0),
            "Revenue Variance": payload.get("revenue_variance", 0),
            "Variance Check": payload.get("variance_check", 0),
        }
    )
    dataframe = pd.DataFrame(rows)
    integer_columns = ["Budget Orders", "Actual Orders"]
    currency_columns = [
        "Budget Price/Order",
        "Actual Price/Order",
        "Budget Revenue",
        "Actual Revenue",
        "Price/AOV Effect",
        "Volume Effect",
        "New/Discontinued Effect",
        "Revenue Variance",
        "Variance Check",
    ]
    styler = (
        dataframe.style
        .format({column: "{:,.0f}" for column in integer_columns}, na_rep="-")
        .format({column: "{:,.2f}" for column in currency_columns}, na_rep="-")
        .apply(
            lambda row: [
                "font-weight: bold; border-top: 2px solid #555;"
                if row["Vehicle Category"] == "TOTAL"
                else ""
                for _ in row
            ],
            axis=1,
        )
        .map(
            lambda value: "color: #d62728;" if isinstance(value, (int, float)) and value < 0 else "",
            subset=["Price/AOV Effect", "Volume Effect", "New/Discontinued Effect"],
        )
    )
    st.markdown("#### Sales Variance Decomposition by Vehicle Category")
    st.caption(
        "Budget Price/Order = Budget Revenue ÷ Budget Orders; Actual Price/Order "
        "= Actual Revenue ÷ Actual Orders. Price effect = (Actual Price/Order − "
        "Budget Price/Order) × Actual Orders; Volume effect = (Actual Orders − "
        "Budget Orders) × Budget Price/Order. "
        "Variance Check must equal zero."
    )
    st.dataframe(styler, use_container_width=True, hide_index=True)
    _render_revenue_variance_visual(dataframe.iloc[:-1].copy())


def _render_gp_decomposition_tables(evidence_records: list[dict]) -> None:
    evidence = next(
        (
            item
            for item in evidence_records
            if item.get("result_type") == "gp_decomposition"
        ),
        None,
    )
    if not evidence:
        return
    payload = evidence.get("compact_payload") or {}
    product_rows = payload.get("product_level") or payload.get("category_analysis") or []
    portfolio = payload.get("portfolio_level") or payload
    if not product_rows or not portfolio:
        return

    product_table = _build_gp_product_table(product_rows, portfolio)
    currency_columns = [
        "Actual Price/Order", "Actual Direct Cost/Order", "Actual Revenue",
        "Actual Direct Cost", "Actual Gross Profit", "Budget Price/Order",
        "Budget Direct Cost/Order", "Budget Revenue", "Budget Direct Cost",
        "Budget Gross Profit",
    ]
    percentage_columns = [
        "Actual GP%", "Budget GP%", "Price Effect (pp)", "Cost Effect (pp)",
        "Check (pp)", "Actual Mix %", "Budget Mix %",
    ]
    styler = (
        product_table.style
        .format({"Actual Orders": "{:,.0f}", "Budget Orders": "{:,.0f}"}, na_rep="-")
        .format({column: "{:,.2f}" for column in currency_columns}, na_rep="-")
        .format({column: "{:+,.2f}" for column in percentage_columns}, na_rep="-")
        .apply(
            lambda row: [
                "font-weight: bold; border-top: 2px solid #555;"
                if row["Vehicle Category"] == "TOTAL" else ""
                for _ in row
            ],
            axis=1,
        )
        .map(
            lambda value: "color: #d62728;" if isinstance(value, (int, float)) and value < 0 else "",
            subset=["Price Effect (pp)", "Cost Effect (pp)", "Check (pp)"],
        )
    )
    st.markdown("#### Product Level Analysis")
    st.caption(
        "Each product is a vehicle category. Revenue is commission_amount; direct "
        "cost is Incentive + Goodwill + Dry-run + Surge. Effects are percentage points."
    )
    st.dataframe(styler, use_container_width=True, hide_index=True)

    st.markdown("#### Portfolio Level Analysis")
    portfolio_table = pd.DataFrame(
        [
            {"Metric": "Budget Revenue", "Formula": "Σ (Budget price × Budget orders)", "Value": _money(portfolio.get("base_revenue"))},
            {"Metric": "Budget Gross Profit", "Formula": "Σ [Budget orders × (Budget price − Budget cost)]", "Value": _money(portfolio.get("base_gross_profit"))},
            {"Metric": "Budget GP%", "Formula": "Budget Gross Profit ÷ Budget Revenue", "Value": _percent(portfolio.get("budget_gp_percentage"))},
            {"Metric": "Mix-only Revenue", "Formula": "Σ (Budget price × Actual orders)", "Value": _money(portfolio.get("mix_only_revenue"))},
            {"Metric": "Mix-only Gross Profit", "Formula": "Σ [Actual orders × (Budget price − Budget cost)]", "Value": _money(portfolio.get("mix_only_gross_profit"))},
            {"Metric": "Mix-only GP%", "Formula": "Mix-only Gross Profit ÷ Mix-only Revenue", "Value": _percent(portfolio.get("mix_only_gp_percentage"))},
            {"Metric": "Price-only Revenue", "Formula": "Σ (Actual price × Actual orders)", "Value": _money(portfolio.get("price_only_revenue"))},
            {"Metric": "Price-only Gross Profit", "Formula": "Σ [Actual orders × (Actual price − Budget cost)]", "Value": _money(portfolio.get("price_only_gross_profit"))},
            {"Metric": "Price-only GP%", "Formula": "Price-only Gross Profit ÷ Price-only Revenue", "Value": _percent(portfolio.get("price_only_gp_percentage"))},
            {"Metric": "Actual Revenue", "Formula": "Σ (Actual price × Actual orders)", "Value": _money(portfolio.get("actual_revenue"))},
            {"Metric": "Actual Gross Profit", "Formula": "Σ [Actual orders × (Actual price − Actual cost)]", "Value": _money(portfolio.get("actual_gross_profit"))},
            {"Metric": "Actual GP%", "Formula": "Actual Gross Profit ÷ Actual Revenue", "Value": _percent(portfolio.get("actual_gp_percentage"))},
        ]
    )
    st.dataframe(portfolio_table, use_container_width=True, hide_index=True)

    effect_table = pd.DataFrame(
        [
            {"Effect Bridge": "Mix Effect", "Percentage Points": portfolio.get("mix_effect_percentage_points"), "Basis Points": portfolio.get("mix_effect_basis_points")},
            {"Effect Bridge": "Price Effect", "Percentage Points": portfolio.get("price_effect_percentage_points"), "Basis Points": portfolio.get("price_effect_basis_points")},
            {"Effect Bridge": "Cost Effect", "Percentage Points": portfolio.get("cost_effect_percentage_points"), "Basis Points": portfolio.get("cost_effect_basis_points")},
            {"Effect Bridge": "Total GP% Change", "Percentage Points": portfolio.get("total_variance_percentage_points"), "Basis Points": portfolio.get("total_variance_basis_points")},
        ]
    )
    st.dataframe(
        effect_table.style.format(
            {"Percentage Points": "{:+,.2f}", "Basis Points": "{:+,.2f}"},
            na_rep="-",
        ),
        use_container_width=True,
        hide_index=True,
    )
    status = portfolio.get("reconciliation_status", "UNKNOWN")
    difference = portfolio.get("reconciliation_difference")
    st.caption(
        f"Bridge reconciliation: {status}; difference: "
        f"{float(difference or 0):.8f} percentage points."
    )
    _render_gp_visual(product_rows, portfolio)


def _render_forecast_table(evidence_records: list[dict]) -> None:
    evidence = next(
        (item for item in evidence_records if item.get("result_type") == "forecast"),
        None,
    )
    if not evidence:
        return
    payload = evidence.get("compact_payload") or {}
    rows = payload.get("forecast_summary") or []
    category_rows = payload.get("category_forecast_summary") or []
    if not rows:
        return
    table = pd.DataFrame(
        [
            {
                "Forecast Period": item.get("forecast_period"),
                "Forecast Orders": item.get("forecast_orders"),
                "Forecast Revenue": item.get("forecast_revenue"),
                "Forecast AOV": item.get("forecast_average_order_value"),
                "Source Periods": ", ".join(item.get("source_periods") or []),
            }
            for item in rows
        ]
    )
    st.markdown("#### Monthly Rolling Forecast")
    st.caption(
        f"Method: {payload.get('method', 'Rolling forecast')} | "
        f"Window: {payload.get('rolling_window', 3)} months. Revenue uses commission_amount only."
    )
    st.dataframe(
        table.style.format(
            {"Forecast Orders": "{:,.0f}", "Forecast Revenue": "{:,.2f}", "Forecast AOV": "{:,.2f}"},
            na_rep="-",
        ),
        use_container_width=True,
        hide_index=True,
    )
    history = pd.DataFrame(payload.get("historical_summary") or [])
    forecast = pd.DataFrame(rows)
    if not history.empty:
        historical_chart = history[["period", "total_revenue"]].rename(
            columns={"period": "Period", "total_revenue": "Revenue"}
        )
        historical_chart["Series"] = "Actual"
        forecast_chart = forecast[["forecast_period", "forecast_revenue"]].rename(
            columns={"forecast_period": "Period", "forecast_revenue": "Revenue"}
        )
        forecast_chart["Series"] = "Forecast"
        trend = pd.concat([historical_chart, forecast_chart], ignore_index=True)
        chart = (
            alt.Chart(trend)
            .mark_line(point=True)
            .encode(
                x=alt.X("Period:N", title="Month"),
                y=alt.Y("Revenue:Q", title="Revenue (INR)"),
                color=alt.Color("Series:N", title=None),
                tooltip=["Period:N", "Series:N", alt.Tooltip("Revenue:Q", format=",.2f")],
            )
            .properties(title="Actual History and Rolling Revenue Forecast", height=340)
        )
        st.altair_chart(chart, use_container_width=True)
    if category_rows:
        category_table = pd.DataFrame(category_rows)
        st.markdown("##### Category-wise Forecast")
        st.dataframe(
            category_table[["vehicle_category", "forecast_period", "forecast_orders", "forecast_revenue", "forecast_average_order_value"]]
            .rename(columns={
                "vehicle_category": "Vehicle Category", "forecast_period": "Forecast Period",
                "forecast_orders": "Forecast Orders", "forecast_revenue": "Forecast Revenue",
                "forecast_average_order_value": "Forecast AOV",
            })
            .style.format({"Forecast Orders": "{:,.0f}", "Forecast Revenue": "{:,.2f}", "Forecast AOV": "{:,.2f}"}),
            use_container_width=True,
            hide_index=True,
        )
        first_period = rows[0].get("forecast_period")
        first = category_table[category_table["forecast_period"] == first_period]
        category_chart = (
            alt.Chart(first)
            .mark_bar()
            .encode(
                x=alt.X("forecast_revenue:Q", title="Forecast Revenue (INR)"),
                y=alt.Y("vehicle_category:N", sort="-x", title="Vehicle Category"),
                tooltip=["vehicle_category:N", alt.Tooltip("forecast_revenue:Q", format=",.2f")],
            )
            .properties(title=f"Category Forecast Revenue — {first_period}", height=340)
        )
        st.altair_chart(category_chart, use_container_width=True)


def _render_kpi_visual(dataframe: pd.DataFrame) -> None:
    st.markdown("##### KPI Visualizations")
    revenue_chart = (
        alt.Chart(dataframe)
        .mark_bar()
        .encode(
            x=alt.X("Revenue:Q", title="Revenue (INR)"),
            y=alt.Y("Vehicle Category:N", sort="-x", title="Vehicle Category"),
            tooltip=[
                alt.Tooltip("Vehicle Category:N"),
                alt.Tooltip("Revenue:Q", format=",.2f"),
                alt.Tooltip("Completed Orders:Q", format=",.0f"),
                alt.Tooltip("AOV:Q", format=",.2f"),
            ],
        )
        .properties(title="Revenue by Vehicle Category", height=340)
    )
    rate_data = dataframe.melt(
        id_vars=["Vehicle Category"],
        value_vars=["Fulfillment %", "Cancellation %"],
        var_name="KPI",
        value_name="Percentage",
    )
    rate_chart = (
        alt.Chart(rate_data)
        .mark_bar()
        .encode(
            x=alt.X("Vehicle Category:N", title="Vehicle Category"),
            y=alt.Y("Percentage:Q", title="Rate (%)"),
            color=alt.Color("KPI:N", title=None),
            xOffset="KPI:N",
            tooltip=["Vehicle Category:N", "KPI:N", alt.Tooltip("Percentage:Q", format=".2f")],
        )
        .properties(title="Fulfillment and Cancellation Rates", height=320)
    )
    st.altair_chart(revenue_chart, use_container_width=True)
    st.altair_chart(rate_chart, use_container_width=True)


def _render_revenue_variance_visual(dataframe: pd.DataFrame) -> None:
    st.markdown("##### Revenue Variance Visualizations")
    variance_chart = (
        alt.Chart(dataframe)
        .mark_bar()
        .encode(
            x=alt.X("Revenue Variance:Q", title="Revenue Variance (INR)"),
            y=alt.Y("Vehicle Category:N", sort="-x", title="Vehicle Category"),
            color=alt.condition(
                alt.datum["Revenue Variance"] >= 0,
                alt.value("#2ca02c"), alt.value("#d62728"),
            ),
            tooltip=[
                "Vehicle Category:N",
                alt.Tooltip("Revenue Variance:Q", format=",.2f"),
                alt.Tooltip("Price/AOV Effect:Q", format=",.2f"),
                alt.Tooltip("Volume Effect:Q", format=",.2f"),
            ],
        )
        .properties(title="Revenue Variance by Vehicle Category", height=350)
    )
    effect_data = dataframe.melt(
        id_vars=["Vehicle Category"],
        value_vars=["Price/AOV Effect", "Volume Effect", "New/Discontinued Effect"],
        var_name="Driver", value_name="Impact",
    )
    effect_chart = (
        alt.Chart(effect_data)
        .mark_bar()
        .encode(
            x=alt.X("Impact:Q", title="Impact (INR)"),
            y=alt.Y("Vehicle Category:N", title="Vehicle Category"),
            color=alt.Color("Driver:N", title="Variance Driver"),
            tooltip=["Vehicle Category:N", "Driver:N", alt.Tooltip("Impact:Q", format=",.2f")],
        )
        .properties(title="Price, Volume and Portfolio-change Effects", height=350)
    )
    st.altair_chart(variance_chart, use_container_width=True)
    st.altair_chart(effect_chart, use_container_width=True)


def _render_pnl_visual(actual: dict, budget: dict) -> None:
    accounts = (
        ("Revenue", "revenue"), ("Gross Profit", "gross_profit"),
        ("EBITDA", "ebitda"), ("EBIT", "ebit"),
        ("EBT", "ebt"), ("Net Profit", "net_profit"),
    )
    chart_rows = [
        {"Account": label, "Scenario": scenario, "Amount": values.get(key, 0)}
        for label, key in accounts
        for scenario, values in (("Actual", actual), ("Budget", budget))
    ]
    chart = (
        alt.Chart(pd.DataFrame(chart_rows))
        .mark_bar()
        .encode(
            x=alt.X("Account:N", sort=[label for label, _ in accounts], title="P&L Account"),
            y=alt.Y("Amount:Q", title="Amount (INR)"),
            color=alt.Color("Scenario:N", title=None),
            xOffset="Scenario:N",
            tooltip=["Account:N", "Scenario:N", alt.Tooltip("Amount:Q", format=",.2f")],
        )
        .properties(title="Actual vs Budget P&L", height=350)
    )
    st.markdown("##### P&L Visualization")
    st.altair_chart(chart, use_container_width=True)


def _render_gp_visual(product_rows: list[dict], portfolio: dict) -> None:
    bridge_rows = []
    running = float(portfolio.get("budget_gp_percentage") or 0)
    bridge_rows.append({"Step": "Budget GP%", "Start": 0, "End": running, "Type": "Total"})
    for label, key in (
        ("Mix Effect", "mix_effect_percentage_points"),
        ("Price Effect", "price_effect_percentage_points"),
        ("Cost Effect", "cost_effect_percentage_points"),
    ):
        effect = float(portfolio.get(key) or 0)
        next_value = running + effect
        bridge_rows.append({
            "Step": label, "Start": min(running, next_value),
            "End": max(running, next_value),
            "Type": "Increase" if effect >= 0 else "Decrease",
        })
        running = next_value
    bridge_rows.append({
        "Step": "Actual GP%", "Start": 0,
        "End": float(portfolio.get("actual_gp_percentage") or running), "Type": "Total",
    })
    order = ["Budget GP%", "Mix Effect", "Price Effect", "Cost Effect", "Actual GP%"]
    bridge = (
        alt.Chart(pd.DataFrame(bridge_rows))
        .mark_bar()
        .encode(
            x=alt.X("Step:N", sort=order, title="Bridge Step"),
            y=alt.Y("Start:Q", title="GP% / Percentage-point Movement"),
            y2="End:Q",
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(domain=["Increase", "Decrease", "Total"], range=["#2ca02c", "#d62728", "#1f77b4"]),
                title=None,
            ),
            tooltip=["Step:N", alt.Tooltip("Start:Q", format=".2f"), alt.Tooltip("End:Q", format=".2f")],
        )
        .properties(title="GP% Budget-to-Actual Bridge", height=350)
    )
    gp_rows = [
        {"Vehicle Category": item.get("vehicle_category"), "Scenario": scenario, "GP%": item.get(key, 0)}
        for item in product_rows
        for scenario, key in (("Actual", "actual_gp_percentage"), ("Budget", "budget_gp_percentage"))
    ]
    category_chart = (
        alt.Chart(pd.DataFrame(gp_rows))
        .mark_bar()
        .encode(
            x=alt.X("Vehicle Category:N", title="Vehicle Category"),
            y=alt.Y("GP%:Q", title="GP%"),
            color=alt.Color("Scenario:N", title=None),
            xOffset="Scenario:N",
            tooltip=["Vehicle Category:N", "Scenario:N", alt.Tooltip("GP%:Q", format=".2f")],
        )
        .properties(title="Actual vs Budget GP% by Vehicle Category", height=330)
    )
    st.markdown("##### GP% Decomposition Visualizations")
    st.altair_chart(bridge, use_container_width=True)
    st.altair_chart(category_chart, use_container_width=True)


def _build_gp_product_table(product_rows: list[dict], portfolio: dict) -> pd.DataFrame:
    rows = []
    for item in product_rows:
        rows.append(
            {
                "Vehicle Category": item.get("vehicle_category") or item.get("product"),
                "Actual Orders": item.get("actual_volume"),
                "Actual Price/Order": item.get("actual_price_per_unit"),
                "Actual Direct Cost/Order": item.get("actual_cost_per_unit"),
                "Actual Revenue": item.get("actual_revenue"),
                "Actual Direct Cost": item.get("actual_direct_cost"),
                "Actual Gross Profit": item.get("actual_gross_profit"),
                "Actual GP%": item.get("actual_gp_percentage"),
                "Budget Orders": item.get("budget_volume"),
                "Budget Price/Order": item.get("budget_price_per_unit"),
                "Budget Direct Cost/Order": item.get("budget_cost_per_unit"),
                "Budget Revenue": item.get("budget_revenue"),
                "Budget Direct Cost": item.get("budget_direct_cost"),
                "Budget Gross Profit": item.get("budget_gross_profit"),
                "Budget GP%": item.get("budget_gp_percentage"),
                "Price Effect (pp)": item.get("price_effect_percentage_points"),
                "Cost Effect (pp)": item.get("cost_effect_percentage_points"),
                "Check (pp)": item.get("check_percentage_points"),
                "Mix Indicator": "Favorable" if item.get("mix_indicator") == 1 else "Unfavorable",
                "Actual Mix %": item.get("actual_mix_percentage"),
                "Budget Mix %": item.get("budget_mix_percentage"),
            }
        )
    actual_orders = sum(float(item.get("actual_volume") or 0) for item in product_rows)
    budget_orders = sum(float(item.get("budget_volume") or 0) for item in product_rows)
    rows.append(
        {
            "Vehicle Category": "TOTAL",
            "Actual Orders": actual_orders,
            "Actual Price/Order": _ratio(portfolio.get("actual_revenue"), actual_orders),
            "Actual Direct Cost/Order": _ratio(
                float(portfolio.get("actual_revenue") or 0) - float(portfolio.get("actual_gross_profit") or 0),
                actual_orders,
            ),
            "Actual Revenue": portfolio.get("actual_revenue"),
            "Actual Direct Cost": float(portfolio.get("actual_revenue") or 0) - float(portfolio.get("actual_gross_profit") or 0),
            "Actual Gross Profit": portfolio.get("actual_gross_profit"),
            "Actual GP%": portfolio.get("actual_gp_percentage"),
            "Budget Orders": budget_orders,
            "Budget Price/Order": _ratio(portfolio.get("base_revenue"), budget_orders),
            "Budget Direct Cost/Order": _ratio(
                float(portfolio.get("base_revenue") or 0) - float(portfolio.get("base_gross_profit") or 0),
                budget_orders,
            ),
            "Budget Revenue": portfolio.get("base_revenue"),
            "Budget Direct Cost": float(portfolio.get("base_revenue") or 0) - float(portfolio.get("base_gross_profit") or 0),
            "Budget Gross Profit": portfolio.get("base_gross_profit"),
            "Budget GP%": portfolio.get("budget_gp_percentage"),
            "Price Effect (pp)": portfolio.get("price_effect_percentage_points"),
            "Cost Effect (pp)": portfolio.get("cost_effect_percentage_points"),
            "Check (pp)": portfolio.get("reconciliation_difference"),
            "Mix Indicator": portfolio.get("reconciliation_status"),
            "Actual Mix %": 100.0,
            "Budget Mix %": 100.0,
        }
    )
    return pd.DataFrame(rows)


def _ratio(numerator, denominator: float) -> float | None:
    if not isinstance(numerator, (int, float)) or not denominator:
        return None
    return float(numerator) / denominator


def _money(value) -> str:
    if value is None:
        return "-"
    return f"INR {float(value):,.2f}"


def _percent(value) -> str:
    if value is None:
        return "-"
    return f"{float(value):,.2f}%"


def _signed(value, sign: int):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) * sign
    return None


def _display(value, currency: bool) -> str:
    if value is None:
        return "N/A"
    return f"INR {value:,.2f}" if currency else f"{value:,.2f}%"


if __name__ == "__main__":
    main()
