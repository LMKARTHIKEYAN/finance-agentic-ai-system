"""Create polished visual PDF attachments for scheduled finance reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.pipelines import ScheduledReport


NAVY = colors.HexColor("#16324F")
BLUE = colors.HexColor("#1F77B4")
TEAL = colors.HexColor("#18A999")
ORANGE = colors.HexColor("#F4A261")
RED = colors.HexColor("#D9534F")
LIGHT = colors.HexColor("#F3F6F9")
MUTED = colors.HexColor("#65758B")


class PdfReportTool:
    """Render a ScheduledReport as a management-ready visual PDF."""

    def create(self, report: ScheduledReport, output_path: str | Path) -> Path:
        if not isinstance(report, ScheduledReport):
            raise TypeError("report must be a ScheduledReport.")
        operations = report.results.get("operations")
        if operations is None:
            raise ValueError("The report has no operations result for visualization.")
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
            fontSize=22, leading=27, textColor=NAVY, alignment=TA_LEFT,
            spaceAfter=4,
        ))
        styles.add(ParagraphStyle(
            name="ReportTitleCompact", parent=styles["ReportTitle"],
            fontSize=17, leading=21,
        ))
        styles.add(ParagraphStyle(
            name="Subtitle", parent=styles["Normal"], fontSize=9.5,
            leading=13, textColor=MUTED, spaceAfter=12,
        ))
        styles.add(ParagraphStyle(
            name="Section", parent=styles["Heading2"], fontName="Helvetica-Bold",
            fontSize=14, leading=18, textColor=NAVY, spaceBefore=8, spaceAfter=7,
        ))
        styles.add(ParagraphStyle(
            name="BodySmall", parent=styles["BodyText"], fontSize=9,
            leading=13, textColor=NAVY,
        ))

        doc = SimpleDocTemplate(
            str(destination), pagesize=landscape(A4),
            rightMargin=14 * mm, leftMargin=14 * mm,
            topMargin=14 * mm, bottomMargin=14 * mm,
            title=report.title, author="Finance Agentic AI System",
        )
        story: list[Any] = [
            Paragraph(
                report.title.replace("&", "&amp;"),
                styles["ReportTitleCompact"]
                if len(report.title) > 48 else styles["ReportTitle"],
            ),
            Paragraph(
                f"Data cutoff: {report.data_cutoff} | Revenue basis: completed-order commission_amount | Status: {'VALIDATED' if report.validated else 'REVIEW'}",
                styles["Subtitle"],
            ),
            self._kpi_cards(operations),
            Spacer(1, 7 * mm),
            Table(
                [[self._status_chart(operations), self._vehicle_chart(operations)]],
                colWidths=[82 * mm, 174 * mm],
                style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]),
            ),
            Spacer(1, 5 * mm),
            Paragraph("Management summary", styles["Section"]),
            Paragraph(self._executive_summary(report), styles["BodySmall"]),
        ]
        if report.report_type == "monthly":
            story.extend(self._monthly_pages(report, styles))
        story.extend([
            PageBreak(),
            Paragraph("Detailed performance", styles["ReportTitle"]),
            Paragraph("Revenue and service performance by vehicle category", styles["Subtitle"]),
            self._detail_table(operations),
        ])
        doc.build(story, onFirstPage=self._page, onLaterPages=self._page)
        return destination

    @staticmethod
    def _kpi_cards(operations: Any) -> Table:
        cards = [
            ("Total orders", f"{operations.total_orders:,.0f}"),
            ("Completed", f"{operations.completed_orders:,.0f}"),
            ("Commission revenue", f"INR {operations.total_revenue:,.2f}"),
            ("Commission AOV", f"INR {operations.average_order_value:,.2f}"),
            ("Fulfillment", f"{operations.fulfillment_percentage:.2f}%"),
            ("Cancellation", f"{operations.cancellation_percentage:.2f}%"),
        ]
        label_style = ParagraphStyle(
            "CardLabel", fontName="Helvetica", fontSize=8,
            textColor=MUTED, leading=10,
        )
        value_style = ParagraphStyle(
            "CardValue", fontName="Helvetica-Bold", fontSize=14,
            textColor=NAVY, leading=17,
        )
        labels = [Paragraph(label, label_style) for label, _ in cards]
        values = [Paragraph(value, value_style) for _, value in cards]
        table = Table(
            [labels, values], colWidths=[42.5 * mm] * 6,
            rowHeights=[8 * mm, 15 * mm],
        )
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
            ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#D9E2EC")),
            ("INNERGRID", (0, 0), (-1, -1), 3, colors.white),
            ("VALIGN", (0, 0), (-1, 0), "BOTTOM"),
            ("VALIGN", (0, 1), (-1, 1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ]))
        return table

    @staticmethod
    def _status_chart(operations: Any) -> Drawing:
        completed = max(int(operations.completed_orders), 0)
        cancelled = max(int(operations.cancelled_orders), 0)
        other = max(int(operations.total_orders) - completed - cancelled, 0)
        drawing = Drawing(230, 160)
        drawing.add(String(5, 145, "Order status mix", fontName="Helvetica-Bold", fontSize=12, fillColor=NAVY))
        pie = Pie()
        pie.x, pie.y, pie.width, pie.height = 12, 20, 115, 115
        pie.data = [completed, cancelled, other]
        pie.labels = ["Completed", "Cancelled", "Other"]
        pie.slices[0].fillColor = TEAL
        pie.slices[1].fillColor = RED
        pie.slices[2].fillColor = ORANGE
        pie.sideLabels = True
        pie.simpleLabels = False
        drawing.add(pie)
        return drawing

    @staticmethod
    def _vehicle_chart(operations: Any) -> Drawing:
        rows = sorted(
            operations.vehicle_summary,
            key=lambda row: float(row.get("total_revenue", 0)), reverse=True,
        )[:7]
        drawing = Drawing(485, 160)
        drawing.add(String(5, 145, "Commission revenue by vehicle category", fontName="Helvetica-Bold", fontSize=12, fillColor=NAVY))
        chart = VerticalBarChart()
        chart.x, chart.y, chart.width, chart.height = 45, 34, 420, 94
        chart.data = [[float(row.get("total_revenue", 0)) / 100000 for row in rows]]
        chart.categoryAxis.categoryNames = [str(row.get("vehicle_category", ""))[:16] for row in rows]
        chart.categoryAxis.labels.angle = 25
        chart.categoryAxis.labels.fontSize = 7
        chart.valueAxis.valueMin = 0
        chart.valueAxis.labels.fontSize = 7
        chart.valueAxis.labelTextFormat = "%.1f"
        chart.bars[0].fillColor = BLUE
        chart.bars[0].strokeColor = BLUE
        drawing.add(chart)
        drawing.add(String(5, 70, "INR lakh", fontName="Helvetica", fontSize=7, fillColor=MUTED))
        return drawing

    @staticmethod
    def _detail_table(operations: Any) -> Table:
        rows = [["Vehicle category", "Orders", "Completed", "Commission revenue", "AOV", "Fulfillment", "Cancellation"]]
        for item in sorted(operations.vehicle_summary, key=lambda row: float(row.get("total_revenue", 0)), reverse=True):
            rows.append([
                str(item.get("vehicle_category", "")),
                f"{int(item.get('total_orders', 0)):,}",
                f"{int(item.get('completed_orders', 0)):,}",
                f"INR {float(item.get('total_revenue', 0)):,.2f}",
                f"INR {float(item.get('average_order_value', 0)):,.2f}",
                f"{float(item.get('fulfillment_percentage', 0)):.2f}%",
                f"{float(item.get('cancellation_percentage', 0)):.2f}%",
            ])
        table = Table(rows, repeatRows=1, colWidths=[53 * mm, 25 * mm, 28 * mm, 45 * mm, 32 * mm, 32 * mm, 32 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9E2EC")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        return table

    def _monthly_pages(self, report: ScheduledReport, styles: Any) -> list[Any]:
        pnl = report.results.get("pnl")
        variance = report.results.get("variance")
        gp = report.results.get("gp_decomposition")
        forecast = report.results.get("forecast")
        if not all((pnl, variance, gp, forecast)):
            raise ValueError("Monthly PDF requires P&L, revenue variance, GP decomposition, and forecast results.")
        report_month = report.title.rsplit(" - ", 1)[-1]
        return [
            PageBreak(),
            Paragraph("Actual vs Budget P&amp;L", styles["ReportTitle"]),
            Paragraph(f"{report_month} management profit and loss statement", styles["Subtitle"]),
            self._pnl_table(self._value(pnl, "summary")),
            Spacer(1, 5 * mm),
            Paragraph(
                "Positive revenue and profit variances are favourable. Negative cost variances are favourable.",
                styles["BodySmall"],
            ),
            PageBreak(),
            Paragraph("Revenue variance analysis", styles["ReportTitle"]),
            Paragraph("Actual commission revenue versus budget revenue", styles["Subtitle"]),
            self._variance_cards(variance),
            Spacer(1, 5 * mm),
            self._variance_chart(variance),
            Spacer(1, 4 * mm),
            self._variance_table(variance),
            PageBreak(),
            Paragraph("Category-wise revenue variance", styles["ReportTitleCompact"]),
            Paragraph(
                "Vehicle-category bridge: budget to actual commission revenue through price/AOV, volume, and portfolio effects",
                styles["Subtitle"],
            ),
            self._category_variance_table(variance),
            PageBreak(),
            Paragraph("GP% decomposition", styles["ReportTitle"]),
            Paragraph("Portfolio margin bridge: mix, price, and cost effects", styles["Subtitle"]),
            self._gp_cards(gp),
            Spacer(1, 4 * mm),
            self._gp_stage_table(gp),
            Spacer(1, 6 * mm),
            Table(
                [[self._gp_chart(gp), self._gp_table(gp)]],
                colWidths=[130 * mm, 120 * mm],
                style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]),
            ),
            Spacer(1, 5 * mm),
            Paragraph(
                f"Reconciliation status: <b>{self._value(gp, 'reconciliation_status')}</b> | Difference: {float(self._value(gp, 'reconciliation_difference')):.4f} percentage points",
                styles["BodySmall"],
            ),
            PageBreak(),
            Paragraph("Category-wise GP% decomposition", styles["ReportTitleCompact"]),
            Paragraph(
                "Vehicle-category unit economics and GP% bridge using volume, price/AOV, and direct cost per completed order",
                styles["Subtitle"],
            ),
            self._category_gp_table(gp),
            PageBreak(),
            Paragraph("Rolling forecast", styles["ReportTitle"]),
            Paragraph("Three-month simple moving-average forecast using commission revenue", styles["Subtitle"]),
            Table(
                [[self._forecast_chart(forecast), self._forecast_table(forecast)]],
                colWidths=[145 * mm, 110 * mm],
                style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]),
            ),
            Spacer(1, 7 * mm),
            Paragraph(
                "Forecast method: Simple Moving Average Rolling Forecast. Source periods roll forward after each forecast month.",
                styles["BodySmall"],
            ),
        ]

    @staticmethod
    def _pnl_table(summary: dict[str, Any]) -> Table:
        actual, budget, variance = summary["actual"], summary["budget"], summary["variance"]
        definitions = [
            ("Revenue", "revenue", "revenue_variance", "revenue_variance_percentage"),
            ("Direct cost", "direct_cost", "direct_cost_variance", "direct_cost_variance_percentage"),
            ("Gross profit", "gross_profit", "gross_profit_variance", "gross_profit_variance_percentage"),
            ("Gross margin %", "gross_margin_percentage", "gross_margin_percentage_point_variance", None),
            ("Sales & marketing", "sales_marketing", "sales_marketing_variance", "sales_marketing_variance_percentage"),
            ("Other OPEX", "other_opex", "other_opex_variance", "other_opex_variance_percentage"),
            ("EBITDA", "ebitda", "ebitda_variance", "ebitda_variance_percentage"),
            ("Depreciation", "depreciation", "depreciation_variance", "depreciation_variance_percentage"),
            ("EBIT", "ebit", "ebit_variance", "ebit_variance_percentage"),
            ("Interest", "interest", "interest_variance", "interest_variance_percentage"),
            ("EBT", "ebt", "ebt_variance", "ebt_variance_percentage"),
            ("Income tax", "income_tax", "income_tax_variance", "income_tax_variance_percentage"),
            ("Net profit", "net_profit", "net_profit_variance", "net_profit_variance_percentage"),
        ]
        rows = [["P&L line", "Actual", "Budget", "Variance", "Variance % / pp"]]
        for label, key, variance_key, percent_key in definitions:
            percentage_line = key == "gross_margin_percentage"
            actual_value = f"{float(actual[key]):.2f}%" if percentage_line else f"INR {float(actual[key]):,.2f}"
            budget_value = f"{float(budget[key]):.2f}%" if percentage_line else f"INR {float(budget[key]):,.2f}"
            variance_value = f"{float(variance[variance_key]):.2f} pp" if percentage_line else f"INR {float(variance[variance_key]):,.2f}"
            pct = f"{float(variance[percent_key]):.2f}%" if percent_key else f"{float(variance[variance_key]):.2f} pp"
            rows.append([label, actual_value, budget_value, variance_value, pct])
        table = Table(rows, repeatRows=1, colWidths=[52 * mm, 48 * mm, 48 * mm, 50 * mm, 43 * mm])
        table.setStyle(PdfReportTool._financial_table_style())
        return table

    @staticmethod
    def _variance_cards(variance: Any) -> Table:
        cards = [
            ("Actual revenue", f"INR {float(PdfReportTool._value(variance, 'actual_revenue')):,.2f}"),
            ("Budget revenue", f"INR {float(PdfReportTool._value(variance, 'budget_revenue')):,.2f}"),
            ("Revenue variance", f"INR {float(PdfReportTool._value(variance, 'revenue_variance')):,.2f}"),
            ("Order variance", f"{int(PdfReportTool._value(variance, 'order_variance')):,}"),
        ]
        return PdfReportTool._simple_cards(cards, 61 * mm)

    @staticmethod
    def _variance_chart(variance: Any) -> Drawing:
        effects = [
            float(PdfReportTool._value(variance, "price_effect")) / 100000,
            float(PdfReportTool._value(variance, "volume_effect")) / 100000,
            float(PdfReportTool._value(variance, "new_discontinued_effect")) / 100000,
        ]
        drawing = Drawing(700, 155)
        chart = VerticalBarChart()
        chart.x, chart.y, chart.width, chart.height = 70, 36, 585, 100
        chart.data = [effects]
        chart.categoryAxis.categoryNames = ["Price/AOV effect", "Volume effect", "New/discontinued"]
        chart.categoryAxis.labels.fontSize = 8
        chart.valueAxis.labels.fontSize = 8
        chart.valueAxis.valueMin = min(0, min(effects) * 1.2)
        chart.valueAxis.valueMax = max(effects) * 1.15 if max(effects) else 1
        chart.bars[0].fillColor = BLUE
        drawing.add(chart)
        drawing.add(String(5, 85, "INR lakh", fontName="Helvetica", fontSize=8, fillColor=MUTED))
        return drawing

    @staticmethod
    def _variance_table(variance: Any) -> Table:
        rows = [["Driver", "Amount", "Interpretation"],
            ["Price/AOV effect", f"INR {float(PdfReportTool._value(variance, 'price_effect')):,.2f}", "Lower AOV reduced revenue"],
            ["Volume effect", f"INR {float(PdfReportTool._value(variance, 'volume_effect')):,.2f}", "Higher completed orders increased revenue"],
            ["New/discontinued effect", f"INR {float(PdfReportTool._value(variance, 'new_discontinued_effect')):,.2f}", "No portfolio entry/exit effect"],
            ["Reconciliation check", f"INR {float(PdfReportTool._value(variance, 'variance_check')):,.2f}", "PASS" if abs(float(PdfReportTool._value(variance, 'variance_check'))) < 0.01 else "REVIEW"],
        ]
        table = Table(rows, colWidths=[60 * mm, 55 * mm, 125 * mm])
        table.setStyle(PdfReportTool._financial_table_style())
        return table

    @staticmethod
    def _category_variance_table(variance: Any) -> Table:
        rows = [[
            "Vehicle category", "Budget qty", "Actual qty",
            "Budget AOV", "Actual AOV", "Budget revenue", "Actual revenue",
            "Price/AOV effect", "Volume effect", "New/disc.", "Variance", "Check",
        ]]
        source = PdfReportTool._value(variance, "vehicle_variance_summary")
        for item in sorted(source, key=lambda row: float(row["revenue_variance"]), reverse=True):
            rows.append([
                item["vehicle_category"], f"{int(item['budget_orders']):,}",
                f"{int(item['actual_orders']):,}",
                f"INR {float(item['budget_aov']):,.2f}",
                f"INR {float(item['actual_aov']):,.2f}",
                f"INR {float(item['budget_revenue']):,.2f}",
                f"INR {float(item['actual_revenue']):,.2f}",
                f"INR {float(item['price_effect']):,.2f}",
                f"INR {float(item['volume_effect']):,.2f}",
                f"INR {float(item['new_discontinued_effect']):,.2f}",
                f"INR {float(item['revenue_variance']):,.2f}",
                f"INR {float(item['variance_check']):,.2f}",
            ])
        rows.append([
            "TOTAL",
            f"{int(PdfReportTool._value(variance, 'budget_orders')):,}",
            f"{int(PdfReportTool._value(variance, 'actual_orders')):,}",
            f"INR {float(PdfReportTool._value(variance, 'budget_aov')):,.2f}",
            f"INR {float(PdfReportTool._value(variance, 'actual_aov')):,.2f}",
            f"INR {float(PdfReportTool._value(variance, 'budget_revenue')):,.2f}",
            f"INR {float(PdfReportTool._value(variance, 'actual_revenue')):,.2f}",
            f"INR {float(PdfReportTool._value(variance, 'price_effect')):,.2f}",
            f"INR {float(PdfReportTool._value(variance, 'volume_effect')):,.2f}",
            f"INR {float(PdfReportTool._value(variance, 'new_discontinued_effect')):,.2f}",
            f"INR {float(PdfReportTool._value(variance, 'revenue_variance')):,.2f}",
            f"INR {float(PdfReportTool._value(variance, 'variance_check')):,.2f}",
        ])
        table = Table(
            rows, repeatRows=1,
            colWidths=[27 * mm, 16 * mm, 16 * mm, 21 * mm, 21 * mm, 28 * mm,
                       28 * mm, 27 * mm, 27 * mm, 23 * mm, 27 * mm, 20 * mm],
        )
        table.setStyle(PdfReportTool._financial_table_style())
        table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 5.6),
            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#E2E8F0")),
        ]))
        return table

    @staticmethod
    def _gp_cards(gp: Any) -> Table:
        cards = [
            ("Budget GP%", f"{float(PdfReportTool._value(gp, 'budget_gp_percentage')):.2f}%"),
            ("Actual GP%", f"{float(PdfReportTool._value(gp, 'actual_gp_percentage')):.2f}%"),
            ("Total movement", f"{float(PdfReportTool._value(gp, 'total_variance_percentage_points')):.2f} pp"),
            ("Reconciliation", str(PdfReportTool._value(gp, "reconciliation_status"))),
        ]
        return PdfReportTool._simple_cards(cards, 61 * mm)

    @staticmethod
    def _gp_chart(gp: Any) -> Drawing:
        values = [
            float(PdfReportTool._value(gp, "mix_effect_percentage_points")),
            float(PdfReportTool._value(gp, "price_effect_percentage_points")),
            float(PdfReportTool._value(gp, "cost_effect_percentage_points")),
        ]
        drawing = Drawing(355, 190)
        drawing.add(String(5, 173, "GP% movement by driver", fontName="Helvetica-Bold", fontSize=11, fillColor=NAVY))
        chart = VerticalBarChart()
        chart.x, chart.y, chart.width, chart.height = 52, 40, 280, 115
        chart.data = [values]
        chart.categoryAxis.categoryNames = ["Mix", "Price", "Cost"]
        chart.valueAxis.valueMin = min(0, min(values) * 1.2)
        chart.valueAxis.valueMax = max(values) * 1.15
        chart.bars[0].fillColor = TEAL
        drawing.add(chart)
        drawing.add(String(5, 90, "Percentage points", fontName="Helvetica", fontSize=7, fillColor=MUTED))
        return drawing

    @staticmethod
    def _gp_table(gp: Any) -> Table:
        rows = [["Driver", "Impact (pp)", "Impact (bps)"],
            ["Mix", f"{float(PdfReportTool._value(gp, 'mix_effect_percentage_points')):.4f}", f"{float(PdfReportTool._value(gp, 'mix_effect_basis_points')):.2f}"],
            ["Price", f"{float(PdfReportTool._value(gp, 'price_effect_percentage_points')):.4f}", f"{float(PdfReportTool._value(gp, 'price_effect_basis_points')):.2f}"],
            ["Cost", f"{float(PdfReportTool._value(gp, 'cost_effect_percentage_points')):.4f}", f"{float(PdfReportTool._value(gp, 'cost_effect_basis_points')):.2f}"],
            ["Total", f"{float(PdfReportTool._value(gp, 'total_variance_percentage_points')):.4f}", f"{float(PdfReportTool._value(gp, 'total_variance_basis_points')):.2f}"],
        ]
        table = Table(rows, colWidths=[42 * mm, 36 * mm, 36 * mm])
        table.setStyle(PdfReportTool._financial_table_style())
        return table

    @staticmethod
    def _gp_stage_table(gp: Any) -> Table:
        rows = [
            ["GP% bridge stage", "Revenue", "Gross profit", "GP%"],
            ["Budget", f"INR {float(PdfReportTool._value(gp, 'base_revenue')):,.2f}",
             f"INR {float(PdfReportTool._value(gp, 'base_gross_profit')):,.2f}",
             f"{float(PdfReportTool._value(gp, 'budget_gp_percentage')):.4f}%"],
            ["Mix only", f"INR {float(PdfReportTool._value(gp, 'mix_only_revenue')):,.2f}",
             f"INR {float(PdfReportTool._value(gp, 'mix_only_gross_profit')):,.2f}",
             f"{float(PdfReportTool._value(gp, 'mix_only_gp_percentage')):.4f}%"],
            ["Price only (budget cost)", f"INR {float(PdfReportTool._value(gp, 'price_only_revenue')):,.2f}",
             f"INR {float(PdfReportTool._value(gp, 'price_only_gross_profit')):,.2f}",
             f"{float(PdfReportTool._value(gp, 'price_only_gp_percentage')):.4f}%"],
            ["Actual", f"INR {float(PdfReportTool._value(gp, 'actual_revenue')):,.2f}",
             f"INR {float(PdfReportTool._value(gp, 'actual_gross_profit')):,.2f}",
             f"{float(PdfReportTool._value(gp, 'actual_gp_percentage')):.4f}%"],
        ]
        table = Table(rows, colWidths=[62 * mm, 55 * mm, 55 * mm, 35 * mm])
        table.setStyle(PdfReportTool._financial_table_style())
        return table

    @staticmethod
    def _category_gp_table(gp: Any) -> Table:
        rows = [[
            "Vehicle category", "Budget qty", "Actual qty", "Budget AOV", "Actual AOV",
            "Budget cost/order", "Actual cost/order", "Budget GP%", "Actual GP%",
            "Price (pp)", "Cost (pp)", "Check (pp)",
        ]]
        source = PdfReportTool._value(gp, "category_analysis")
        for item in sorted(source, key=lambda row: float(row["actual_revenue"]), reverse=True):
            rows.append([
                item["vehicle_category"],
                f"{int(item['budget_volume']):,}", f"{int(item['actual_volume']):,}",
                f"INR {float(item['budget_price_per_unit']):,.2f}",
                f"INR {float(item['actual_price_per_unit']):,.2f}",
                f"INR {float(item['budget_cost_per_unit']):,.2f}",
                f"INR {float(item['actual_cost_per_unit']):,.2f}",
                f"{float(item['budget_gp_percentage']):.2f}%",
                f"{float(item['actual_gp_percentage']):.2f}%",
                f"{float(item['price_effect_percentage_points']):.4f}",
                f"{float(item['cost_effect_percentage_points']):.4f}",
                f"{float(item['check_percentage_points']):.4f}",
            ])
        table = Table(
            rows, repeatRows=1,
            colWidths=[28 * mm, 17 * mm, 17 * mm, 22 * mm, 22 * mm, 26 * mm,
                       26 * mm, 21 * mm, 21 * mm, 20 * mm, 20 * mm, 19 * mm],
        )
        table.setStyle(PdfReportTool._financial_table_style())
        table.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 5.6)]))
        return table

    @staticmethod
    def _forecast_chart(forecast: Any) -> Drawing:
        rows = PdfReportTool._value(forecast, "forecast_summary")
        values = [float(item["forecast_revenue"]) / 100000 for item in rows]
        drawing = Drawing(400, 210)
        drawing.add(String(5, 193, "Forecast commission revenue", fontName="Helvetica-Bold", fontSize=11, fillColor=NAVY))
        chart = VerticalBarChart()
        chart.x, chart.y, chart.width, chart.height = 55, 45, 320, 125
        chart.data = [values]
        chart.categoryAxis.categoryNames = [item["forecast_period"] for item in rows]
        chart.valueAxis.valueMin = 0
        chart.valueAxis.valueMax = max(values) * 1.2
        chart.bars[0].fillColor = BLUE
        drawing.add(chart)
        drawing.add(String(5, 100, "INR lakh", fontName="Helvetica", fontSize=8, fillColor=MUTED))
        return drawing

    @staticmethod
    def _forecast_table(forecast: Any) -> Table:
        rows = [["Period", "Orders", "Revenue", "AOV"]]
        for item in PdfReportTool._value(forecast, "forecast_summary"):
            rows.append([
                item["forecast_period"], f"{int(item['forecast_orders']):,}",
                f"INR {float(item['forecast_revenue']):,.2f}",
                f"INR {float(item['forecast_average_order_value']):,.2f}",
            ])
        table = Table(rows, colWidths=[26 * mm, 27 * mm, 45 * mm, 30 * mm])
        table.setStyle(PdfReportTool._financial_table_style())
        return table

    @staticmethod
    def _simple_cards(cards: list[tuple[str, str]], width: float) -> Table:
        labels = [Paragraph(label, ParagraphStyle("MetricLabel", fontName="Helvetica", fontSize=8, textColor=MUTED)) for label, _ in cards]
        values = [Paragraph(value, ParagraphStyle("MetricValue", fontName="Helvetica-Bold", fontSize=13, textColor=NAVY, leading=16)) for _, value in cards]
        table = Table([labels, values], colWidths=[width] * len(cards), rowHeights=[8 * mm, 14 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
            ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#D9E2EC")),
            ("INNERGRID", (0, 0), (-1, -1), 3, colors.white),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return table

    @staticmethod
    def _financial_table_style() -> TableStyle:
        return TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("ALIGN", (0, 0), (-1, 0), "LEFT"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9E2EC")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])

    @staticmethod
    def _value(container: Any, key: str) -> Any:
        return container[key] if isinstance(container, dict) else getattr(container, key)

    @staticmethod
    def _executive_summary(report: ScheduledReport) -> str:
        commentary = report.results.get("commentary")
        summary = getattr(commentary, "executive_summary", "")
        risks = getattr(commentary, "risks", [])
        text = (summary or "Performance was calculated from validated Snowflake data.").replace("₹", "INR ")
        if risks:
            text += "<br/><br/><b>Management attention:</b> " + " ".join(str(item) for item in risks[:3])
        return text

    @staticmethod
    def _page(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        width, _ = landscape(A4)
        canvas.setStrokeColor(colors.HexColor("#D9E2EC"))
        canvas.line(14 * mm, 10 * mm, width - 14 * mm, 10 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(14 * mm, 6 * mm, "Finance Agentic AI System | Confidential")
        canvas.drawRightString(width - 14 * mm, 6 * mm, f"Page {doc.page}")
        canvas.restoreState()
