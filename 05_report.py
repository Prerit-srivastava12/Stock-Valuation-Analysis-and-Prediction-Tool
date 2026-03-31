# ------------------------------------------------------------
#  05_report.py
#  Generates the unified valuation + prediction report
# ------------------------------------------------------------

import pandas as pd
import os
from datetime import datetime
from config import TICKER, COMPANY_NAME, OUTPUT_DIR, TERMINAL_GROWTH_RATE

os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_verdict(current_price, intrinsic_value) -> tuple[str, float]:
    """
    Returns (verdict_string, margin_of_safety_pct)
    """
    if not current_price or not intrinsic_value:
        return "INSUFFICIENT DATA", 0.0

    margin = (intrinsic_value - current_price) / intrinsic_value * 100

    if margin > 15:
        verdict = "  UNDERVALUED"
    elif margin < -15:
        verdict = "  OVERVALUED"
    else:
        verdict = "   FAIRLY VALUED"

    return verdict, round(margin, 2)


def generate_report(val_results: dict, pred_results: dict) -> str:
    """
    Assembles the full text report and saves it.
    Returns the report as a string.
    """
    now   = datetime.now().strftime("%Y-%m-%d %H:%M")
    dcf   = val_results.get("dcf", {})
    rel   = val_results.get("relative", pd.DataFrame())
    sens  = val_results.get("sensitivity", pd.DataFrame())

    current_price    = val_results.get("current_price")
    intrinsic_value  = dcf.get("intrinsic_value")

    verdict, margin  = compute_verdict(current_price, intrinsic_value)

    horizon_preds = pred_results.get("horizon_predictions", {})
    metrics       = pred_results.get("metrics", pd.DataFrame())

    lines = []
    L = lines.append

    # ── HEADER ──────────────────────────────────────────────
    L("=" * 65)
    L(f"  STOCK VALUATION & PREDICTION REPORT")
    L(f"  {COMPANY_NAME}  ({TICKER})")
    L(f"  Generated: {now}")
    L("=" * 65)

    # ── CURRENT SNAPSHOT ────────────────────────────────────
    L("\n CURRENT MARKET SNAPSHOT")
    L(f"  Current Market Price:     ₹{current_price:,.2f}" if current_price else "  Current Price: N/A")

    if not rel.empty and TICKER in rel.index:
        row = rel.loc[TICKER]
        L(f"  Market Cap:               ₹{val_results['dcf'].get('enterprise_value_cr', 'N/A')} Cr (EV)")
        L(f"  P/E (Trailing):           {row.get('P/E', 'N/A')}")
        L(f"  P/B:                      {row.get('P/B', 'N/A')}")
        L(f"  EV/EBITDA:                {row.get('EV/EBITDA', 'N/A')}")

    # ── DCF VALUATION ────────────────────────────────────────
    L("\n📐 DCF INTRINSIC VALUATION")
    if "error" in dcf:
        L(f"    {dcf['error']}")
    else:
        L(f"  WACC Used:                {dcf.get('wacc')}%")
        L(f"  FCF Growth Rate (hist.):  {dcf.get('growth_rate_used')}%")
        L(f"  Terminal Growth Rate:     {dcf.get('terminal_growth')}%")
        L(f"  Terminal Value (% of EV): {dcf.get('terminal_value_pct')}%")
        L(f"  Enterprise Value:         ₹{dcf.get('enterprise_value_cr')} Cr")
        L(f"  ─────────────────────────────────────────────")
        L(f"  ► Intrinsic Value/Share:  ₹{intrinsic_value:,.2f}" if intrinsic_value else "  ► N/A")

    # ── SENSITIVITY ──────────────────────────────────────────
    L("\n SENSITIVITY ANALYSIS (Intrinsic Value ₹/share)")
    if not sens.empty:
        L(sens.to_string())
    else:
        L("  No sensitivity data.")

    # ── RELATIVE VALUATION ──────────────────────────────────
    L("\n🔍 RELATIVE VALUATION vs PEERS")
    if not rel.empty:
        display_cols = ["Company", "Price", "P/E", "P/B", "EV/EBITDA", "ROE (%)", "Net Margin (%)"]
        available    = [c for c in display_cols if c in rel.columns]
        L(rel[available].to_string())
        L(f"\n  Peer Comparison Signals:")
        L(val_results.get("rel_verdict", "N/A"))
    else:
        L("  No peer data available.")

    # ── ML PREDICTION ────────────────────────────────────────
    L("\n ML PRICE PREDICTIONS")
    if "error" in pred_results:
        L(f"    {pred_results['error']}")
    else:
        L(f"  Current Price: ₹{current_price:,.2f}" if current_price else "  N/A")
        L(f"  {'Horizon':<15} {'LR':>10} {'RF':>10} {'GB':>10} {'Ensemble':>12}")
        L(f"  {'-'*57}")
        horizon_labels = {63: "3 months", 126: "6 months", 252: "12 months"}
        for h, preds in horizon_preds.items():
            label = horizon_labels.get(h, f"{h}d")
            lr  = f"₹{preds.get('Linear Regression', 'N/A'):,.0f}" if preds.get('Linear Regression') else "N/A"
            rf  = f"₹{preds.get('Random Forest', 'N/A'):,.0f}" if preds.get('Random Forest') else "N/A"
            gb  = f"₹{preds.get('Gradient Boosting', 'N/A'):,.0f}" if preds.get('Gradient Boosting') else "N/A"
            ens = f"₹{preds.get('Ensemble (Avg)', 'N/A'):,.0f}" if preds.get('Ensemble (Avg)') else "N/A"
            L(f"  {label:<15} {lr:>10} {rf:>10} {gb:>10} {ens:>12}")

        L(f"\n  Model Accuracy (on {int(len(pred_results.get('df_features', []))*0.2)} test samples):")
        if not metrics.empty:
            L(metrics.to_string())

    # ── FINAL VERDICT ────────────────────────────────────────
    L("\n" + "=" * 65)
    L(f"  FINAL VERDICT:  {verdict}")
    if intrinsic_value and current_price:
        L(f"  DCF Intrinsic Value:      ₹{intrinsic_value:,.2f}")
        L(f"  Current Market Price:     ₹{current_price:,.2f}")
        L(f"  Margin of Safety:         {margin:+.1f}%")
        if margin > 0:
            L(f"  → Market is pricing the stock {margin:.1f}% BELOW intrinsic value")
        else:
            L(f"  → Market is pricing the stock {abs(margin):.1f}% ABOVE intrinsic value")
    L("=" * 65)
    L(f"\n  DISCLAIMER: This report is for educational/research purposes only.")
    L(f"   ML predictions reflect historical patterns, not guaranteed future returns.")
    L(f"   Always conduct independent due diligence before investment decisions.")

    report = "\n".join(lines)

    # Save to file
    path = os.path.join(OUTPUT_DIR, f"{TICKER.replace('.','_')}_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n Report saved: {path}")
    return report


def print_report(report: str):
    print("\n" + report)
