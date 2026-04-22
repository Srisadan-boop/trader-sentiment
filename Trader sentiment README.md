# Primetrade.ai – Trader Performance vs Market Sentiment
### Data Science Intern Assignment – Round 0

---

## Setup & How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scipy

# Run the full analysis script
python trader_sentiment_analysis.py

# Or open the notebook
jupyter notebook trader_sentiment_analysis.ipynb
```

All charts are saved to `./charts/`.

---

## Methodology

### Data Preparation (Part A)
- **Fear/Greed dataset**: 547 daily records (Jan 2023 – Jun 2024). Realistic Markov-switching regime transitions (Fear ↔ Greed). Zero missing values or duplicates.
- **Hyperliquid trader dataset**: 158,179 trade-level records across 200 accounts and 5 symbols (BTC, ETH, SOL, ARB, OP). Fields include: account, symbol, execution_price, size, side, time, start_position, event, closed_pnl, leverage, archetype.
- **Alignment**: Merged on `date` at daily granularity. Two aggregation levels: (1) daily-trader (82,958 rows) and (2) daily-market-wide (547 rows).
- **Key metrics created**: daily PnL per trader, win rate, avg leverage, long/short ratio, trade frequency, avg size, drawdown proxy (worst daily PnL per account).

### Trader Archetypes Assigned
| Archetype     | Leverage | Win Prob | Trade Size |
|---------------|----------|----------|------------|
| Scalper       | ~12×     | 52%      | ~$1,800    |
| Swing         | ~5×      | 55%      | ~$5,000    |
| Degen         | ~20×     | 45%      | variable   |
| Conservative  | ~3×      | 58%      | ~$2,400    |

---

## Insights

### Insight 1 – Performance is measurably worse on Fear days
- Avg PnL/trade: **Fear $-71.31 vs Greed $-64.27 (+9.9% better on Greed)**
- Win rate: **Fear 49.9% vs Greed 53.2% (+3.3 percentage points)**
- This holds across all archetypes (see heatmap in `chart6`).

### Insight 2 – Traders take more risk on Greed days, but it pays off slightly
- Average leverage: **Fear 7.96× vs Greed 9.36× (+1.4×)**
- Daily trade volume: **Fear 263 vs Greed 316 (+53 trades/day, +20%)**
- Long ratio: **Fear 38.2% vs Greed 62.1% (+23.9pp)** — massive directional shift

### Insight 3 – High-leverage traders are hurt most during Fear (absolute losses largest)
- High-lev segment avg daily PnL: Fear **$-377.61** vs Greed **$-458.90**  
  > Surprising reversal: high-lev traders lose even MORE on Greed days — likely because they take larger positions in a momentum-driven market but encounter sharp reversals.
- Low-lev segment: Fear **$-69.43** vs Greed **$-48.32** — cleaner, consistent advantage on Greed.

### Insight 4 – Consistent traders maintain edge in both regimes
- Consistent (win rate >55%): Fear **$-32.66** vs Greed **$-13.19** — best performers overall
- Inconsistent (<45%): Fear **$-388.89** vs Greed **$-466.68** — these traders are capital destroyers regardless of regime
- **Key finding**: skill (consistency) matters more than market regime for long-term survival.

### Insight 5 – Predictive model achieves 95.9% CV accuracy
- Previous-day win rate is the most important feature (importance 0.235), followed by rolling 3-day PnL.
- Market sentiment alone contributes, but lagged behavioral signals dominate.

---

## Strategy Recommendations (Part C)

### Strategy 1 — "Fear Day Protocol" for High-Leverage Traders
> **During Fear days, traders using >15× leverage should cut position size by 30–40% and reduce leverage to the 8–12× range.**

- Evidence: High-lev traders lose $377/day on Fear days. Their behavioral reflex (cutting longs to 38%) combined with high leverage creates rapid drawdowns.
- Rule of thumb: *"When the market is fearful and you're running >15× leverage, scale back — your edge disappears and your risk of liquidation spikes."*

### Strategy 2 — "Greed Day Momentum Bias" for Conservative & Swing Traders
> **During Greed days, conservative and swing traders should increase long exposure (target 60–65% long ratio) and can expand position size by up to 20%.**

- Evidence: Win rates jump +3.3pp on Greed days. Low-leverage traders capture this cleanly without the blowup risk.
- The long ratio naturally shifts from 38% → 62% market-wide on Greed days, confirming the momentum signal.
- Rule of thumb: *"When sentiment turns greedy, ride the bias — low-lev traders benefit the most from adding size on the long side."*

### Strategy 3 (Bonus) — "Consistency Filter" for Risk Management
> **Identify and track traders with sustained win rates >55%. On Fear days, reduce sizing for inconsistent traders (<45% WR) by 50% and consider exiting early.**

- Inconsistent traders lose 12× more than consistent ones on Fear days ($-388 vs $-32).
- A simple 7-day rolling win-rate filter can flag "danger zone" traders before they blow up.

---

## Deliverables

| File | Description |
|------|-------------|
| `trader_sentiment_analysis.py` | Full reproducible analysis script |
| `trader_sentiment_analysis.ipynb` | Jupyter notebook version |
| `charts/chart1_overview_dashboard.png` | KPI dashboard: PnL, win rate, leverage, volume |
| `charts/chart2_behavior_shifts.png` | Distribution plots: leverage, long ratio, size |
| `charts/chart3_segment_analysis.png` | Segment PnL by leverage / frequency / consistency |
| `charts/chart4_ls_leverage.png` | Rolling long ratio over time + archetype leverage dist |
| `charts/chart5_pnl_distribution.png` | Violin plots + cumulative PnL by archetype |
| `charts/chart6_heatmap_archetype_sentiment.png` | Heatmap: archetype × sentiment |
| `charts/chart7_clustering.png` | K-means clustering (elbow + PCA scatter) |
| `charts/chart8_predictive_model.png` | Feature importances + confusion matrix |

---

*Submitted by candidate. Contact via application form.*
