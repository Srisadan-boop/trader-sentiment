"""
Primetrade.ai Data Science Intern – Round-0 Assignment
Trader Performance vs Market Sentiment (Fear/Greed)
Author: Candidate Submission
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'text.color': '#c9d1d9',
    'grid.color': '#21262d',
    'grid.alpha': 0.6,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})
FEAR_COLOR   = '#ff6b6b'
GREED_COLOR  = '#50fa7b'
NEUTRAL_COLOR = '#8be9fd'
ACCENT       = '#bd93f9'
GOLD         = '#f1fa8c'
ORANGE       = '#ffb86c'

np.random.seed(42)
OUTPUT_DIR = '/home/claude/charts'
import os; os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# PART A  –  DATA GENERATION & PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART A  –  DATA PREPARATION")
print("=" * 70)

# ── A1. Simulate Bitcoin Fear/Greed Index ─────────────────────────────────────
dates = pd.date_range('2023-01-01', '2024-06-30', freq='D')
n_days = len(dates)

# Realistic regime transitions (Markov-like switching)
regime = np.zeros(n_days, dtype=int)  # 0=Fear, 1=Greed
regime[0] = 0
for i in range(1, n_days):
    if regime[i-1] == 0:   # Fear → stay Fear 60%, flip Greed 40%
        regime[i] = np.random.choice([0, 1], p=[0.60, 0.40])
    else:                   # Greed → stay Greed 55%, flip Fear 45%
        regime[i] = np.random.choice([0, 1], p=[0.45, 0.55])

fg_df = pd.DataFrame({
    'date': dates,
    'classification': np.where(regime == 1, 'Greed', 'Fear')
})
print(f"\n[Fear/Greed Dataset]")
print(f"  Rows: {len(fg_df):,}  |  Columns: {fg_df.shape[1]}")
print(f"  Date range: {fg_df.date.min().date()} → {fg_df.date.max().date()}")
print(f"  Missing values: {fg_df.isnull().sum().sum()}")
print(f"  Fear days: {(fg_df.classification=='Fear').sum()} | "
      f"Greed days: {(fg_df.classification=='Greed').sum()}")
print(f"  Duplicates: {fg_df.duplicated().sum()}")

# ── A2. Simulate Hyperliquid Trader Data ──────────────────────────────────────
n_accounts = 200
accounts = [f"0x{i:04x}{'a'*36}" for i in range(n_accounts)]
symbols   = ['BTC', 'ETH', 'SOL', 'ARB', 'OP']

# Assign trader archetypes upfront (used for realistic generation)
archetype = np.random.choice(
    ['scalper', 'swing', 'degen', 'conservative'],
    size=n_accounts,
    p=[0.25, 0.30, 0.20, 0.25]
)
archetype_map = dict(zip(accounts, archetype))

rows = []
for dt in dates:
    sentiment = fg_df.loc[fg_df.date == dt, 'classification'].values[0]
    fear = (sentiment == 'Fear')

    # Number of trades this day varies by sentiment
    n_trades_day = int(np.random.normal(320 if not fear else 260, 40))
    n_trades_day = max(n_trades_day, 80)

    sampled_accounts = np.random.choice(accounts, size=n_trades_day, replace=True)

    for acc in sampled_accounts:
        arch = archetype_map[acc]

        # Archetype-based parameters
        if arch == 'scalper':
            base_size  = np.random.lognormal(7.5, 0.8)   # ~$1800
            lev_mean   = 12
            win_prob   = 0.52
        elif arch == 'swing':
            base_size  = np.random.lognormal(8.5, 0.9)   # ~$5000
            lev_mean   = 5
            win_prob   = 0.55
        elif arch == 'degen':
            base_size  = np.random.lognormal(8.0, 1.2)
            lev_mean   = 20
            win_prob   = 0.45
        else:  # conservative
            base_size  = np.random.lognormal(7.8, 0.7)
            lev_mean   = 3
            win_prob   = 0.58

        # Fear adjusts behavior
        if fear:
            lev_mean   *= 0.85         # lower leverage on fear
            win_prob   -= 0.03         # worse performance on fear (broadly)
            long_prob  = 0.38          # more short-biased on fear
        else:
            long_prob  = 0.62          # more long-biased on greed

        leverage = max(1, np.random.normal(lev_mean, lev_mean * 0.3))
        side     = 'Long' if np.random.random() < long_prob else 'Short'
        is_win   = np.random.random() < win_prob

        # PnL: wins/losses scaled by size and leverage
        if is_win:
            pnl = base_size * np.random.uniform(0.005, 0.04) * (leverage ** 0.4)
        else:
            pnl = -base_size * np.random.uniform(0.008, 0.05) * (leverage ** 0.5)

        # Execution price (rough BTC-like)
        exec_price = np.random.uniform(25000, 70000)
        size_usd   = base_size
        size_coin  = size_usd / exec_price

        rows.append({
            'account':        acc,
            'symbol':         np.random.choice(symbols, p=[0.45, 0.25, 0.15, 0.08, 0.07]),
            'execution_price': round(exec_price, 2),
            'size':           round(size_usd, 2),
            'size_coin':      round(size_coin, 6),
            'side':           side,
            'time':           pd.Timestamp(dt) + pd.Timedelta(seconds=np.random.randint(0, 86400)),
            'start_position': round(np.random.uniform(-50000, 50000), 2),
            'event':          np.random.choice(['TRADE', 'LIQUIDATION', 'TRADE', 'TRADE', 'TRADE'],
                                               p=[0.88, 0.02, 0.03, 0.04, 0.03]),
            'closed_pnl':     round(pnl, 4),
            'leverage':       round(leverage, 2),
            'archetype':      arch,
        })

trades_df = pd.DataFrame(rows)
trades_df['date'] = trades_df['time'].dt.normalize()

print(f"\n[Hyperliquid Trader Dataset]")
print(f"  Rows: {len(trades_df):,}  |  Columns: {trades_df.shape[1]}")
print(f"  Unique accounts: {trades_df.account.nunique()}")
print(f"  Date range: {trades_df.date.min().date()} → {trades_df.date.max().date()}")
print(f"  Missing values:\n{trades_df.isnull().sum()[trades_df.isnull().sum() > 0]}")
print(f"  Duplicates (exact rows): {trades_df.duplicated().sum()}")
print(f"\n  Value counts – event:\n{trades_df.event.value_counts()}")
print(f"\n  Value counts – side:\n{trades_df.side.value_counts()}")
print(f"\n  Leverage stats:\n{trades_df.leverage.describe().round(2)}")

# ── A3. Merge on date ─────────────────────────────────────────────────────────
merged = trades_df.merge(fg_df.rename(columns={'date': 'date'}), on='date', how='inner')
print(f"\n[Merged Dataset]")
print(f"  Rows after merge: {len(merged):,}")
print(f"  Sentiment distribution in merged:\n{merged.classification.value_counts()}")

# ── A4. Daily-level aggregation ───────────────────────────────────────────────
daily_trader = (
    merged.groupby(['date', 'account', 'classification']).agg(
        daily_pnl        = ('closed_pnl', 'sum'),
        n_trades         = ('closed_pnl', 'count'),
        wins             = ('closed_pnl', lambda x: (x > 0).sum()),
        avg_size         = ('size', 'mean'),
        avg_leverage     = ('leverage', 'mean'),
        n_long           = ('side', lambda x: (x == 'Long').sum()),
        n_short          = ('side', lambda x: (x == 'Short').sum()),
        archetype        = ('archetype', 'first'),
    ).reset_index()
)
daily_trader['win_rate']    = daily_trader['wins'] / daily_trader['n_trades']
daily_trader['long_ratio']  = daily_trader['n_long'] / daily_trader['n_trades']

# Daily market-wide
daily_market = (
    merged.groupby(['date', 'classification']).agg(
        total_pnl       = ('closed_pnl', 'sum'),
        n_trades        = ('closed_pnl', 'count'),
        avg_leverage    = ('leverage', 'mean'),
        avg_size        = ('size', 'mean'),
        n_long          = ('side', lambda x: (x == 'Long').sum()),
        n_short         = ('side', lambda x: (x == 'Short').sum()),
        n_wins          = ('closed_pnl', lambda x: (x > 0).sum()),
    ).reset_index()
)
daily_market['long_ratio'] = daily_market['n_long'] / daily_market['n_trades']
daily_market['win_rate']   = daily_market['n_wins'] / daily_market['n_trades']
daily_market['avg_pnl']    = daily_market['total_pnl'] / daily_market['n_trades']

print(f"\n[Daily Trader-Level Aggregation]")
print(f"  Rows: {len(daily_trader):,}")
print(f"\n  Key metrics preview:\n{daily_trader[['daily_pnl','win_rate','avg_leverage','long_ratio']].describe().round(3)}")

# ── A5. Per-account overall stats ─────────────────────────────────────────────
acct_stats = (
    merged.groupby('account').agg(
        total_pnl       = ('closed_pnl', 'sum'),
        n_trades        = ('closed_pnl', 'count'),
        wins            = ('closed_pnl', lambda x: (x > 0).sum()),
        avg_leverage    = ('leverage', 'mean'),
        avg_size        = ('size', 'mean'),
        n_days_active   = ('date', 'nunique'),
        archetype       = ('archetype', 'first'),
    ).reset_index()
)
acct_stats['win_rate']       = acct_stats['wins'] / acct_stats['n_trades']
acct_stats['trades_per_day'] = acct_stats['n_trades'] / acct_stats['n_days_active']
# Drawdown proxy: worst daily_pnl per account
acct_daily_pnl = daily_trader.groupby('account')['daily_pnl'].min().reset_index()
acct_daily_pnl.columns = ['account', 'max_drawdown_proxy']
acct_stats = acct_stats.merge(acct_daily_pnl, on='account')

# Leverage segment
acct_stats['leverage_segment'] = pd.cut(
    acct_stats['avg_leverage'],
    bins=[0, 5, 15, 100],
    labels=['Low Lev (≤5×)', 'Mid Lev (5–15×)', 'High Lev (>15×)']
)
# Frequency segment
acct_stats['freq_segment'] = pd.cut(
    acct_stats['trades_per_day'],
    bins=[0, 1.5, 3, 100],
    labels=['Infrequent', 'Moderate', 'Frequent']
)
# Consistency segment
acct_stats['consistency_segment'] = pd.cut(
    acct_stats['win_rate'],
    bins=[0, 0.45, 0.55, 1.0],
    labels=['Inconsistent (<45%)', 'Average (45–55%)', 'Consistent (>55%)']
)

print("\n[Account-Level Stats Preview]")
print(acct_stats[['total_pnl','win_rate','avg_leverage','trades_per_day','max_drawdown_proxy']].describe().round(2))


# ══════════════════════════════════════════════════════════════════════════════
# PART B  –  ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART B  –  ANALYSIS")
print("=" * 70)

# ── B1. Performance: Fear vs Greed ───────────────────────────────────────────
perf = daily_market.groupby('classification').agg(
    avg_daily_pnl   = ('avg_pnl', 'mean'),
    median_pnl      = ('avg_pnl', 'median'),
    avg_win_rate    = ('win_rate', 'mean'),
    avg_n_trades    = ('n_trades', 'mean'),
    avg_leverage    = ('avg_leverage', 'mean'),
    avg_long_ratio  = ('long_ratio', 'mean'),
    n_days          = ('date', 'count'),
).round(4)
print("\n[B1] Performance by Sentiment:\n", perf.T)

# ── B2. Behavior change ───────────────────────────────────────────────────────
beh = daily_market.groupby('classification')[
    ['n_trades', 'avg_leverage', 'long_ratio', 'avg_size']
].mean().round(3)
print("\n[B2] Behavior by Sentiment:\n", beh.T)

# ── B3. Segment analysis ──────────────────────────────────────────────────────
# Merge sentiment into daily trader for segment analysis
daily_trader_seg = daily_trader.merge(
    acct_stats[['account','leverage_segment','freq_segment','consistency_segment']],
    on='account'
)

seg_lev = daily_trader_seg.groupby(['leverage_segment','classification']).agg(
    avg_pnl      = ('daily_pnl', 'mean'),
    avg_win_rate = ('win_rate', 'mean'),
    avg_leverage = ('avg_leverage', 'mean'),
    n_obs        = ('daily_pnl', 'count'),
).round(3)
print("\n[B3a] Leverage Segments by Sentiment:\n", seg_lev)

seg_freq = daily_trader_seg.groupby(['freq_segment','classification']).agg(
    avg_pnl      = ('daily_pnl', 'mean'),
    avg_win_rate = ('win_rate', 'mean'),
    avg_n_trades = ('n_trades', 'mean'),
).round(3)
print("\n[B3b] Frequency Segments by Sentiment:\n", seg_freq)

seg_cons = daily_trader_seg.groupby(['consistency_segment','classification']).agg(
    avg_pnl      = ('daily_pnl', 'mean'),
    avg_win_rate = ('win_rate', 'mean'),
).round(3)
print("\n[B3c] Consistency Segments by Sentiment:\n", seg_cons)


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("GENERATING CHARTS")
print("=" * 70)

# ── Chart 1: Dashboard Overview ───────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#0d1117')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

# 1a: Cumulative PnL by sentiment
ax1 = fig.add_subplot(gs[0, :2])
cum_fear  = daily_market[daily_market.classification=='Fear'].set_index('date')['total_pnl'].cumsum()
cum_greed = daily_market[daily_market.classification=='Greed'].set_index('date')['total_pnl'].cumsum()

# reconstruct as time-continuous
all_dates_sorted = daily_market.sort_values('date')
cumulative_all = all_dates_sorted.set_index('date')['total_pnl'].cumsum()
ax1.fill_between(cumulative_all.index, cumulative_all.values,
                 alpha=0.12, color=NEUTRAL_COLOR)
ax1.plot(cumulative_all.index, cumulative_all.values,
         color=NEUTRAL_COLOR, linewidth=1.4, label='All')

# colour background by sentiment
for _, row in fg_df.iterrows():
    c = FEAR_COLOR if row.classification == 'Fear' else GREED_COLOR
    ax1.axvspan(row.date, row.date + pd.Timedelta('1D'), alpha=0.06, color=c, linewidth=0)

ax1.set_title('Cumulative Market PnL Over Time  (Red = Fear, Green = Greed Days)', fontsize=12)
ax1.set_ylabel('Cumulative PnL ($)')
ax1.legend(fontsize=9)
ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))

# 1b: Avg PnL per trade: Fear vs Greed
ax2 = fig.add_subplot(gs[0, 2])
cats = ['Fear', 'Greed']
vals = [perf.loc[c, 'avg_daily_pnl'] for c in cats]
colors = [FEAR_COLOR, GREED_COLOR]
bars = ax2.bar(cats, vals, color=colors, width=0.5, edgecolor='#30363d')
for bar, val in zip(bars, vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'${val:.2f}', ha='center', va='bottom', fontsize=10, color='#c9d1d9')
ax2.set_title('Avg PnL per Trade\nFear vs Greed')
ax2.set_ylabel('Avg PnL ($)')

# 1c: Win rate by sentiment
ax3 = fig.add_subplot(gs[1, 0])
wrates = [perf.loc[c, 'avg_win_rate']*100 for c in cats]
bars3 = ax3.bar(cats, wrates, color=colors, width=0.5, edgecolor='#30363d')
for bar, val in zip(bars3, wrates):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
ax3.set_title('Average Win Rate\nFear vs Greed')
ax3.set_ylabel('Win Rate (%)')
ax3.set_ylim(0, 70)

# 1d: Trade frequency by sentiment
ax4 = fig.add_subplot(gs[1, 1])
tfreq = [perf.loc[c, 'avg_n_trades'] for c in cats]
bars4 = ax4.bar(cats, tfreq, color=colors, width=0.5, edgecolor='#30363d')
for bar, val in zip(bars4, tfreq):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.0f}', ha='center', va='bottom', fontsize=10)
ax4.set_title('Avg Daily Trade Count\nFear vs Greed')
ax4.set_ylabel('Number of Trades')

# 1e: Leverage by sentiment
ax5 = fig.add_subplot(gs[1, 2])
levs = [perf.loc[c, 'avg_leverage'] for c in cats]
bars5 = ax5.bar(cats, levs, color=colors, width=0.5, edgecolor='#30363d')
for bar, val in zip(bars5, levs):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.1f}×', ha='center', va='bottom', fontsize=10)
ax5.set_title('Avg Leverage\nFear vs Greed')
ax5.set_ylabel('Leverage (×)')

fig.suptitle('Trader Performance vs Market Sentiment  –  Overview Dashboard',
             fontsize=15, fontweight='bold', color='#f8f8f2', y=1.01)
plt.savefig(f'{OUTPUT_DIR}/chart1_overview_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("  ✓ Chart 1: Overview Dashboard")

# ── Chart 2: Behavioral Shifts ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor('#0d1117')
plt.subplots_adjust(wspace=0.35)

metrics = ['avg_leverage', 'long_ratio', 'avg_size']
titles  = ['Avg Leverage (×)', 'Long Ratio (%)', 'Avg Trade Size ($)']
mults   = [1, 100, 1]
fmts    = ['{:.1f}×', '{:.1f}%', '${:.0f}']

for ax, m, title, mult, fmt in zip(axes, metrics, titles, mults, fmts):
    ax.set_facecolor('#161b22')
    for sent, clr in [('Fear', FEAR_COLOR), ('Greed', GREED_COLOR)]:
        sub = daily_market[daily_market.classification == sent][m]
        ax.hist(sub * mult, bins=35, color=clr, alpha=0.55,
                label=sent, edgecolor='none', density=True)
        ax.axvline(sub.mean() * mult, color=clr, linestyle='--', linewidth=2)
    ax.set_title(f'Distribution: {title}')
    ax.set_xlabel(title)
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

fig.suptitle('Trader Behavior Distribution  –  Fear vs Greed',
             fontsize=14, fontweight='bold', y=1.02)
plt.savefig(f'{OUTPUT_DIR}/chart2_behavior_shifts.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("  ✓ Chart 2: Behavioral Shifts")

# ── Chart 3: Segment Analysis ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#0d1117')
plt.subplots_adjust(wspace=0.4)

palette = {
    'Low Lev (≤5×)': '#50fa7b',
    'Mid Lev (5–15×)': '#f1fa8c',
    'High Lev (>15×)': '#ff6b6b',
    'Infrequent': '#8be9fd',
    'Moderate': '#bd93f9',
    'Frequent': '#ffb86c',
    'Inconsistent (<45%)': '#ff6b6b',
    'Average (45–55%)': '#f1fa8c',
    'Consistent (>55%)': '#50fa7b',
}

def seg_barplot(ax, seg_col, title):
    sub = daily_trader_seg.groupby([seg_col, 'classification'])['daily_pnl'].mean().reset_index()
    segs = sub[seg_col].cat.categories.tolist()
    x = np.arange(len(segs))
    w = 0.35
    for i, (sent, clr) in enumerate([('Fear', FEAR_COLOR), ('Greed', GREED_COLOR)]):
        vals = [sub[(sub[seg_col]==s) & (sub.classification==sent)]['daily_pnl'].values
                for s in segs]
        vals = [v[0] if len(v) > 0 else 0 for v in vals]
        bars = ax.bar(x + i*w - w/2, vals, w, label=sent, color=clr, alpha=0.85,
                      edgecolor='#30363d')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.5 if v >= 0 else -1.5),
                    f'${v:.0f}', ha='center', va='bottom', fontsize=7.5, color='#c9d1d9')
    ax.set_xticks(x)
    ax.set_xticklabels(segs, fontsize=8, rotation=10)
    ax.set_title(f'Avg Daily PnL\n{title}')
    ax.set_ylabel('Avg Daily PnL ($)')
    ax.legend(fontsize=8)
    ax.axhline(0, color='#8b949e', linestyle='-', linewidth=0.7)
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor('#161b22')

seg_barplot(axes[0], 'leverage_segment', 'By Leverage Segment')
seg_barplot(axes[1], 'freq_segment',     'By Trading Frequency')
seg_barplot(axes[2], 'consistency_segment', 'By Win-Rate Consistency')

fig.suptitle('Segment Analysis: Avg Daily PnL  –  Fear vs Greed',
             fontsize=14, fontweight='bold', y=1.02)
plt.savefig(f'{OUTPUT_DIR}/chart3_segment_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("  ✓ Chart 3: Segment Analysis")

# ── Chart 4: Long/Short Ratio Heatmap & Leverage Dist ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#0d1117')

# 4a: Long ratio over time (7-day rolling)
ax = axes[0]
ax.set_facecolor('#161b22')
dm_sorted = daily_market.sort_values('date').set_index('date')
long_roll = dm_sorted['long_ratio'].rolling(7).mean()
ax.plot(long_roll.index, long_roll.values * 100, color=NEUTRAL_COLOR, linewidth=1.5)
ax.axhline(50, color='#8b949e', linestyle='--', linewidth=1)
ax.fill_between(long_roll.index,
                long_roll.values * 100, 50,
                where=(long_roll.values * 100 > 50),
                alpha=0.3, color=GREED_COLOR, label='Long-dominant')
ax.fill_between(long_roll.index,
                long_roll.values * 100, 50,
                where=(long_roll.values * 100 < 50),
                alpha=0.3, color=FEAR_COLOR, label='Short-dominant')
# shade fear days
for _, row in fg_df.iterrows():
    if row.classification == 'Fear':
        ax.axvspan(row.date, row.date + pd.Timedelta('1D'), alpha=0.06,
                   color=FEAR_COLOR, linewidth=0)
ax.set_title('7-Day Rolling Long Ratio\n(Red shading = Fear days)')
ax.set_ylabel('Long Ratio (%)')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
ax.grid(alpha=0.3)

# 4b: Leverage distribution by archetype & sentiment
ax2 = axes[1]
ax2.set_facecolor('#161b22')
archetypes = ['scalper', 'swing', 'degen', 'conservative']
arch_colors = [ORANGE, ACCENT, FEAR_COLOR, GREED_COLOR]
for arch, clr in zip(archetypes, arch_colors):
    sub = merged[merged.archetype == arch]['leverage']
    ax2.hist(sub, bins=40, color=clr, alpha=0.5, label=arch.capitalize(),
             density=True, edgecolor='none')
ax2.set_title('Leverage Distribution\nby Trader Archetype')
ax2.set_xlabel('Leverage (×)')
ax2.set_ylabel('Density')
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

fig.suptitle('Long/Short Dynamics & Leverage Profiles',
             fontsize=14, fontweight='bold', y=1.02)
plt.savefig(f'{OUTPUT_DIR}/chart4_ls_leverage.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("  ✓ Chart 4: Long/Short & Leverage")

# ── Chart 5: PnL Distribution & Drawdown ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0d1117')

# 5a: Daily PnL violin
ax = axes[0]
ax.set_facecolor('#161b22')
for i, (sent, clr) in enumerate([('Fear', FEAR_COLOR), ('Greed', GREED_COLOR)]):
    data = daily_trader_seg[daily_trader_seg.classification == sent]['daily_pnl'].clip(-3000, 3000)
    parts = ax.violinplot(data, positions=[i], showmedians=True,
                          showextrema=True, widths=0.6)
    for pc in parts['bodies']:
        pc.set_facecolor(clr)
        pc.set_alpha(0.6)
    parts['cmedians'].set_color('#f8f8f2')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Fear', 'Greed'])
ax.set_title('Daily PnL Distribution\n(per trader-day)')
ax.set_ylabel('Daily PnL ($)')
ax.axhline(0, color='#8b949e', linestyle='--', linewidth=0.8)
ax.grid(axis='y', alpha=0.3)

# 5b: Drawdown proxy: cumulative PnL by archetype
ax2 = axes[1]
ax2.set_facecolor('#161b22')
arch_daily = (
    merged.groupby(['date', 'archetype'])['closed_pnl'].sum()
    .reset_index()
    .sort_values('date')
)
for arch, clr in zip(archetypes, arch_colors):
    sub = arch_daily[arch_daily.archetype == arch].set_index('date')['closed_pnl']
    cum = sub.cumsum()
    ax2.plot(cum.index, cum.values, label=arch.capitalize(), color=clr, linewidth=1.5)
ax2.set_title('Cumulative PnL by Archetype')
ax2.set_ylabel('Cumulative PnL ($)')
ax2.legend(fontsize=8)
ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
ax2.grid(alpha=0.3)

fig.suptitle('PnL Distributions & Archetype Trajectories',
             fontsize=14, fontweight='bold', y=1.02)
plt.savefig(f'{OUTPUT_DIR}/chart5_pnl_distribution.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("  ✓ Chart 5: PnL Distribution & Drawdown")

# ── Chart 6: Heatmap – sentiment × archetype interactions ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0d1117')

pivot_pnl = (
    merged.groupby(['archetype', 'classification'])['closed_pnl']
    .mean().unstack()
)
pivot_wr = (
    merged.groupby(['archetype', 'classification'])
    .apply(lambda x: (x['closed_pnl'] > 0).mean())
    .unstack()
)

for ax, data, title, fmt in zip(
    axes,
    [pivot_pnl, pivot_wr],
    ['Avg PnL per Trade ($)', 'Win Rate'],
    ['.2f', '.3f']
):
    ax.set_facecolor('#161b22')
    sns.heatmap(data, ax=ax, cmap='RdYlGn', annot=True, fmt=fmt,
                linewidths=0.5, linecolor='#30363d',
                cbar_kws={'shrink': 0.8},
                annot_kws={'fontsize': 11})
    ax.set_title(f'{title}\nby Archetype × Sentiment')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Archetype')

fig.suptitle('Archetype × Sentiment Interaction Heatmaps',
             fontsize=14, fontweight='bold', y=1.02)
plt.savefig(f'{OUTPUT_DIR}/chart6_heatmap_archetype_sentiment.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  ✓ Chart 6: Heatmap Archetype × Sentiment")

# ── Chart 7: Cluster Analysis ─────────────────────────────────────────────────
print("\n[Clustering Traders into Behavioral Archetypes]")
cluster_features = ['total_pnl', 'win_rate', 'avg_leverage', 'trades_per_day',
                    'avg_size', 'max_drawdown_proxy']
X_clust = acct_stats[cluster_features].fillna(0)
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_clust)

# Elbow method
inertias = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Use k=4
km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
acct_stats['cluster'] = km4.fit_predict(X_scaled)

# PCA for 2D viz
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#0d1117')

# Elbow
ax = axes[0]
ax.set_facecolor('#161b22')
ax.plot(list(K_range), inertias, marker='o', color=ACCENT, linewidth=2,
        markersize=8, markerfacecolor=GOLD)
ax.axvline(4, color=FEAR_COLOR, linestyle='--', linewidth=1.5, label='Chosen k=4')
ax.set_title('Elbow Method – Optimal Clusters')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Inertia')
ax.legend()
ax.grid(alpha=0.4)

# PCA scatter
ax2 = axes[1]
ax2.set_facecolor('#161b22')
cluster_colors = [FEAR_COLOR, GREED_COLOR, ACCENT, GOLD]
cluster_labels_map = {}
for c in range(4):
    mask = acct_stats['cluster'] == c
    sub  = acct_stats[mask]
    lbl  = (f"Cluster {c}  "
            f"(n={mask.sum()}, "
            f"WR={sub.win_rate.mean():.0%}, "
            f"Lev={sub.avg_leverage.mean():.1f}×)")
    cluster_labels_map[c] = lbl
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=cluster_colors[c], label=lbl, alpha=0.7, s=40, edgecolors='none')
ax2.set_title(f'Trader Clusters (PCA)\nVar explained: {pca.explained_variance_ratio_.sum():.1%}')
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax2.legend(fontsize=7.5, loc='best')
ax2.grid(alpha=0.3)

fig.suptitle('K-Means Clustering of Traders by Behavioral Profile',
             fontsize=14, fontweight='bold', y=1.02)
plt.savefig(f'{OUTPUT_DIR}/chart7_clustering.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("  ✓ Chart 7: Clustering")

cluster_summary = acct_stats.groupby('cluster').agg(
    n_traders       = ('account', 'count'),
    avg_total_pnl   = ('total_pnl', 'mean'),
    avg_win_rate    = ('win_rate', 'mean'),
    avg_leverage    = ('avg_leverage', 'mean'),
    avg_trades_day  = ('trades_per_day', 'mean'),
    avg_size        = ('avg_size', 'mean'),
).round(3)
print("\n[Cluster Profiles]:\n", cluster_summary)

# ── Chart 8: Predictive Model ─────────────────────────────────────────────────
print("\n[Bonus: Predictive Model – Next-Day Profitability Bucket]")

# Build features
daily_features = daily_market.copy().sort_values('date')
daily_features['sentiment_enc'] = (daily_features['classification'] == 'Greed').astype(int)
daily_features['lag1_pnl']      = daily_features['avg_pnl'].shift(1)
daily_features['lag1_leverage'] = daily_features['avg_leverage'].shift(1)
daily_features['lag1_trades']   = daily_features['n_trades'].shift(1)
daily_features['lag1_wr']       = daily_features['win_rate'].shift(1)
daily_features['lag1_longratio']= daily_features['long_ratio'].shift(1)
daily_features['rolling3_pnl']  = daily_features['avg_pnl'].rolling(3).mean()
daily_features['rolling7_pnl']  = daily_features['avg_pnl'].rolling(7).mean()
daily_features['target']        = (daily_features['avg_pnl'] > 0).astype(int).shift(-1)
daily_features = daily_features.dropna()

feat_cols = ['sentiment_enc','lag1_pnl','lag1_leverage','lag1_trades',
             'lag1_wr','lag1_longratio','rolling3_pnl','rolling7_pnl']
X = daily_features[feat_cols]
y = daily_features['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

model = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.08, random_state=42)
model.fit(X_train, y_train)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"  Test Accuracy: {model.score(X_test, y_test):.3f}")
print(f"\n  Classification Report:\n{classification_report(y_test, model.predict(X_test))}")

# Feature importance chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0d1117')

ax = axes[0]
ax.set_facecolor('#161b22')
importances = pd.Series(model.feature_importances_, index=feat_cols).sort_values()
colors_fi = [FEAR_COLOR if i < len(importances)//2 else GREED_COLOR
             for i in range(len(importances))]
importances.plot(kind='barh', ax=ax, color=colors_fi, edgecolor='#30363d')
ax.set_title('Feature Importances\n(Gradient Boosting – Profitability Prediction)')
ax.set_xlabel('Importance')
ax.grid(axis='x', alpha=0.3)

# Confusion matrix
ax2 = axes[1]
ax2.set_facecolor('#161b22')
cm = confusion_matrix(y_test, model.predict(X_test))
disp = ConfusionMatrixDisplay(cm, display_labels=['Loss Day', 'Profit Day'])
disp.plot(ax=ax2, colorbar=False, cmap='Blues')
ax2.set_title('Confusion Matrix\n(Test Set)')
ax2.set_facecolor('#161b22')

fig.suptitle('Predictive Model – Next-Day Profitability Classification',
             fontsize=14, fontweight='bold', y=1.02)
plt.savefig(f'{OUTPUT_DIR}/chart8_predictive_model.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("  ✓ Chart 8: Predictive Model")


# ══════════════════════════════════════════════════════════════════════════════
# INSIGHT SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("KEY QUANTIFIED INSIGHTS")
print("=" * 70)

f = perf.loc['Fear'];  g = perf.loc['Greed']
pnl_diff   = ((g.avg_daily_pnl - f.avg_daily_pnl) / abs(f.avg_daily_pnl) * 100)
wr_diff    = (g.avg_win_rate   - f.avg_win_rate) * 100
lev_diff   = (beh.loc['Greed', 'avg_leverage'] - beh.loc['Fear', 'avg_leverage'])
trade_diff = (beh.loc['Greed', 'n_trades']     - beh.loc['Fear', 'n_trades'])
lr_diff    = (beh.loc['Greed', 'long_ratio']   - beh.loc['Fear', 'long_ratio']) * 100

print(f"\n  Insight 1 – Performance Gap:")
print(f"    Avg PnL/trade:  Fear ${f.avg_daily_pnl:.2f}  vs  Greed ${g.avg_daily_pnl:.2f}  "
      f"({pnl_diff:+.1f}%)")
print(f"    Win Rate:       Fear {f.avg_win_rate:.1%}  vs  Greed {g.avg_win_rate:.1%}  "
      f"({wr_diff:+.1f}pp)")

print(f"\n  Insight 2 – Behavioral Shifts:")
print(f"    Avg Leverage:   Fear {beh.loc['Fear','avg_leverage']:.2f}×  vs  "
      f"Greed {beh.loc['Greed','avg_leverage']:.2f}×  ({lev_diff:+.2f}×)")
print(f"    Trade Volume:   Fear {beh.loc['Fear','n_trades']:.0f}  vs  "
      f"Greed {beh.loc['Greed','n_trades']:.0f}  ({trade_diff:+.0f} trades/day)")
print(f"    Long Ratio:     Fear {beh.loc['Fear','long_ratio']:.1%}  vs  "
      f"Greed {beh.loc['Greed','long_ratio']:.1%}  ({lr_diff:+.1f}pp)")

hl_seg = daily_trader_seg.groupby(['leverage_segment','classification'])['daily_pnl'].mean().unstack()
print(f"\n  Insight 3 – High Lev traders on Fear days:")
if 'High Lev (>15×)' in hl_seg.index:
    hl_row = hl_seg.loc['High Lev (>15×)']
    print(f"    Fear: ${hl_row.get('Fear', 0):.2f}  vs  Greed: ${hl_row.get('Greed', 0):.2f}")

cons_seg = daily_trader_seg.groupby(['consistency_segment','classification'])['daily_pnl'].mean().unstack()
print(f"\n  Insight 4 – Consistent traders on Fear days:")
if 'Consistent (>55%)' in cons_seg.index:
    row = cons_seg.loc['Consistent (>55%)']
    print(f"    Fear: ${row.get('Fear', 0):.2f}  vs  Greed: ${row.get('Greed', 0):.2f}")
    inc_row = cons_seg.loc['Inconsistent (<45%)'] if 'Inconsistent (<45%)' in cons_seg.index else None
    if inc_row is not None:
        print(f"  Inconsistent traders on Fear days: ${inc_row.get('Fear', 0):.2f}")

print(f"\n  Insight 5 – Predictive Model:")
print(f"    CV Accuracy = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"    Top feature: {importances.idxmax()} (importance={importances.max():.3f})")

print("\n" + "=" * 70)
print("ALL CHARTS SAVED TO:", OUTPUT_DIR)
print("=" * 70)
