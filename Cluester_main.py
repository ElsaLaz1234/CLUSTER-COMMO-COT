#COT features
from Data_.data_loader import COT_data_cat
df_COT = COT_data_cat.copy()
df_COT.index = pd.to_datetime(df_COT.index)
df_COT = df_COT.sort_index()

latest_name = df_COT.groupby('CFTC_Contract_Market_Code')['Market_and_Exchange_Names'].last()
df_COT['Commodity'] = df_COT['CFTC_Contract_Market_Code'].map(latest_name)

#MM_ratio
long_mm = df_COT['M_Money_Positions_Long_All']
short_mm = df_COT['M_Money_Positions_Short_All']
total_mm = (long_mm + short_mm).replace(0, np.nan)

MM_ratio = df_COT[['Commodity']].copy()
MM_ratio['MM_ratio'] = (long_mm - short_mm) / total_mm

#ΔMM_ratio
MM_ratio['ΔMM_ratio'] = MM_ratio.groupby('Commodity')['MM_ratio'].diff(4)

#Construction Objects
# Object 1 - COT positioning cluester
features = ['MM_ratio', 'ΔMM_ratio']
from PARAM_Config import rolling_window_COT

def rolling_zscore(x, window=rolling_window_COT):
    mean = x.rolling(window, min_periods=window).mean()
    std = x.rolling(window, min_periods=window).std()
    return (x - mean) / std.replace(0, np.nan)

z_COT = MM_ratio[['Commodity']].copy()
for f in features:
    z_COT["z_" + f] = MM_ratio.groupby('Commodity')[f].transform(rolling_zscore)

z_COT = z_COT.dropna()

def get_z_t(date):
    z_t = z_COT.loc[date].set_index('Commodity')
    return z_t[['z_MM_ratio', 'z_ΔMM_ratio']]

# Object 2 - return co-movement cluester
from Data_.data_loader import COT_data_p 
from PARAM_Config import rolling_window_r
df_rt = COT_data_p.copy()
df_rt.index = pd.to_datetime(df_rt.index)
df_rt = df_rt.sort_index()

r = np.log(df_rt / df_rt.shift(1))

def get_x_t(date, window=rolling_window_r):
    past = r.loc[:date].tail(window)         
    if len(past) < window or past.isna().any().any():
        return None                          
    x_t = past.iloc[::-1].T                 
    x_t.columns = [f'r_t-{k}' for k in range(window)]
    return x_t

#Hierarchical Cluestering
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from PARAM_Config import k_c

def distance_COT(z_t):
    D = squareform(pdist(z_t.values, metric="euclidean"))
    return pd.DataFrame(D, index=z_t.index, columns=z_t.index)

def distance_ret(x_t):
    rho = x_t.T.corr(method='spearman')
    d = np.sqrt(np.clip(2 * (1 - rho.to_numpy()), 0, None))
    np.fill_diagonal(d, 0)
    return pd.DataFrame(d, index=rho.index, columns=rho.columns)

k_range = range(2, 9)       
coph_threshold = 0.75

def cluster_week(d):
    d_condensed = squareform(d.values, checks=False)
    link = linkage(d_condensed, method='ward')
    labels = fcluster(link, t=k_c, criterion='maxclust')
    coph, _ = cophenet(link, d_condensed)
    sil = {
        k: silhouette_score(d.values, fcluster(link, t=k, criterion='maxclust'), metric='precomputed')
        for k in k_range
        }
    return pd.Series(labels, index=d.index), coph, sil

common_dates = z_COT.index.unique().intersection(r.index)

c_COT, c_ret, diagnostics = {}, {}, []

for date in common_dates:
    z_t = get_z_t(date)
    x_t = get_x_t(date)
    if x_t is None or len(z_t) != 16:
        continue                                

    labels_COT, coph_cot, sil_cot = cluster_week(distance_COT(z_t))
    labels_ret, coph_ret, sil_ret = cluster_week(distance_ret(x_t))

    c_COT[date] = labels_COT
    c_ret[date] = labels_ret

    diagnostics.append({
        'date': date,
        'coph_cot': coph_cot,
        'coph_ret': coph_ret,
        'sil_cot': sil_cot[k_c],
        'sil_ret': sil_ret[k_c],
        'flag_cot': coph_cot < coph_threshold,
        'flag_ret': coph_ret < coph_threshold,
        **{f'sil_cot_k{k}': v for k, v in sil_cot.items()},
        **{f'sil_ret_k{k}': v for k, v in sil_ret.items()},
    })

c_ret = pd.DataFrame(c_ret).T
diagnostics = pd.DataFrame(diagnostics).set_index('date')

#Lead-Lag Test: NW T stat , Lift, Peak lag
import statsmodels.api as sm
from sklearn.metrics import adjusted_rand_score
from PARAM_Config import lags, ref_lag, lag_labels

c_COT = pd.DataFrame(c_COT).T.sort_index()
c_ret = c_ret.sort_index()

cot_to_ticker = {
    'COCOA - ICE FUTURES U.S.': 'CC=F',
    'COFFEE C - ICE FUTURES U.S.': 'KC=F',
    'COPPER- #1 - COMMODITY EXCHANGE INC.': 'HG=F',
    'CORN - CHICAGO BOARD OF TRADE': 'ZC=F',
    'COTTON NO. 2 - ICE FUTURES U.S.': 'CT=F',
    'FEEDER CATTLE - CHICAGO MERCANTILE EXCHANGE': 'GF=F',
    'GASOLINE RBOB - NEW YORK MERCANTILE EXCHANGE': 'RB=F',
    'GOLD - COMMODITY EXCHANGE INC.': 'GC=F',
    'LEAN HOGS - CHICAGO MERCANTILE EXCHANGE': 'HE=F',
    'LIVE CATTLE - CHICAGO MERCANTILE EXCHANGE': 'LE=F',
    'NAT GAS NYME - NEW YORK MERCANTILE EXCHANGE': 'NG=F',
    'SILVER - COMMODITY EXCHANGE INC.': 'SI=F',
    'SOYBEANS - CHICAGO BOARD OF TRADE': 'ZS=F',
    'SUGAR NO. 11 - ICE FUTURES U.S.': 'SB=F',
    'WHEAT-SRW - CHICAGO BOARD OF TRADE': 'ZW=F',
    'WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE': 'CL=F',
}
c_COT = c_COT.rename(columns=cot_to_ticker)
c_ret = c_ret[c_COT.columns]

dates_cl = c_COT.index

max_lag = max(lags)
ari = {}
for i in range(max_lag, len(dates_cl)):         
    t = dates_cl[i]
    ari[t] = {
        h: adjusted_rand_score(c_COT.iloc[i - h].values, c_ret.iloc[i].values)
        for h in lags
    }
ari = pd.DataFrame(ari).T                        

def nw_tstat(series, nw_lags=rolling_window_r):
    y = series.dropna().values
    x = np.ones(len(y))
    res = sm.OLS(y, x).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})
    return res.params[0], res.tvalues[0]

results = []
for h in lags:
    mean_ari, t_ari = nw_tstat(ari[h])
    lift, t_lift = nw_tstat(ari[h] - ari[0])
    results.append({
        'lag': lag_labels[h],
        'h_weeks': h,
        'mean_ari': mean_ari,
        'nw_tstat': t_ari,
        'lift': lift,
        'nw_tstat_lift': t_lift if h != 0 else np.nan,
    })

results = pd.DataFrame(results).set_index('h_weeks')
peak_lag = results['mean_ari'].idxmax()

#Subperiod Analysi - ARI
from PARAM_Config import subperiods, calm_periods, crisis_periods

sub_results = []
for name, (start, end) in subperiods.items():
    ari_sub = ari.loc[start:end]
    row = {
        'period': name,
        'type': 'crisis' if name in crisis_periods else 'calm',
        'n_weeks': len(ari_sub),
    }
    for h in lags:
        row[lag_labels[h]] = ari_sub[h].mean()
    for h in lags[1:]:
        row[f'lift_{lag_labels[h]}'] = ari_sub[h].mean() - ari_sub[0].mean()
    sub_results.append(row)

sub_results = pd.DataFrame(sub_results).set_index('period')
sub_by_type = sub_results.groupby('type').mean(numeric_only=True)

#COT Category - ARI
from PARAM_Config import COT_category

cat_columns = {
    'MM': ('M_Money_Positions_Long_All', 'M_Money_Positions_Short_All'),
    'PM': ('Prod_Merc_Positions_Long_All', 'Prod_Merc_Positions_Short_All'),
    'SD': ('Swap_Positions_Long_All', 'Swap__Positions_Short_All'),
    'OR': ('Other_Rept_Positions_Long_All', 'Other_Rept_Positions_Short_All'),
}

cat_features = df_COT[['Commodity']].copy()
for cat, (col_long, col_short) in cat_columns.items():
    long_x = df_COT[col_long]
    short_x = df_COT[col_short]
    total_x = (long_x + short_x).replace(0, np.nan)
    cat_features[f'{cat}_ratio'] = (long_x - short_x) / total_x
    cat_features[f'Δ{cat}_ratio'] = cat_features.groupby('Commodity')[f'{cat}_ratio'].diff(4)

z_cat = cat_features[['Commodity']].copy()
for f in cat_features.columns.drop('Commodity'):
    z_cat['z_' + f] = cat_features.groupby('Commodity')[f].transform(rolling_zscore)

versions = {
    'M1_MM': ['z_MM_ratio', 'z_ΔMM_ratio'],
    'M2_PM': ['z_PM_ratio', 'z_ΔPM_ratio'],
    'M3_SD': ['z_SD_ratio', 'z_ΔSD_ratio'],
    'M4_OR': ['z_OR_ratio', 'z_ΔOR_ratio'],
    'M5_combined': [f'z_{p}{cat}_ratio' for cat in cat_columns for p in ['', 'Δ']],
}

def cluster_labels(d):
    link = linkage(squareform(d.values, checks=False), method='ward')
    return pd.Series(fcluster(link, t=k_c, criterion='maxclust'), index=d.index)

c_versions = {}
for version, feats in versions.items():
    labels = {}
    for date in c_ret.index:
        z_t = z_cat.loc[date].set_index('Commodity')[feats].dropna()
        if len(z_t) != 16:
            continue
        labels[date] = cluster_labels(distance_COT(z_t))
    c_v = pd.DataFrame(labels).T.reindex(c_ret.index)
    c_versions[version] = c_v.rename(columns=cot_to_ticker)[c_ret.columns]

cat_results = []
for version, c_v in c_versions.items():
    ari_v = {}
    for i in range(max_lag, len(c_ret)):
        if c_v.iloc[i - max_lag:i + 1].isna().any().any():
            continue
        ari_v[c_ret.index[i]] = {
            h: adjusted_rand_score(c_v.iloc[i - h].values, c_ret.iloc[i].values)
            for h in lags
        }
    ari_v = pd.DataFrame(ari_v).T

    for h in lags:
        mean_ari, t_ari = nw_tstat(ari_v[h])
        lift, t_lift = nw_tstat(ari_v[h] - ari_v[0])
        cat_results.append({
            'version': version,
            'lag': lag_labels[h],
            'mean_ari': mean_ari,
            'nw_tstat': t_ari,
            'lift': lift,
            'nw_tstat_lift': t_lift if h != 0 else np.nan,
            'n_weeks': len(ari_v),
        })

cat_results = pd.DataFrame(cat_results).set_index(['version', 'lag'])

#Graphs/Dengramm
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram
from PARAM_Config import sector_c, tickers_label, COT_category

results_dir = Path('Results')
results_dir.mkdir(exist_ok=True)

ticker_sector = {
    'CL=F': 'Energy', 'NG=F': 'Energy', 'RB=F': 'Energy',
    'GC=F': 'Metals', 'SI=F': 'Metals', 'HG=F': 'Metals',
    'ZC=F': 'Grains', 'ZW=F': 'Grains', 'ZS=F': 'Grains',
    'KC=F': 'Softs', 'SB=F': 'Softs', 'CC=F': 'Softs', 'CT=F': 'Softs',
    'LE=F': 'Livestocks', 'HE=F': 'Livestocks', 'GF=F': 'Livestocks',
}
lag_order = [lag_labels[h] for h in lags]

#Denogramm
def plot_dendrogram(d, ax, title):
    tickers_d = [cot_to_ticker.get(x, x) for x in d.index]
    link = linkage(squareform(d.values, checks=False), method='ward')
    dendrogram(link, labels=[tickers_label[t][0] for t in tickers_d],
               ax=ax, leaf_rotation=90, color_threshold=link[-(k_c - 1), 2])
    name_to_sector = {tickers_label[t][0]: ticker_sector[t] for t in tickers_d}
    for lbl in ax.get_xticklabels():
        lbl.set_color(sector_c[name_to_sector[lbl.get_text()]])
    ax.set_title(title, fontsize=10)

t_star = ari[ref_lag].idxmax()
i_star = c_ret.index.get_loc(t_star)

for h in lags:
    date_cot = c_ret.index[i_star - h]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_dendrogram(distance_COT(get_z_t(date_cot)), axes[0],
                    f'COT clusters at t-h = {date_cot.date()}')
    plot_dendrogram(distance_ret(get_x_t(t_star)), axes[1],
                    f'Return clusters at t = {t_star.date()}')
    fig.suptitle(f'{lag_labels[h]} | ARI = {ari.loc[t_star, h]:.3f}', fontsize=12)
    handles = [plt.Line2D([0], [0], color=c, lw=6) for c in sector_c.values()]
    fig.legend(handles, sector_c.keys(), loc='lower center', ncol=5)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(results_dir / f'dendrogram_{lag_labels[h]}.png', dpi=200)
    plt.close(fig)

#Bar charts
def plot_bars(df, title, filename):
    ax = df[lag_order].plot(kind='bar', figsize=(12, 5), width=0.8)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('Mean ARI')
    ax.set_title(title)
    ax.legend(title='Lag')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(results_dir / filename, dpi=200)
    plt.close()

plot_bars(sub_results, 'Mean ARI by subperiod', 'bar_ari_subperiod.png')

cat_mean = cat_results['mean_ari'].unstack('lag')[lag_order]
plot_bars(cat_mean, 'Mean ARI by COT category', 'bar_ari_category.png')

#Heatmaps - t-stats
def plot_heatmap(df, title, filename):
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(df) + 2))
    im = ax.imshow(df.values, cmap='RdBu_r', vmin=-3, vmax=3, aspect='auto')
    ax.set_xticks(range(df.shape[1]), df.columns)
    ax.set_yticks(range(df.shape[0]), df.index)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f'{df.iloc[i, j]:.2f}', ha='center', va='center', fontsize=9)
    fig.colorbar(im, ax=ax, label='Newey-West t-stat')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(results_dir / filename, dpi=200)
    plt.close(fig)

sub_tstats = pd.DataFrame({
    name: {lag_labels[h]: nw_tstat(ari.loc[start:end, h])[1] for h in lags}
    for name, (start, end) in subperiods.items()
}).T[lag_order]
plot_heatmap(sub_tstats, 'NW t-stat of mean ARI by subperiod (indicative)', 'heatmap_tstat_subperiod.png')

cat_tstats = cat_results['nw_tstat'].unstack('lag')[lag_order]
plot_heatmap(cat_tstats, 'NW t-stat of mean ARI by COT category', 'heatmap_tstat_category.png')

#Histogram
s = ari[ref_lag].dropna()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(s, bins=40, color='steelblue', edgecolor='white')
ax.axvline(s.mean(), color='red', linestyle='--', label=f'Mean = {s.mean():.3f}')
ax.axvline(s.median(), color='black', linestyle=':', label=f'Median = {s.median():.3f}')
ax.axvline(0, color='grey', linewidth=0.8)

ax.set_xlabel('Weekly ARI')
ax.set_ylabel('Number of weeks')
ax.set_title(f'Distribution of weekly ARI at {lag_labels[ref_lag]} '
             f'(std = {s.std():.3f}, % > 0 = {(s > 0).mean():.0%})')
ax.legend()

fig.tight_layout()
fig.savefig(results_dir / f'histogram_ari_{lag_labels[ref_lag]}.png', dpi=200)
plt.close(fig)

#Tables
def save_table(df, title, filename, col_width=2.2):
    fig, ax = plt.subplots(figsize=(col_width * (df.shape[1] + 1), 0.5 * len(df) + 1.5))
    ax.axis('off')
    table = ax.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title(title, fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(results_dir / f'{filename}.pdf')
    fig.savefig(results_dir / f'{filename}.png', dpi=200)
    plt.close(fig)

fmt = lambda s: ', '.join(f'{lag_labels[h]}: {v:.4f}' for h, v in s.items())
best_cat = cat_results.xs(lag_labels[ref_lag], level='lag')['lift'].idxmax()

summary = pd.DataFrame({
    'Definition': [
        'Mean ARI between COT and return clusters at lag h',
        'Significance of mean ARI, corrected for serial autocorrelation',
        'Predictive gain relative to the contemporaneous lag h=0',
        'Lag maximizing mean ARI',
        'Signal during crisis vs calm periods (descriptive)',
        'M1_MM (baseline) vs M2_PM vs M3_SD vs M4_OR vs M5_combined',
    ],
    'Result': [
        fmt(results['mean_ari']),
        fmt(results['nw_tstat']),
        fmt(results['lift'].iloc[1:]),
        f'h* = {peak_lag} weeks',
        f"h=0 | calm: {sub_by_type.loc['calm', 'h=0']:.4f}, crisis: {sub_by_type.loc['crisis', 'h=0']:.4f}",
        f"Highest lift at {lag_labels[ref_lag]}: {best_cat}",
    ],
}, index=['Mean ARI_h', 'Newey-West t-stat', 'lift_h', 'Peak lag h*', 'ARI by subperiod', 'ARI by COT category'])
save_table(summary, 'Lead-lag test: summary of results', 'summary_table', col_width=5)

cat_table = cat_results.round(4).reset_index()
cat_table['version'] = cat_table['version'].map(
    lambda v: f"{v} ({COT_category.get(v.split('_')[1], 'Combined')})")
save_table(cat_table.set_index('version'), 'ARI by COT category', 'cat_results_table')

save_table(sub_by_type.round(4), 'ARI by period type: calm vs crisis', 'sub_by_type_table')
