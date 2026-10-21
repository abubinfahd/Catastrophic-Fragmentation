"""Statistical tests added to the notebook's saved per-pair disparity values."""
import itertools

import numpy as np
import pandas as pd
from scipy import stats


def holm(values):
    values = np.asarray(values, dtype=float)
    adjusted = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    previous = 0.
    for position, index in enumerate(finite[np.argsort(values[finite])]):
        previous = max(previous, min(1., (len(finite) - position) * values[index]))
        adjusted[index] = previous
    return adjusted


def confidence_interval(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('At least two finite observations are required')
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / np.sqrt(len(values))
    margin = float(stats.t.ppf(.975, len(values) - 1)) * se
    return dict(N=len(values), Mean=mean, SD=sd, SE=se, CI_Lower=mean-margin, CI_Upper=mean+margin)


def analyze(raw, folder):
    if raw.duplicated(['Domain', 'Model', 'Pair']).any():
        raise ValueError('Duplicate domain/model/pair observations')
    if not np.isfinite(raw['Compute_Disparity']).all():
        raise ValueError('Nonfinite disparity observations')
    intervals, comparisons, domain_tests = [], [], []
    for (domain, model), group in raw.groupby(['Domain', 'Model']):
        intervals.append(dict(Domain=domain, Model=model, **confidence_interval(group.Compute_Disparity)))
    for domain, group in raw.groupby('Domain'):
        wide = group.pivot(index='Pair', columns='Model', values='Compute_Disparity').sort_index()
        if wide.isna().any().any():
            raise ValueError('Model comparisons require measurements of the same pairs')
        for a, b in itertools.combinations(wide.columns, 2):
            difference = np.round(wide[a].to_numpy() - wide[b].to_numpy(), 2)
            t = stats.ttest_rel(wide[a], wide[b])
            w = stats.wilcoxon(difference, method='asymptotic', zero_method='wilcox') if np.any(difference) else None
            identical = not np.any(difference)
            comparisons.append(dict(Domain=domain, Model_A=a, Model_B=b, N=len(wide),
                                    Mean_Difference=float((wide[a]-wide[b]).mean()),
                                    T=float(t.statistic) if not identical else 0.,
                                    T_p=float(t.pvalue) if not identical else 1.,
                                    Wilcoxon_W=float(w.statistic) if w else 0.,
                                    Wilcoxon_p=float(w.pvalue) if w else 1.))
    # Corpus rows are unrelated across domains: use independent Welch tests.
    for model, group in raw.groupby('Model'):
        domains = {name: rows.Compute_Disparity.to_numpy() for name, rows in group.groupby('Domain')}
        for a, b in itertools.combinations(domains, 2):
            result = stats.ttest_ind(domains[a], domains[b], equal_var=False)
            domain_tests.append(dict(Model=model, Domain_A=a, Domain_B=b, Welch_T=float(result.statistic),
                                     Degrees_Of_Freedom=float(result.df), p=float(result.pvalue)))
    intervals = pd.DataFrame(intervals)
    comparisons = pd.DataFrame(comparisons)
    domain_tests = pd.DataFrame(domain_tests)
    for column in ['T_p', 'Wilcoxon_p']:
        if not comparisons.empty:
            comparisons[column + '_Holm'] = holm(comparisons[column])
            comparisons[column + '_Underflow'] = (comparisons[column] == 0) & np.isfinite(comparisons['T'] if column == 'T_p' else comparisons['Wilcoxon_W'])
    if not domain_tests.empty:
        domain_tests['p_Holm'] = holm(domain_tests.p)
        domain_tests['p_Underflow'] = domain_tests.p == 0
    intervals.to_csv(folder / 'confidence_intervals.csv', index=False)
    comparisons.to_csv(folder / 'paired_model_tests.csv', index=False)
    domain_tests.to_csv(folder / 'domain_tests.csv', index=False)
    return intervals, comparisons, domain_tests
