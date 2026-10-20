import numpy as np

sales = np.array([
    [120, 125, 130, 128, 135, 140, 138, 145],
    [80, 82, 79, 85, 88, 90, 86, 92],
    [150, 148, 155, 160, 158, 165, 220, 170],
    [60, 65, 62, 58, 64, 63, 61, 66],
    [100, 105, 102, 108, 110, 112, 109, 115],
], dtype=float)
region_names = np.array(["Bishkek", "Osh", "Karakol", "Tokmok", "Jalal-Abad"])

# Validate data dimensions and values.
assert sales.shape == (region_names.size, 8)
assert np.all(np.isfinite(sales)) and np.all(sales >= 0)

# Tasks 1–2: aggregate over the eight weekly observations in each row.
total = np.sum(sales, axis=1)
weekly_mean = np.mean(sales, axis=1)
weekly_median = np.median(sales, axis=1)
std = np.std(sales, axis=1, ddof=0)
q1 = np.percentile(sales, 25, axis=1)
q3 = np.percentile(sales, 75, axis=1)
iqr = q3 - q1
cv = std / weekly_mean * 100

# Task 3: Tukey IQR fences and a vectorized outlier mask.
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
outlier_mask = (sales < lower[:, None]) | (sales > upper[:, None])
outlier_region_idx, outlier_week_idx = np.where(outlier_mask)
outlier_values = sales[outlier_region_idx, outlier_week_idx]

# Task 4: revenue shares.
store_total = np.sum(total)
assert store_total != 0
share = total / store_total * 100

# Task 5: compare mean weekly sales in the first and last four weeks.
first_mean = np.mean(sales[:, :4], axis=1)
last_mean = np.mean(sales[:, 4:], axis=1)
change_abs = last_mean - first_mean
change_pct = np.divide(
    change_abs,
    first_mean,
    out=np.full_like(change_abs, np.nan),
    where=first_mean != 0,
) * 100

# Task 6: exclude flagged values for a sensitivity comparison only.
sales_without_outliers = np.where(outlier_mask, np.nan, sales)
clean_total = np.nansum(sales_without_outliers, axis=1)
clean_mean = np.nanmean(sales_without_outliers, axis=1)

# Correctness checks.
assert np.isclose(total.sum(), sales.sum())
assert np.isclose(share.sum(), 100)
assert outlier_mask.sum() == 1
assert np.isclose(clean_total.sum(), sales.sum() - outlier_values.sum())
assert np.all(change_abs > 0)

# Compact numerical output; full interpretation is in the accompanying report.
print("Region | total | mean | median | std | IQR | CV% | share%")
print(np.column_stack((total, weekly_mean, weekly_median, std, iqr, cv, share)))
print("Region order:", region_names)
print("IQR outlier (region, week, value):",
      region_names[outlier_region_idx], outlier_week_idx + 1, outlier_values)
print("First-half mean:", first_mean)
print("Second-half mean:", last_mean)
print("Absolute change:", change_abs)
print("Percent change:", change_pct)
print("Total revenue before/after outlier exclusion:", total, clean_total)
print("Mean before/after outlier exclusion:", weekly_mean, clean_mean)
print("Store revenue before/after exclusion:", store_total, clean_total.sum())
