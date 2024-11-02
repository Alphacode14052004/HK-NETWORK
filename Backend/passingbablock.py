import numpy as np
import scipy.stats as stats
from scipy.odr import ODR, Model, RealData
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# Data
mri_data = np.array([
    5.3, 5.6, 4.5, 5.2, 5.7, 5.6, 4.8, 7.3, 5.7, 5.5, 7.1, 5.4, 5.7, 5.5, 4.6,
    5.8, 6.8, 5.3, 5.8, 5.3, 5.9, 4.7, 7.4, 5.5, 5.4, 7.2, 5.5, 5.6, 4.8, 5.5,
    5.9, 6.7, 5.6, 5.8, 5.4, 5.8, 4.6, 6.1, 5.8, 5.5, 5.9, 5.7, 4.5, 6.1, 5.8,
    6.4, 5.2, 4.7, 5.7, 5.8
])

usg_data = np.array([
    5.6, 5.5, 4.5, 5.3, 5.5, 5.4, 4.6, 7.1, 5.5, 5.7, 6.9, 5.5, 5.8, 5.3, 4.4,
    5.6, 6.5, 5.2, 5.6, 5.2, 6.1, 4.5, 7.2, 5.4, 5.2, 7.1, 5.4, 5.3, 4.7, 5.4,
    5.7, 6.6, 5.4, 5.5, 5.3, 5.7, 4.3, 6.3, 5.6, 5.3, 5.7, 5.5, 4.3, 5.8, 5.6,
    6.2, 5.3, 4.5, 5.4, 5.6
])

def passing_bablok(x, y):
    """Compute Passing-Bablok regression."""
    n = len(x)
    slopes = []

    for i in range(n):
        for j in range(i + 1, n):
            if x[i] != x[j]:  # Avoid division by zero
                slope = (y[j] - y[i]) / (x[j] - x[i])
                slopes.append(slope)

    if not slopes:  # Check if slopes list is empty
        return 0, 0  # Default values in case of failure

    slopes.sort()
    b1 = np.median(slopes)
    b0 = np.median(y - b1 * x)

    return b0, b1

def deming_regression(x, y, ratio=1.0):
    """Compute Deming regression using Orthogonal Distance Regression."""
    def f(B, x):
        return B[0] + B[1] * x

    linear = Model(f)
    data = RealData(x, y, sx=np.ones_like(x) * ratio, sy=np.ones_like(y) * ratio)
    odr = ODR(data, linear, beta0=[0., 1.])
    output = odr.run()

    return output.beta[0], output.beta[1]

def equivariant_pb(x, y):
    """Compute Equivariant Passing-Bablok regression."""
    x_std = (x - np.mean(x)) / np.std(x)
    y_std = (y - np.mean(y)) / np.std(y)

    b0_std, b1_std = passing_bablok(x_std, y_std)

    if np.std(x) == 0:  # Prevent division by zero
        return np.mean(y), 0

    b1 = b1_std * (np.std(y) / np.std(x))
    b0 = np.mean(y) - b1 * np.mean(x)

    return b0, b1

def calculate_performance_metrics(x, y, threshold=None):
    """Calculate accuracy and sensitivity (recall)."""
    if threshold is None:
        threshold = np.median(np.concatenate([x, y]))

    x_binary = (x > threshold).astype(int)
    y_binary = (y > threshold).astype(int)

    accuracy = accuracy_score(x_binary, y_binary)
    sensitivity = recall_score(x_binary, y_binary, zero_division=0)

    correlation = np.corrcoef(x, y)[0, 1]
    mape = np.mean(np.abs((x - y) / x)) * 100
    rmse = np.sqrt(np.mean((x - y)**2))

    # Calculate ICC
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    mean_all = (mean_x + mean_y) / 2

    ssb = n * ((mean_x - mean_all)**2 + (mean_y - mean_all)**2)
    ssw = np.sum((x - mean_x)**2) + np.sum((y - mean_y)**2)

    icc = (ssb - ssw) / (ssb + ssw) if (ssb + ssw) != 0 else 0

    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'correlation': correlation,
        'mape': mape,
        'rmse': rmse,
        'icc': icc
    }

def create_comparison_plots(x, y, threshold=None):
    """Create comprehensive comparison plots."""
    pb_b0, pb_b1 = passing_bablok(x, y)
    dem_b0, dem_b1 = deming_regression(x, y)
    epb_b0, epb_b1 = equivariant_pb(x, y)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)

    # Regression plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(x, y, alpha=0.5)
    x_range = np.linspace(min(x), max(x), 100)

    ax1.plot(x_range, pb_b0 + pb_b1 * x_range, 'r-', label='Passing-Bablok')
    ax1.plot(x_range, dem_b0 + dem_b1 * x_range, 'g-', label='Deming')
    ax1.plot(x_range, epb_b0 + epb_b1 * x_range, 'b-', label='Equivariant PB')
    ax1.plot(x_range, x_range, 'k--', label='Identity')

    ax1.set_xlabel('MRI Measurements (cm)')
    ax1.set_ylabel('USG Measurements (cm)')
    ax1.legend()
    ax1.set_title('Method Comparison: MRI vs USG')

    # Bland-Altman plot
    ax2 = fig.add_subplot(gs[0, 1])
    mean = (x + y) / 2
    diff = y - x

    md = np.mean(diff)
    sd = np.std(diff)

    ax2.scatter(mean, diff, alpha=0.5)
    ax2.axhline(md, color='k', linestyle='-', label=f'Mean difference: {md:.3f}')
    ax2.axhline(md + 1.96 * sd, color='k', linestyle='--', 
                label=f'95% limits: {md + 1.96 * sd:.3f}')
    ax2.axhline(md - 1.96 * sd, color='k', linestyle='--', 
                label=f'{md - 1.96 * sd:.3f}')

    ax2.set_xlabel('Mean of MRI and USG (cm)')
    ax2.set_ylabel('Difference (USG - MRI) (cm)')
    ax2.legend()
    ax2.set_title('Bland-Altman Plot')

    # Box plot
    ax3 = fig.add_subplot(gs[1, :])
    box_data = [x, y]
    ax3.boxplot(box_data, labels=['MRI', 'USG'])
    ax3.set_title('Distribution of Measurements')

    # Add individual points
        # Add individual points
    for i, data in enumerate(box_data):
        y_data = np.random.normal(0, 0.04, size=len(data)) + data  # Add some jitter
        ax3.scatter([i + 1] * len(data), y_data, color='orange', alpha=0.5)

    ax3.set_ylabel('Measurement (cm)')
    ax3.set_title('Distribution of MRI and USG Measurements')

    plt.tight_layout()
    plt.show()

# Run comparison plots and calculate performance metrics
performance_metrics = calculate_performance_metrics(mri_data, usg_data)
create_comparison_plots(mri_data, usg_data)

print("Performance Metrics:")
for metric, value in performance_metrics.items():
    print(f"{metric}: {value:.4f}")
