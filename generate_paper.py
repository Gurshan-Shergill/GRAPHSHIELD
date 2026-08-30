import os
import matplotlib.pyplot as plt

def generate_report_figures():
    """
    Generates high-resolution vector/PNG charts locally to avoid broken image URLs.
    """
    os.makedirs("generated_figures", exist_ok=True)

    # Figure 1: Real-time Data Ingestion Architecture Pipeline
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.text(0.15, 0.5, 'Event Ingestion', bbox=dict(boxstyle='round,pad=0.5', fc='#E8EEFF', ec='#357ABD'), ha='center', va='center', fontweight='bold')
    ax.annotate('', xy=(0.4, 0.5), xytext=(0.28, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.55, 0.5, 'Stream Processor', bbox=dict(boxstyle='round,pad=0.5', fc='#E8EEFF', ec='#357ABD'), ha='center', va='center', fontweight='bold')
    ax.annotate('', xy=(0.78, 0.5), xytext=(0.68, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.90, 0.5, 'Data Warehouse', bbox=dict(boxstyle='round,pad=0.5', fc='#E8EEFF', ec='#357ABD'), ha='center', va='center', fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("generated_figures/figure_1_pipeline.png", dpi=300)
    plt.close()

    # Figure 4: Conversion Funnel Dynamic
    stages = ['Stage 1: Visitors\n(100k)', 'Stage 2: Engagement\n(52k)', 'Stage 3: Checkout\n(24k)', 'Stage 4: Purchase\n(10k)']
    units = [100000, 52000, 24000, 10000]
    colors = ['#1F3A5F', '#2A5B8C', '#357ABD', '#4A90E2']

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(stages[::-1], units[::-1], color=colors[::-1], height=0.55)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1000, bar.get_y() + bar.get_height()/2, f'{width:,}', va='center', fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title('Four-Stage Conversion Funnel Dynamics', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("generated_figures/figure_4_funnel.png", dpi=300)
    plt.close()

    # Figure 12: Strategic Risk Profile
    categories = ['Algo Shift', 'Ad Inflation', 'Churn', 'Latency']
    risk_levels = [3, 3.5, 2, 1]
    risk_colors = ['#E74C3C', '#E74C3C', '#F39C12', '#2ECC71']

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(categories, risk_levels, color=risk_colors, width=0.4)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Low', 'Med', 'High'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title('Strategic Risk Vulnerability Profile', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("generated_figures/figure_12_risk.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_report_figures()
    print("Figures successfully generated in 'generated_figures/' directory.")