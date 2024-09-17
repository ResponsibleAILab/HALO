import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb, to_hex


llama_accuracies = [0.44, 0.45, 0.43, 0.44, 0.45, 0.46, 0.44, 0.45, 0.43, 0.44, 0.45, 0.46, 0.44, 0.45, 0.47, 0.44, 0.43, 0.44, 0.45, 0.44]
halo_llama_accuracies = [0.60, 0.62, 0.59, 0.60, 0.61, 0.62, 0.64, 0.61, 0.60, 0.59, 0.61, 0.62, 0.60, 0.63, 0.65, 0.62, 0.60, 0.61, 0.62, 0.60]

chatgpt_accuracies = [0.55, 0.55, 0.55, 0.55, 0.56, 0.57, 0.58, 0.56, 0.55, 0.55, 0.56, 0.57, 0.56, 0.57, 0.58, 0.55, 0.56, 0.57, 0.56, 0.55]
halo_chatgpt_accuracies = [0.70, 0.71, 0.69, 0.69, 0.70, 0.71, 0.72, 0.69, 0.70, 0.68, 0.70, 0.71, 0.69, 0.72, 0.74, 0.71, 0.68, 0.69, 0.70, 0.70]

mistral_accuracies = [ 0.37, 0.38, 0.36, 0.37, 0.38, 0.39, 0.37, 0.38, 0.36, 0.37, 0.38, 0.39, 0.37, 0.38, 0.40, 0.37, 0.36, 0.37, 0.38, 0.37]
halo_mistral_accuracies =[0.50, 0.52, 0.49, 0.50, 0.51, 0.52, 0.54, 0.51, 0.50, 0.49, 0.51, 0.52, 0.50, 0.53, 0.55, 0.52, 0.50, 0.51, 0.52, 0.50]

subjects = [
    "Anaesthesia", "Anatomy", "Biochemistry", "Dental", "ENT", "Forensic",
    "Medicine", "Microbiology", "Ophthalmology", "Orthopaedics", "Pathology",
    "Pediatrics", "Pharmacology", "Physiology", "Psychiatry", "Radiology",
    "Skin", "Preventive Med.", "Surgery", "General"
]

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig.patch.set_facecolor('#f5f5f5')

# Function to plot each graph
def plot_graph(ax, data1, data2, label1, label2, title, xlabel=False):
    sns.set(style="whitegrid")
    accuracy_df = pd.DataFrame({
        'Subject': subjects,
        label1: data1,
        label2: data2
    })
    melted_df = pd.melt(accuracy_df, id_vars=['Subject'], var_name='Model', value_name='Accuracy')

    palette = {label1: '#cdcc2e', label2: '#6dc7cb'}
    barplot = sns.barplot(x='Subject', y='Accuracy', hue='Model', data=melted_df, palette=palette, dodge=True, ax=ax)

    def darken_color(color, factor=0.7):
        rgb = to_rgb(color)
        darkened_rgb = [x * factor for x in rgb]
        return to_hex(darkened_rgb)

    patterns = {label1: 'xxx', label2: '//'}
    for bar, model in zip(barplot.patches, melted_df['Model']):
        bar.set_hatch(patterns[model])
        darker_color = darken_color(bar.get_facecolor())
        bar.set_edgecolor(darker_color)

        height = bar.get_height()
        ax.annotate(f'{height:.2f}', (bar.get_x() + bar.get_width() / 2, height),
                    textcoords="offset points", xytext=(0, 10), ha='center', va='bottom',
                    fontsize=10, weight='bold', color='black')

    ax.set_facecolor('white')
    spine_color = '#d9d9d9'
    for spine in ax.spines.values():
        spine.set_color(spine_color)

    ax.set_ylim(0.2, None)
    ax.set_ylabel('Accuracy', fontsize=12, weight='bold')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel('')
    else:
        ax.set_xlabel('')

    ax.legend(title='', title_fontsize='13', fontsize=18, loc='upper center', bbox_to_anchor=(0.8, 0.3), ncol=2 , prop={'weight': 'bold'})

    # Make y-ticks bold
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(12)

plot_graph(axes[0], chatgpt_accuracies, halo_chatgpt_accuracies, 'ChatGPT-3.5', 'HALO + ChatGPT-3.5', 'ChatGPT-3.5 vs HALO')
plot_graph(axes[1], llama_accuracies, halo_llama_accuracies, 'LLaMA-3.1', 'HALO + LLaMA-3.1', 'LLaMA-3.1 vs HALO')
plot_graph(axes[2], mistral_accuracies, halo_mistral_accuracies, 'Mistral 7B', 'HALO + Mistral 7B', 'Mistral 7B vs HALO', xlabel=True)


plt.xticks(rotation=45, ha='right', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig('combined_accuracy_comparison.svg', format='svg', facecolor='#f5f5f5')
plt.show()
