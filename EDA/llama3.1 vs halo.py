import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# List of subjects
subjects = [
    "Anaesthesia", "Anatomy", "Biochemistry", "Dental", "ENT", "Forensic",
    "Medicine", "Microbiology", "Ophthalmology", "Orthopaedics", "Pathology",
    "Pediatrics", "Pharmacology", "Physiology", "Psychiatry", "Radiology",
    "Skin", "Preventive Med.", "Surgery", "General"
]

llama_accuracies = [    0.44, 0.45, 0.43, 0.44, 0.45,0.46, 0.44, 0.45, 0.43, 0.44,0.45, 0.46, 0.44, 0.45, 0.47,0.44, 0.43, 0.44, 0.45, 0.44]
halo_accuracies = [0.60, 0.62, 0.59, 0.60, 0.61, 0.62, 0.64, 0.61, 0.60, 0.59, 0.61, 0.62, 0.60, 0.63, 0.65, 0.62, 0.60, 0.61, 0.62, 0.60]
# Create a DataFrame for accuracy comparison
accuracy_df = pd.DataFrame({
    'Subject': subjects,
    'LLaMa-3.1': llama_accuracies,
    'HALO': halo_accuracies
})

# Melt the DataFrame for use with Seaborn
melted_df = pd.melt(accuracy_df, id_vars=['Subject'], var_name='Model', value_name='Accuracy')

# Set the background color
plt.figure(figsize=(18, 7), facecolor='#f5f5f5')

# Create a custom color palette
palette = {'LLaMa-3.1': '#cdcc2e', 'HALO': '#6dc7cb'}

# Plotting
sns.set(style="whitegrid")

# Create a barplot with custom colors
barplot = sns.barplot(x='Subject', y='Accuracy', hue='Model', data=melted_df, palette=palette, dodge=True)

# Add titles and labels
#plt.title('Accuracy: ChatGPT-3.5 vs HALO Model', fontsize=20, weight='bold')
plt.xlabel('', fontsize=1, weight='bold')
plt.ylabel('Accuracy', fontsize=16, weight='bold')
plt.xticks(rotation=45, ha='right', fontsize=14, weight='bold')
plt.yticks(fontsize=12, weight='bold')

# Adding data labels
for p in barplot.patches:
    barplot.annotate(format(p.get_height(), '.2f'),
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center',
                     xytext=(0, 10),
                     textcoords='offset points',
                     fontsize=10, weight='bold', color='black')

# Customize gridlines
barplot.grid(True, which='both', linestyle='--', linewidth=0.5)
barplot.set_axisbelow(True)

# Customize legend to be below the plot
plt.legend(title='Models', title_fontsize='13', fontsize='12', loc='upper center', bbox_to_anchor=(0.8, 0.3), ncol=2)

# Save the plot as an image file
plt.tight_layout()
plt.savefig('accuracy_comparison.svg', format='svg') 

# Show the plot
plt.show()