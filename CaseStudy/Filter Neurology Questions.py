import pandas as pd

# Load the dataset
df = pd.read_json('C:/Users/asume/Downloads/SummuProject/data/dev.json', lines=True)

# Define relevant keywords for filtering
keywords = ['dementia', 'aphasia', 'MCI', 'mild cognitive impairment', 'cognitive decline']

# Filter the dataset for relevant questions
filtered_df = df[
    df['question'].str.contains('|'.join(keywords), case=False) | 
    df['subject_name'].str.contains('Medicine|Psychiatry|General', case=False, na=False)
]

# Save the filtered questions to a new JSON file
filtered_df.to_json('C:/Users/asume/Downloads/SummuProject/data/filtered_questions.json', orient='records', lines=True)

# Optionally, you can display the filtered questions
print(filtered_df[['question', 'subject_name', 'topic_name']])
