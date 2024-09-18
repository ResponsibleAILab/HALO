import json

with open('dev.json', 'r') as file:
    content = file.read()

json_objects = content.split('\n')  
data = []
for obj in json_objects:
    if obj.strip(): 
        data.append(json.loads(obj))

df = pd.DataFrame(data)

batch_size = 100  

num_batches = (len(df) + batch_size - 1) // batch_size


for i in range(num_batches):
    start_index = i * batch_size
    end_index = min((i + 1) * batch_size, len(df))
    batch_df = df.iloc[start_index:end_index]
    
    batch_data = batch_df.to_dict(orient='records')
    
    with open(f'dev_batch_{i+1}.json', 'w') as file:
        json.dump(batch_data, file, indent=4)
    
    print(f'Saved batch {i+1} with {end_index - start_index} questions.')
