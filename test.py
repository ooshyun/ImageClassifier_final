import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

print(cat_to_name)    