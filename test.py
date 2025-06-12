import  json
with open("Data/my_list.json", "r") as f:
    loaded_list = json.load(f)

with open("Data/dropbox_links.json", "r") as f:
    loaded_list2 = json.load(f)

first_5 = loaded_list[:5]
print(first_5)
first_55 = loaded_list2[:5]
print(first_55)