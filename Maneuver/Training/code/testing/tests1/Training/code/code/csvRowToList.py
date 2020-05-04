import csv

csv_file = 'sample.csv'

names = []
description = []
price = []

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        names.append(row.get('name'))
        description.append(row.get('description'))
        price.append(row.get('price'))

print(names)
## ['Apples', 'White Bread', 'Wholemeal Bread']

print(description)
## ['A bag of 3 apples', 'A loaf of white bread', 'A loag of wholemeal bread']

print(price)
## ['1.75', '1.90', '1.45']
