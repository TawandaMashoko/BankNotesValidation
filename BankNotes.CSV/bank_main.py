import csv
import random

# Number of rows to generate
num_rows = 10000


# Function to generate a random float between a range
def generate_random_value(min_value, max_value):
    return round(random.uniform(min_value, max_value), 4)


# Generate the data
data = []
for _ in range(num_rows):
    variance = generate_random_value(-10, 10)
    skewness = generate_random_value(-10, 10)
    curtosis = generate_random_value(-10, 10)
    entropy = generate_random_value(-10, 10)
    label = random.choice([0, 1])  # Randomly choose 0 (not counterfeit) or 1 (counterfeit)
    data.append([variance, skewness, curtosis, entropy, label])

# Header for the CSV file
header = ['variance', 'skewness', 'curtosis', 'entropy', 'class']

# Write the data to a CSV file
with open('banknotes_large.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)

print(f"CSV file 'banknotes_large.csv' with {num_rows} rows created successfully.")
