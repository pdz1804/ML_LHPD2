import psutil

# Get available memory in GB
available_memory = psutil.virtual_memory().available / (1024 ** 3)
total_memory = psutil.virtual_memory().total / (1024 ** 3)

print(f"Available Memory: {available_memory:.2f} GB")
print(f"Total Memory: {total_memory:.2f} GB")
