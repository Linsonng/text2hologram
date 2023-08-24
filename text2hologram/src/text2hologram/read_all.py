import os

# Get the current directory
current_dir = os.getcwd()

# Loop through every file in the current directory
for file in os.listdir(current_dir):
    # If the file is a Python file
    if file.endswith('.py'):
        # Open the file
        with open(file, 'r') as f:
            # Read the contents
            contents = f.read()

            # Print the contents
            print(f'--- Contents of {file} ---\n{contents}\n')
