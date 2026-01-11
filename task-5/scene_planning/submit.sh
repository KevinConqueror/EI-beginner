#!/bin/bash

# Check if a filename argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <name>_<id>"
    exit 1
fi

# Assign the first argument to a variable
dirname=$1

# Check if the directory exists
if [ -d "$dirname" ]; then
    read -p "Directory '$dirname' already exists. Do you want to delete it? (y/n): " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -rf "$dirname"
        echo "Directory '$dirname' has been deleted."
    else
        echo "Operation aborted."
        exit 1
    fi
fi

# Create the directory
mkdir "$dirname"
echo "Directory '$dirname' has been created."

# Move the files into the directory
cp -r python_package "$dirname"
echo "Folder 'python_package' has been copied to '$dirname'."

# Copy the 'conf' folder into the directory
cp -r conf "$dirname"
echo "Folder 'conf' has been copied to '$dirname'."

# Copy start_llm.sh and submmit.sh into the directory
cp start_llm.sh env.bash "$dirname"
echo "Files 'start_llm.sh' 'env.bash' have been copied to '$dirname'."

# Create a compressed tar.gz file of the directory
tar -czf "${dirname}.tar.gz" "$dirname"
echo "Directory '$dirname' has been compressed into '${dirname}.tar.gz'."