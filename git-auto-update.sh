#!/bin/bash

# Variables
REPO_URL="https://github.com/chiranjeevi-205/chiru.git"  # Replace with your repository URL
REPO_NAME=$(basename -s .git $REPO_URL)  # Extract the repository name from the URL

# Step 1: Check if the repository directory exists
if [ ! -d "$REPO_NAME" ]; then
    echo "Repository directory does not exist. Cloning the repository..."
    git clone $REPO_URL
    if [ $? -eq 0 ]; then
        echo "Repository successfully cloned."
    else
        echo "Error occurred during cloning the repository."
        exit 1
    fi
else
    echo "Repository directory exists. Checking for changes..."
    cd $REPO_NAME

    # Fetch the latest changes from the remote
    git fetch origin main

    # Compare local and remote branches
    LOCAL=$(git rev-parse @)
    REMOTE=$(git rev-parse @{u})

    if [ $LOCAL != $REMOTE ]; then
        echo "Changes detected in the repository. Pulling the latest changes..."
        git pull origin main
        if [ $? -eq 0 ]; then
            echo "Repository successfully updated."
        else
            echo "Error occurred during pulling the repository."
            exit 1
        fi
    fi
fi

echo "Git operation completed."
