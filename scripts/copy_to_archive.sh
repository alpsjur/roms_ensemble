#!/bin/bash

# Define variables
START_YEAR=2012      # Starting year
END_YEAR=2021        # Ending year
MONTH="02"           # Month 
DAY="02"             # Day 
SOURCE_DIR="/lustre/storeB/project/fou/hi/roms_hindcast/norkyst_v3/sdepth"
DEST_DIR="/lustre/storeB/users/ansju8054/roms-ting/roms_ensemble/files/archive"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Loop through the specified range of years
for YEAR in $(seq $START_YEAR $END_YEAR); do
    # Construct the file path based on the year, month, and day
    FILE_PATH="$SOURCE_DIR/$YEAR/$MONTH/norkyst800-${YEAR}${MONTH}${DAY}.nc"
    
    # Check if the file exists before copying
    if [ -f "$FILE_PATH" ]; then
        echo "Copying $FILE_PATH to $DEST_DIR"
        cp "$FILE_PATH" "$DEST_DIR"
    else
        echo "File not found: $FILE_PATH"
    fi
done

echo "File copying process completed."