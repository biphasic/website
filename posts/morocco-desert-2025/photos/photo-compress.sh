#!/bin/bash
# compress_photos.sh
# Reduce image size by half and lower JPEG quality

# Create an output folder
mkdir -p compressed

# Loop through all JPG images
for img in *.jpg; do
  # Skip if no JPGs are found
  [ -e "$img" ] || continue

  echo "Processing $img ..."
  
  # Resize to 50% and reduce quality to 75%
  convert "$img" -resize 50% -quality 75 "compressed/$img"
done

echo "✅ Compression complete! Files saved in ./compressed"
