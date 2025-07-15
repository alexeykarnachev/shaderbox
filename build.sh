#!/bin/bash
set -e

echo "Building ShaderBox distributions for itch.io..."
echo "================================================"

# Clean and create dist directory
rm -rf dist
mkdir -p dist

# Create temporary build directories
mkdir -p /tmp/shaderbox-build-windows
mkdir -p /tmp/shaderbox-build-linux

# Build Windows distribution
echo "Building Windows distribution..."

# Copy source files to temp Windows dir
cp -r shaderbox /tmp/shaderbox-build-windows/
cp pyproject.toml /tmp/shaderbox-build-windows/
cp uv.lock /tmp/shaderbox-build-windows/
cp .python-version /tmp/shaderbox-build-windows/
[ -f LICENSE ] && cp LICENSE /tmp/shaderbox-build-windows/
[ -f README.md ] && cp README.md /tmp/shaderbox-build-windows/

# Copy Windows scripts
cp scripts/run.bat /tmp/shaderbox-build-windows/
cp scripts/README.md /tmp/shaderbox-build-windows/

# Create Windows archive
cd /tmp
zip -r shaderbox-windows.zip shaderbox-build-windows/
mv shaderbox-windows.zip "$OLDPWD/dist/"
cd - > /dev/null

echo "✓ Windows distribution created: dist/shaderbox-windows.zip"

# Build Linux distribution
echo "Building Linux distribution..."

# Copy source files to temp Linux dir
cp -r shaderbox /tmp/shaderbox-build-linux/
cp pyproject.toml /tmp/shaderbox-build-linux/
cp uv.lock /tmp/shaderbox-build-linux/
cp .python-version /tmp/shaderbox-build-linux/
[ -f LICENSE ] && cp LICENSE /tmp/shaderbox-build-linux/
[ -f README.md ] && cp README.md /tmp/shaderbox-build-linux/

# Copy Linux scripts
cp scripts/run.sh /tmp/shaderbox-build-linux/
cp scripts/README.md /tmp/shaderbox-build-linux/

# Make Linux script executable
chmod +x /tmp/shaderbox-build-linux/run.sh

# Create Linux archive
cd /tmp
tar -czf shaderbox-linux.tar.gz shaderbox-build-linux/
mv shaderbox-linux.tar.gz "$OLDPWD/dist/"
cd - > /dev/null

echo "✓ Linux distribution created: dist/shaderbox-linux.tar.gz"

# Clean up temp directories
rm -rf /tmp/shaderbox-build-windows /tmp/shaderbox-build-linux

echo
echo "Build complete!"
echo "Distribution files created in 'dist/' directory"
echo
echo "Upload instructions:"
echo "- Upload shaderbox-windows.zip for Windows users"
echo "- Upload shaderbox-linux.tar.gz for Linux users"
echo
echo "Users should:"
echo "1. Extract the archive"
echo "2. Run run.bat (Windows) or ./run.sh (Linux) to start the application"
