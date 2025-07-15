#!/bin/bash
set -e

# itch.io upload script using butler
# Requires butler to be installed: https://itch.io/docs/butler/

# Configuration file
CONFIG_FILE="itch-config"
VERSION_FILE="pyproject.toml"

# Default configuration
ITCH_USER="your-username"
ITCH_GAME="shaderbox"
WINDOWS_CHANNEL="windows"
LINUX_CHANNEL="linux"

# Load configuration from file if it exists
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to extract version from pyproject.toml
get_version() {
    if [ -n "$CUSTOM_VERSION" ]; then
        echo "$CUSTOM_VERSION"
    else
        grep '^version = ' "$VERSION_FILE" | sed 's/version = "\(.*\)"/\1/'
    fi
}

# Check if butler is installed
check_butler() {
    if ! command -v butler &> /dev/null; then
        print_error "butler is not installed!"
        echo
        echo "Please install butler from: https://itch.io/docs/butler/"
        echo
        echo "Quick install:"
        echo "curl -L -o butler.zip https://broth.itch.ovh/butler/linux-amd64/LATEST/archive/default"
        echo "unzip butler.zip"
        echo "chmod +x butler"
        echo "sudo mv butler /usr/local/bin/"
        exit 1
    fi
}

# Check if user is logged in to butler
check_butler_auth() {
    # Try to get butler status - it should show credentials info if logged in
    local status_output=$(butler status 2>&1)

    if echo "$status_output" | grep -q "not logged in\|no credentials\|invalid credentials"; then
        print_error "You are not logged in to butler!"
        echo
        echo "Please login first:"
        echo "butler login"
        exit 1
    fi

    # If we get here, butler is logged in
    print_status "Butler authentication: OK"
}

# Check if distribution files exist
check_distributions() {
    if [ ! -f "dist/shaderbox-windows.zip" ] || [ ! -f "dist/shaderbox-linux.tar.gz" ]; then
        print_error "Distribution files not found!"
        echo
        echo "Please run ./build.sh first to create distribution files."
        exit 1
    fi
}

# Validate configuration
validate_config() {
    if [ "$ITCH_USER" = "your-username" ]; then
        print_error "Please configure your itch.io settings!"
        echo
        echo "Create a 'itch-config' file with your settings:"
        echo "cp itch-config.example itch-config"
        echo "# Then edit itch-config with your username and game slug"
        echo
        echo "Or edit the ITCH_USER variable directly in this script."
        exit 1
    fi
}

# Upload function
upload_to_itch() {
    local file="$1"
    local channel="$2"
    local version="$3"

    print_status "Uploading $file to channel '$channel'..."

    butler push "$file" "$ITCH_USER/$ITCH_GAME:$channel" --userversion "$version"

    if [ $? -eq 0 ]; then
        print_status "✓ Successfully uploaded $file"
    else
        print_error "✗ Failed to upload $file"
        exit 1
    fi
}

# Main function
main() {
    echo "itch.io Upload Script for ShaderBox"
    echo "===================================="
    echo

    # Validate configuration
    validate_config

    # Check prerequisites
    check_butler
    check_butler_auth
    check_distributions

    # Get version
    VERSION=$(get_version)
    if [ -z "$VERSION" ]; then
        print_error "Could not extract version from $VERSION_FILE"
        exit 1
    fi

    print_status "Project: $ITCH_USER/$ITCH_GAME"
    print_status "Version: $VERSION"
    print_status "Windows file: dist/shaderbox-windows.zip"
    print_status "Linux file: dist/shaderbox-linux.tar.gz"
    echo

    # Confirm upload
    read -p "Do you want to upload to itch.io? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Upload cancelled."
        exit 0
    fi

    echo
    print_status "Starting upload..."
    echo

    # Upload Windows version
    upload_to_itch "dist/shaderbox-windows.zip" "$WINDOWS_CHANNEL" "$VERSION"

    # Upload Linux version
    upload_to_itch "dist/shaderbox-linux.tar.gz" "$LINUX_CHANNEL" "$VERSION"

    echo
    print_status "All uploads completed successfully!"
    echo
    print_status "Your game page: https://$ITCH_USER.itch.io/$ITCH_GAME"
    echo
    print_status "Next steps:"
    echo "1. Go to your itch.io game page"
    echo "2. Update the game description and screenshots"
    echo "3. Set pricing and availability"
    echo "4. Publish the game"
}

# Show usage if help is requested
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  --dry-run      Show what would be uploaded without actually uploading"
    echo
    echo "Configuration:"
    echo "  Edit the ITCH_USER and ITCH_GAME variables at the top of this script"
    echo
    echo "Prerequisites:"
    echo "  1. Install butler: https://itch.io/docs/butler/"
    echo "  2. Login to butler: butler login"
    echo "  3. Run ./build.sh to create distribution files"
    echo
    exit 0
fi

# Dry run mode
if [ "$1" = "--dry-run" ]; then
    echo "DRY RUN MODE - No actual uploads will be performed"
    echo
    validate_config
    check_distributions

    VERSION=$(get_version)

    echo "Would upload:"
    echo "  dist/shaderbox-windows.zip -> $ITCH_USER/$ITCH_GAME:$WINDOWS_CHANNEL (v$VERSION)"
    echo "  dist/shaderbox-linux.tar.gz -> $ITCH_USER/$ITCH_GAME:$LINUX_CHANNEL (v$VERSION)"
    echo
    exit 0
fi

# Run main function
main
