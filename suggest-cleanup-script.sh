#!/bin/bash

# Disk Space Cleanup and Debugging Script
# This script helps identify and clean up disk space issues on Linux systems

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_header() {
    echo -e "\n${BLUE}================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================${NC}"
}

# Function to show disk usage
show_disk_usage() {
    print_header "CURRENT DISK USAGE"
    df -h | grep -E "(Filesystem|/dev/)"
    echo ""
    
    # Check if root partition is > 90% full
    root_usage=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$root_usage" -gt 90 ]; then
        print_error "Root partition is ${root_usage}% full - CRITICAL!"
    elif [ "$root_usage" -gt 80 ]; then
        print_warning "Root partition is ${root_usage}% full - needs attention"
    else
        print_status "Root partition usage is healthy (${root_usage}%)"
    fi
}

# Function to identify large directories
identify_large_dirs() {
    print_header "IDENTIFYING LARGE DIRECTORIES"
    
    print_info "Checking common cache and temporary directories..."
    
    directories=(
        "$HOME/.cache"
        "$HOME/.npm" 
        "$HOME/.local/lib/python*/site-packages"
        "$HOME/.config"
        "$HOME/Downloads"
        "$HOME/.local/share"
        "/tmp"
        "/var/log"
        "/var/cache"
    )
    
    for dir in "${directories[@]}"; do
        if [ -d "$dir" ]; then
            size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            echo "  $dir: $size"
        fi
    done
    
    print_info "\nTop 10 largest directories in home folder:"
    du -sh "$HOME"/* 2>/dev/null | sort -hr | head -10
}

# Function to identify specific large files
identify_large_files() {
    print_header "IDENTIFYING LARGE FILES"
    
    print_info "Searching for files larger than 100MB..."
    find "$HOME" -type f -size +100M -exec ls -lh {} + 2>/dev/null | head -20
    
    print_info "\nLargest log files:"
    find "$HOME" -name "*.log" -type f -exec du -sh {} + 2>/dev/null | sort -hr | head -10
    
    print_info "\nLarge installer files:"
    find "$HOME" \( -name "*.deb" -o -name "*.AppImage" -o -name "*.zip" -o -name "*.tar.gz" \) -type f -exec ls -lh {} + 2>/dev/null | head -10
}

# Function to analyze cache directories in detail
analyze_caches() {
    print_header "CACHE ANALYSIS"
    
    # Check Poetry cache
    if [ -d "$HOME/.cache/pypoetry" ]; then
        poetry_size=$(du -sh "$HOME/.cache/pypoetry" 2>/dev/null | cut -f1)
        print_info "Poetry cache: $poetry_size"
        print_info "  Artifacts: $(find "$HOME/.cache/pypoetry/artifacts" -type f 2>/dev/null | wc -l) files"
        print_info "  Cache: $(find "$HOME/.cache/pypoetry/cache" -type f 2>/dev/null | wc -l) files"
    fi
    
    # Check npm cache
    if [ -d "$HOME/.npm" ]; then
        npm_size=$(du -sh "$HOME/.npm" 2>/dev/null | cut -f1)
        print_info "NPM cache: $npm_size"
    fi
    
    # Check pip cache
    if [ -d "$HOME/.cache/pip" ]; then
        pip_size=$(du -sh "$HOME/.cache/pip" 2>/dev/null | cut -f1)
        print_info "Pip cache: $pip_size"
    fi
    
    # Check config directories
    if [ -d "$HOME/.config" ]; then
        print_info "\nTop config directories:"
        du -sh "$HOME/.config"/* 2>/dev/null | sort -hr | head -10
    fi
}

# Function to perform safe cleanup
perform_cleanup() {
    print_header "SAFE CLEANUP OPTIONS"
    
    total_freed=0
    
    # Clean pip cache
    if command -v pip &> /dev/null; then
        print_info "Cleaning pip cache..."
        before_pip=$(du -sb "$HOME/.cache/pip" 2>/dev/null | cut -f1 || echo "0")
        pip cache purge 2>/dev/null || true
        after_pip=$(du -sb "$HOME/.cache/pip" 2>/dev/null | cut -f1 || echo "0")
        freed_pip=$((before_pip - after_pip))
        if [ $freed_pip -gt 0 ]; then
            print_status "Freed $(numfmt --to=iec $freed_pip) from pip cache"
            total_freed=$((total_freed + freed_pip))
        fi
    fi
    
    # Clean npm cache
    if command -v npm &> /dev/null; then
        print_info "Cleaning npm cache..."
        before_npm=$(du -sb "$HOME/.npm" 2>/dev/null | cut -f1 || echo "0")
        npm cache clean --force 2>/dev/null || true
        after_npm=$(du -sb "$HOME/.npm" 2>/dev/null | cut -f1 || echo "0")
        freed_npm=$((before_npm - after_npm))
        if [ $freed_npm -gt 0 ]; then
            print_status "Freed $(numfmt --to=iec $freed_npm) from npm cache"
            total_freed=$((total_freed + freed_npm))
        fi
    fi
    
    # Clean Poetry cache (with confirmation)
    if [ -d "$HOME/.cache/pypoetry" ]; then
        poetry_size=$(du -sb "$HOME/.cache/pypoetry" 2>/dev/null | cut -f1 || echo "0")
        if [ $poetry_size -gt 1073741824 ]; then  # > 1GB
            print_warning "Poetry cache is large: $(numfmt --to=iec $poetry_size)"
            read -p "Clean Poetry cache? This will slow down future poetry installs (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                if command -v poetry &> /dev/null; then
                    poetry cache clear pypi --all 2>/dev/null || true
                fi
                rm -rf "$HOME/.cache/pypoetry" 2>/dev/null || true
                print_status "Freed $(numfmt --to=iec $poetry_size) from Poetry cache"
                total_freed=$((total_freed + poetry_size))
            fi
        fi
    fi
    
    # Clean browser caches
    print_info "Cleaning browser caches..."
    
    # Chrome cache
    if [ -d "$HOME/.config/google-chrome/Default/Service Worker/CacheStorage" ]; then
        before_chrome=$(du -sb "$HOME/.config/google-chrome/Default/Service Worker/CacheStorage" 2>/dev/null | cut -f1 || echo "0")
        rm -rf "$HOME/.config/google-chrome/Default/Service Worker/CacheStorage" 2>/dev/null || true
        if [ $before_chrome -gt 0 ]; then
            print_status "Freed $(numfmt --to=iec $before_chrome) from Chrome cache"
            total_freed=$((total_freed + before_chrome))
        fi
    fi
    
    # Clean application logs
    print_info "Cleaning application logs..."
    
    # Remove logs older than 30 days
    find "$HOME/.config" -name "*.log" -type f -mtime +30 -delete 2>/dev/null || true
    
    # Clean editor caches
    for editor in "Cursor" "Code" "Windsurf"; do
        if [ -d "$HOME/.config/$editor/logs" ]; then
            before_logs=$(du -sb "$HOME/.config/$editor/logs" 2>/dev/null | cut -f1 || echo "0")
            rm -rf "$HOME/.config/$editor/logs" 2>/dev/null || true
            if [ $before_logs -gt 0 ]; then
                print_status "Freed $(numfmt --to=iec $before_logs) from $editor logs"
                total_freed=$((total_freed + before_logs))
            fi
        fi
    done
    
    # System cleanup (requires sudo)
    print_info "System cleanup (requires sudo)..."
    if sudo -n true 2>/dev/null; then
        sudo apt autoremove --purge -y 2>/dev/null || true
        sudo apt autoclean 2>/dev/null || true
        print_status "System packages cleaned"
    else
        print_warning "Skipping system cleanup (no sudo access)"
    fi
    
    # Summary
    if [ $total_freed -gt 0 ]; then
        print_status "Total space freed: $(numfmt --to=iec $total_freed)"
    else
        print_info "No significant space was freed"
    fi
}

# Function to suggest manual cleanup
suggest_manual_cleanup() {
    print_header "MANUAL CLEANUP SUGGESTIONS"
    
    # Large downloads
    if [ -d "$HOME/Downloads" ]; then
        large_downloads=$(find "$HOME/Downloads" -type f -size +50M 2>/dev/null)
        if [ -n "$large_downloads" ]; then
            print_info "Large files in Downloads (>50MB):"
            echo "$large_downloads" | xargs ls -lh 2>/dev/null
            print_warning "Consider removing old installers and downloads"
        fi
    fi
    
    # Old Python packages
    print_info "\nPython package locations:"
    find "$HOME/.local/lib" -name "site-packages" -type d 2>/dev/null | while read dir; do
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "  $dir: $size"
    done
    
    # Large config directories
    print_info "\nLargest config directories:"
    du -sh "$HOME/.config"/* 2>/dev/null | sort -hr | head -5
    
    print_warning "\nConsider cleaning:"
    echo "  - Old Docker images: docker system prune -a"
    echo "  - Old snap packages: sudo snap list --all | grep disabled"
    echo "  - Journal logs: sudo journalctl --vacuum-size=100M"
    echo "  - Temp files: sudo find /tmp -type f -atime +7 -delete"
}

# Main execution
main() {
    echo -e "${GREEN}🧹 Disk Space Cleanup and Debugging Script${NC}"
    echo -e "${GREEN}===========================================${NC}\n"
    
    if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        echo "Usage: $0 [options]"
        echo "Options:"
        echo "  --analyze-only    Only analyze disk usage, don't clean"
        echo "  --clean-safe      Perform safe automatic cleanup"
        echo "  --clean-all       Perform aggressive cleanup (interactive)"
        echo "  --help           Show this help message"
        exit 0
    fi
    
    # Store initial disk usage
    initial_usage=$(df / | awk 'NR==2 {print $3}')
    
    # Always show current status
    show_disk_usage
    identify_large_dirs
    identify_large_files
    analyze_caches
    
    # Determine cleanup level
    if [ "$1" = "--analyze-only" ]; then
        suggest_manual_cleanup
        print_info "Analysis complete. Use --clean-safe or --clean-all to perform cleanup."
    elif [ "$1" = "--clean-safe" ]; then
        perform_cleanup
        show_disk_usage
    elif [ "$1" = "--clean-all" ]; then
        perform_cleanup
        suggest_manual_cleanup
        show_disk_usage
    else
        # Interactive mode
        echo ""
        read -p "Perform safe cleanup? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            perform_cleanup
        fi
        suggest_manual_cleanup
        show_disk_usage
    fi
    
    # Show final summary
    final_usage=$(df / | awk 'NR==2 {print $3}')
    if [ $initial_usage -gt $final_usage ]; then
        freed=$((initial_usage - final_usage))
        print_status "Total disk space freed: $(numfmt --to=iec $((freed * 1024)))"
    fi
    
    print_status "Cleanup completed!"
}

# Run main function with all arguments
main "$@"
