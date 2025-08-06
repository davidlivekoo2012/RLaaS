#!/bin/bash

# Bash script to fix line endings in the RLaaS project

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--dry-run]"
            echo "  --dry-run    Show what would be changed without making changes"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}RLaaS Line Ending Fix Script${NC}"
echo -e "${GREEN}=============================${NC}"

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}Running in DRY RUN mode - no files will be modified${NC}"
fi

# File patterns that should have LF line endings
LF_PATTERNS=(
    "*.py" "*.md" "*.txt" "*.yml" "*.yaml" "*.json" 
    "*.toml" "*.cfg" "*.ini" "*.sh" "*.bash" "*.sql"
    "Dockerfile" "Makefile" ".gitignore" ".gitattributes"
    ".editorconfig"
)

# File patterns that should have CRLF line endings (none for Unix systems)
CRLF_PATTERNS=(
    "*.bat" "*.cmd" "*.ps1"
)

convert_line_endings() {
    local file_path="$1"
    local target_ending="$2"
    
    if [ ! -f "$file_path" ]; then
        echo -e "  ${RED}File not found: $file_path${NC}"
        return
    fi
    
    # Check if file is binary
    if file "$file_path" | grep -q "binary"; then
        echo -e "  ${GRAY}Skipping binary file: $file_path${NC}"
        return
    fi
    
    # Detect current line endings
    if [ ! -s "$file_path" ]; then
        echo -e "  ${GRAY}Skipping empty file: $file_path${NC}"
        return
    fi
    
    local has_crlf=$(grep -c $'\r$' "$file_path" 2>/dev/null || true)
    local has_lf=$(grep -c $'[^\r]$' "$file_path" 2>/dev/null || true)
    
    local current_ending="Unknown"
    if [ "$has_crlf" -gt 0 ] && [ "$has_lf" -eq 0 ]; then
        current_ending="CRLF"
    elif [ "$has_lf" -gt 0 ] && [ "$has_crlf" -eq 0 ]; then
        current_ending="LF"
    elif [ "$has_crlf" -gt 0 ] && [ "$has_lf" -gt 0 ]; then
        current_ending="Mixed"
    fi
    
    if [ "$current_ending" = "$target_ending" ]; then
        echo -e "  ${GREEN}OK: $file_path ($current_ending)${NC}"
        return
    fi
    
    echo -e "  ${YELLOW}Converting: $file_path ($current_ending -> $target_ending)${NC}"
    
    if [ "$DRY_RUN" = false ]; then
        if [ "$target_ending" = "LF" ]; then
            # Convert to LF (Unix line endings)
            if command -v dos2unix >/dev/null 2>&1; then
                dos2unix "$file_path" 2>/dev/null
            else
                # Fallback method using sed
                sed -i 's/\r$//' "$file_path"
            fi
        elif [ "$target_ending" = "CRLF" ]; then
            # Convert to CRLF (Windows line endings)
            if command -v unix2dos >/dev/null 2>&1; then
                unix2dos "$file_path" 2>/dev/null
            else
                # Fallback method using sed
                sed -i 's/$/\r/' "$file_path"
            fi
        fi
    fi
}

process_files() {
    local patterns=("$@")
    local target_ending="$1"
    shift
    patterns=("$@")
    
    echo -e "\n${CYAN}Processing files for $target_ending line endings...${NC}"
    
    for pattern in "${patterns[@]}"; do
        echo -e "  ${GRAY}Pattern: $pattern${NC}"
        
        # Find files matching pattern, excluding common directories
        find . -name "$pattern" -type f \
            -not -path "./.git/*" \
            -not -path "./__pycache__/*" \
            -not -path "./node_modules/*" \
            -not -path "./.venv/*" \
            -not -path "./venv/*" \
            -not -path "./build/*" \
            -not -path "./dist/*" \
            | while read -r file; do
                convert_line_endings "$file" "$target_ending"
            done
    done
}

# Main execution
echo -e "\n${BLUE}Starting line ending conversion...${NC}"

# Process LF files
process_files "LF" "${LF_PATTERNS[@]}"

# Process CRLF files (only on Windows or if specifically requested)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    process_files "CRLF" "${CRLF_PATTERNS[@]}"
fi

echo -e "\n${GREEN}Line ending conversion completed!${NC}"

if [ "$DRY_RUN" = true ]; then
    echo -e "\n${YELLOW}To actually fix the files, run without --dry-run parameter:${NC}"
    echo -e "  ${NC}./scripts/fix-line-endings.sh${NC}"
else
    echo -e "\n${CYAN}Next steps:${NC}"
    echo -e "${NC}1. Review the changes with: git diff${NC}"
    echo -e "${NC}2. Stage the changes with: git add .${NC}"
    echo -e "${NC}3. Commit the changes with: git commit -m 'Fix line endings'${NC}"
fi
