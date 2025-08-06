# Line Endings Configuration Guide

This guide explains how to handle line ending issues in the RLaaS project, especially when working on Windows.

## The Problem

Git on Windows can automatically convert line endings between Unix-style (LF) and Windows-style (CRLF), which can cause warnings like:

```
warning: in the working copy of 'src/rlaas/ui/__init__.py', LF will be replaced by CRLF the next time Git touches it
```

## Solution

We've configured the project to use consistent line endings across all platforms:

### 1. Git Configuration

The project includes:
- **`.gitattributes`** - Defines line ending rules for different file types
- **`.editorconfig`** - Ensures consistent editor settings

### 2. Recommended Git Settings

Run these commands in your project directory:

```bash
# Disable automatic CRLF conversion
git config core.autocrlf false

# Set default line ending to LF
git config core.eol lf

# Apply the .gitattributes rules
git add .gitattributes .editorconfig
```

### 3. Fix Existing Files

#### On Windows (PowerShell):
```powershell
# Dry run to see what would be changed
.\scripts\fix-line-endings.ps1 -DryRun

# Actually fix the files
.\scripts\fix-line-endings.ps1
```

#### On Linux/Mac (Bash):
```bash
# Make script executable (Linux/Mac only)
chmod +x scripts/fix-line-endings.sh

# Dry run to see what would be changed
./scripts/fix-line-endings.sh --dry-run

# Actually fix the files
./scripts/fix-line-endings.sh
```

### 4. Manual Fix (Alternative)

If you prefer to fix manually:

```bash
# Refresh the index to apply .gitattributes rules
git add --renormalize .

# Check what changed
git status

# Commit the line ending fixes
git commit -m "Fix line endings according to .gitattributes"
```

## File Type Rules

Our `.gitattributes` file enforces these rules:

### LF (Unix) Line Endings:
- Python files (`.py`)
- Markdown files (`.md`)
- YAML files (`.yml`, `.yaml`)
- JSON files (`.json`)
- Configuration files (`.toml`, `.cfg`, `.ini`)
- Shell scripts (`.sh`, `.bash`)
- Docker files
- Makefiles

### CRLF (Windows) Line Endings:
- Batch files (`.bat`, `.cmd`)
- PowerShell scripts (`.ps1`)

### Binary Files:
- Images, archives, executables are treated as binary

## Editor Configuration

The `.editorconfig` file ensures:
- UTF-8 encoding
- Consistent indentation (4 spaces for Python, 2 for YAML/JSON)
- Automatic trailing whitespace removal
- Final newline insertion

## Troubleshooting

### Still Getting Warnings?

1. **Check your Git version**: Ensure you're using Git 2.10+
2. **Verify configuration**:
   ```bash
   git config --list | grep -E "(autocrlf|eol)"
   ```
3. **Re-apply attributes**:
   ```bash
   git add --renormalize .
   git commit -m "Normalize line endings"
   ```

### Editor Issues?

1. **VS Code**: Install the EditorConfig extension
2. **PyCharm**: EditorConfig support is built-in
3. **Vim**: Install the editorconfig-vim plugin

### Docker Issues?

The Dockerfile uses multi-stage builds and should handle line endings correctly. If you encounter issues:

1. Ensure your Docker version is recent (19.03+)
2. Use the development target for local development:
   ```bash
   docker build --target development -t rlaas:dev .
   ```

## Best Practices

1. **Always use the provided scripts** to fix line endings
2. **Configure your editor** to respect `.editorconfig`
3. **Run the line ending fix script** before committing large changes
4. **Use consistent Git settings** across your team

## Team Setup

For team consistency, everyone should run:

```bash
# Clone the repository
git clone <repository-url>
cd rlaas

# Configure Git
git config core.autocrlf false
git config core.eol lf

# Fix any existing line ending issues
# Windows:
.\scripts\fix-line-endings.ps1

# Linux/Mac:
./scripts/fix-line-endings.sh
```

This ensures everyone has the same line ending configuration and prevents future issues.
