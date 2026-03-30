#!/bin/bash
# Package paper for arXiv submission
# Creates a clean zip with all necessary LaTeX files and figures

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/release"
PAPER_DIR="$OUTPUT_DIR/arxiv_paper"

echo "=== Packaging for arXiv ==="

# Clean and create output directory
rm -rf "$PAPER_DIR"
mkdir -p "$PAPER_DIR"

# Copy main LaTeX file
cp "$PROJECT_ROOT/paper/main.tex" "$PAPER_DIR/"

# Copy figures (PDF format for arXiv)
mkdir -p "$PAPER_DIR/figures"
cp "$PROJECT_ROOT/outputs/figures/"*.pdf "$PAPER_DIR/figures/"

# Update graphicspath in the tex file for the new structure
# arXiv expects figures in the same directory or a subdirectory
sed 's|{..\/outputs\/figures\/}|{figures/}|g' "$PROJECT_ROOT/paper/main.tex" > "$PAPER_DIR/main.tex"

# Create zip file
cd "$OUTPUT_DIR"
rm -f arxiv_submission.zip
zip -r arxiv_submission.zip arxiv_paper/

echo ""
echo "=== arXiv package created ==="
echo "Location: $OUTPUT_DIR/arxiv_submission.zip"
echo "Contents:"
unzip -l arxiv_submission.zip
echo ""
echo "Note: quantikz package is included in standard TeX Live distributions."
echo "If arXiv compilation fails, you may need to include quantikz.sty manually."
