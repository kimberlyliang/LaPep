#!/bin/bash
# Script to remove wt_halflife.json from git history

set -e

echo "⚠️  WARNING: This will rewrite git history!"
echo "This will remove pretrained/predictors/wt/wt_halflife.json from ALL commits."
echo ""
read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

echo "Removing file from git history..."

# Method 1: Use git filter-branch (built-in, slower but works)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch pretrained/predictors/wt/wt_halflife.json" \
  --prune-empty --tag-name-filter cat -- --all

echo ""
echo "✅ File removed from history!"
echo ""
echo "Next steps:"
echo "  1. Verify: git log --all -- pretrained/predictors/wt/wt_halflife.json"
echo "  2. Force push: git push --force origin main"
echo "  3. Force push: git push --force huggingface main"
echo ""
echo "⚠️  WARNING: Force push rewrites remote history. Make sure all collaborators"
echo "   are aware and have pulled/synced their local repositories."

