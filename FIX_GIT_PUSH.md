# Fix Git Push - Remove Large File from History

## Current Issue

The file `pretrained/predictors/wt/wt_halflife.json` (86MB) is in git history and HuggingFace rejects pushes with files >10MB.

## Solution: Remove from History

### Step 1: Commit or Stash Unstaged Changes

You have unstaged changes. Choose one:

**Option A: Commit them first**
```bash
git add .
git commit -m "Clean up repository - remove unnecessary files"
```

**Option B: Stash them temporarily**
```bash
git stash
# After filter-branch, restore with: git stash pop
```

### Step 2: Remove File from History

```bash
# Suppress warning
export FILTER_BRANCH_SQUELCH_WARNING=1

# Remove file from all commits
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch pretrained/predictors/wt/wt_halflife.json" \
  --prune-empty --tag-name-filter cat -- --all
```

This will take a few minutes as it rewrites all commits.

### Step 3: Clean Up Backup Refs

```bash
# Remove backup refs created by filter-branch
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

### Step 4: Force Push

```bash
# Push to GitHub
git push --force origin main

# Push to HuggingFace
git push --force huggingface main
```

## Alternative: Simpler Approach (If Filter-Branch Fails)

If filter-branch is too slow or problematic:

```bash
# 1. Create a fresh branch from before the file was added
git checkout -b main-clean 08d7dc5

# 2. Cherry-pick commits without the large file
git cherry-pick 5011cfa  # fixing the eval script
git cherry-pick a6d3f4a  # adding new prompts  
git cherry-pick 9fb9d70  # changing to train for 30 epochs
git cherry-pick 8176e8b  # Merge remote changes
git cherry-pick dab9a5d  # Remove training checkpoints
git cherry-pick ae39deb  # adding new updates with wildtypes

# 3. Push the clean branch
git push origin main-clean
git push huggingface main-clean

# 4. Make it the new main (if desired)
git push --force origin main-clean:main
git push --force huggingface main-clean:main
```

## Verify File is Removed

After filter-branch, verify:
```bash
git log --all -- pretrained/predictors/wt/wt_halflife.json
# Should return nothing
```

## Quick Commands (Copy-Paste)

```bash
# 1. Commit unstaged changes
git add .
git commit -m "Repository cleanup"

# 2. Remove from history
export FILTER_BRANCH_SQUELCH_WARNING=1
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch pretrained/predictors/wt/wt_halflife.json" \
  --prune-empty --tag-name-filter cat -- --all

# 3. Clean up
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 4. Force push
git push --force origin main
git push --force huggingface main
```

