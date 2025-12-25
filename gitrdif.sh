#!/bin/bash

# gitrdif.sh - Generate a diff between local and remote branch
# Output: gitrdiff.md in the project root

# Get current branch name
BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Fetch latest from remote without merging
echo "Fetching latest from origin/$BRANCH..."
git fetch origin "$BRANCH" 2>/dev/null

# Check if remote branch exists
if ! git rev-parse --verify "origin/$BRANCH" > /dev/null 2>&1; then
    echo "Remote branch origin/$BRANCH not found. Using origin/main..."
    REMOTE_BRANCH="origin/main"
else
    REMOTE_BRANCH="origin/$BRANCH"
fi

# Output file
OUTPUT="gitrdiff.md"

# Generate the diff
echo "Generating diff: local $BRANCH vs $REMOTE_BRANCH..."

{
    echo "# Git Diff Report"
    echo ""
    echo "> [!WARNING]"
    echo "> **PENDING REMOTE CHANGES DETECTED**"
    echo "> This report shows changes that exist on the remote branch ($REMOTE_BRANCH) but have NOT yet been pulled locally."
    echo "> DO NOT confuse these with your local work. These are the updates you will receive after running \`git pull\`."
    echo ""
    echo "**Generated**: $(date)"
    echo ""
    echo "**Local Branch**: $BRANCH"
    echo ""
    echo "**Comparing Against**: $REMOTE_BRANCH"
    echo ""
    echo "---"
    echo ""
    
    # NEW: Show uncommitted changes first (working directory)
    echo "## Uncommitted Changes (working directory)"
    echo ""
    echo "### Modified/Staged Files"
    echo ""
    echo '```'
    git status --short 2>/dev/null || echo "(clean)"
    echo '```'
    echo ""
    
    # Check if there are any uncommitted changes
    if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
        echo "### Uncommitted Diff"
        echo ""
        echo '```diff'
        git diff 2>/dev/null
        git diff --cached 2>/dev/null
        echo '```'
        echo ""
    fi
    
    # NEW: Show contents of untracked files (new files not yet staged)
    UNTRACKED=$(git ls-files --others --exclude-standard 2>/dev/null)
    if [ -n "$UNTRACKED" ]; then
        echo "### New Untracked Files"
        echo ""
        for file in $UNTRACKED; do
            # Skip binary files and very large files
            if [ -f "$file" ] && file "$file" | grep -q text; then
                LINES=$(wc -l < "$file" 2>/dev/null || echo "0")
                if [ "$LINES" -lt 500 ]; then
                    echo "#### \`$file\`"
                    echo ""
                    echo '```'
                    cat "$file" 2>/dev/null
                    echo '```'
                    echo ""
                else
                    echo "#### \`$file\` ($LINES lines - truncated)"
                    echo ""
                    echo '```'
                    head -100 "$file" 2>/dev/null
                    echo "... ($LINES total lines)"
                    echo '```'
                    echo ""
                fi
            fi
        done
    fi
    
    echo "---"
    echo ""
    
    # Show commits that are different
    echo "## Commits Ahead (local changes not on remote)"
    echo ""
    echo '```'
    git log --oneline "$REMOTE_BRANCH..HEAD" 2>/dev/null || echo "(none)"
    echo '```'
    echo ""
    
    echo "## Commits Behind (REMOTE UPDATES PENDING)"
    echo "These commits exist on origin but are NOT in your local branch yet."
    echo ""
    echo '```'
    git log --oneline "HEAD..$REMOTE_BRANCH" 2>/dev/null || echo "(none)"
    echo '```'
    echo ""
    
    echo "---"
    echo ""
    echo "## File Changes (UPDATES YOU WILL RECEIVE)"
    echo "This shows what will change in your local files after you pull."
    echo ""
    echo '```'
    # Show diff from local perspective to remote
    git diff --stat HEAD "$REMOTE_BRANCH" 2>/dev/null || echo "(no changes)"
    echo '```'
    echo ""
    
    echo "---"
    echo ""
    echo "## Full Diff of Pending Remote Updates"
    echo ""
    echo "Green (+) = lines that will be ADDED to your local files"
    echo "Red (-) = lines that will be REMOVED from your local files"
    echo ""
    echo '```diff'
    # Show diff from local perspective to remote
    git diff HEAD "$REMOTE_BRANCH" 2>/dev/null || echo "(no diff)"
    echo '```'
    
} > "$OUTPUT"

echo "Done! Created $OUTPUT"
echo ""
echo "Summary:"
echo "  Uncommitted files: $(git status --short 2>/dev/null | wc -l | tr -d ' ')"
echo "  YOUR unpushed commits: $(git log --oneline "$REMOTE_BRANCH..HEAD" 2>/dev/null | wc -l | tr -d ' ')"
echo "  Remote commits to pull: $(git log --oneline "HEAD..$REMOTE_BRANCH" 2>/dev/null | wc -l | tr -d ' ')"


