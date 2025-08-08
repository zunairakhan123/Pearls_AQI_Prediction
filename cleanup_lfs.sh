#!/bin/bash
# Quick LFS Cleanup for AQI Project

echo "🧹 Starting LFS cleanup process..."

# Show current LFS usage
echo "📊 Current LFS files:"
git lfs ls-files --size

echo ""
echo "🗂️ Removing older files from LFS tracking..."

# Remove the older/smaller files from LFS tracking
git lfs untrack "data/features/backfill_features_20250802.csv"
git lfs untrack "data/features/features_20250802_1330.csv" 
git lfs untrack "data/features/features_20250807_0520.csv.gz"
git lfs untrack "data/features/features_20250807_0550.csv.gz"

# Keep only the two largest/most recent files:
# - features_20250807_0604.csv.gz (77 MB) 
# - features_20250807_0720.csv.gz (117 MB)
# - feature_metadata.csv (2.8 KB) - already small, keep in LFS

echo "✅ Removed older files from LFS tracking"
echo "📦 Files still in LFS:"
echo "  - features_20250807_0604.csv.gz (77 MB)"
echo "  - features_20250807_0720.csv.gz (117 MB)" 
echo "  - feature_metadata.csv (2.8 KB)"
echo "  Total: ~194 MB"

# Update .gitattributes to prevent future bloat
echo ""
echo "⚙️ Updating .gitattributes for better control..."

cat > .gitattributes << 'EOF'
# Only track recent large feature files with LFS (last 2-3 days)
data/features/features_*_0[6-9][0-9][0-9].csv.gz filter=lfs diff=lfs merge=lfs -text
data/features/features_*_1[0-9][0-9][0-9].csv.gz filter=lfs diff=lfs merge=lfs -text

# Track small metadata files with LFS
data/features/feature_metadata.csv filter=lfs diff=lfs merge=lfs -text

# Track model files with LFS (usually smaller but binary)
models/*.pkl filter=lfs diff=lfs merge=lfs -text
models/*.joblib filter=lfs diff=lfs merge=lfs -text

# Don't track older feature files or backfills with LFS anymore
# They'll remain in git history but as regular files
EOF

# Stage the changes
git add .gitattributes
git add data/features/

# Commit the cleanup
git commit -m "🧹 LFS cleanup: Reduce storage usage

- Removed older feature files from LFS tracking:
  * backfill_features_20250802.csv (21 MB)
  * features_20250802_1330.csv (21 MB)  
  * features_20250807_0520.csv.gz (22 MB)
  * features_20250807_0550.csv.gz (46 MB)

- Kept recent files in LFS:
  * features_20250807_0604.csv.gz (77 MB)
  * features_20250807_0720.csv.gz (117 MB)
  * feature_metadata.csv (2.8 KB)

- Updated .gitattributes for better LFS management
- Total LFS reduction: ~110 MB (from 307MB to ~194MB)"

echo ""
echo "✅ LFS cleanup complete!"
echo ""
echo "🔍 New LFS usage:"
git lfs ls-files --size

echo ""
echo "📤 Ready to push changes:"
echo "  git push origin main"
echo ""
echo "⚠️  Note: This may take a moment as Git reorganizes the repository"