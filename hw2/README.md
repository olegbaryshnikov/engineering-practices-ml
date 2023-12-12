**git lfs have been chosen**

**To track specific file or extension:**
```
# specific file
git lfs track file.ext 
# extension
git lfs track *.ext 
```

**To add new or update any files use ordinary git commands:**
```
git add <path-to-files>
git commit
git push
```

**To use previous (or current) versions of whole reposiotry use ordinary git commands:**
```
# install git lfs before the next step
# clone current version
git clone https://github.com/olegbaryshnikov/engineering-practices-ml

# go to created dir, revert changes to chosen commit
git reset --hard cac0e929c0df6602a56a8269461ea36ae3e88db9
```