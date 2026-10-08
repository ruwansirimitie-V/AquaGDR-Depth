# Creating the GitHub repository

This project directory is separate from the research workspace. Create the
remote repository only after checking the publication and license checklist.

## Before the first push

1. Replace the top-level `LICENSE` placeholder with a license approved by all
   code authors and check compatibility with the UniFuse MIT notice.
2. Fill in the author and final paper details in `CITATION.cff`.
3. Add the upstream UniFuse URL and exact revision to
   `third_party/unifuse/UPSTREAM.md`.
4. Verify that every included file can be redistributed. The dataset, split
   lists, checkpoints, and run outputs are deliberately not included.
5. Search for private paths, usernames, tokens, and internal experiment logs.
6. Run the tests and do a clean-environment smoke test after data/checkpoints
   are available.

## Create the remote repository in GitHub

On GitHub, select **New repository**, choose the name `AquaGDR-Depth`, set the
visibility you intend, and create it **without** auto-generating a README,
license, or `.gitignore` because those files already exist locally. Keep it
private until the release checklist is complete if code/data permission is
still being reviewed.

## Connect this local folder and push

Open PowerShell in this folder, replace the URL with the GitHub URL shown on
your repository page, and run:

```powershell
cd D:\UDBF-Depth-v3\AquaGDR-Depth
git add README.md LICENSE CITATION.cff environment.yml .gitignore configs aquagdr tools docs tests third_party
git status --short
git diff --cached --stat
git commit -m "Prepare AquaGDR-Depth release"
git remote add origin https://github.com/<YOUR-ACCOUNT>/AquaGDR-Depth.git
git push -u origin main
```

Inspect `git status` and the staged diff before committing. Do not use
`git add .` until you have reviewed the ignore rules and checked that no data,
weights, or private files have entered the folder.

## Later updates

```powershell
git status
git add <specific-files>
git diff --cached
git commit -m "Describe the change"
git push
```

Git stores local history. `git push` publishes the committed changes to the
remote repository. Deleting a file in a later commit does not erase it from
earlier Git history, so never commit secrets or restricted data in the first
place.
