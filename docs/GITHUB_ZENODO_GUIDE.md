# Upload to your existing GitHub repository and get a Zenodo DOI

Repository: https://github.com/Farhan191091/Reducing-tumor-independent-shortcut-learning-in-brain-tumor-

## Part A. Put the files into the repository (GitHub Desktop, no command line)
1. Install GitHub Desktop (https://desktop.github.com) and sign in as Farhan191091.
2. File > Clone repository > choose `Reducing-tumor-independent-shortcut-learning-in-brain-tumor-` > Local path e.g. `C:\GitHub` > Clone.
3. Unzip `brain-tumor-mri-shortcut.zip`. Open the unzipped folder `brain-tumor-mri-shortcut`, select EVERYTHING inside it (Ctrl+A, including the hidden file `.zenodo.json`; in Windows Explorer enable View > Hidden items) and copy it into `C:\GitHub\Reducing-tumor-independent-shortcut-learning-in-brain-tumor-`. If Windows asks, choose "Replace" (for an existing README or LICENSE).
4. Back in GitHub Desktop, about 1,400 changed files appear. Summary: `Code and data for the JNM submission` > Commit to main > Push origin.
5. Check on github.com that the README with figures is shown and that the repository is Public (Settings > General > Danger Zone > Change visibility, if needed).

## Part B. Zenodo DOI
1. https://zenodo.org > Log in > Log in with GitHub.
2. Menu under your name > GitHub > Sync now > switch the repository ON.
3. On GitHub: Releases > Create a new release > Tag `v1.0.0` > Title `v1.0.0 (manuscript submission)` > Publish release.
4. After 1 to 5 minutes, refresh the Zenodo GitHub page; a DOI such as 10.5281/zenodo.1234567 appears. Open it and check title, the four authors and license (MIT).

## Part C. Send back the Zenodo DOI (the "concept DOI" if two are shown).

## Problems
* Zenodo shows no DOI: the switch must be ON before the release; switch it on and publish release v1.0.1.
* GitHub Desktop warns about large files: no file is above 25 MB; check that you copied the right folder.
