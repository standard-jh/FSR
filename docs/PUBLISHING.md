# Publishing

This directory is already initialized as a local git repository.

Current local state:

```bash
git log --oneline -1
```

To publish manually:

```bash
cd /home/juhwan/Documents/sr/latent_sr/FSR
```

Create an empty GitHub repository named:

```text
FSR
```

Then push:

```bash
git remote add origin https://github.com/<your-username>/FSR.git
git push -u origin main
```

If the GitHub CLI is installed and authenticated:

```bash
gh repo create FSR --public --source=. --remote=origin --push
```

If the repository should stay private:

```bash
gh repo create FSR --private --source=. --remote=origin --push
```

Notes:

- Checkpoints are not committed. They are ignored by `.gitignore`.
- Raw datasets are not committed.
- Representative images, raw metric CSVs, and compact result tables are committed.
