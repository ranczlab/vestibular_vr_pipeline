# Collaborative Git Workflow for Shared Repositories

This guide describes a clean, low-conflict workflow for two or more collaborators working on the same repository.

---

## 1. Create a Feature Branch

Always start your work on a separate branch:

```bash
git switch -c <your-inits>/<task>
```

Example:
```bash
git switch -c ea/refactor-analysis-script
```

---

## 2. Commit Small, Logical Changes

Commit frequently, each time your code reaches a meaningful state.

---

## 3. Sync With `main` Before Pushing

Regularly pull updates and rebase to integrate others' work:

```bash
git fetch origin
git rebase origin/main
```

Resolve any conflicts locally before pushing.

---

## 4. Push and Open a Pull Request

```bash
git push -u origin <your-inits>/<task>
```

Open a Pull Request (PR) on GitHub for review and merging.

---

## 5. Merge (Prefer Squash Merge)

Squash-merge PRs into `main` to keep history clean and readable.

After merging, update your local main:

```bash
git switch main
git fetch origin
git rebase origin/main
```

---

## 6. One-Time Global Configuration

To reduce merge headaches:

```bash
git config --global pull.rebase true
git config --global rebase.autoStash true
```

---

## 7. When Both Edit the Same File

- Communicate early about who edits which section.
- Rebase often.
- Resolve conflicts locally:

```bash
git status
git checkout --ours path/to/file      # keep your version
git checkout --theirs path/to/file    # take theirs
git add path/to/file
git rebase --continue                 # or git commit if merging
```

---

## 8. Repo Hygiene

- Use consistent formatting and linting: **Black**, **isort**, **ruff**.
- Add a `pre-commit` config to enforce this automatically.
- Protect `main`: require PRs and reviews, no direct pushes.
- Prefer **squash merges**.
- Split large files if you often touch the same code.

---

### Summary

✅ **Branches per feature**  
✅ **Rebase frequently**  
✅ **PRs with squash merges**  
✅ **Code formatting + reviews**  
✅ **Protected `main`**

This keeps your collaboration smooth and your history clean.
