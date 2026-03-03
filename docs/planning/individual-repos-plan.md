# Splitting Code and Docs into Separate Repos

## Goal

Separate the codebase into two repos with different access levels:
- **Forge** (code) — wider access (public or larger team)
- **Forge-docs** (docs) — restricted access (core team only)

---

## Approach: Git Submodules

The docs repo is linked as a submodule inside the code repo. Locally they appear as one project; on GitHub they're two separate repos with independent access controls.

### Local view (after clone with submodules)

```
Forge/
├── src/
├── configs/
├── forge.py
├── .gitmodules          ← tells git where to find submodules
└── docs/                ← submodule → Forge-docs repo
    ├── planning/
    ├── analysis/
    ├── paper/
    └── archive/
```

### GitHub view

```
Forge/
├── src/              ← normal folder
├── configs/          ← normal folder
├── forge.py        ← normal file
└── docs @ a3f8b2c    ← link to Forge-docs repo at specific commit
```

---

## How It Works

### `.gitmodules` file (auto-created, committed to code repo)

```ini
[submodule "docs"]
    path = docs
    url = git@github.com:NU-Deep-Learning/Forge-docs.git
```

Git reads this file to know which submodules exist, where they go, and where to fetch them from.

### Cloning

```bash
# Option A: all at once
git clone --recurse-submodules git@github.com:NU-Deep-Learning/Forge.git

# Option B: two steps
git clone git@github.com:NU-Deep-Learning/Forge.git
cd Forge
git submodule update --init

# Option C: docs repo directly (no code)
git clone git@github.com:NU-Deep-Learning/Forge-docs.git
```

### Access scenarios

| User | Code repo access | Docs repo access | What happens |
|------|-----------------|-----------------|--------------|
| Public contributor | Yes | No | Clone succeeds. `docs/` is empty. Code works fine. |
| Core team member | Yes | Yes | Clone succeeds. `docs/` populated with full docs. |
| Docs-only reviewer | No | Yes | Can clone docs repo directly. |

---

## Setup Steps

### 1. Create the docs repo on GitHub

- Create `NU-Deep-Learning/Forge-docs` as a **private** repo
- Add only core team members as collaborators

### 2. Move docs to the new repo

```bash
# From the Forge root
cd /path/to/Forge

# Copy docs to a temp location
cp -r docs /tmp/forge-docs

# Initialize docs as its own repo
cd /tmp/forge-docs
git init
git add .
git commit -m "Initial docs migration from Forge"
git remote add origin git@github.com:NU-Deep-Learning/Forge-docs.git
git push -u origin main
```

### 3. Replace docs/ with submodule in the code repo

```bash
cd /path/to/Forge

# Remove docs from git tracking (keeps local files until submodule replaces them)
git rm -r docs
git commit -m "Remove docs/ in preparation for submodule"

# Add docs back as a submodule
git submodule add git@github.com:NU-Deep-Learning/Forge-docs.git docs
git commit -m "Add docs as submodule (separate private repo)"
git push
```

### 4. Verify

```bash
# .gitmodules should exist
cat .gitmodules

# docs/ should contain the full docs
ls docs/

# git status should show clean
git status
```

---

## Day-to-Day Workflow

### Making changes to docs

```bash
cd docs/
# You're now in the Forge-docs repo
git add .
git commit -m "Update analysis docs"
git push

# Back in the main repo, update the submodule pointer
cd ..
git add docs
git commit -m "Update docs submodule"
git push
```

### Pulling latest docs

```bash
git submodule update --remote docs
```

### Key thing to remember

Submodules pin to a specific commit. After pushing changes in the docs repo, you also need to update the pointer in the code repo (`git add docs && git commit`). This is the main friction point with submodules.

---

## Alternative: Two Fully Separate Repos

If submodule friction is too annoying, just keep them completely independent:

```
~/Projects/
├── Forge/           ← code repo
└── Forge-docs/      ← docs repo
```

- Simpler git workflow (no submodule commands)
- No shared directory tree (relative links between code and docs won't work)
- Each repo is fully independent

---

## Recommendation

Start with **submodules** — it preserves the current directory structure and relative links. If the two-commit workflow (docs commit + pointer update) becomes annoying, switch to separate repos later. The migration is trivial either way.
