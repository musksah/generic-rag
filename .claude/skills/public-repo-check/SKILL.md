---
name: public-repo-check
description: Scan staged changes, or the whole tracked tree, for secrets, credentials, personal data and private documents before committing or pushing to this public repo. Use before any commit that adds files or data, or when the user asks whether the repo is safe to share.
argument-hint: [--all]
---

# Public repo check

This repo is public. Anything committed stays in the git history even after a
later commit deletes it.

Scope: staged changes (`git diff --cached`) by default. With `--all`, use
`git ls-files`, and also check history with `git log -p --all -S '<pattern>'`
for anything found.

Look for:

- **Credentials:** Google API keys `AIza...`, service-account JSON
  (`"type": "service_account"`, `private_key`), OpenAI `sk-...`, `.env` files,
  connection strings with passwords
- **Local data:** `chroma_data*/` and `.chroma*/` directories, `*.sqlite3`,
  notebook outputs, logs. `.gitignore` covers the Chroma dirs, `.venv` and `.env`.
  Flag anything similar that slips through.
- **Documents:** anything new under `documents/`. Ask the user whether it is
  public or licensed for redistribution. The State of the Union text is public domain.
- **Personal data:** emails, phone numbers and ID numbers in sample data or fixtures

Project ids belong in `GOOGLE_CLOUD_PROJECT`, not in code. Flag any that get hardcoded.

Report findings by severity with file and line. Show only the first 4
characters of any secret. For a real credential, tell the user to rotate it
first. Removing it from the repo is not enough. Never rewrite history or
force-push yourself.
