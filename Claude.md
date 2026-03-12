##Workflow Orchestration

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## .claude/ Directory

The .claude/ directory provides a spacee for Claude to store helpful files. Please read the contents of this directory to get caught up to speedd at the beginning of each session. 

The `lessons.md` file contains important lessons to remember -- especially after making a mistake and noting the correction. 

The `session_notes_<N>.md` files contain session notes from each session we have. Reading this in chronological order will give you the context of what we've done for each subsequent session. 

## Session Notes

At the end of each session, write a session notes file to `.claude/session_notes_<N>.md` at the project root, where N is the next session number. 

Follow the style of existing notes in that directory (if any exist). 

Capture: what was done, state at end of session, and TODOs for next session.

Note: `.claude/` is gitignored (see `.gitignore`), so session notes are local only.

## Gitignored Directories

The following project-specific paths are gitignored:
- `.claude/` — Claude session notes and local AI context
- `.data/` — raw and processed datasets
- `.models/` — saved model artifacts
- `mlruns/`, `mlartifacts/`, `mlflow.db` — MLflow local tracking

## Plan Node Default

Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)

If something goes sideways, STOP and re-plan immediately don't keep pushing

Use plan mode for verification steps, not just building

Write detailed specs upfront to reduce ambiguity

## Subagent Strategy

Use subagents liberally to keep main context window clean

Offload research, exploration, and parallel analysis to subagents

For complex problems, throw more compute at it via subagents

One tack per subagent for focused execution

## Self-Improvement Loop

After ANY correction from the user: update .claude/lessons.md with the pattern

Write rules for yourself that prevent the same mistake

Ruthlessly iterate on these lessons until mistake rate drops

Review lessons at session start for relevant project

## Verification Before Done

Never mark a task complete without proving it works

Diff behavior between main and your changes when relevant

Ask yourself: "Would a staff engineer approve this?"

Review your own work before presenting it back to me and make neccessary adjustments. 

## Demand Elegance (Balanced)

For non-trivial changes: pause and ask "is there a more elegant way?"

If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"

Skip this for simple, obvious fixes don't over-engineer

Challenge your own work before presenting it

## Core Principles

**Simplicity First**: Make every change as simple as possible. Impact minimal code. Don't over-engineer or make code overly complex.

**Readability**: Focus on code being readable and understandable to a junior dev. Be liberal with comments explaining what chunks of code do.

**Flat & Wide Over Deep**: Where possible, avoid writing code that requires multiple layers of import/reference. We can always refactor later to make patterns "industry ready", but this is not always the most readable (see above).

**No Laziness**: Find root causes. No temporary fixes. Senior developer standards.

**Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

**Notebooks run from project root** - VS Code setting `jupyter.notebookFileRoot` is set

**Data in `.data/`** - Raw datasets, gitignored

**Models in `.models/`** - Saved artifacts, gitignored

## Coding Conventions

**Readability is the default — performance is the exception**: Always ask "would a junior dev find this confusing to read?" If yes, rewrite it. Only sacrifice readability for a significant, measurable performance gain (e.g. vectorized operations). Never use clever or compact code just to look smart.

**List comprehensions**: Fine for simple one-liners (`[x.strip() for x in items]`). Use a plain `for` loop with `.append()` when the body is multi-line or involves conditional logic.

**List initialization**: Initialize empty lists immediately above the loop that fills them. Never use type-annotated local variable syntax (`records: list[PaperRecord] = []`) — plain `records = []` is cleaner. Same for scalar accumulators (`year = None`).

**Private functions**: Use `_` prefix for any function not intended as a public entry point. If it's only called from within the module, it should be private.

**Function design**: Don't split a function in two just because the steps are distinct. If the second is only ever called by the first and adds no independent value, merge them. Function names must reflect what they actually return.

**Configure-then-use**: A configure function should only configure — no return values. Compute derived values separately after calling it.

**Log message prefixes**: Every `logger.*` call and `raise` must be prefixed with a `_MOD` or `_SCRIPT` constant (e.g. `_MOD = "[pubmed.py]"`). Add this constant at the top of every new module or script from the start.

**Session notes**: Only write session notes when explicitly asked. User will ask (e.g. "end of session", "write session notes").