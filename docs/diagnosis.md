# Data quality diagnosis

## Question

How many days per week may an employee work remotely?

## What the system answered

The answer says the current policy allows up to two remote days per week. It also says the superseded 2024 policy allows up to three remote days per week, and it calls that a conflict. The cited files are `01_remote_work_policy.md` section 2 and `04_remote_work_policy_2024_superseded.md` section 2.

## Was it retrieval?

No. Search returned the weekly-limit section of both remote-work files. The current file is the closest match and the superseded file is second. Both chunks are about the same rule, so both are reasonable results.

## Was it generation?

No. The prompt says to answer only from the retrieved chunks and to say when they conflict. The three-day limit is written in the superseded file. The model is repeating that text, not inventing it.

## Root cause

The corpus still contains an outdated copy of the remote-work policy. The current policy allows two remote days, and both the manager and People Operations have to approve the arrangement in writing. The 2024 file allows three remote days, and the manager's written approval is enough. The conflicting answer comes from `data/policies/04_remote_work_policy_2024_superseded.md`, not from chunking, search, reranking, or the Foundry call.

## What would fix it

Stop indexing the superseded file, then index the policies again. This note does not make that change. The lab asks for the issue to be found, and that file is the issue.
