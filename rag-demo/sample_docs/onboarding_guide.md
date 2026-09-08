# New Engineer Onboarding Guide

## Week 1: Access and Setup

Your manager will submit access requests for GitHub, the internal wiki,
and the staging environment on your first day. Access typically clears
within 4 hours but can take up to 2 business days for systems requiring
security team approval, such as production database read access.

Install the standard toolchain using the `setup.sh` script in the
`onboarding` repository. This installs the pinned versions of the
language runtimes, the internal CLI (`acme-cli`), and pre-commit hooks
that run linting and secret-scanning before every commit.

## Week 2: First Contribution

Every new engineer is assigned a "good first issue" labeled ticket in
their team's board. The expectation is not to ship it perfectly — it is
to go through the full review and deploy process once with a buddy
engineer, including how staging deploys are promoted to production via
the release pipeline.

Code review at Acme requires at least one approval from someone outside
your immediate team for changes touching shared libraries. Changes
scoped to a single service only need one approval from any team member.

## Week 3-4: Ramping Up

By the end of the first month, engineers are expected to be on the
on-call rotation shadow schedule (observing, not carrying the pager).
Full on-call rotation eligibility begins after 60 days, and requires
having completed the incident response training module in the internal
learning platform.

## Who to Ask

Questions about tooling go to #eng-platform on Slack. Questions about
team-specific processes go to your onboarding buddy first, and to your
manager if your buddy is unavailable. HR and benefits questions go
through the HR portal, not Slack.
