"""Treyrin research model: shared language backbone and decision-model experiments."""

__version__ = "0.1.0-research"

# --- first-release finalization assembly (2026-10-26) ---------------------------
# The D06 goal-integration modules (goal_control/goal_state/intake/migration) are
# copied byte-identical from research/treyrin-deepseek-work-2026-10-26/D06/integration
# into this package. intake.py imports ``goal_state`` under its plain top-level
# module name, so this package registers that same module object under the plain
# name before intake is imported. No source text is rewritten to hide the seam;
# the alias lives only in sys.modules and is listed in CANDIDATE-ASSEMBLY.md.
# These are research contracts, not a capability claim.
import sys as _sys

from . import goal_state as _goal_state

_sys.modules.setdefault("goal_state", _goal_state)

from .goal_control import (GoalActionHead, HeadContractError, goal_feature_rows,
                           joint_softmax, legality_mask)
from .goal_metadata import goal_metadata, parse_goal_metadata
from .goal_state import GoalRecord, GoalState, GoalStateError, MigrationPlan
from .intake import (GoalProposal, IntakeError, IntakeJournal, intake_proposal,
                     parse_goal_proposal)
from .migration import (MigrationError, NamedMigrationReport, apply_named_migration,
                        check_optimizer_moments, plan_named_migration, snapshot_moments)

__all__ = [
    "GoalActionHead", "HeadContractError", "goal_feature_rows", "joint_softmax", "legality_mask",
    "goal_metadata", "parse_goal_metadata",
    "GoalRecord", "GoalState", "GoalStateError", "MigrationPlan",
    "GoalProposal", "IntakeError", "IntakeJournal", "intake_proposal", "parse_goal_proposal",
    "MigrationError", "NamedMigrationReport", "apply_named_migration",
    "check_optimizer_moments", "plan_named_migration", "snapshot_moments",
]
