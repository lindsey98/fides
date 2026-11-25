# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Integrity labeling (by data source) and security policies for AgentDojo / AgentDyn.

The tutorial assigns integrity labels by *source* (e.g. an email is trusted iff its sender
is ``@contoso.com``). We generalize that deterministically to the benchmark: each suite has
a registry of **untrusted-source tools** -- tools that surface data from attacker-reachable
channels (emails, chat messages, web pages, reviews, files, search results). A tool result
from such a tool is labeled ``UNTRUSTED``; the loop joins it into the running context label.

Each suite also has a registry of **sink tools** -- tools with an external side effect or
that exfiltrate data (send money / email / message, post, share, invite, transfer, push,
delete, update credentials). A ``Policy`` inspects the labeled trace and raises
``PolicyViolation`` to stop a policy-violating action.

Registries are authoritative per suite; a name-pattern heuristic is the generic fallback so
unlisted / AgentDyn tools still get a reasonable label. This is the "generic + extensible
per-suite" design.
"""

from __future__ import annotations

import json
import re
from typing import Any

from fides_defense.core import (
    Action,
    Label,
    Policy,
    PolicyViolation,
    ToolCall,
    label_is_untrusted,
)

# --------------------------------------------------------------------------------------
# Per-suite registries
# --------------------------------------------------------------------------------------

# Tools whose output originates from an attacker-reachable channel -> UNTRUSTED integrity.
UNTRUSTED_SOURCES: dict[str, set[str]] = {
    "banking": {
        "read_file",
        "get_most_recent_transactions",
        "get_scheduled_transactions",
    },
    "slack": {
        "read_channel_messages",
        "read_inbox",
        "get_webpage",
    },
    "travel": {
        "get_rating_reviews_for_hotels",
        "get_rating_reviews_for_restaurants",
        "get_rating_reviews_for_car_rental",
        "search_calendar_events",
        "get_day_calendar_events",
    },
    "workspace": {
        "get_unread_emails",
        "get_received_emails",
        "get_sent_emails",
        "get_draft_emails",
        "search_emails",
        "search_contacts_by_name",
        "search_contacts_by_email",
        "search_calendar_events",
        "get_day_calendar_events",
        "get_file_by_id",
        "list_files",
        "search_files",
        "search_files_by_filename",
    },
    "shopping": {
        "get_shopping_linked_unread_emails",
        "get_shopping_linked_received_emails",
        "search_product",
        "view_order_history",
        "get_invoice",
        "view_cart",
        "view_available_discounts",
        "read_file",
        "list_directory",
        "browse_webpage",
        "download_file_through_url",
        "download_file_through_id",
        "search_calendar_events",
        "get_day_calendar_events",
    },
    "github": {
        "get_received_emails",
        "get_sent_emails",
        "get_github_linked_unread_emails",
        "get_github_repository_information",
        "read_file",
        "list_directory",
        "browse_webpage",
        "download_file_through_url",
        "download_file_through_id",
        "git_pull",
        "git_clone",
        "git_get_linked_ssh_keys",
    },
    "dailylife": {
        "get_unread_emails",
        "get_received_emails",
        "get_sent_emails",
        "search_emails",
        "browse_webpage",
        "download_file_through_url",
        "download_file_through_id",
        "read_file",
        "list_directory",
        "search_calendar_events",
        "get_day_calendar_events",
    },
}

# Tools with an external side effect / that exfiltrate -> policy-protected sinks.
SINKS: dict[str, set[str]] = {
    "banking": {
        "send_money",
        "schedule_transaction",
        "update_scheduled_transaction",
        "update_password",
        "update_user_info",
    },
    "slack": {
        "send_direct_message",
        "send_channel_message",
        "post_webpage",
        "add_user_to_channel",
        "invite_user_to_slack",
        "remove_user_from_slack",
    },
    "travel": {
        "reserve_hotel",
        "reserve_car_rental",
        "reserve_restaurant",
        "create_calendar_event",
        "cancel_calendar_event",
        "send_email",
    },
    "workspace": {
        "send_email",
        "delete_email",
        "create_calendar_event",
        "cancel_calendar_event",
        "reschedule_calendar_event",
        "add_calendar_event_participants",
        "create_file",
        "delete_file",
        "append_to_file",
        "share_file",
    },
    "shopping": {
        "send_money",
        "send_email",
        "refund_order",
        "checkout_selected_cart",
        "update_shopping_account_password",
        "update_shopping_account_user_info",
        "create_file",
        "create_dir",
        "delete_node",
        "move_node",
        "copy_node",
        "input_to_webpage",
        "create_calendar_event",
        "cancel_calendar_event",
        "reschedule_calendar_event",
        "add_calendar_event_participants",
    },
    "github": {
        "send_email",
        "git_create_repo",
        "git_delete_repo",
        "git_transfer_repo_ownership",
        "git_invite_collaborators",
        "git_push",
        "git_issue",
        "git_star",
        "git_unstar",
        "git_add_ssh_key",
        "git_delete_ssh_key",
        "update_github_account_password",
        "create_file",
        "create_dir",
        "delete_node",
        "move_node",
        "copy_node",
        "input_to_webpage",
    },
    "dailylife": {
        "send_money",
        "verify_transaction",
        "send_email",
        "delete_email",
        "create_file",
        "create_dir",
        "delete_node",
        "move_node",
        "copy_node",
        "input_to_webpage",
        "create_calendar_event",
        "cancel_calendar_event",
        "reschedule_calendar_event",
        "add_calendar_event_participants",
    },
}

# Generic name-pattern fallback for tools not in a suite registry (e.g. AgentDyn additions
# or a newly added suite). Authoritative registries above take precedence.
_SOURCE_PATTERNS = re.compile(
    r"(^read_|_emails?$|^get_.*(email|message|inbox|review|transaction|webpage|repository)"
    r"|inbox|message|review|webpage|browse|download|search|list_)",
    re.IGNORECASE,
)
_SINK_PATTERNS = re.compile(
    r"(^send_|^post_|^share_|^invite_|^remove_|^delete_|^create_|^update_|^reserve_|^checkout"
    r"|^refund|^cancel_|^reschedule_|^add_|^move_|^copy_|^git_(push|create|delete|transfer|invite|issue|star|unstar|add_ssh|delete_ssh)"
    r"|transfer|input_to_webpage)",
    re.IGNORECASE,
)


def is_untrusted_source(suite: str, tool_name: str) -> bool:
    """Whether a tool's output should be labeled UNTRUSTED (attacker-reachable channel)."""
    registry = UNTRUSTED_SOURCES.get(suite)
    if registry is not None and tool_name in registry:
        return True
    if registry is not None:
        # Suite is known: registry is authoritative for integrity-relevant reads. Fall back
        # to the heuristic only for tools the registry has not classified either way.
        if tool_name in SINKS.get(suite, set()):
            return False
    return bool(_SOURCE_PATTERNS.search(tool_name))


def is_sink(suite: str, tool_name: str) -> bool:
    """Whether a tool is a policy-protected sink (external side effect / exfiltration)."""
    registry = SINKS.get(suite)
    if registry is not None and tool_name in registry:
        return True
    if registry is not None and tool_name in UNTRUSTED_SOURCES.get(suite, set()):
        return False
    return bool(_SINK_PATTERNS.search(tool_name))


# --------------------------------------------------------------------------------------
# Policies  (inspect the labeled trace; raise PolicyViolation to stop an action)
# --------------------------------------------------------------------------------------

_URL_RE = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$\-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)


def _args_text(args: dict[str, Any]) -> str:
    try:
        return json.dumps(args, default=str)
    except Exception:
        return str(args)


def contains_url(text: str) -> bool:
    return _URL_RE.search(text) is not None


def make_policy(name: str, suite: str) -> Policy:
    """Build a policy closure for a suite.

    Policies (faithful to the tutorial's dynamic taint-tracking):

    - ``no_untrusted_sink`` (default): once untrusted data has entered the context, no sink
      tool may be called. This is the deterministic integrity policy -- an injected
      instruction can only cause harm through a sink, and under attack the context is
      untrusted, so every policy-violating sink call is stopped.
    - ``no_untrusted_url``: a faithful port of the tutorial's ``policy_no_untrusted_url`` --
      block a sink only when the context is untrusted *and* the arguments contain a URL
      (stops exfiltration via link preview / click), allowing other sink calls through.
    - ``none``: taint-track only, never block (useful to measure the undefended planner).
    """

    def policy(trace: list[tuple[Action, Label]]) -> None:
        if not trace:
            return
        action, label = trace[-1]
        if not isinstance(action, ToolCall):
            return
        if not is_sink(suite, action.name):
            return

        if name == "none":
            return

        if name == "no_untrusted_url":
            if label_is_untrusted(label) and contains_url(_args_text(action.arguments)):
                raise PolicyViolation(
                    f"untrusted context flows a URL into sink `{action.name}`"
                )
            return

        # Default: no_untrusted_sink
        if label_is_untrusted(label):
            raise PolicyViolation(
                f"untrusted data must not flow into sink `{action.name}`"
            )

    return policy


POLICIES = ("no_untrusted_sink", "no_untrusted_url", "none")
