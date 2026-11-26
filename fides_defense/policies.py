# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Integrity labeling (by data source) and security policies for AgentDojo / AgentDyn.

The tutorial assigns integrity labels by *source* (e.g. an email is trusted iff its sender
is ``@contoso.com``). We generalize that deterministically to the benchmark: each suite has
a registry of **untrusted-source tools**.

Labeling principle: a tool's output is ``UNTRUSTED`` **only if it surfaces content authored
by a third party / attacker** -- inbound emails and chat messages from others, web pages,
product reviews and listings, downloaded-by-URL files, shared/received documents, calendar
invites from others. The **user's own records are trusted**: their cart, order history,
invoices for their own orders, account/profile info, their own files and directory listings,
their own sent emails and drafts, their own contacts, and calendar events they created.
Over-tainting the user's own data would make FIDES's utility cost a labeling artifact rather
than a property of the method (and, with variable hiding on, would needlessly route the
user's own data through the quarantined LLM). The untrusted lists below are grounded in each
suite's AgentDojo injection vectors (``threat_model.VECTOR_CLASS``): the kept tools are the
channels those vectors actually ride in.

Each suite also has a registry of **sink tools** -- tools with an external side effect or
that exfiltrate data (send money / email / message, post, share, invite, transfer, push,
delete, update credentials). A ``Policy`` inspects the labeled trace and raises
``PolicyViolation`` to stop a policy-violating action.

For a **known** suite the registry is authoritative (a tool not listed as untrusted is
trusted); the name-pattern heuristic is used only as a fallback for unknown suites / tools.
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

# Tools whose output surfaces third-party / attacker-authored content -> UNTRUSTED.
# The user's OWN records (cart, orders, invoices, account info, own files/dirs, sent/draft
# mail, contacts, own calendar) are deliberately NOT listed here -- they are trusted.
UNTRUSTED_SOURCES: dict[str, set[str]] = {
    "banking": {
        "read_file",                   # bills / landlord notices authored by third parties
        "get_most_recent_transactions",  # incoming transactions (subject set by the sender)
        # trusted: get_scheduled_transactions (the user's own scheduled payments)
    },
    "slack": {
        "read_channel_messages",       # messages from other users
        "read_inbox",                  # DMs from other users
        "get_webpage",                 # external web content
    },
    "travel": {
        "get_rating_reviews_for_hotels",       # third-party reviews
        "get_rating_reviews_for_restaurants",
        "get_rating_reviews_for_car_rental",
        # trusted: calendar reads (the user's own calendar; not a travel injection vector)
    },
    "workspace": {
        "get_unread_emails",           # inbound mail from others
        "get_received_emails",
        "search_emails",               # searches the received corpus
        "get_day_calendar_events",     # calendar invites can be created by others (vectors)
        "search_calendar_events",
        "get_file_by_id",              # drive files can be shared/received (drive vectors)
        "search_files",
        "list_files",
        "search_files_by_filename",
        # trusted: get_sent_emails, get_draft_emails (own), search_contacts_* (own contacts)
    },
    "shopping": {
        "get_shopping_linked_unread_emails",   # inbound mail from others
        "get_shopping_linked_received_emails",
        "search_product",              # seller-authored listings / reviews
        "browse_webpage",              # external web content
        "download_file_through_url",   # file fetched from an external URL
        # trusted: view_cart, view_order_history, get_invoice, view_available_discounts,
        # read_file, list_directory, download_file_through_id, calendar reads (own data)
    },
    "github": {
        "get_received_emails",         # inbound mail from others
        "get_github_linked_unread_emails",
        "get_github_repository_information",  # repo / issue / review content from others
        "browse_webpage",              # external web content
        "download_file_through_url",   # file fetched from an external URL
        "git_pull",                    # pulls others' commits
        "git_clone",
        # trusted: get_sent_emails, read_file, list_directory, download_file_through_id,
        # git_get_linked_ssh_keys (own data). injection_files_* are co-located (own files).
    },
    "dailylife": {
        "get_unread_emails",           # inbound mail from others
        "get_received_emails",
        "search_emails",
        "browse_webpage",              # web vectors (injection_web_*)
        "download_file_through_url",   # file fetched from an external URL
        "get_day_calendar_events",     # calendar vectors (injection_calendar_*)
        "search_calendar_events",
        # trusted: get_sent_emails, read_file, list_directory, download_file_through_id (own).
        # injection_web_file_* / injection_file_* are co-located (own files).
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
    """Whether a tool's output should be labeled UNTRUSTED (third-party-authored content).

    For a known suite the registry is authoritative: a tool not listed is trusted (so the
    user's own-data tools are never re-tainted by the heuristic). The name-pattern fallback
    applies only to unknown suites / tools.
    """
    registry = UNTRUSTED_SOURCES.get(suite)
    if registry is not None:
        return tool_name in registry
    return bool(_SOURCE_PATTERNS.search(tool_name))


def is_sink(suite: str, tool_name: str) -> bool:
    """Whether a tool is a policy-protected sink (external side effect / exfiltration).

    Authoritative per known suite; name-pattern fallback for unknown suites / tools.
    """
    registry = SINKS.get(suite)
    if registry is not None:
        return tool_name in registry
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
