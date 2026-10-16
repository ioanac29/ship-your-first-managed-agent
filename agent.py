# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""
The SRE Agent. Seven functions, each a single Managed Agents concept.
Fill them in during the workshop. Everything else is in provided.py.
"""
import json
import time
import uuid

import anthropic
import streamlit as st
from anthropic.lib import files_from_dir

from provided import DATA, SYSTEM, TOOLS, metrics, deploys, diff

client = anthropic.Anthropic()


# ── 1. Agent ──────────────────────────────────────────────────────────────
# What the agent IS: model, system prompt, tools, skills. Create once, reuse forever.
# Hint: client.beta.skills.create(display_title=..., files=files_from_dir(...))
#       then client.beta.agents.create(..., skills=[...]). display_title must be
#       unique per org — suffix it with uuid like setup_environment() does.
@st.cache_resource
def setup_agent() -> str:
    skill = client.skills.create(
        display_name=f"Incident Triage Runbook {uuid.uuid4().hex[:6]}",
        files=files_from_dir("incident-triage-runbook"),
    )
    agent = client.beta.agents.create(
        name="SRE Agent",
        model="claude-opus-5-5",
        system=SYSTEM,
        tools=TOOLS,
        skills=[{"type": "custom", "skill_id": skill.id, "version": "latest"}],
    )
    return agent.id


# ── 2. Environment ────────────────────────────────────────────────────────
# Where the agent's container runs. Create once, reuse forever.
# Hint: client.beta.environments.create(name=..., config={"type": "cloud", ...})
@st.cache_resource
def setup_environment() -> str:
    env = client.beta.environments.create(
        name=f"sre-agent-{uuid.uuid4().hex[:6]}",
        config={
            "type": "cloud",
            "networking": {"type": "limited", "allow_package_managers": True},
        },
    )
    return env.id


# ── 3. Upload the log ─────────────────────────────────────────────────────
# Push data/app.log to the Files API so sessions can mount it.
# Hint: client.beta.files.upload(file=open(...))
@st.cache_resource
def upload_log() -> str:
    with open(DATA / "app.log", "rb") as f:
        return client.files.upload(file=f).id


# ── 4. Session ────────────────────────────────────────────────────────────
# Bind agent + environment, mount the log under /mnt/session/uploads/.
# Hint: client.beta.sessions.create(agent=agent_id, environment_id=..., resources=[...])
def start_session(agent_id: str, env_id: str, log_file_id: str) -> str:
    session = client.beta.sessions.create(
        agent=agent_id,
        environment_id=env_id,
        resources=[{"type": "file", "file_id": log_file_id, "mount_path": "/app.log"}],
        budget={"type": "limit", "max_list_cost": {"amount": "200", "currency": "USD"}},  # $2.00 cap
    )
    return session.id


# ── 5. Stream loop ────────────────────────────────────────────────────────
# Open the event stream, send the user's message, yield events. When you see
# agent.custom_tool_use, call handle_tool() and post the result back.
# Hint: with client.beta.sessions.events.stream(session_id) as s: ...events.send(...)
def stream_reply(session_id: str, user_text: str):
    # Open the stream first, then send — otherwise the first events can be missed.
    with client.beta.sessions.events.stream(session_id) as stream:
        client.beta.sessions.events.send(
            session_id,
            events=[{"type": "user.message", "content": [{"type": "text", "text": user_text}]}],
        )
        for ev in stream:
            if ev.type == "agent.custom_tool_use":
                result = handle_tool(ev.name, ev.input)
                client.beta.sessions.events.send(
                    session_id,
                    events=[{
                        "type": "user.custom_tool_result",
                        "custom_tool_use_id": ev.id,
                        "content": [{"type": "text", "text": result}],
                    }],
                )
            yield ev
            # requires_action means the agent is waiting on us (a tool result), so keep going.
            # Any other idle reason (end_turn, budget_reached, ...) or termination ends the turn.
            if ev.type == "session.status_terminated":
                return
            if ev.type == "session.status_idle" and ev.stop_reason.type != "requires_action":
                if ev.stop_reason.type == "budget_reached":
                    st.warning("This session hit its $2 budget cap — start a new session to keep going.")
                return


# ── 6. Local tool handlers ────────────────────────────────────────────────
# When the cloud agent calls get_metrics / get_recent_deploys / get_diff,
# answer from `metrics` / `deploys` / `diff` (already loaded from data/).
def handle_tool(name: str, args: dict) -> str:
    # Always return a string — an exception here would leave the agent waiting on a result forever.
    if name == "get_metrics":
        service, metric = args.get("service", ""), args.get("metric", "")
        if service not in metrics:
            return f"no service {service!r}; available: {', '.join(metrics)}"
        if metric not in metrics[service]:
            return f"no metric {metric!r} for {service}; available: {', '.join(metrics[service])}"
        return json.dumps(metrics[service][metric])
    if name == "get_recent_deploys":
        return deploys
    if name == "get_diff":
        commit = args.get("commit", "").strip().lower()
        full_sha = diff.split()[1]  # diff.txt starts with "commit <full sha>"
        if len(commit) >= 7 and full_sha.startswith(commit):
            return diff
        return f"no diff for commit {commit!r}; only {full_sha[:7]} is available"
    return f"unknown tool {name!r}"


# ── 7. Delete session ─────────────────────────────────────────────────────
# Sessions are real cloud resources — clean them up.
# Hint: client.beta.sessions.delete(session_id)
def delete_session(session_id: str) -> None:
    # A running session can't be deleted: interrupt it, then wait briefly for it to settle.
    if client.beta.sessions.retrieve(session_id).status == "running":
        client.beta.sessions.events.send(session_id, events=[{"type": "user.interrupt"}])
    for _ in range(25):
        if client.beta.sessions.retrieve(session_id).status != "running":
            break
        time.sleep(0.2)
    client.beta.sessions.delete(session_id)
