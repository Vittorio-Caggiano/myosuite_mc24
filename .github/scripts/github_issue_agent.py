#!/usr/bin/env python3
"""
GitHub Issue Analysis Agent
Uses Claude API with tool_use to analyze GitHub issues and autonomously
create PRs with fixes. Claude drives the process using GitHub API tools.
"""

import base64
import json
import re
import time
import requests
import anthropic
from typing import Dict, List, Optional

# Timeout for all HTTP requests (seconds)
REQUEST_TIMEOUT = 30
# Max file size to return to Claude (characters)
MAX_FILE_SIZE = 15000
# Max agent turns for implementation
MAX_AGENT_TURNS = 20


# --- Tool definitions for Claude ---
IMPLEMENTATION_TOOLS = [
    {
        "name": "search_code",
        "description": "Search for code patterns in the repository using GitHub Code Search. Limited to 3 searches per session — use wisely with specific queries.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g., 'Copyright 2025', 'def old_function')"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "list_files",
        "description": "List all files in the repository. Returns the full file tree. Prefer list_directory for browsing specific folders.",
        "input_schema": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name": "list_directory",
        "description": "List files and subdirectories in a specific directory. Use this to explore the repository structure before reading files. Much faster than list_files for targeted browsing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to repo root (e.g., 'src', 'myosuite/envs'). Use '' for root."
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the content of a specific file (not a directory). Use list_directory to explore directories first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to repo root (e.g., 'src/main.py')"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "create_branch",
        "description": "Create a new git branch from main. Call this ONCE before writing any files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch_name": {
                    "type": "string",
                    "description": "Branch name (e.g., 'bot/fix-issue-3')"
                }
            },
            "required": ["branch_name"]
        }
    },
    {
        "name": "write_file",
        "description": "Write or update a file on a branch. Provide the COMPLETE new file content. Each call creates a separate commit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to repo root"
                },
                "content": {
                    "type": "string",
                    "description": "Complete new file content"
                },
                "branch": {
                    "type": "string",
                    "description": "Branch to write to"
                },
                "message": {
                    "type": "string",
                    "description": "Commit message for this file change"
                }
            },
            "required": ["path", "content", "branch", "message"]
        }
    },
    {
        "name": "create_pull_request",
        "description": "Create a pull request from a branch to main. Call this AFTER all files have been written.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "PR title (under 70 chars)"
                },
                "body": {
                    "type": "string",
                    "description": "PR description in markdown"
                },
                "branch": {
                    "type": "string",
                    "description": "Source branch name"
                }
            },
            "required": ["title", "body", "branch"]
        }
    },
    {
        "name": "post_comment",
        "description": "Post a comment on the GitHub issue.",
        "input_schema": {
            "type": "object",
            "properties": {
                "body": {
                    "type": "string",
                    "description": "Comment body in markdown"
                }
            },
            "required": ["body"]
        }
    },
]


class GitHubTools:
    """Executes GitHub API operations on behalf of Claude's tool calls."""

    MAX_SEARCHES = 3  # Limit search_code calls to prevent endless searching

    def __init__(self, github_token: str, owner: str, repo: str, issue_number: int):
        self.owner = owner
        self.repo = repo
        self.issue_number = issue_number
        self.search_count = 0
        self.headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
        }
        self.base_url = f"https://api.github.com/repos/{owner}/{repo}"

    def _check_rate_limit(self, resp):
        """Check GitHub API rate limit and wait if needed."""
        remaining = resp.headers.get("X-RateLimit-Remaining")
        if remaining and int(remaining) < 5:
            reset = int(resp.headers.get("X-RateLimit-Reset", 0))
            wait = max(reset - int(time.time()), 1)
            print(f"    Rate limit low ({remaining} remaining), waiting {wait}s...")
            time.sleep(min(wait, 60))

    def search_code(self, query: str) -> str:
        self.search_count += 1
        if self.search_count > self.MAX_SEARCHES:
            return (f"Search limit reached ({self.MAX_SEARCHES} searches used). "
                    "Use list_directory and read_file to find remaining files, "
                    "then proceed to create_branch and write_file.")
        search_headers = {**self.headers, "Accept": "application/vnd.github.text-match+json"}
        params = {"q": f"{query} repo:{self.owner}/{self.repo}", "per_page": 30}
        resp = requests.get("https://api.github.com/search/code",
                            headers=search_headers, params=params, timeout=REQUEST_TIMEOUT)
        self._check_rate_limit(resp)
        if resp.status_code == 403:
            return "Search rate limited. Use list_directory and read_file instead."
        if resp.status_code != 200:
            return f"Search failed: {resp.status_code}. Use list_directory and read_file instead."
        items = resp.json().get("items", [])
        results = []
        for item in items:
            fragments = [tm.get("fragment", "") for tm in item.get("text_matches", [])]
            results.append({"path": item["path"], "matches": fragments})
        return json.dumps(results, indent=2)

    def list_files(self) -> str:
        resp = requests.get(f"{self.base_url}/git/trees/main?recursive=1",
                            headers=self.headers, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return f"Failed to list files: {resp.status_code}"
        paths = [item["path"] for item in resp.json().get("tree", []) if item["type"] == "blob"]
        return json.dumps(paths)

    def list_directory(self, path: str) -> str:
        """List contents of a specific directory."""
        clean_path = path.strip("/").strip()
        if not clean_path or clean_path == ".":
            clean_path = ""
        url = f"{self.base_url}/contents/{clean_path}" if clean_path else f"{self.base_url}/contents/"
        resp = requests.get(url, headers=self.headers, params={"ref": "main"},
                            timeout=REQUEST_TIMEOUT)
        self._check_rate_limit(resp)
        if resp.status_code != 200:
            return f"Directory not found: {path} ({resp.status_code})"
        data = resp.json()
        if not isinstance(data, list):
            return f"Not a directory (it's a file). Use read_file instead: {path}"
        entries = []
        for item in sorted(data, key=lambda x: (x["type"] != "dir", x["name"])):
            kind = "dir" if item["type"] == "dir" else "file"
            size_info = f" ({item.get('size', 0)} bytes)" if kind == "file" else ""
            entries.append(f"  {kind}: {item['name']}{size_info}")
        return f"Contents of '{path or '/'}':\n" + "\n".join(entries)

    def read_file(self, path: str) -> str:
        resp = requests.get(f"{self.base_url}/contents/{path}",
                            headers=self.headers, params={"ref": "main"}, timeout=REQUEST_TIMEOUT)
        self._check_rate_limit(resp)
        if resp.status_code != 200:
            return f"File not found: {path} ({resp.status_code})"
        data = resp.json()
        size = data.get("size", 0)
        if size > 500000:
            return f"File too large to read ({size} bytes): {path}"
        content = base64.b64decode(data["content"]).decode("utf-8")
        if len(content) > MAX_FILE_SIZE:
            return content[:MAX_FILE_SIZE] + f"\n\n... [TRUNCATED — file is {len(content)} chars, showing first {MAX_FILE_SIZE}]"
        return content

    def create_branch(self, branch_name: str) -> str:
        ref_resp = requests.get(f"{self.base_url}/git/ref/heads/main",
                                headers=self.headers, timeout=REQUEST_TIMEOUT)
        if ref_resp.status_code != 200:
            return f"Failed to get main ref: {ref_resp.status_code}"
        sha = ref_resp.json()["object"]["sha"]
        resp = requests.post(
            f"{self.base_url}/git/refs", headers=self.headers,
            json={"ref": f"refs/heads/{branch_name}", "sha": sha},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 201:
            return f"Branch '{branch_name}' created successfully"
        elif resp.status_code == 422:
            return f"Branch '{branch_name}' already exists"
        return f"Failed to create branch: {resp.status_code} {resp.text}"

    def write_file(self, path: str, content: str, branch: str, message: str) -> str:
        existing = requests.get(
            f"{self.base_url}/contents/{path}",
            headers=self.headers, params={"ref": branch}, timeout=REQUEST_TIMEOUT,
        )
        payload = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": branch,
        }
        if existing.status_code == 200:
            payload["sha"] = existing.json()["sha"]

        resp = requests.put(f"{self.base_url}/contents/{path}",
                            headers=self.headers, json=payload, timeout=REQUEST_TIMEOUT)
        self._check_rate_limit(resp)
        if resp.status_code in (200, 201):
            return f"File '{path}' written successfully"
        return f"Failed to write '{path}': {resp.status_code} {resp.text}"

    def create_pull_request(self, title: str, body: str, branch: str) -> str:
        resp = requests.post(
            f"{self.base_url}/pulls", headers=self.headers,
            json={"title": title, "body": body, "head": branch, "base": "main"},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 201:
            url = resp.json()["html_url"]
            return f"PR created: {url}"
        return f"Failed to create PR: {resp.status_code} {resp.text}"

    def post_comment(self, body: str) -> str:
        resp = requests.post(
            f"{self.base_url}/issues/{self.issue_number}/comments",
            headers=self.headers, json={"body": body}, timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 201:
            return "Comment posted"
        return f"Failed to post comment: {resp.status_code}"

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """Dispatch a tool call to the appropriate method."""
        dispatch = {
            "search_code": lambda: self.search_code(tool_input["query"]),
            "list_files": lambda: self.list_files(),
            "list_directory": lambda: self.list_directory(tool_input["path"]),
            "read_file": lambda: self.read_file(tool_input["path"]),
            "create_branch": lambda: self.create_branch(tool_input["branch_name"]),
            "write_file": lambda: self.write_file(
                tool_input["path"], tool_input["content"],
                tool_input["branch"], tool_input["message"]
            ),
            "create_pull_request": lambda: self.create_pull_request(
                tool_input["title"], tool_input["body"], tool_input["branch"]
            ),
            "post_comment": lambda: self.post_comment(tool_input["body"]),
        }
        fn = dispatch.get(tool_name)
        if fn:
            try:
                return fn()
            except Exception as e:
                return f"Error executing {tool_name}: {e}"
        return f"Unknown tool: {tool_name}"


class GitHubIssueAgent:
    def __init__(self, github_token: str, anthropic_api_key: str):
        self.github_token = github_token
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.github_headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
        }

    def get_repository_context(self, owner: str, repo: str) -> Dict:
        """Gather repository context including README, structure, and recent commits."""
        base_url = f"https://api.github.com/repos/{owner}/{repo}"
        context = {"repo_info": {}, "readme": "", "structure": [], "recent_commits": []}

        resp = requests.get(base_url, headers=self.github_headers, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            d = resp.json()
            context["repo_info"] = {
                "name": d.get("name"), "description": d.get("description"),
                "language": d.get("language"), "topics": d.get("topics", []),
            }

        try:
            r = requests.get(f"{base_url}/readme", headers=self.github_headers, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                context["readme"] = base64.b64decode(r.json()["content"]).decode("utf-8")
        except Exception:
            pass

        try:
            r = requests.get(f"{base_url}/git/trees/main?recursive=1", headers=self.github_headers, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                context["structure"] = [i["path"] for i in r.json().get("tree", [])[:100]]
        except Exception:
            pass

        try:
            r = requests.get(f"{base_url}/commits?per_page=10", headers=self.github_headers, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                context["recent_commits"] = [
                    {"sha": c["sha"][:7], "message": c["commit"]["message"],
                     "author": c["commit"]["author"]["name"]}
                    for c in r.json()
                ]
        except Exception:
            pass

        return context

    def get_issue_details(self, owner: str, repo: str, issue_number: int) -> Dict:
        """Fetch detailed information about a specific issue."""
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
        resp = requests.get(url, headers=self.github_headers, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            raise Exception(f"Failed to fetch issue: {resp.status_code}")

        d = resp.json()
        comments = []
        if d.get("comments", 0) > 0:
            cr = requests.get(d["comments_url"], headers=self.github_headers, timeout=REQUEST_TIMEOUT)
            if cr.status_code == 200:
                comments = [
                    {"author": c["user"]["login"], "body": c["body"], "created_at": c["created_at"]}
                    for c in cr.json()
                ]

        return {
            "number": d["number"], "title": d["title"], "body": d["body"],
            "state": d["state"],
            "labels": [l["name"] for l in d.get("labels", [])],
            "assignees": [a["login"] for a in d.get("assignees", [])],
            "created_at": d["created_at"], "updated_at": d["updated_at"],
            "comments": comments,
        }

    def analyze_issue(self, owner: str, repo: str, issue_number: int,
                      extended_thinking: bool = True) -> Dict:
        """Analyze an issue using Claude and provide a resolution plan."""
        print(f"Fetching repository context for {owner}/{repo}...")
        repo_context = self.get_repository_context(owner, repo)

        print(f"Fetching issue #{issue_number} details...")
        issue = self.get_issue_details(owner, repo, issue_number)

        print("Analyzing issue with Claude...")

        prompt = f"""You are a GitHub issue analysis agent. Analyze the following issue and provide a structured action plan.

REPOSITORY CONTEXT:
Repository: {owner}/{repo}
Description: {repo_context['repo_info'].get('description', 'N/A')}
Primary Language: {repo_context['repo_info'].get('language', 'N/A')}
Topics: {', '.join(repo_context['repo_info'].get('topics', []))}

README Summary:
{repo_context['readme'][:2000] if repo_context['readme'] else 'No README available'}

Repository Structure (key files):
{chr(10).join(repo_context['structure'][:50])}

Recent Commits:
{chr(10).join([f"- {c['sha']}: {c['message'][:100]}" for c in repo_context['recent_commits']])}

ISSUE DETAILS:
Issue #{issue['number']}: {issue['title']}
State: {issue['state']}
Labels: {', '.join(issue['labels']) if issue['labels'] else 'None'}
Created: {issue['created_at']}

Description:
{issue['body']}

Comments ({len(issue['comments'])}):
{chr(10).join([f"- {c['author']}: {c['body'][:200]}" for c in issue['comments'][:5]])}

Provide:
1. **Issue Classification**: bug, feature request, documentation, etc.
2. **Severity Assessment**: critical/high/medium/low and why
3. **Root Cause Analysis**: potential root causes based on repo context
4. **Action Required**: yes/no and explanation
5. **Resolution Plan**: step-by-step with specific files, code changes, testing
6. **Estimated Effort**: hours/days
7. **Related Issues**: common patterns in similar projects"""

        kwargs = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if extended_thinking:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 3000}

        response = self.client.messages.create(**kwargs)

        analysis_text = ""
        thinking_text = ""
        for block in response.content:
            if block.type == "thinking":
                thinking_text = block.thinking
            elif block.type == "text":
                analysis_text = block.text

        return {
            "issue_number": issue_number,
            "issue_title": issue["title"],
            "repository": f"{owner}/{repo}",
            "analysis": analysis_text,
            "thinking_process": thinking_text if extended_thinking else None,
            "model_used": response.model,
            "tokens_used": {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
            },
        }

    def post_analysis_comment(self, owner: str, repo: str, issue_number: int,
                              analysis: Dict):
        """Post the analysis as a comment on the GitHub issue."""
        comment_body = f"""## AI Analysis Report

{analysis['analysis']}

---
*Generated by Claude Issue Agent*
*Model: {analysis['model_used']} | Tokens: {analysis['tokens_used']['input']}->{analysis['tokens_used']['output']}*
"""
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
        resp = requests.post(url, headers=self.github_headers, json={"body": comment_body}, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 201:
            print(f"Analysis posted to issue #{issue_number}")
            return True
        print(f"Failed to post comment: {resp.status_code}")
        return False

    def implement_issue(self, owner: str, repo: str, issue_number: int,
                        analysis_result: Dict) -> Optional[str]:
        """
        Agentic implementation: Claude autonomously uses tools to search code,
        read files, create a branch, write fixes, and open a PR.
        """
        tools = GitHubTools(self.github_token, owner, repo, issue_number)
        issue = self.get_issue_details(owner, repo, issue_number)

        system_prompt = f"""You are an autonomous code implementation agent. Your job is to resolve
GitHub issue #{issue_number} by creating a pull request with the necessary changes.

CRITICAL: You have a STRICT budget of {MAX_AGENT_TURNS} turns. You MUST create a PR within this budget.
Do NOT spend all turns searching — prioritize action over exhaustive discovery.

Available tools: search_code (max 3 uses), list_directory, list_files, read_file,
create_branch, write_file, create_pull_request, post_comment

Follow this workflow IN ORDER:

PHASE 1 — DISCOVER (turns 1-4 max):
- Use list_directory to browse relevant directories (preferred over search_code)
- Use search_code with 1-2 targeted queries for specific patterns (you only get 3 searches total)
- Use read_file to read files you plan to change
- Focus on files mentioned in the analysis below — do not search aimlessly

PHASE 2 — IMPLEMENT (start by turn 5 at the latest):
- Use create_branch to create branch 'bot/fix-issue-{issue_number}'
- Use write_file for each file that needs changes (provide COMPLETE file content)
- Only change what is necessary to resolve the issue

PHASE 3 — FINALIZE:
- Use create_pull_request to open a PR (include 'Closes #{issue_number}' in body)
- Use post_comment to notify on the issue with a link to the PR

RULES:
- You MUST call create_branch by turn 5
- You MUST call create_pull_request before running out of turns
- Do NOT use read_file on directories — use list_directory instead
- If search returns poor results, use list_directory to browse and find files manually
- It is better to fix most files than to search forever trying to find every file"""

        user_msg = f"""Resolve this issue:

ISSUE #{issue['number']}: {issue['title']}
Body: {issue['body'] or '(no description)'}

ANALYSIS:
{analysis_result['analysis'][:3000]}

Start with 1-2 searches or list_directory calls, read the key files, then create a branch and implement the fix."""

        messages = [{"role": "user", "content": user_msg}]
        pr_url = None
        branch_created = False
        start_time = time.time()
        time_limit = 8 * 60  # 8 minutes max for the agent loop

        for turn in range(MAX_AGENT_TURNS):
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                print(f"  Time limit reached ({int(elapsed)}s). Stopping agent.")
                tools.post_comment(
                    f"Agent timed out after {int(elapsed)}s and {turn} turns. "
                    "The issue may be too complex for automated resolution."
                )
                break

            print(f"  Agent turn {turn + 1}/{MAX_AGENT_TURNS} ({int(elapsed)}s elapsed)...")

            try:
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=16000,
                    system=system_prompt,
                    tools=IMPLEMENTATION_TOOLS,
                    messages=messages,
                )
            except Exception as e:
                print(f"  Claude API error: {e}")
                tools.post_comment(f"Agent encountered an API error: {e}")
                break

            # Check if Claude is done (no more tool calls)
            if response.stop_reason == "end_turn":
                print("  Agent finished.")
                for block in response.content:
                    if block.type == "text":
                        print(f"  Final: {block.text[:200]}")
                        if "PR created:" in block.text:
                            match = re.search(r'https://github\.com/\S+', block.text)
                            if match:
                                pr_url = match.group()
                break

            # Process tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    # Log tool call (truncate long values for readability)
                    short_input = {k: (v[:60] + "...") if isinstance(v, str) and len(v) > 60 else v
                                   for k, v in block.input.items()}
                    print(f"    Tool: {block.name}({json.dumps(short_input)})")
                    result = tools.execute(block.name, block.input)
                    # Track branch creation
                    if block.name == "create_branch" and "successfully" in result:
                        branch_created = True
                    # Capture PR URL from create_pull_request
                    if block.name == "create_pull_request" and "PR created:" in result:
                        match = re.search(r'https://github\.com/\S+', result)
                        if match:
                            pr_url = match.group()
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result[:8000],  # limit result size
                    })
                elif block.type == "text" and block.text:
                    print(f"    Think: {block.text[:150]}")

            messages.append({"role": "assistant", "content": response.content})

            # Phase enforcement: nudge Claude if it hasn't started implementing
            nudge = None
            if turn >= 4 and not branch_created:
                remaining = MAX_AGENT_TURNS - turn - 1
                nudge = (
                    f"URGENT: You have used {turn + 1} turns without creating a branch. "
                    f"Only {remaining} turns remain. You MUST call create_branch NOW, "
                    "then write_file for each change, then create_pull_request. "
                    "Stop searching and start implementing immediately."
                )
                print(f"    NUDGE: {nudge}")
            elif turn >= 2 and not branch_created and tools.search_count >= 2:
                nudge = (
                    "You've done enough searching. Move to implementation: "
                    "read the files you need, then create_branch and write_file."
                )

            if nudge:
                tool_results.append({"type": "text", "text": nudge})

            messages.append({"role": "user", "content": tool_results})

        print(f"  Agent completed in {int(time.time() - start_time)}s, {turn + 1} turns")
        return pr_url
