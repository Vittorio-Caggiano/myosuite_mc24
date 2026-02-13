#!/usr/bin/env python3
"""
GitHub Issue Analysis Agent
Uses Claude API with tool_use to analyze GitHub issues and autonomously
create PRs with fixes. Claude drives the process using GitHub API tools.
"""

import base64
import json
import re
import requests
import anthropic
from typing import Dict, List, Optional


# --- Tool definitions for Claude ---
IMPLEMENTATION_TOOLS = [
    {
        "name": "search_code",
        "description": "Search for code patterns in the repository using GitHub Code Search. Use this to find ALL files that contain a specific pattern (e.g., copyright notices, specific function names, outdated imports).",
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
        "description": "List all files in the repository. Returns the full file tree.",
        "input_schema": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name": "read_file",
        "description": "Read the content of a specific file in the repository.",
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

    def __init__(self, github_token: str, owner: str, repo: str, issue_number: int):
        self.owner = owner
        self.repo = repo
        self.issue_number = issue_number
        self.headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
        }
        self.base_url = f"https://api.github.com/repos/{owner}/{repo}"

    def search_code(self, query: str) -> str:
        search_headers = {**self.headers, "Accept": "application/vnd.github.text-match+json"}
        params = {"q": f"{query} repo:{self.owner}/{self.repo}", "per_page": 100}
        resp = requests.get("https://api.github.com/search/code", headers=search_headers, params=params)
        if resp.status_code != 200:
            return f"Search failed: {resp.status_code}"
        items = resp.json().get("items", [])
        results = []
        for item in items:
            fragments = [tm.get("fragment", "") for tm in item.get("text_matches", [])]
            results.append({"path": item["path"], "matches": fragments})
        return json.dumps(results, indent=2)

    def list_files(self) -> str:
        resp = requests.get(f"{self.base_url}/git/trees/main?recursive=1", headers=self.headers)
        if resp.status_code != 200:
            return f"Failed to list files: {resp.status_code}"
        paths = [item["path"] for item in resp.json().get("tree", []) if item["type"] == "blob"]
        return json.dumps(paths)

    def read_file(self, path: str) -> str:
        resp = requests.get(f"{self.base_url}/contents/{path}", headers=self.headers, params={"ref": "main"})
        if resp.status_code != 200:
            return f"File not found: {path} ({resp.status_code})"
        content = base64.b64decode(resp.json()["content"]).decode("utf-8")
        return content

    def create_branch(self, branch_name: str) -> str:
        # Get main HEAD
        ref_resp = requests.get(f"{self.base_url}/git/ref/heads/main", headers=self.headers)
        if ref_resp.status_code != 200:
            return f"Failed to get main ref: {ref_resp.status_code}"
        sha = ref_resp.json()["object"]["sha"]
        # Create branch
        resp = requests.post(
            f"{self.base_url}/git/refs",
            headers=self.headers,
            json={"ref": f"refs/heads/{branch_name}", "sha": sha},
        )
        if resp.status_code == 201:
            return f"Branch '{branch_name}' created successfully"
        elif resp.status_code == 422:
            return f"Branch '{branch_name}' already exists"
        return f"Failed to create branch: {resp.status_code} {resp.text}"

    def write_file(self, path: str, content: str, branch: str, message: str) -> str:
        # Check if file exists to get its SHA (needed for updates)
        existing = requests.get(
            f"{self.base_url}/contents/{path}",
            headers=self.headers,
            params={"ref": branch},
        )
        payload = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": branch,
        }
        if existing.status_code == 200:
            payload["sha"] = existing.json()["sha"]

        resp = requests.put(f"{self.base_url}/contents/{path}", headers=self.headers, json=payload)
        if resp.status_code in (200, 201):
            return f"File '{path}' written successfully"
        return f"Failed to write '{path}': {resp.status_code} {resp.text}"

    def create_pull_request(self, title: str, body: str, branch: str) -> str:
        resp = requests.post(
            f"{self.base_url}/pulls",
            headers=self.headers,
            json={"title": title, "body": body, "head": branch, "base": "main"},
        )
        if resp.status_code == 201:
            url = resp.json()["html_url"]
            return f"PR created: {url}"
        return f"Failed to create PR: {resp.status_code} {resp.text}"

    def post_comment(self, body: str) -> str:
        resp = requests.post(
            f"{self.base_url}/issues/{self.issue_number}/comments",
            headers=self.headers,
            json={"body": body},
        )
        if resp.status_code == 201:
            return "Comment posted"
        return f"Failed to post comment: {resp.status_code}"

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """Dispatch a tool call to the appropriate method."""
        dispatch = {
            "search_code": lambda: self.search_code(tool_input["query"]),
            "list_files": lambda: self.list_files(),
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

        resp = requests.get(base_url, headers=self.github_headers)
        if resp.status_code == 200:
            d = resp.json()
            context["repo_info"] = {
                "name": d.get("name"), "description": d.get("description"),
                "language": d.get("language"), "topics": d.get("topics", []),
            }

        try:
            r = requests.get(f"{base_url}/readme", headers=self.github_headers)
            if r.status_code == 200:
                context["readme"] = base64.b64decode(r.json()["content"]).decode("utf-8")
        except Exception:
            pass

        try:
            r = requests.get(f"{base_url}/git/trees/main?recursive=1", headers=self.github_headers)
            if r.status_code == 200:
                context["structure"] = [i["path"] for i in r.json().get("tree", [])[:100]]
        except Exception:
            pass

        try:
            r = requests.get(f"{base_url}/commits?per_page=10", headers=self.github_headers)
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
        resp = requests.get(url, headers=self.github_headers)
        if resp.status_code != 200:
            raise Exception(f"Failed to fetch issue: {resp.status_code}")

        d = resp.json()
        comments = []
        if d.get("comments", 0) > 0:
            cr = requests.get(d["comments_url"], headers=self.github_headers)
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
        resp = requests.post(url, headers=self.github_headers, json={"body": comment_body})
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

You have tools to interact with the repository. Follow this workflow:

1. SEARCH: Use search_code to find ALL files affected by this issue.
   Run multiple searches with different queries to be thorough.
2. READ: Use read_file to read the current content of each affected file.
3. BRANCH: Use create_branch to create a new branch (use 'bot/fix-issue-{issue_number}').
4. WRITE: Use write_file to update each file that needs changes.
   Provide the COMPLETE updated file content, not just the diff.
   Only change what is necessary to resolve the issue.
5. PR: Use create_pull_request to open a PR.
   Include 'Closes #{issue_number}' in the body.
6. COMMENT: Use post_comment to notify on the issue with a link to the PR.

Be thorough - search with multiple queries to find ALL affected files.
For example, for copyright updates, search for 'Copyright', the old year, etc.
Do NOT skip files. Read each one and update it if needed."""

        user_msg = f"""Resolve this issue:

ISSUE #{issue['number']}: {issue['title']}
Body: {issue['body'] or '(no description)'}

ANALYSIS:
{analysis_result['analysis'][:3000]}

Start by searching for all affected files, then implement the fix."""

        messages = [{"role": "user", "content": user_msg}]
        max_turns = 30
        pr_url = None

        for turn in range(max_turns):
            print(f"  Agent turn {turn + 1}...")

            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=16000,
                system=system_prompt,
                tools=IMPLEMENTATION_TOOLS,
                messages=messages,
            )

            # Check if Claude is done (no more tool calls)
            if response.stop_reason == "end_turn":
                print("  Agent finished.")
                # Extract any final text
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
                    print(f"    Tool: {block.name}({json.dumps({k: v[:80] if isinstance(v, str) and len(v) > 80 else v for k, v in block.input.items()})})")
                    result = tools.execute(block.name, block.input)
                    # Capture PR URL from create_pull_request
                    if block.name == "create_pull_request" and "PR created:" in result:
                        match = re.search(r'https://github\.com/\S+', result)
                        if match:
                            pr_url = match.group()
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result[:10000],  # limit result size
                    })
                elif block.type == "text" and block.text:
                    print(f"    Think: {block.text[:150]}")

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        return pr_url
