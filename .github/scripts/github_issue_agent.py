#!/usr/bin/env python3
"""
GitHub Issue Analysis Agent
Uses Claude API to analyze GitHub issues with repository context
and provide actionable resolution plans.
"""

import os
import anthropic
import requests
from typing import Dict, List, Optional
import json

class GitHubIssueAgent:
    def __init__(self, github_token: str, anthropic_api_key: str):
        """
        Initialize the GitHub Issue Agent

        Args:
            github_token: GitHub personal access token
            anthropic_api_key: Anthropic API key for Claude
        """
        self.github_token = github_token
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.github_headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }

    def get_repository_context(self, owner: str, repo: str) -> Dict:
        """
        Gather repository context including README, structure, and recent commits
        """
        base_url = f"https://api.github.com/repos/{owner}/{repo}"

        context = {
            "repo_info": {},
            "readme": "",
            "structure": [],
            "recent_commits": []
        }

        # Get repository info
        response = requests.get(base_url, headers=self.github_headers)
        if response.status_code == 200:
            repo_data = response.json()
            context["repo_info"] = {
                "name": repo_data.get("name"),
                "description": repo_data.get("description"),
                "language": repo_data.get("language"),
                "topics": repo_data.get("topics", [])
            }

        # Get README
        try:
            readme_response = requests.get(
                f"{base_url}/readme",
                headers=self.github_headers
            )
            if readme_response.status_code == 200:
                readme_data = readme_response.json()
                # Decode base64 content
                import base64
                context["readme"] = base64.b64decode(
                    readme_data["content"]
                ).decode('utf-8')
        except Exception as e:
            print(f"Could not fetch README: {e}")

        # Get repository tree (file structure)
        try:
            tree_response = requests.get(
                f"{base_url}/git/trees/main?recursive=1",
                headers=self.github_headers
            )
            if tree_response.status_code == 200:
                tree_data = tree_response.json()
                context["structure"] = [
                    item["path"] for item in tree_data.get("tree", [])[:100]
                ]
        except Exception as e:
            print(f"Could not fetch repository structure: {e}")

        # Get recent commits
        try:
            commits_response = requests.get(
                f"{base_url}/commits?per_page=10",
                headers=self.github_headers
            )
            if commits_response.status_code == 200:
                commits_data = commits_response.json()
                context["recent_commits"] = [
                    {
                        "sha": commit["sha"][:7],
                        "message": commit["commit"]["message"],
                        "author": commit["commit"]["author"]["name"]
                    }
                    for commit in commits_data
                ]
        except Exception as e:
            print(f"Could not fetch recent commits: {e}")

        return context

    def get_issue_details(self, owner: str, repo: str, issue_number: int) -> Dict:
        """
        Fetch detailed information about a specific issue
        """
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
        response = requests.get(url, headers=self.github_headers)

        if response.status_code == 200:
            issue_data = response.json()

            # Get comments
            comments = []
            if issue_data.get("comments", 0) > 0:
                comments_url = issue_data["comments_url"]
                comments_response = requests.get(
                    comments_url,
                    headers=self.github_headers
                )
                if comments_response.status_code == 200:
                    comments_data = comments_response.json()
                    comments = [
                        {
                            "author": comment["user"]["login"],
                            "body": comment["body"],
                            "created_at": comment["created_at"]
                        }
                        for comment in comments_data
                    ]

            return {
                "number": issue_data["number"],
                "title": issue_data["title"],
                "body": issue_data["body"],
                "state": issue_data["state"],
                "labels": [label["name"] for label in issue_data.get("labels", [])],
                "assignees": [assignee["login"] for assignee in issue_data.get("assignees", [])],
                "created_at": issue_data["created_at"],
                "updated_at": issue_data["updated_at"],
                "comments": comments
            }
        else:
            raise Exception(f"Failed to fetch issue: {response.status_code}")

    def analyze_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        extended_thinking: bool = True
    ) -> Dict:
        """
        Analyze an issue using Claude and provide a resolution plan
        """
        print(f"Fetching repository context for {owner}/{repo}...")
        repo_context = self.get_repository_context(owner, repo)

        print(f"Fetching issue #{issue_number} details...")
        issue = self.get_issue_details(owner, repo, issue_number)

        print("Analyzing issue with Claude...")

        prompt = f"""You are a GitHub issue analysis agent. Your task is to analyze the following issue in the context of the repository and provide a structured action plan.

REPOSITORY CONTEXT:
-------------------
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
--------------
Issue #{issue['number']}: {issue['title']}
State: {issue['state']}
Labels: {', '.join(issue['labels']) if issue['labels'] else 'None'}
Created: {issue['created_at']}

Description:
{issue['body']}

Comments ({len(issue['comments'])}):
{chr(10).join([f"- {c['author']}: {c['body'][:200]}" for c in issue['comments'][:5]])}

ANALYSIS TASK:
--------------
Please analyze this issue and provide:

1. **Issue Classification**: Categorize the issue (bug, feature request, documentation, question, etc.)

2. **Severity Assessment**: Rate the severity (critical, high, medium, low) and explain why

3. **Root Cause Analysis**: Based on the repository context, identify potential root causes

4. **Action Required**: Determine if this issue requires action (yes/no) and explain

5. **Resolution Plan**: Provide a detailed, step-by-step plan to resolve this issue, including:
   - Specific files that likely need to be modified
   - Code changes or implementation approach
   - Testing recommendations
   - Documentation updates needed

6. **Estimated Effort**: Estimate the effort required (hours/days)

7. **Related Issues**: Identify if this might be related to other common issues in similar projects

Please structure your response in a clear, actionable format."""

        kwargs = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        }
        if extended_thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": 3000
            }
        response = self.client.messages.create(**kwargs)

        analysis_text = ""
        thinking_text = ""

        for block in response.content:
            if block.type == "thinking":
                thinking_text = block.thinking
            elif block.type == "text":
                analysis_text = block.text

        result = {
            "issue_number": issue_number,
            "issue_title": issue['title'],
            "repository": f"{owner}/{repo}",
            "analysis": analysis_text,
            "thinking_process": thinking_text if extended_thinking else None,
            "model_used": response.model,
            "tokens_used": {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens
            }
        }

        return result

    def post_analysis_comment(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        analysis: Dict
    ):
        """
        Post the analysis as a comment on the GitHub issue
        """
        comment_body = f"""## AI Analysis Report

{analysis['analysis']}

---
*Generated by Claude Issue Agent*
*Model: {analysis['model_used']} | Tokens: {analysis['tokens_used']['input']}->{analysis['tokens_used']['output']}*
"""

        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
        response = requests.post(
            url,
            headers=self.github_headers,
            json={"body": comment_body}
        )

        if response.status_code == 201:
            print(f"Analysis posted to issue #{issue_number}")
            return True
        else:
            print(f"Failed to post comment: {response.status_code}")
            return False


    def get_file_content(self, owner: str, repo: str, path: str, ref: str = "main") -> Optional[str]:
        """Fetch a file's content from the repository."""
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        response = requests.get(url, headers=self.github_headers, params={"ref": ref})
        if response.status_code == 200:
            import base64
            return base64.b64decode(response.json()["content"]).decode("utf-8")
        return None

    def generate_implementation(
        self,
        owner: str,
        repo: str,
        issue: Dict,
        analysis: str,
        repo_context: Dict,
    ) -> Optional[Dict]:
        """
        Ask Claude to generate concrete file changes to resolve the issue.

        Returns a dict with keys: changes (list of file edits), pr_title, pr_body.
        """
        file_hints = [p for p in repo_context.get("structure", []) if not p.endswith("/")]

        prompt = f"""You are a code implementation agent. Based on the following issue and analysis,
generate the EXACT file changes needed to resolve this issue.

REPOSITORY: {owner}/{repo}
ISSUE #{issue['number']}: {issue['title']}

ISSUE BODY:
{issue['body'] or '(no description)'}

ANALYSIS:
{analysis}

REPOSITORY FILES:
{chr(10).join(file_hints[:80])}

INSTRUCTIONS:
1. Identify which files need to be modified or created.
2. For each file, provide the COMPLETE new file content (not a diff).
3. Only change what is necessary to resolve the issue.
4. Keep changes minimal and focused.

Respond with ONLY a JSON object (no markdown fences) in this exact format:
{{
  "pr_title": "Short PR title (under 70 chars)",
  "pr_body": "Description of what changes were made and why",
  "changes": [
    {{
      "path": "relative/file/path",
      "description": "What was changed in this file"
    }}
  ]
}}

List the files that need changing. I will then provide their current contents
so you can generate the updated versions."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        plan_text = ""
        for block in response.content:
            if block.type == "text":
                plan_text = block.text
                break

        try:
            plan = json.loads(plan_text)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{[\s\S]*\}', plan_text)
            if match:
                plan = json.loads(match.group())
            else:
                print("Failed to parse implementation plan from Claude response")
                return None

        # Fetch current file contents and ask Claude to generate updated versions
        file_contents = {}
        for change in plan.get("changes", []):
            path = change["path"]
            content = self.get_file_content(owner, repo, path)
            if content is not None:
                file_contents[path] = content
            else:
                file_contents[path] = None  # new file

        files_context = ""
        for path, content in file_contents.items():
            if content is not None:
                files_context += f"\n--- FILE: {path} ---\n{content}\n--- END FILE ---\n"
            else:
                files_context += f"\n--- FILE: {path} (NEW FILE) ---\n--- END FILE ---\n"

        implementation_prompt = f"""Based on the plan below, generate the COMPLETE updated content for each file.

ISSUE #{issue['number']}: {issue['title']}
ISSUE BODY: {issue['body'] or '(no description)'}

PLAN:
PR Title: {plan['pr_title']}
PR Body: {plan['pr_body']}
Files to change:
{json.dumps(plan['changes'], indent=2)}

CURRENT FILE CONTENTS:
{files_context}

Respond with ONLY a JSON object (no markdown fences) in this exact format:
{{
  "files": [
    {{
      "path": "relative/file/path",
      "content": "COMPLETE new file content here"
    }}
  ]
}}

IMPORTANT:
- Provide the COMPLETE file content, not just the changed parts.
- For existing files, include ALL original content with only the necessary modifications.
- For new files, provide the full content."""

        impl_response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=16000,
            messages=[{"role": "user", "content": implementation_prompt}],
        )

        impl_text = ""
        for block in impl_response.content:
            if block.type == "text":
                impl_text = block.text
                break

        try:
            impl = json.loads(impl_text)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{[\s\S]*\}', impl_text)
            if match:
                impl = json.loads(match.group())
            else:
                print("Failed to parse implementation from Claude response")
                return None

        return {
            "pr_title": plan["pr_title"],
            "pr_body": plan["pr_body"],
            "changes": plan["changes"],
            "files": impl["files"],
        }

    def create_branch(self, owner: str, repo: str, branch_name: str, base_ref: str = "main") -> bool:
        """Create a new branch from base_ref."""
        url = f"https://api.github.com/repos/{owner}/{repo}/git/ref/heads/{base_ref}"
        response = requests.get(url, headers=self.github_headers)
        if response.status_code != 200:
            print(f"Failed to get base ref: {response.status_code}")
            return False

        base_sha = response.json()["object"]["sha"]

        url = f"https://api.github.com/repos/{owner}/{repo}/git/refs"
        response = requests.post(
            url,
            headers=self.github_headers,
            json={"ref": f"refs/heads/{branch_name}", "sha": base_sha},
        )
        if response.status_code == 201:
            print(f"Created branch: {branch_name}")
            return True
        else:
            print(f"Failed to create branch: {response.status_code} {response.text}")
            return False

    def commit_files(
        self, owner: str, repo: str, branch: str, files: List[Dict], message: str
    ) -> bool:
        """
        Commit multiple file changes to a branch in a single commit
        using the Git Data API (trees + commits).
        """
        base_url = f"https://api.github.com/repos/{owner}/{repo}"

        # 1. Get the latest commit SHA on the branch
        ref_resp = requests.get(
            f"{base_url}/git/ref/heads/{branch}", headers=self.github_headers
        )
        if ref_resp.status_code != 200:
            print(f"Failed to get branch ref: {ref_resp.status_code}")
            return False
        latest_sha = ref_resp.json()["object"]["sha"]

        # 2. Get the tree SHA of that commit
        commit_resp = requests.get(
            f"{base_url}/git/commits/{latest_sha}", headers=self.github_headers
        )
        base_tree_sha = commit_resp.json()["tree"]["sha"]

        # 3. Create blobs for each file and build tree entries
        tree_items = []
        for f in files:
            blob_resp = requests.post(
                f"{base_url}/git/blobs",
                headers=self.github_headers,
                json={"content": f["content"], "encoding": "utf-8"},
            )
            if blob_resp.status_code != 201:
                print(f"Failed to create blob for {f['path']}: {blob_resp.status_code}")
                return False
            tree_items.append(
                {
                    "path": f["path"],
                    "mode": "100644",
                    "type": "blob",
                    "sha": blob_resp.json()["sha"],
                }
            )

        # 4. Create a new tree
        tree_resp = requests.post(
            f"{base_url}/git/trees",
            headers=self.github_headers,
            json={"base_tree": base_tree_sha, "tree": tree_items},
        )
        if tree_resp.status_code != 201:
            print(f"Failed to create tree: {tree_resp.status_code}")
            return False
        new_tree_sha = tree_resp.json()["sha"]

        # 5. Create a new commit
        commit_resp = requests.post(
            f"{base_url}/git/commits",
            headers=self.github_headers,
            json={
                "message": message,
                "tree": new_tree_sha,
                "parents": [latest_sha],
            },
        )
        if commit_resp.status_code != 201:
            print(f"Failed to create commit: {commit_resp.status_code}")
            return False
        new_commit_sha = commit_resp.json()["sha"]

        # 6. Update the branch reference
        ref_update = requests.patch(
            f"{base_url}/git/refs/heads/{branch}",
            headers=self.github_headers,
            json={"sha": new_commit_sha},
        )
        if ref_update.status_code == 200:
            print(f"Committed {len(files)} file(s) to {branch}")
            return True
        else:
            print(f"Failed to update ref: {ref_update.status_code}")
            return False

    def create_pull_request(
        self, owner: str, repo: str, title: str, body: str, head: str, base: str = "main"
    ) -> Optional[str]:
        """Create a pull request. Returns the PR URL or None."""
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        response = requests.post(
            url,
            headers=self.github_headers,
            json={"title": title, "body": body, "head": head, "base": base},
        )
        if response.status_code == 201:
            pr_url = response.json()["html_url"]
            print(f"Created PR: {pr_url}")
            return pr_url
        else:
            print(f"Failed to create PR: {response.status_code} {response.text}")
            return None

    def implement_issue(
        self, owner: str, repo: str, issue_number: int, analysis_result: Dict
    ) -> Optional[str]:
        """
        Full implementation flow: generate changes -> create branch -> commit -> open PR.
        Returns the PR URL or None.
        """
        print(f"\nGenerating implementation for issue #{issue_number}...")

        repo_context = self.get_repository_context(owner, repo)
        issue = self.get_issue_details(owner, repo, issue_number)

        impl = self.generate_implementation(
            owner, repo, issue, analysis_result["analysis"], repo_context
        )
        if not impl or not impl.get("files"):
            print("No implementation changes generated.")
            return None

        branch_name = f"bot/fix-issue-{issue_number}"

        print(f"Creating branch {branch_name}...")
        if not self.create_branch(owner, repo, branch_name):
            return None

        commit_msg = f"fix: {impl['pr_title']} (#{issue_number})"
        print(f"Committing {len(impl['files'])} file(s)...")
        if not self.commit_files(owner, repo, branch_name, impl["files"], commit_msg):
            return None

        pr_body = f"""## Summary
{impl['pr_body']}

### Changes
{chr(10).join(f"- **{c['path']}**: {c['description']}" for c in impl['changes'])}

Closes #{issue_number}

---
*Generated automatically by Claude Issue Agent*"""

        print("Opening pull request...")
        pr_url = self.create_pull_request(
            owner, repo, impl["pr_title"], pr_body, branch_name
        )

        if pr_url:
            # Post a comment on the issue linking to the PR
            comment = f"I've created a pull request with a proposed fix: {pr_url}"
            requests.post(
                f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments",
                headers=self.github_headers,
                json={"body": comment},
            )

        return pr_url
