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
        
        Args:
            owner: Repository owner
            repo: Repository name
            issue_number: Issue number to analyze
            extended_thinking: Use Claude's extended thinking capability
        
        Returns:
            Dictionary containing analysis and action plan
        """
        print(f"Fetching repository context for {owner}/{repo}...")
        repo_context = self.get_repository_context(owner, repo)
        
        print(f"Fetching issue #{issue_number} details...")
        issue = self.get_issue_details(owner, repo, issue_number)
        
        print("Analyzing issue with Claude...")
        
        # Construct the prompt for Claude
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

        # Call Claude API
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
        
        # Extract the analysis from response
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
        comment_body = f"""## 🤖 AI Analysis Report

{analysis['analysis']}

---
*Generated by Claude Issue Agent*
*Model: {analysis['model_used']} | Tokens: {analysis['tokens_used']['input']}→{analysis['tokens_used']['output']}*
"""
        
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
        response = requests.post(
            url,
            headers=self.github_headers,
            json={"body": comment_body}
        )
        
        if response.status_code == 201:
            print(f"✓ Analysis posted to issue #{issue_number}")
            return True
        else:
            print(f"✗ Failed to post comment: {response.status_code}")
            return False


    def get_file_content(self, owner: str, repo: str, path: str, ref: str = "main") -> Optional[str]:
        """Fetch a file's content from the repository."""
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        response = requests.get(url, headers=self.github_headers, params={"ref": ref})
        if response.status_code == 200:
            import base64
            return base64.b64decode(response.json()["content"]).decode("utf-8")
        return None

    def search_code(self, owner: str, repo: str, query: str, per_page: int = 100) -> List[Dict]:
        """
        Search for code patterns in the repository using GitHub Code Search API.
        Returns list of {path, matches} for files containing the query.
        """
        search_headers = {**self.github_headers, "Accept": "application/vnd.github.text-match+json"}
        url = "https://api.github.com/search/code"
        params = {"q": f"{query} repo:{owner}/{repo}", "per_page": per_page}
        response = requests.get(url, headers=search_headers, params=params)
        if response.status_code == 200:
            data = response.json()
            results = []
            for item in data.get("items", []):
                match_info = {
                    "path": item["path"],
                    "matches": []
                }
                for tm in item.get("text_matches", []):
                    match_info["matches"].append(tm.get("fragment", ""))
                results.append(match_info)
            print(f"  Code search for '{query}': found {len(results)} files")
            return results
        else:
            print(f"  Code search failed: {response.status_code}")
            return []

    def get_full_tree(self, owner: str, repo: str, ref: str = "main") -> List[str]:
        """Get the full file tree of the repository (all file paths)."""
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}?recursive=1"
        response = requests.get(url, headers=self.github_headers)
        if response.status_code == 200:
            return [item["path"] for item in response.json().get("tree", [])
                    if item["type"] == "blob"]
        return []

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
        Uses a 3-step approach:
          1. Ask Claude what search queries to run to find affected files
          2. Run those searches, collect matching files
          3. Ask Claude to generate the actual changes

        Returns a dict with keys: changes, files, pr_title, pr_body.
        """
        import re

        all_files = self.get_full_tree(owner, repo)
        print(f"Repository has {len(all_files)} files total")

        # Step 1: Ask Claude what to search for
        search_prompt = f"""You are a code implementation agent. For the issue below, I need to find
ALL files in the repository that need to be modified.

REPOSITORY: {owner}/{repo}
ISSUE #{issue['number']}: {issue['title']}
ISSUE BODY: {issue['body'] or '(no description)'}

ANALYSIS SUMMARY:
{analysis[:2000]}

ALL REPOSITORY FILES ({len(all_files)} files):
{chr(10).join(all_files)}

Tell me:
1. What code search queries should I run to find ALL affected files?
   (e.g., for copyright updates, search for "Copyright" and year patterns)
2. Are there specific file paths from the list above that should be included?

Respond with ONLY a JSON object (no markdown fences):
{{
  "search_queries": ["query1", "query2"],
  "explicit_paths": ["path/to/file1", "path/to/file2"],
  "pr_title": "Short PR title (under 70 chars)",
  "pr_body": "Description of what changes will be made and why"
}}"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": search_prompt}],
        )

        plan_text = ""
        for block in response.content:
            if block.type == "text":
                plan_text = block.text
                break

        try:
            plan = json.loads(plan_text)
        except json.JSONDecodeError:
            match = re.search(r'\{[\s\S]*\}', plan_text)
            if match:
                plan = json.loads(match.group())
            else:
                print("Failed to parse search plan from Claude response")
                return None

        # Step 2: Run the searches and collect all affected file paths
        affected_paths = set(plan.get("explicit_paths", []))

        for query in plan.get("search_queries", []):
            results = self.search_code(owner, repo, query)
            for r in results:
                affected_paths.add(r["path"])

        # Filter to paths that actually exist in the tree
        valid_paths = [p for p in affected_paths if p in all_files]
        print(f"Found {len(valid_paths)} files to potentially modify")

        if not valid_paths:
            print("No files found to modify.")
            return None

        # Step 3: Fetch all affected files and ask Claude to generate changes
        # Process in batches to stay within API limits
        MAX_FILES_PER_BATCH = 15
        all_changes = []
        all_impl_files = []

        for batch_start in range(0, len(valid_paths), MAX_FILES_PER_BATCH):
            batch_paths = valid_paths[batch_start:batch_start + MAX_FILES_PER_BATCH]
            batch_num = batch_start // MAX_FILES_PER_BATCH + 1
            total_batches = (len(valid_paths) + MAX_FILES_PER_BATCH - 1) // MAX_FILES_PER_BATCH
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch_paths)} files)...")

            files_context = ""
            for path in batch_paths:
                content = self.get_file_content(owner, repo, path)
                if content is not None:
                    files_context += f"\n--- FILE: {path} ---\n{content}\n--- END FILE ---\n"

            implementation_prompt = f"""You are a code implementation agent. Apply the MINIMAL changes needed
to resolve the issue for EACH file below.

ISSUE #{issue['number']}: {issue['title']}
ISSUE BODY: {issue['body'] or '(no description)'}

TASK: {plan['pr_body']}

CURRENT FILE CONTENTS:
{files_context}

For each file, decide if it actually needs changes. If a file does NOT need
changes (e.g., the search matched but the content is already correct), skip it.

Respond with ONLY a JSON object (no markdown fences):
{{
  "files": [
    {{
      "path": "relative/file/path",
      "content": "COMPLETE updated file content",
      "description": "What was changed"
    }}
  ]
}}

IMPORTANT:
- Only include files that ACTUALLY need changes.
- Provide the COMPLETE file content for each changed file.
- Make ONLY the changes needed to resolve the issue — nothing else."""

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
                match = re.search(r'\{[\s\S]*\}', impl_text)
                if match:
                    impl = json.loads(match.group())
                else:
                    print(f"  Failed to parse batch {batch_num} response, skipping")
                    continue

            for f in impl.get("files", []):
                all_impl_files.append({"path": f["path"], "content": f["content"]})
                all_changes.append({"path": f["path"], "description": f.get("description", "Updated")})

        if not all_impl_files:
            print("No file changes were generated.")
            return None

        print(f"Total files to change: {len(all_impl_files)}")

        return {
            "pr_title": plan["pr_title"],
            "pr_body": plan["pr_body"],
            "changes": all_changes,
            "files": all_impl_files,
        }

    def create_branch(self, owner: str, repo: str, branch_name: str, base_ref: str = "main") -> bool:
        """Create a new branch from base_ref."""
        # Get the SHA of the base branch
        url = f"https://api.github.com/repos/{owner}/{repo}/git/ref/heads/{base_ref}"
        response = requests.get(url, headers=self.github_headers)
        if response.status_code != 200:
            print(f"Failed to get base ref: {response.status_code}")
            return False

        base_sha = response.json()["object"]["sha"]

        # Create the new branch
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
        Full implementation flow: generate changes → create branch → commit → open PR.
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
