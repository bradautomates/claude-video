#!/usr/bin/env python3
"""
Batch import finished media-knowledge markdown pages into an AFFiNE workspace.

This is a reusable template. Copy it, set the env vars (or hardcode them), and
run it after each batch of KB pages is ready. It expects:

- AFFINE_BASE_URL      - e.g. https://mars.golden-hops.ts.net
- AFFINE_COOKIE        - session cookie string (see below)
- AFFINE_WORKSPACE_ID  - target workspace UUID
- KB_ROOT              - path to the markdown KB (default: ~/Personal/media-knowledge)

The script:
1. Opens a single long-lived stdio MCP connection to affine-mcp (avoids repeated auth / 429s).
2. Imports markdown files from videos/, channels/, persons/, and concepts/.
3. Strips YAML frontmatter and converts [[path/file.md|Title]] wikilinks to [[Title]].
4. Adds each imported doc under a sidebar folder (Videos, Channels, Persons, Concepts).

To get the session cookie once, sign in via REST and read the cookie jar:

    curl -c cookies.txt -H 'Content-Type: application/json' \
      -d '{"email":"admin@noblecloud.dev","password":"YOUR_PASSWORD"}' \
      https://mars.golden-hops.ts.net/api/auth/sign-in
    export AFFINE_COOKIE=$(awk '/affine_session/{printf "affine_session=%s; ", $NF} /affine_csrf_token/{printf "affine_csrf_token=%s; ", $NF} /affine_user_id/{printf "affine_user_id=%s", $NF}' cookies.txt)

Folder IDs are discovered at runtime via `list_organize_nodes`. If a folder does not
exist, create it first with `create_folder` and move it under the parent with
`move_organize_node`, using a fractional index like "a1" (not "a00").
"""
import os, sys, json, subprocess, threading, queue, time, re, glob

KB_ROOT = os.environ.get("KB_ROOT", os.path.expanduser("~/Personal/media-knowledge"))
WORKSPACE_ID = os.environ.get("AFFINE_WORKSPACE_ID", "")
ENV = {
    "AFFINE_BASE_URL": os.environ.get("AFFINE_BASE_URL", ""),
    "AFFINE_COOKIE": os.environ.get("AFFINE_COOKIE", ""),
    "AFFINE_WORKSPACE_ID": WORKSPACE_ID,
}
SERVER = os.environ.get("AFFINE_MCP_SERVER", "/Users/noblecloud/.local/bin/affine-mcp")

FOLDER_NAMES = {
    "videos": "Videos",
    "channels": "Channels",
    "persons": "Persons",
    "concepts": "Concepts",
}

class AffineMcpClient:
    def __init__(self):
        self.proc = subprocess.Popen(
            [SERVER],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env={**os.environ, **ENV}, bufsize=1,
        )
        self._q = queue.Queue()
        self._id = 0
        threading.Thread(target=self._reader, daemon=True).start()
        self._init()

    def _reader(self):
        for line in self.proc.stdout:
            line = line.strip()
            if line:
                try:
                    self._q.put(json.loads(line))
                except json.JSONDecodeError:
                    pass

    def _send(self, msg):
        self.proc.stdin.write(json.dumps(msg, separators=(',', ':')) + "\n")
        self.proc.stdin.flush()

    def _next_id(self):
        self._id += 1
        return self._id

    def _init(self):
        self._send({
            "jsonrpc": "2.0", "id": self._next_id(), "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "kb-import", "version": "1.0"},
            }
        })
        resp = self._recv(timeout=15)
        if resp.get("error"):
            raise RuntimeError(f"init failed: {resp['error']}")
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _recv(self, timeout=120):
        return self._q.get(timeout=timeout)

    def call(self, method, params, timeout=120):
        req_id = self._next_id()
        self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        while True:
            resp = self._recv(timeout=timeout)
            if resp.get("id") == req_id:
                return resp

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.wait(timeout=5)


def strip_frontmatter(text):
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return text


def extract_title(text):
    m = re.search(r'^title:\s*(.+)$', text, re.M)
    if m:
        return m.group(1).strip()
    m = re.search(r'^#\s+(.+)$', text, re.M)
    if m:
        return m.group(1).strip()
    return None


def convert_wikilinks(text, title_map):
    def repl(m):
        path = m.group(1).strip()
        alt = (m.group(2) or "").strip()
        if path in title_map:
            return f"[[{title_map[path]}]]"
        if path.endswith(".md"):
            no_ext = path[:-3]
            if no_ext in title_map:
                return f"[[{title_map[no_ext]}]]"
        return f"[[{alt or os.path.splitext(os.path.basename(path))[0].replace('-', ' ').title()}]]"
    return re.sub(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]', repl, text)


def get_text_content(resp):
    result = resp.get("result", {})
    content = result.get("content", [])
    if content:
        for item in content:
            if item.get("type") == "text":
                text = item.get("text", "")
                if text:
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return {"raw": text}
    return result


def list_docs(client):
    resp = client.call("tools/call", {
        "name": "list_docs",
        "arguments": {"workspaceId": WORKSPACE_ID, "first": 200}
    })
    data = get_text_content(resp)
    return {edge["node"]["title"]: edge["node"]["id"] for edge in data.get("edges", []) if edge["node"].get("title")}


def list_folders(client):
    resp = client.call("tools/call", {
        "name": "list_organize_nodes",
        "arguments": {"workspaceId": WORKSPACE_ID}
    })
    data = get_text_content(resp)
    return {node["data"]: node["id"] for node in data.get("nodes", []) if node.get("type") == "folder"}


def create_doc(client, title, markdown):
    resp = client.call("tools/call", {
        "name": "create_doc_from_markdown",
        "arguments": {"workspaceId": WORKSPACE_ID, "title": title, "markdown": markdown}
    })
    data = get_text_content(resp)
    if not data or not data.get("ok"):
        print(f"  FAILED create {title}: {json.dumps(data)}", file=sys.stderr)
        return None
    return data.get("docId")


def add_to_folder(client, doc_id, folder_id):
    client.call("tools/call", {
        "name": "add_organize_link",
        "arguments": {"workspaceId": WORKSPACE_ID, "folderId": folder_id, "type": "doc", "targetId": doc_id}
    })


def import_files(client, subdir, folder_id, existing):
    pattern = os.path.join(KB_ROOT, subdir, "*.md")
    files = sorted(glob.glob(pattern))
    print(f"\n=== {subdir} ({len(files)} files) -> {folder_id} ===")
    for fpath in files:
        raw = open(fpath).read()
        title = extract_title(raw)
        if not title:
            print(f"  SKIP no title: {fpath}")
            continue
        if title in existing:
            print(f"  EXISTS {title}; organizing existing {existing[title]}")
            doc_id = existing[title]
        else:
            markdown = convert_wikilinks(strip_frontmatter(raw), existing)
            print(f"  CREATE {title}")
            doc_id = create_doc(client, title, markdown)
            if doc_id:
                existing[title] = doc_id
            time.sleep(2)
        if doc_id:
            add_to_folder(client, doc_id, folder_id)
            time.sleep(2)


def main():
    if not all(ENV.values()):
        print("Set AFFINE_BASE_URL, AFFINE_COOKIE, and AFFINE_WORKSPACE_ID", file=sys.stderr)
        sys.exit(1)
    client = AffineMcpClient()
    try:
        existing = list_docs(client)
        folders = list_folders(client)
        print(f"Existing docs: {len(existing)}, folders: {list(folders.keys())}")
        for subdir, folder_name in FOLDER_NAMES.items():
            folder_id = folders.get(folder_name)
            if not folder_id:
                print(f"  WARNING: folder '{folder_name}' not found, skipping {subdir}")
                continue
            import_files(client, subdir, folder_id, existing)
    finally:
        client.close()


if __name__ == "__main__":
    main()
