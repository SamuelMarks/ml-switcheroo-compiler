import urllib.request
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

repos = []
page = 1
while True:
    url = f"https://api.github.com/users/SamuelMarks/repos?per_page=100&page={page}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, context=ctx) as response:
            data = json.loads(response.read().decode())
            if not data:
                break
            repos.extend(data)
            page += 1
    except Exception as e:
        print(f"Error: {e}")
        break

target_repos = []
for r in repos:
    name = r["name"]
    if name.startswith("zero-") or name in [
        "ml-switcheroo-ir",
        "ml-framework-snapshots",
    ]:
        target_repos.append(r)

print("Target Repos found:")
for r in sorted(target_repos, key=lambda x: x["name"]):
    print(f"- {r['name']}: {r['description']}")
