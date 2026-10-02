"""Headless-execute the notebook with nbclient against the workspace venv.
Writes an executed copy + a per-cell result log. Exit 0 iff no cell errored."""
import nbformat as nbf
from nbclient import NotebookClient
import sys, json, time

NB_PATH = "Dynamic_Systems_v2_notebook.ipynb"
EXECPATH = "Dynamic_Systems_v2_notebook_executed.ipynb"
LOG = "exec_log.json"

nb = nbf.read(NB_PATH, as_version=4)
client = NotebookClient(nb, timeout=1800, kernel_name="dsvenv",
                        allow_errors=False)
t0 = time.time()
client.execute()
nbf.write(nb, EXECPATH)

# per-cell log: index, type, error?, output char count
log = []
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    err = None
    out_chars = 0
    for o in c.get("outputs", []):
        if o.get("output_type") == "error":
            err = o.get("ename", "?") + ": " + o.get("evalue", "")
        elif o.get("output_type") == "stream":
            out_chars += len(o.get("text", ""))
        elif o.get("output_type") in ("execute_result", "display_data"):
            out_chars += len(o.get("data", {}).get("text/plain", ""))
    log.append({"cell": i, "error": err, "out_chars": out_chars})

with open(LOG, "w") as f:
    json.dump(log, f, indent=2)

nerr = sum(1 for r in log if r["error"])
print(f"Executed {len(nb.cells)} cells in {time.time()-t0:.0f}s. Errors: {nerr}")
for r in log:
    if r["error"]:
        print("  ERROR cell", r["cell"], ":", r["error"][:200])
sys.exit(1 if nerr else 0)
