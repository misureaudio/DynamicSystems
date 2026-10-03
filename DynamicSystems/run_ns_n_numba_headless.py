"""Headless-execute NS_N_Dependence_numba.ipynb (Numba build) with nbclient,
pinned to the WORKSPACE VENV.  Identical to run_ns_n_headless.py except the
notebook / executed / log paths point at the Numba variant.

STANDING RULE: only D:\\Source\\hermes-dir\\.venv (interpreter AND modules) -- never
the system python.  The kernel command is hard-pinned to the absolute venv
interpreter (no kernelspec, no PATH), because an unactivated shell would otherwise
resolve bare `python` to the system Python 3.11.
"""
import os, sys, json, time
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager

VENV_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv",
                                       "Scripts", "python.exe"))
if not os.path.exists(VENV_PY):
    sys.exit(f"FATAL: venv interpreter not found at {VENV_PY}")
print(f"Kernel pinned to: {VENV_PY}", flush=True)

NB_PATH = "NS_N_Dependence_numba.ipynb"
EXECPATH = "NS_N_Dependence_numba_executed.ipynb"
LOG = "ns_n_numba_exec_log.json"

class VenvKernelManager(KernelManager):
    kernel_cmd = [VENV_PY, "-m", "ipykernel_launcher", "-f", "{connection_file}"]

nb = nbf.read(NB_PATH, as_version=4)
client = NotebookClient(nb, timeout=7200,
                        kernel_name="dsvenv",
                        kernel_manager_class=VenvKernelManager,
                        allow_errors=False)
t0 = time.time()
client.execute()
nbf.write(nb, EXECPATH)

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
