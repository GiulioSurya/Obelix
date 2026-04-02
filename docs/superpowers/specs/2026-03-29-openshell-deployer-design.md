# OpenShell Deployer — Design Spec

> Status: APPROVED
> Date: 2026-03-29

## Obiettivo

Deployare un agente Obelix **dentro** una sandbox OpenShell, esponendolo come
server A2A identico a quello creato da `a2a_serve()`. Il client A2A non deve
percepire alcuna differenza tra un server locale e uno deployato in sandbox.

---

## Decisioni

### 1. Entry point: `AgentFactory.a2a_openshell_deploy()`

Metodo separato da `a2a_serve()` perche' il lifecycle e' molto diverso
(creazione provider, BYOC, port forwarding, cleanup). Internamente
avvia lo stesso server A2A (uvicorn + FastAPI + ObelixAgentExecutor) ma
dentro la sandbox.

```python
factory.a2a_serve("agent", port=8002)                  # locale (invariato)
factory.a2a_openshell_deploy("agent", port=8002,       # sandbox OpenShell
    policy="policy.yaml",
    providers=["anthropic"],
)
```

### 2. BashTool: LocalShellExecutor dentro la sandbox

Niente gRPC, niente OpenShellExecutor. I comandi eseguono come subprocess
locale, protetti dal Policy Engine di OpenShell a livello kernel.
Questo permette di eliminare OpenShellExecutor una volta completato il Deployer.

### 3. Trasparenza lato client

Il client A2A si connette a `localhost:<port>` sia con `a2a_serve()` che con
`a2a_openshell_deploy()`. Stessa Agent Card, stesso protocollo, stesse skill.

### 4. CLI-first + TODO per SDK

Il SDK Python (`openshell` v0.0.16) espone solo sandbox CRUD + exec +
inference routing. Per tutto il resto usiamo il CLI via subprocess.
Ogni chiamata CLI nel codice ha un TODO per futura migrazione a SDK.

**Via SDK:**
- Sandbox create/delete/get/wait_ready/health
- Exec (per avviare il server dentro la sandbox)

**Via CLI (TODO: migrare a SDK quando disponibile):**
- `openshell provider create` — registrazione credenziali
- `openshell policy set` — applicazione/aggiornamento policy
- `openshell forward start/stop` — port forwarding

### 5. Provider management: automatico con override

Default: `openshell provider create --from-existing` automaticamente.
Override: l'utente passa nomi di provider gia' registrati in OpenShell.

### 6. BYOC per il codice

Tre modalita':
1. **Default**: il Deployer genera un Dockerfile temporaneo (Python 3.13 + uv +
   src/ + pyproject.toml + entrypoint) e lo passa a OpenShell
2. **Dockerfile custom**: l'utente passa un path a Dockerfile
3. **Immagine pre-buildata**: l'utente passa un riferimento immagine

OpenShell gestisce build + import nativamente con `--from`.

### 7. Avvio server via exec()

`SandboxClient.exec()` con comando shell — stessa cosa del Dockerfile CMD.
Niente cloudpickle: piu' prevedibile, piu' debuggabile.

### 8. Struttura modulo

```
src/obelix/adapters/outbound/openshell/
    __init__.py
    deployer.py
```

### 9. Policy opzionale + hot-reload

Policy non obbligatoria (warning se assente). `update_policy(path)` per
hot-reload delle `network_policies` via CLI.

### 10. Lifecycle: context manager + destroy()

Il Deployer resta vivo (come `a2a_serve()`). Cleanup su Ctrl+C/SIGTERM:
stop forward → delete sandbox → close client. Idempotente.

Funziona da qualsiasi processo Python (locale o containerizzato) che possa
raggiungere il gateway OpenShell e abbia `openshell` CLI nel PATH.

### 11. Logging separato

Log del Deployer (create, policy, forward, cleanup) nel logger Python.
Log dell'agente nella sandbox: `openshell logs <sandbox> --tail` da
un terminale separato.

---

## Architettura

### Componenti

```
AgentFactory                         OpenShellDeployer
├─ a2a_openshell_deploy()            ├─ _ensure_providers()     CLI
│   crea il Deployer,                ├─ _build_image()          Docker/CLI
│   delega tutto a lui               ├─ _create_sandbox()       SDK
│                                    ├─ _start_server()         SDK (exec)
│                                    ├─ _start_forward()        CLI
│                                    ├─ update_policy()         CLI
│                                    ├─ destroy()               SDK + CLI
│                                    └─ __aenter__/__aexit__
│
DeploymentInfo (dataclass, frozen)
├─ sandbox_name: str
├─ endpoint: str
├─ port: int
```

### Flusso deploy

```
a2a_openshell_deploy("agent", port=8002, policy="p.yaml", providers=["anthropic"])
    │
    ▼
1. Validazione pre-deploy
   ├─ SDK openshell importabile?
   ├─ CLI openshell nel PATH?
   ├─ Gateway raggiungibile? (client.health())
   └─ dockerfile e image non entrambi passati?
    │
    ▼
2. _ensure_providers(["anthropic"])
   ├─ openshell provider get anthropic  (esiste gia'?)
   └─ openshell provider create --name anthropic --type claude --from-existing
   # TODO: replace with SDK when Provider CRUD available
    │
    ▼
3. _build_image()
   ├─ Se image= passato: usa quella, skip
   ├─ Se dockerfile= passato: usa quel Dockerfile
   └─ Se niente: genera Dockerfile temporaneo
   └─ Passa a openshell sandbox create --from
    │
    ▼
4. _create_sandbox()
   └─ CLI: openshell sandbox create --from <image/dockerfile>
         --provider anthropic --policy policy.yaml
   └─ SDK: client.wait_ready(name, timeout=120)
   # Nota: --from e' solo CLI (BYOC). SandboxSpec non supporta BYOC.
   # TODO: replace with SDK if SandboxSpec adds image/dockerfile support
    │
    ▼
5. _start_server()
   └─ SDK: client.exec(sandbox_id,
        ["bash", "-c", "uv run python <entrypoint>"])
    │
    ▼
6. _start_forward()
   └─ openshell forward start <port> <sandbox> -d
   # TODO: replace with SDK when port forwarding available
    │
    ▼
7. return DeploymentInfo(
       sandbox_name=...,
       endpoint="http://localhost:8002",
       port=8002
   )
```

### Flusso cleanup (destroy / __aexit__)

```
1. openshell forward stop <port> <sandbox>
   # TODO: replace with SDK
2. client.delete(sandbox_name)        ← SDK
3. client.wait_deleted(sandbox_name)  ← SDK
4. client.close()                     ← SDK
```

---

## Interfaccia pubblica

### OpenShellDeployer

```python
class OpenShellDeployer:
    """Deploys an Obelix agent inside an OpenShell sandbox as an A2A server."""

    def __init__(
        self,
        agent_factory: AgentFactory,
        agent_name: str,
        *,
        port: int = 8002,
        policy: str | None = None,
        providers: list[str] | None = None,
        dockerfile: str | None = None,
        image: str | None = None,
        gateway: str | None = None,
        tls_cert_dir: str | None = None,
        # A2A Agent Card params
        endpoint: str | None = None,
        version: str = "0.1.0",
        description: str | None = None,
        provider_name: str = "Obelix",
        # Subagent composition
        subagents: list[str] | None = None,
        subagent_config: dict | None = None,
    ):

    async def deploy(self) -> DeploymentInfo:
        """Full deploy: providers → image → sandbox → server → forward."""

    async def update_policy(self, policy_path: str) -> bool:
        """Hot-reload network_policies. Returns True on success.
        # TODO: replace with SDK when openshell Python SDK exposes policy set
        """

    async def destroy(self) -> None:
        """Stop forward, delete sandbox, close client. Idempotent."""

    async def __aenter__(self) -> DeploymentInfo:
        return await self.deploy()

    async def __aexit__(self, *exc) -> None:
        await self.destroy()
```

### DeploymentInfo

```python
@dataclass(frozen=True)
class DeploymentInfo:
    sandbox_name: str
    endpoint: str        # "http://localhost:{port}"
    port: int
```

### AgentFactory.a2a_openshell_deploy()

Thin wrapper — crea il Deployer, chiama deploy(), blocca fino a SIGINT:

```python
def a2a_openshell_deploy(
    self,
    agent: str,
    *,
    port: int = 8002,
    policy: str | None = None,
    providers: list[str] | None = None,
    dockerfile: str | None = None,
    image: str | None = None,
    gateway: str | None = None,
    tls_cert_dir: str | None = None,
    endpoint: str | None = None,
    version: str = "0.1.0",
    description: str | None = None,
    provider_name: str = "Obelix",
    subagents: list[str] | None = None,
    subagent_config: dict | None = None,
) -> None:
    """Deploy agent in OpenShell sandbox and block until interrupted.

    Like a2a_serve() but runs inside an OpenShell sandbox.
    On Ctrl+C / SIGTERM, destroys the sandbox and exits.
    """
```

---

## Error handling

### Errori durante il deploy

| Step | Cosa puo' fallire | Comportamento |
|---|---|---|
| Validazione | SDK/CLI mancante, gateway giu' | `RuntimeError`, no cleanup |
| Provider setup | Credenziali mancanti | `RuntimeError`, no cleanup |
| Image build | Docker non attivo, Dockerfile invalido | `RuntimeError`, no cleanup |
| Sandbox create | Spec invalida, timeout | `SandboxError`, cleanup sandbox |
| Start server | Entrypoint sbagliato, import error | `RuntimeError`, cleanup sandbox |
| Port forward | Porta occupata | `RuntimeError`, cleanup sandbox |

**Regola**: se la sandbox e' stata creata, qualsiasi errore successivo
la distrugge prima di propagare l'eccezione. Niente sandbox orfane.

### Edge cases

| Caso | Comportamento |
|---|---|
| `policy=None` | `logger.warning(...)`, prosegue |
| Porta occupata | Errore immediato |
| Gateway non avviato | Messaggio: "Run: openshell gateway start" |
| SDK non installato | `ImportError` |
| CLI non nel PATH | `RuntimeError` |
| SIGINT durante deploy | Cleanup best-effort |
| `destroy()` x2 | Idempotente |
| `dockerfile` + `image` | `ValueError` (mutually exclusive) |

### Validazione pre-deploy

1. SDK `openshell` importabile
2. CLI `openshell` nel PATH (`shutil.which`)
3. Gateway raggiungibile (`client.health()`)
4. `dockerfile` e `image` non entrambi passati

---

## File toccati

| File | Modifica |
|---|---|
| `adapters/outbound/openshell/__init__.py` | Nuovo — export OpenShellDeployer |
| `adapters/outbound/openshell/deployer.py` | Nuovo — classe principale |
| `core/agent/agent_factory.py` | Aggiunta `a2a_openshell_deploy()` |

Nessuna modifica al codice esistente — solo aggiunte.

---

## SDK OpenShell — Inventory (v0.0.16)

### Disponibile nel SDK Python

| Classe | Metodi chiave |
|---|---|
| `SandboxClient` | `create(spec)`, `get(name)`, `list()`, `delete(name)`, `wait_ready(name)`, `exec(id, cmd)`, `exec_stream(id, cmd)`, `exec_python(id, fn)`, `health()` |
| `SandboxSession` | Wrappa client+ref, stessi metodi senza passare id |
| `Sandbox` | Context manager: create/wait_ready/delete on exit |
| `InferenceRouteClient` | `set_cluster(provider, model)`, `get_cluster()` |
| `SandboxSpec` | `policy`, `providers`, `environment`, `template`, `gpu` |
| `TlsConfig` | `ca_path`, `cert_path`, `key_path` |
| `ExecResult` | `exit_code`, `stdout`, `stderr` |

### NON disponibile nel SDK (solo CLI o gRPC raw)

| Capability | CLI | gRPC RPC |
|---|---|---|
| Provider CRUD | `openshell provider create/get/list/delete` | `CreateProvider`, etc. |
| Policy set/get | `openshell policy set/get/list` | `UpdateConfig`, etc. |
| Port forwarding | `openshell forward start/stop/list` | N/A |
| Log streaming | `openshell logs --tail` | `GetSandboxLogs`, `WatchSandbox` |
| SSH sessions | `openshell sandbox connect` | `CreateSshSession` |

> **Regola**: monitorare periodicamente le release del SDK (`openshell` su PyPI
> o repo GitHub NVIDIA/OpenShell) e migrare le chiamate CLI a SDK quando
> i metodi diventano disponibili.

---

## Post-deploy: dismissione OpenShellExecutor

Una volta che il Deployer e' stabile e testato:

1. `OpenShellExecutor` (`adapters/outbound/shell/openshell_executor.py`) diventa
   deprecato — il Deployer copre il suo use case con architettura piu' pulita
2. Gli agenti che usavano `OpenShellExecutor` migrano a `a2a_openshell_deploy()`
   con `LocalShellExecutor` dentro la sandbox
3. Rimuovere `OpenShellExecutor` e il relativo file watcher per policy
4. Il shell-demo (`examples/shell-demo/`) viene riscritto con il Deployer
