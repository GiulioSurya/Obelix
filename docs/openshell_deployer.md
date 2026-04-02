# OpenShell Deployer

> Status: DESIGN IN PROGRESS — non ancora implementato

## Cos'e'

Il Deployer esegue un agente Obelix **dentro una sandbox OpenShell**, esponendolo
come server A2A. Da fuori e' indistinguibile da un server avviato con `a2a_serve()`:
stesso protocollo, stessa Agent Card, stesso CLI client.

La differenza e' che l'agente gira in un ambiente isolato con policy di sicurezza
a livello kernel (filesystem, network, processi).

## Confronto con a2a_serve()

```
a2a_serve()    →  server A2A nel tuo processo locale
a2a_openshell_deploy()   →  server A2A dentro una sandbox OpenShell
```

| | a2a_serve() | a2a_openshell_deploy() |
|---|---|---|
| Dove gira | Processo locale | Sandbox OpenShell |
| Sicurezza | Nessuna (permessi OS) | Policy kernel (filesystem, network, processi) |
| BashTool | Qualsiasi executor | LocalShellExecutor (protetto da policy) |
| API key | Nel tuo .env / environment | Iniettate dal gateway OpenShell (non nel filesystem) |
| Requisiti | `uv sync --extra serve` | Docker + Gateway OpenShell |
| Client A2A | `localhost:<port>` | `localhost:<port>` (via port forwarding) |

## Prerequisiti

- Docker attivo (Docker Desktop su Windows/Mac, Docker Engine su Linux)
- Gateway OpenShell avviato: `openshell gateway start`
- Dipendenze: `uv sync --extra serve --extra openshell`

## Quick Start

```python
from obelix.core.agent.agent_factory import AgentFactory

factory = AgentFactory()
factory.register(name="my_agent", cls=MyAgent)

# Deploy in sandbox (invece di a2a_serve)
factory.a2a_deploy(
    "my_agent",
    port=8002,
    policy="policy.yaml",       # policy di sicurezza OpenShell
    providers=["anthropic"],    # credenziali LLM (auto-detect da env locale)
)
```

Poi connettiti normalmente:
```bash
uv run python examples/cli_client.py --url http://localhost:8002
```

## Come funziona

```
1. Provider setup    openshell provider create --from-existing
                     (registra le API key nel gateway)

2. Image build       Docker build dell'immagine con codice + dipendenze
                     (o genera Dockerfile automaticamente se non fornito)

3. Sandbox create    openshell sandbox create --from <image>
                     --provider anthropic --policy policy.yaml
                     (crea sandbox con policy e credenziali)

4. Server start      Avvia a2a_serve() dentro la sandbox via exec()
                     (uvicorn + FastAPI + A2A protocol)

5. Port forward      openshell forward start <port> <sandbox>
                     (rende il server raggiungibile su localhost)
```

## Immagine Docker

Il Deployer supporta tre modalita':

### 1. Generazione automatica (default)

Se non passi `dockerfile` ne' `image`, il Deployer genera un Dockerfile
minimale basato sul progetto corrente:

```python
factory.a2a_openshell_deploy("my_agent", port=8002, policy="policy.yaml")
# → genera Dockerfile temporaneo, builda, passa a OpenShell
```

Il Dockerfile generato contiene: Python 3.13, uv, il tuo codice (`src/`,
`pyproject.toml`, `uv.lock`), e le dipendenze installate. L'entrypoint
avvia `a2a_serve()` con la configurazione passata a `a2a_openshell_deploy()`.

### 2. Dockerfile custom

Per immagini con requisiti particolari (pacchetti di sistema, tool extra):

```python
factory.a2a_openshell_deploy("my_agent", port=8002,
    dockerfile="examples/shell-demo/Dockerfile",
)
```

### 3. Immagine pre-buildata

Per deploy rapidi o immagini da registry:

```python
factory.a2a_openshell_deploy("my_agent", port=8002,
    image="my-registry.com/my-agent:latest",
)
```

## Parametri a2a_openshell_deploy()

```python
factory.a2a_deploy(
    agent: str,                      # nome dell'agente registrato
    *,
    port: int = 8002,                # porta del server A2A
    policy: str | None = None,       # path a policy YAML OpenShell
    providers: list[str] | None,     # nomi provider (auto-create con --from-existing)
    dockerfile: str | None = None,   # path a Dockerfile custom
    image: str | None = None,        # immagine pre-buildata
    # ... stessi parametri di a2a_serve per Agent Card:
    endpoint: str | None = None,
    version: str = "0.1.0",
    description: str | None = None,
)
```

## Logging

I log del Deployer (creazione sandbox, policy, forwarding, cleanup) appaiono
nel terminale dove hai lanciato `a2a_openshell_deploy()`.

I log dell'agente nella sandbox (uvicorn, Obelix, tool execution) sono
separati. Per vederli, apri un altro terminale:

```bash
# Log in tempo reale
openshell logs <sandbox-name> --tail

# Filtra per livello
openshell logs <sandbox-name> --tail --level warn
```

## Policy hot-reload

Le `network_policies` possono essere aggiornate a runtime senza ricreare
la sandbox:

```python
# Da codice, tramite il deployment object
await deployment.update_policy("new-policy.yaml")

# Oppure da CLI
openshell policy set <sandbox-name> --policy new-policy.yaml --wait
```

I campi statici (`filesystem_policy`, `process`, `landlock`) sono immutabili
dopo la creazione — per cambiarli serve un nuovo deploy.

## BashTool dentro la sandbox

Quando l'agente gira nella sandbox, il BashTool usa automaticamente
`LocalShellExecutor` — i comandi eseguono come subprocess normali, ma il
Policy Engine di OpenShell li intercetta a livello kernel:

```
LLM genera: bash(command="rm -rf /etc")
    → LocalShellExecutor → subprocess.run("rm -rf /etc")
        → Policy Engine (kernel): /etc e' deny → EPERM
            → exit_code: 1, stderr: "Permission denied"
```

L'agente non sa di essere in una sandbox. Prova, fallisce, gestisce l'errore.

## Policy

La policy controlla cosa l'agente puo' fare nel sandbox.
Vedi `docs/openshell_integration.md` per il formato e gli esempi.

Campi statici (immutabili dopo creazione):
- `filesystem_policy` — path read-only / read-write
- `process` — run_as_user/group
- `landlock` — compatibility mode

Campo dinamico (aggiornabile a runtime):
- `network_policies` — endpoint whitelist, binary matching

## Provider e credenziali

Le API key non sono mai nel filesystem della sandbox. Il gateway OpenShell
le inietta come variabili d'ambiente nel processo del sandbox.

```python
# Automatico: legge dal tuo environment locale
factory.a2a_openshell_deploy("agent", providers=["anthropic"])

# Manuale: usa provider gia' registrati in OpenShell
# (creati con: openshell provider create --name my-key --type generic --credential API_KEY=sk-...)
factory.a2a_openshell_deploy("agent", providers=["my-key"])
```

## Cleanup

Il Deployer resta vivo e gestisce il lifecycle della sandbox.
Su Ctrl+C, SIGTERM o uscita dal context manager, la sandbox viene distrutta automaticamente.

```python
# Context manager (raccomandato)
async with factory.a2a_openshell_deploy("agent", port=8002, policy="p.yaml") as deployment:
    print(f"Agent running at {deployment.endpoint}")
# sandbox distrutto automaticamente

# Oppure: gestione manuale
deployment = await factory.a2a_openshell_deploy("agent", port=8002, policy="p.yaml")
# ...
await deployment.destroy()  # cleanup esplicito
```

Non richiede containerizzazione — funziona da qualsiasi processo Python
che possa raggiungere il gateway OpenShell e abbia il CLI `openshell` nel PATH.

## Limitazioni note

- **Solo Linux/macOS**: il SDK OpenShell non ha wheel Windows
- **Docker richiesto**: per build immagine e gateway
- **Comandi multilinea**: il SDK gRPC rifiuta newline nei comandi
  (workaround: salvare su file + eseguire)

<!-- TODO: aggiornare man mano che il design si finalizza -->
