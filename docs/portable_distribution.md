# Portable Phoenix distribution

This is the pragmatic seminar/workshop fallback when hosting Phoenix over the
network is blocked by workstation firewalls, VLAN isolation, or university Wi-Fi
rules.

The intended seminar workflow is:

```text
Instructor builds zip once
        ↓
Instructor uploads zip to GitLab/Nextcloud/ILIAS/Moodle
        ↓
Participants download zip
        ↓
Participants extract zip
        ↓
Participants double-click Phoenix.bat or Phoenix.app
```

Participants do **not** need Git, conda, Python, or terminal commands.

Phoenix is a Streamlit/PyBaMM app. A true single-file executable is possible in
principle, but it is fragile for this dependency stack. The supported local
fallback is therefore a **portable app folder**:

```text
PhoenixPortable*
├── Phoenix.bat or Phoenix.app
├── app/      # Phoenix source tree
└── env/      # packed Python/conda environment
```

Users run one launcher. The launcher starts Streamlit locally on
`127.0.0.1:8501` and opens a browser. No incoming network port is needed.

## Windows package

Build on Windows from an Anaconda Prompt:

```bat
scripts\package_phoenix_portable_windows.bat
```

Output:

```text
dist\PhoenixPortableWindows\
dist\PhoenixPortableWindows.zip
```

Test:

```bat
dist\PhoenixPortableWindows\Phoenix.bat
```

Give participants `PhoenixPortableWindows.zip`. They extract it and run
`Phoenix.bat`. The zip also contains `README_FIRST.txt` with the same simple
instructions.

## macOS package

There are two macOS routes.

### Route A: native portable Mac package

This is the closest equivalent to the Windows portable folder, but it must be
built on macOS because the packed Python environment contains macOS binaries.

Build on macOS from a conda-enabled shell:

```bash
bash scripts/package_phoenix_portable_macos.sh
```

Output:

```text
dist/PhoenixPortableMac/
dist/PhoenixPortableMac.zip
```

Test:

```bash
open dist/PhoenixPortableMac/Phoenix.app
```

Give participants `PhoenixPortableMac.zip`. They extract it and open
`Phoenix.app`. The zip also contains `README_FIRST.txt` with the same simple
instructions.

If macOS blocks the unsigned app, right-click `Phoenix.app` and choose
**Open**. This is normal for an unnotarized local app bundle.

### Route B: Mac bootstrap zip built from Windows/Linux

Use this if you do not have access to a Mac. This zip can be created on Windows
or Linux and given to Mac users. It does **not** contain a prebuilt macOS Python
environment. Instead, the user's Mac downloads micromamba and creates the
environment on first launch.

Build from any platform:

```bash
python scripts/package_phoenix_macos_bootstrap.py
```

Output:

```text
dist/PhoenixMacBootstrap.zip
```

Participants:

1. Download `PhoenixMacBootstrap.zip`.
2. Extract it.
3. Open the `PhoenixMac` folder.
4. Double-click `Run Phoenix.command`.
5. Wait for the first-run environment install.

Requirements for participants:

- macOS on Apple Silicon or Intel.
- Internet access on first launch.
- No Git.
- No existing Python.
- No existing conda.

If macOS blocks the command, right-click `Run Phoenix.command` and choose
**Open**.

## Important limitations

- Build Windows packages on Windows and macOS packages on macOS.
- A Mac bootstrap zip can be built from Windows/Linux, but the Mac environment
  is installed on the user's Mac at first launch.
- The zip files will be large because they include PyBaMM, NumPy, SciPy,
  Streamlit, matplotlib, and solver dependencies.
- The first launch may take longer because `conda-unpack` fixes paths inside
  the portable environment.
- This is a seminar/workshop distribution method, not a polished public desktop
  product.
- If you later want a real signed `.exe` or notarized `.app`, treat that as a
  separate packaging project.
