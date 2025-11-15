# ProyectoIDI2

Setup Virtual Environments
- source venv/bin/activate
- venv\Scripts\activate

## Virtual Environment

---

### Windows

```powershell
# Activate the venv (PowerShell)
.\.venv\Scripts\Activate

# Restore dependencies
python -m pip install -r requirements.txt

# When done, deactivate
deactivate
```

**PowerShell note:** If activating a script is blocked, run once as administrator:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### macOS Linux

```bash
# Activate the venv
source .venv/bin/activate
# Restore dependencies
python -m pip install -r requirements.txt

# When done, deactivate
deactivate
```

---

**Note:** If you need to install a new dependency please run the following commands with the venv active

```bash
# Install packages (example)
pip install libary

# Verify installed packages
pip list

# Freeze dependencies
pip freeze > requirements.txt

```