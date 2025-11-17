# ProyectoIDI2

## Virtual Environment

### Windows

```powershell
# Activate the venv (PowerShell)
.\.env\Scripts\Activate

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
source .env/bin/activate
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

## Dependencies

- **Numpy:** For array management
- **Pytorch:** Used to create the dataset with all the augmentations
- **CV2:** For image processing
- **PyWavelets:** Used to get PRNU from the images
