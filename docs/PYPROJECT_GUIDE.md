# 📦 Complete Guide to `pyproject.toml`

A comprehensive guide to configuring your Python project with `pyproject.toml` - the modern standard for Python packaging and project configuration.

## 🎯 What is pyproject.toml?

`pyproject.toml` is a configuration file used by:
- **Packaging tools** (for building and distributing your project)
- **Development tools** (linters, type checkers, formatters, etc.)
- **Build systems** (setuptools, hatchling, poetry, etc.)

## 📋 Table Structure Overview

The `pyproject.toml` file contains three main TOML tables:

| Table | Purpose | Required | Description |
|---|---|---|---|
| `[build-system]` | ✅ **Required** | Yes | Declares which build backend and dependencies to use |
| `[project]` | 📦 **Metadata** | Recommended | Project metadata (name, version, dependencies, etc.) |
| `[tool]` | 🔧 **Tools** | Optional | Tool-specific configurations (black, mypy, pytest, etc.) |

---

## 🏗️ Build System Configuration

### Why [build-system] is Essential

> ⚠️ **Important:** The `[build-system]` table should **always** be present, regardless of which build backend you use.

This table defines:
- **Which build tool** to use for your project
- **Dependencies needed** to build your project

### Common Build Backend Examples

#### 🚀 **Hatchling** (Recommended for new projects)
```toml
[build-system]
requires = ["hatchling >= 1.26"]
build-backend = "hatchling.build"
```

#### 🔧 **setuptools** (Traditional, widely supported)
```toml
[build-system]
requires = ["setuptools >= 61.0", "wheel"]
build-backend = "setuptools.build_meta"
```

#### ✨ **Flit** (Lightweight)
```toml
[build-system]
requires = ["flit_core >=3.2,<4"]
build-backend = "flit_core.buildapi"
```

#### 🎨 **PDM** (Modern dependency management)
```toml
[build-system]
requires = ["pdm-backend"]
build-backend = "pdm.backend"
```

---

## 📦 Project Metadata Configuration

The `[project]` table contains your project's basic information and is understood by most modern build backends.

### 🔄 Static vs Dynamic Metadata

**Static:** You directly specify the value
```toml
[project]
version = "1.0.0"
requires-python = ">= 3.8"
```

**Dynamic:** Build backend computes the value
```toml
[project]
dynamic = ["version"]  # Read from __version__ or Git tag
```

---

### ✅ Required Fields

#### 📛 **name** (Required)
The name of your project on PyPI.

```toml
[project]
name = "my-awesome-project"
```

**Naming Rules:**
- ✅ ASCII letters, digits, underscores `_`, hyphens `-`, periods `.`
- ❌ Cannot start or end with `_`, `-`, or `.`
- 🔄 Case-insensitive: `Cool-Stuff` = `cool.stuff` = `COOL_STUFF`

#### 🏷️ **version** (Required)
Project version following [semantic versioning](https://semver.org/).

```toml
[project]
version = "2024.1.0"
```

**Version Examples:**
- `1.0.0` - Standard release
- `2024.1.0a1` - Alpha release
- `1.2.3rc1` - Release candidate
- `0.1.0.dev1` - Development version

---

### 📚 Dependencies and Requirements

#### 🔗 **dependencies**
Core dependencies required for your project to run.

```toml
[project]
dependencies = [
    "httpx",
    "requests >= 2.28.0",
    "pydantic >= 2.0.0, < 3.0.0",
    "python-dotenv ~= 1.0.0",
    "django > 4.0; python_version >= '3.8'",
]
```

**Dependency Specification Syntax:**
- `package-name` - Latest version
- `package-name >= 1.0` - Minimum version
- `package-name < 2.0` - Maximum version
- `package-name >= 1.0, < 2.0` - Version range
- `package-name ~= 1.4.2` - Compatible release
- `package-name[extra]` - With optional extras
- `package-name; condition` - Conditional dependency

#### 🎯 **optional-dependencies**
Optional features that users can install separately.

```toml
[project.optional-dependencies]
# GUI support
gui = ["PyQt6", "matplotlib"]

# CLI tools
cli = ["click >= 8.0", "rich", "typer"]

# Development tools
dev = [
    "pytest >= 7.0",
    "black",
    "isort",
    "mypy",
    "pre-commit",
]

# Documentation
docs = [
    "mkdocs",
    "mkdocs-material",
    "mkdocstrings",
]

# All optional dependencies
all = ["my-project[gui,cli,dev,docs]"]
```

**Installation Examples:**
```bash
pip install my-project[gui]          # With GUI support
pip install my-project[dev,docs]     # Multiple extras
pip install my-project[all]          # Everything
```

#### 🐍 **requires-python**
Minimum Python version your project supports.

```toml
[project]
requires-python = ">= 3.8"
```

> 💡 **Tip:** Avoid upper bounds like `<= 3.11` unless absolutely necessary. They can cause dependency resolution issues.

---

### 🎭 Project Identity

#### 👥 **authors/maintainers**
People responsible for the project.

```toml
[project]
authors = [
    {name = "Alice Developer", email = "alice@example.com"},
    {name = "Bob Maintainer", email = "bob@example.com"},
    {name = "Contributors Team"},
    {email = "team@example.com"},
]

maintainers = [
    {name = "Current Maintainer", email = "maintainer@example.com"}
]
```

#### 📝 **description**
One-line project description (appears as headline on PyPI).

```toml
[project]
description = "A fast and intuitive web framework for building APIs"
```

#### 📖 **readme**
Longer project description (displayed on PyPI project page).

```toml
[project]
readme = "README.md"
```

**Alternative formats:**
```toml
[project]
readme = {file = "README.rst", content-type = "text/x-rst"}
# or
readme = {text = "My project description", content-type = "text/plain"}
```

---

### ⚖️ Licensing

#### 📄 **license** (Modern PEP 639 Format)
SPDX license expression.

```toml
[project]
license = "MIT"
# or
license = "Apache-2.0"
# or complex expressions
license = "MIT AND (Apache-2.0 OR BSD-2-Clause)"
# or custom license
license = "LicenseRef-My-Custom-License"
```

#### 📁 **license-files**
License files to distribute with your package.

```toml
[project]
license-files = ["LICENSE", "NOTICE", "AUTHORS.md"]
# or with globs
license-files = ["LICEN[CS]E*", "*.LICENSE", "legal/*.txt"]
```

**Build Backend Support for PEP 639:**

| Build Backend | Minimum Version |
|---|---|
| hatchling | 1.27.0+ ✅ |
| setuptools | 77.0.3+ ✅ |
| flit-core | 3.12+ ✅ |
| pdm-backend | 2.4.0+ ✅ |
| poetry-core | ❌ Not yet |

---

### 🔍 Discovery and Classification

#### 🏷️ **keywords**
Help users find your project through search.

```toml
[project]
keywords = ["web", "api", "framework", "async", "fast"]
```

#### 📊 **classifiers**
PyPI classifiers for categorizing your project.

```toml
[project]
classifiers = [
    # Development Status
    "Development Status :: 4 - Beta",
    "Development Status :: 5 - Production/Stable",
    
    # Intended Audience
    "Intended Audience :: Developers",
    "Intended Audience :: System Administrators",
    
    # Topic
    "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
    "Topic :: Software Development :: Libraries :: Python Modules",
    
    # License
    "License :: OSI Approved :: MIT License",
    
    # Python Versions
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    
    # Operating System
    "Operating System :: OS Independent",
    
    # Special classifiers
    "Private :: Do Not Upload",  # Prevents PyPI upload
]
```

#### 🔗 **urls**
Important project links.

```toml
[project.urls]
Homepage = "https://myproject.com"
Documentation = "https://docs.myproject.com"
Repository = "https://github.com/user/myproject"
"Bug Tracker" = "https://github.com/user/myproject/issues"
Changelog = "https://github.com/user/myproject/blob/main/CHANGELOG.md"
"Funding" = "https://github.com/sponsors/user"
"Say Thanks!" = "https://saythanks.io/to/user"
```

**Well-known URL Labels:**
- `Homepage`, `Home`, `Home-page`
- `Download`, `Download-URL`
- `Documentation`, `Docs`
- `Repository`, `Source`, `Source-Code`
- `Bug-Tracker`, `Issue-Tracker`
- `Changelog`, `Release-Notes`
- `Funding`, `Donate`, `Sponsor`

---

### 🖥️ Executable Scripts

#### 📱 **scripts**
Command-line tools (console scripts).

```toml
[project.scripts]
myproject = "myproject.cli:main"
myproject-dev = "myproject.dev:dev_main"
```

After installation:
```bash
myproject --help          # Runs myproject.cli:main()
myproject-dev --serve     # Runs myproject.dev:dev_main()
```

#### 🪟 **gui-scripts**
GUI applications (Windows-specific behavior).

```toml
[project.gui-scripts]
myproject-gui = "myproject.gui:gui_main"
```

**Differences:**
- **Console scripts:** Open terminal on Windows
- **GUI scripts:** Run in background on Windows (no terminal popup)
- **Linux/macOS:** No difference between the two

---

### 🔌 Advanced: Entry Points

Create plugins or extensions for other tools.

```toml
[project.entry-points."pytest11"]
myplugin = "myproject.pytest_plugin"

[project.entry-points."myframework.plugins"]
auth = "myproject.auth:AuthPlugin"
cache = "myproject.cache:CachePlugin"
```

---

## 🛠️ Tool Configuration

The `[tool]` table contains tool-specific configurations.

### 🎨 **Black** (Code formatter)
```toml
[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  migrations
  | .venv
)/
'''
```

### 📐 **isort** (Import sorter)
```toml
[tool.isort]
profile = "black"
line_length = 88
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
```

### 🔍 **mypy** (Type checker)
```toml
[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false
```

### 🧪 **pytest** (Testing framework)
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--verbose",
    "--cov=myproject",
    "--cov-report=html",
    "--cov-report=term-missing",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
]
```

### 📊 **coverage.py** (Code coverage)
```toml
[tool.coverage.run]
source = ["myproject"]
omit = ["tests/*", "migrations/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
]
```

---

## 📄 Complete Example Template

Here's a comprehensive `pyproject.toml` example:

```toml
[build-system]
requires = ["hatchling >= 1.26"]
build-backend = "hatchling.build"

[project]
name = "awesome-web-api"
version = "1.2.0"
description = "A fast and intuitive web framework for building APIs"
readme = "README.md"
license = "MIT"
license-files = ["LICENSE"]
authors = [
    {name = "Alice Developer", email = "alice@company.com"},
    {name = "Bob Maintainer", email = "bob@company.com"},
]
maintainers = [
    {name = "Current Team", email = "team@company.com"}
]
requires-python = ">= 3.8"
keywords = ["web", "api", "framework", "async", "fast"]

dependencies = [
    "fastapi >= 0.104.0",
    "uvicorn[standard] >= 0.24.0",
    "pydantic >= 2.0.0, < 3.0.0",
    "sqlalchemy >= 2.0.0",
    "alembic >= 1.12.0",
    "python-dotenv >= 1.0.0",
]

[project.optional-dependencies]
# Development dependencies
dev = [
    "pytest >= 7.4.0",
    "pytest-cov >= 4.1.0",
    "pytest-asyncio >= 0.21.0",
    "black >= 23.0.0",
    "isort >= 5.12.0",
    "mypy >= 1.5.0",
    "ruff >= 0.1.0",
    "pre-commit >= 3.4.0",
]

# Documentation dependencies
docs = [
    "mkdocs >= 1.5.0",
    "mkdocs-material >= 9.4.0",
    "mkdocstrings[python] >= 0.24.0",
]

# Production extras
redis = ["redis >= 5.0.0"]
postgresql = ["asyncpg >= 0.28.0"]
monitoring = ["prometheus-client >= 0.17.0", "sentry-sdk >= 1.30.0"]

# Meta extras
all = ["awesome-web-api[redis,postgresql,monitoring]"]

[project.urls]
Homepage = "https://awesome-api.com"
Documentation = "https://docs.awesome-api.com"
Repository = "https://github.com/company/awesome-web-api"
"Bug Tracker" = "https://github.com/company/awesome-web-api/issues"
Changelog = "https://github.com/company/awesome-web-api/blob/main/CHANGELOG.md"

[project.scripts]
awesome-api = "awesome_web_api.cli:main"
awesome-migrate = "awesome_web_api.migration:migrate_cli"

[project.gui-scripts]
awesome-api-gui = "awesome_web_api.gui:gui_main"

[project.entry-points."awesome_api.plugins"]
auth = "awesome_web_api.auth:AuthPlugin"
cache = "awesome_web_api.cache:CachePlugin"

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Operating System :: OS Independent",
]

# Tool configurations
[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311', 'py312']
include = '\.pyi?$'

[tool.isort]
profile = "black"
line_length = 88
multi_line_output = 3

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = [
    "--strict-markers",
    "--cov=awesome_web_api",
    "--cov-report=html",
    "--cov-report=term-missing",
]

[tool.coverage.run]
source = ["awesome_web_api"]
omit = ["tests/*"]

[tool.ruff]
line-length = 88
target-version = "py38"
select = ["E", "F", "W", "C", "N"]
```

---

## 💡 Best Practices

### ✅ **Do This:**

1. **Always include `[build-system]`** - Required for modern Python packaging
2. **Use semantic versioning** - `MAJOR.MINOR.PATCH` format
3. **Specify Python version range** - Use `requires-python` appropriately
4. **Pin important dependencies** - Avoid breaking changes
5. **Use optional-dependencies** - Keep core lightweight
6. **Include comprehensive metadata** - Help users discover your project
7. **Add tool configurations** - Standardize your development workflow
8. **Use well-known URL labels** - Better integration with tools

### ❌ **Avoid This:**

1. **Don't omit `[build-system]`** - Your project won't build properly
2. **Don't use overly strict version pins** - `package==1.0.0` unless necessary
3. **Don't include development deps in main `dependencies`** - Use `optional-dependencies`
4. **Don't forget `requires-python`** - Prevents installation on incompatible Python versions
5. **Don't use deprecated license format** - Use SPDX identifiers

### 🔄 **Migration from setup.py/setup.cfg:**

**Old setup.py:**
```python
from setuptools import setup

setup(
    name="myproject",
    version="1.0.0",
    install_requires=["requests"],
)
```

**New pyproject.toml:**
```toml
[build-system]
requires = ["setuptools >= 61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "myproject"
version = "1.0.0"
dependencies = ["requests"]
```

---

## 🔍 Troubleshooting

### Common Issues:

**Error: "No module named 'build'"**
```bash
pip install build
```

**Error: "Invalid build-backend"**
- Check your `build-backend` spelling
- Ensure the backend package is in `requires`

**Error: "License should be a dict/table"**
- Your build backend doesn't support PEP 639
- Update your build backend or use old format

**Dynamic version not working:**
```toml
[project]
dynamic = ["version"]

# For setuptools, add:
[tool.setuptools.dynamic]
version = {attr = "mypackage.__version__"}

# For hatchling, add:
[tool.hatch.version]
path = "mypackage/__init__.py"
```

---

## 📚 Additional Resources

- 📖 [Python Packaging User Guide](https://packaging.python.org/)
- 🔧 [PEP 621 - Storing project metadata in pyproject.toml](https://peps.python.org/pep-0621/)
- ⚖️ [PEP 639 - Improving License Clarity with Better Package Metadata](https://peps.python.org/pep-0639/)
- 🏗️ [Build backends comparison](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- 🔍 [PyPI Classifiers](https://pypi.org/classifiers/)
- 📜 [SPDX License List](https://spdx.org/licenses/)

---

*This guide follows the latest Python packaging standards and best practices as of 2024. Always refer to the official documentation for the most up-to-date information.*