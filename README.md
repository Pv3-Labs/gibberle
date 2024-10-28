# gibberle

## Setup Instructions

1. **Install Python 3.12 or newer**:

   - Download and install Python from the [official Python website](https://www.python.org/downloads/).
   - Ensure to check the option to **'Add Python to PATH'** during installation. (Don't mess this up, man!)

2. **Clone the Repository and Create a Virtual Environment**:

   - Skipping the cloning step here; y'all got it.

   - Create a virtual environment:
     ```bash
     python -m venv venv
     ```

3. **Activate the Virtual Environment**:

   - **For Windows**:

     ```bash
     .\venv\Scripts\activate
     ```

   - **For Linux/macOS**:
     ```bash
     source venv/bin/activate
     ```

   Remember, you'll need to activate the virtual environment each time you open a new terminal session.

4. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Jupyter Notebook

### Running Jupyter Notebook

To start Jupyter Notebook, run:

```bash
jupyter notebook
```

## Web App (Next.js)

```bash
cd web-app
# and
npm i
# and
npm run dev
```

## VS Code Extensions [WIP]

- Auto Close Tag - Jun Han
- ESLint - Microsoft
- Jupyter - Microsoft
- MDX - unified
- MDX Preview - Xiaoyi Chen
- Prettier - Prettier
- Python - Microsoft
- Thunder Client - Thunder Client
