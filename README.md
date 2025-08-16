# Forvia_Project

# Guide: Installing Dependencies with `requirements.txt`

This guide will walk you through the process of setting up a Python environment and installing dependencies listed in a `requirements.txt` file using two different methods: Python's built-in `venv` module and `uv`.

## Prerequisites

- Python installed on your system. (Ideally 3.10)
- Java SDK is installed on your system. (Ideally latest stable version)
- `pip` installed and updated.
- `uv` installed if you plan to use it.

## Method 1: Using Python's `venv`

### Step 1: Create a Virtual Environment

Open your terminal or command prompt and navigate to your project directory. Run the following command to create a virtual environment:

```bash
python -m venv myenv
```

Here, myenv is the name of your virtual environment. You can replace it with any name you prefer.

### Step 2: Activate the Virtual Environment

- On Windows:
```bash
myenv\Scripts\activate
```

- On macOS and Linux:
```bash
source myenv/bin/activate
```

### Step 3: Install Dependencies

With the virtual environment activated, you can now install the dependencies listed in your requirements.txt file. Run the following command:

```bash
pip install -r requirements.txt
```
This command will read the requirements.txt file and install all the specified packages and their dependencies.

## Method 2: Using uv

### Step 1: Install uv
If you haven't already installed uv, you can do so by following the instructions on the official uv GitHub repository.

### Step 2: Create a Virtual Environment

Navigate to your project directory and run the following command to create a virtual environment using uv:
```bash
uv venv myenv
```

### Step 3: Activate the Virtual Environment

- On Windows:
```bash
myenv\Scripts\activate
```

- On macOS and Linux:
```bash
source myenv/bin/activate
```

### Step 4: Install Dependencies
With the virtual environment activated, you can install the dependencies listed in your requirements.txt file using uv. Run the following command:
```bash
uv pip install -r requirements.txt
```

This command will read the requirements.txt file and install all the specified packages and their dependencies.

Both methods allow you to create isolated Python environments and install dependencies efficiently. Using venv is a built-in solution that comes with Python, while uv offers additional features and potentially faster performance. Choose the method that best fits your needs and workflow.

___

### Manual

Execution Order:

1. Filter Bookstates (Optional)
2. Filter Time (Optional)
3. Balancing 
4. Encoding 
5. Clean Correlations 
6. Choose one of the feature importance models (GBM, Lasso, Ridge, Logistic Regression)
7. Filter columns accordingly 
8. Use the prediction model for predictive quality
