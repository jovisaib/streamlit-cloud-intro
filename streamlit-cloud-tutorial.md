# Streamlit Cloud Deployment

---

## Agenda
1. What is Streamlit?
2. Setting up your Streamlit environment
3. Creating a simple Streamlit app
4. Preparing for deployment
5. Deploying to Streamlit Cloud
6. Common issues and troubleshooting
7. Q&A

---

## 1. What is Streamlit?

### Overview
- Open-source Python library for creating web apps with minimal effort
- No frontend experience needed (HTML, CSS, JavaScript)
- Perfect for data scientists and ML engineers to deploy models
- Built-in components for data visualization, inputs, and layout

### Streamlit Cloud
- Hosting platform for Streamlit apps
- Free tier available for public repositories
- Simple deployment from GitHub
- Automatic updates when repository changes

### Why Streamlit Cloud?
- Zero DevOps knowledge required
- No server management
- Easy sharing and collaboration
- Seamless deployment process

---

## 2. Setting up your Streamlit environment

### Prerequisites
- Python
- GitHub account
- Git installed locally

### Creating a development environment
```bash
# Create and activate virtual environment
python -m venv streamlit-env
source streamlit-env/bin/activate  # On Windows: streamlit-env\Scripts\activate

# Install Streamlit
pip install streamlit

# Verify installation
streamlit hello
```

### GitHub Repository Setup
- Create a new repository on GitHub
- Initialize local repository
- Connect to remote repository

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/your-username/your-repo.git
git push -u origin main
```

---

## 3. Creating a simple Streamlit app

### Example App: Data Explorer

```python
# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set page title
st.set_page_config(page_title="Data Explorer", page_icon="📊")

# Header
st.title("Data Explorer")
st.write("A simple app to explore your data")

# Data generation
def generate_data():
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    values = np.random.randn(100).cumsum()
    df = pd.DataFrame({"Date": dates, "Value": values})
    return df

# Sidebar
st.sidebar.header("Controls")
data_size = st.sidebar.slider("Sample Size", 10, 100, 50)

# Generate data
df = generate_data().iloc[:data_size]

# Display data
st.subheader("Raw Data")
st.dataframe(df)

# Visualization
st.subheader("Data Visualization")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df["Date"], df["Value"])
ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.grid(True)
st.pyplot(fig)

# Download options
st.subheader("Download Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="data.csv",
    mime="text/csv"
)
```

### Running your app locally
```bash
streamlit run app.py
```

---

## 4. Preparing for deployment

### Required files

1. `app.py` (your main application file)
2. `requirements.txt` (dependencies)
3. `.gitignore` (optional but recommended)

### Creating requirements.txt
```bash
pip freeze > requirements.txt
```

or create a minimal version:
```
streamlit==1.24.0
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
```

### .gitignore (basic)
```
# Python
__pycache__/
*.py[cod]
*$py.class
venv/
streamlit-env/
streamlit-env

# Environment variables
.env

# System files
.DS_Store
```

### Final repository structure
```
your-streamlit-app/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md (optional but recommended)
```

---

## 5. Deploying to Streamlit Cloud

### Step 1: Sign up for Streamlit Cloud
- Go to [share.streamlit.io](https://share.streamlit.io/)
- Sign in with your GitHub account

### Step 2: Deploy your app
1. Click "New app" button
2. Connect to your GitHub repository
3. Configure your app:
   - Repository: `your-username/your-repo`
   - Branch: `main` (or your preferred branch)
   - Main file path: `app.py`
   - Advanced settings (optional):
     - Python version
     - Package dependencies

### Step 3: Launch
- Click "Deploy"
- Wait for build process (usually 2-3 minutes)
- Access your app at `https://share.streamlit.io/your-username/your-repo/main/app.py`

### Step 4: Share and update
- Share the URL with others
- Any push to your repository will trigger automatic redeployment

---

## 6. Common issues and troubleshooting

### Package dependency issues
- Check requirements.txt for compatibility
- Specify exact versions of dependencies
- Consider using a Pipfile or conda environment file

### Memory limitations
- Free tier has memory constraints
- Optimize data loading and processing
- Use caching with `@st.cache_data` and `@st.cache_resource`

### Secrets management
- Never commit sensitive information
- Use Streamlit's secrets management:
  1. Add secrets via Streamlit Cloud dashboard
  2. Access with `st.secrets['key']`

### Performance optimization
- Minimize rerunning expensive operations
- Use session state for preserving state between reruns
- Implement caching for data loading and processing

---

## 7. Beyond the basics

### Custom theming
```python
# Custom theme in .streamlit/config.toml
[theme]
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Multipage apps
- Create a pages/ directory in your repo
- Add Python files prefixed with numbers
- Each file becomes a separate page in navigation

---

## Resources

### Documentation
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)

### Community
- [Streamlit Forum](https://discuss.streamlit.io/)
- [Streamlit GitHub](https://github.com/streamlit/streamlit)

### Tutorials
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Awesome Streamlit](https://github.com/MarcSkovMadsen/awesome-streamlit)


*Happy Streamlit Deployment!*