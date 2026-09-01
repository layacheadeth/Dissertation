# Use an official lightweight Python image
FROM python:3.11-slim

# Install basic system tools needed for building dependencies (like rank_bm25 or tokenizers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /workspace

# Copy only requirements first to leverage Docker's layer caching
COPY requirements.txt .

# Install PyTorch 2.4.0 (CPU version optimized for x86 Linux) followed by your app dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your modular codebase into the container
COPY . .

# Expose port 8888 in case you want to execute inside the Jupyter notebooks
EXPOSE 8888

# Set default execution to run your main script array processing
CMD ["python3", "main.py"]