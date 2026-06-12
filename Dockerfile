FROM python:3.10-slim

# Install system dependencies if any (usually not needed for pure PyTorch, but good to have basics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces require running as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install
COPY --chown=user:user webapp/requirements.txt $HOME/app/webapp/requirements.txt
RUN pip install --no-cache-dir -r $HOME/app/webapp/requirements.txt

# Copy the rest of the application (including all model folders and webapp)
COPY --chown=user:user . $HOME/app

# Change directory to webapp
WORKDIR $HOME/app/webapp

# Hugging Face Spaces exposes port 7860 by default
EXPOSE 7860

# Run the FastAPI server via uvicorn on port 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
