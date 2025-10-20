# Dockerfile (Final Optimized Version with CPU-only PyTorch)

# Stage 1: Build the Python environment
FROM python:3.11-slim as builder

WORKDIR /app

# Copy requirements.txt first for caching
COPY requirements.txt .

# 3. Install all other dependencies from the modified requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# --- END OF OPTIMIZATION ---

# Stage 2: Create the final, clean image
FROM python:3.11-slim

WORKDIR /app

# Copy the installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
# Also copy the executables (like streamlit)
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy only the necessary application files
COPY app.py .env ./
COPY app ./app
COPY model/capsnet ./model/capsnet
COPY data/raw/ ./data/raw/
COPY output/hackathon/run_states/ ./output/hackathon/run_states/
COPY figures/langraph_test/ ./figures/langraph_test/

# Expose the Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]