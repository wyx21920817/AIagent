FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# Install Python 3.10 and other necessary packages
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3.10-dev \
    lsof \
    net-tools \
    curl \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Create app directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make start.sh executable
RUN chmod +x start.sh

# Create necessary directories
RUN mkdir -p logs reports

# Expose the port the app runs on
EXPOSE 8888

# Create a modified start script that runs in foreground instead of background
RUN sed -i 's/uvicorn app.main:app --host 0.0.0.0 --port $PORT --reload >> logs\/server.log 2>&1 &/uvicorn app.main:app --host 0.0.0.0 --port $PORT --reload/' start.sh && \
    sed -i '/echo $! > .server.pid/d' start.sh && \
    sed -i '/echo "Server started in background. PID: $(cat .server.pid)"/d' start.sh && \
    sed -i '/deactivate/d' start.sh

# Command to run the application
CMD ["/bin/bash", "-c", "./start.sh && tail -f logs/server.log"]
