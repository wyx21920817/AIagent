# AI Doctor Assistant

An intelligent doctor assistant with agent capabilities including:
- Long-term memory storage and retrieval
- Multi-turn dialogue management  
- Tool/API calling abilities
- Medical knowledge integration
- DICOM image processing

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
Create a `.env` file with:
```
OPENAI_API_KEY=your_key_here
```

4. Run the application:
```bash
uvicorn app.main:app --reload
```

## Docker Deployment

### Building the Docker Image

```bash
# Build the Docker image
docker build -t aiagent .
```

### Running the Container

```bash
# Run the container with environment variables
docker run -p 8888:8888 --env-file .env -d aiagent
```



### Docker Compose

Alternatively, you can use docker-compose:

```bash
# Start the application using docker-compose
docker-compose up -d
```
