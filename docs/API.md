# API Documentation

## Endpoints

### Chat

#### POST /chat
Process a chat message and get a response.

**Request:**
```json
{
    "message": "string"
}
```

**Response:**
```json
{
    "reply": "string",
    "topic": "string",
    "knowledge_articles": ["string"],
    "catalog_items": ["string"],
    "entities": [["string", "string"]],
    "sentiment": {
        "score": "float",
        "label": "string"
    }
}
```

### Settings

#### GET /api/settings
Get user settings.

**Response:**
```json
{
    "sessionTimeout": "integer",
    "entityRecognition": "boolean",
    "sentimentAnalysis": "boolean",
    "responseStyle": "string",
    "confidenceThreshold": "float",
    "contextMemory": "boolean",
    "autoSummarization": "boolean",
    "profanityFilter": "boolean",
    "languageSupport": ["string"],
    "memoryDuration": "string"
}
```

#### POST /api/settings
Update user settings.

**Request:**
```json
{
    "sessionTimeout": "integer",
    "entityRecognition": "boolean",
    "sentimentAnalysis": "boolean",
    "responseStyle": "string",
    "confidenceThreshold": "float",
    "contextMemory": "boolean",
    "autoSummarization": "boolean",
    "profanityFilter": "boolean",
    "languageSupport": ["string"],
    "memoryDuration": "string"
}
```

### Tickets

#### POST /add_ticket
Create a new ticket.

**Request:**
```json
{
    "user_id": "string",
    "description": "string",
    "priority": "string",
    "status": "string"
}
```

### Monitoring

#### GET /health
Get application health status.

**Response:**
```json
{
    "status": "string",
    "components": {
        "database": "string",
        "nlp_pipeline": "string"
    },
    "timestamp": "float"
}
```

#### GET /metrics
Get application metrics.

**Response:**
```json
{
    "requests": {
        "total": "integer",
        "errors": "integer",
        "average_duration": "float"
    },
    "cache": {
        "hits": "integer",
        "misses": "integer"
    }
}
```

## WebSocket Events

### Client to Server

- `connect`: Connect to WebSocket
- `disconnect`: Disconnect from WebSocket
- `join`: Join a chat room
- `leave`: Leave a chat room
- `chat_message`: Send a chat message
- `ticket_update`: Update ticket status

### Server to Client

- `connection_established`: Connection confirmation
- `connection_closed`: Disconnection confirmation
- `status`: Room join/leave status
- `new_message`: New chat message
- `error`: Error message
- `notification`: System notification 