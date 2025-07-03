# Route Planner Agent

A comprehensive route planning agent built with LlamaIndex, Couchbase vector search, and Capella AI model services for intelligent travel planning and recommendations.

## Features

- **Intelligent Route Planning**: Find optimal routes between destinations using semantic search
- **Multi-Modal Transportation**: Compare car, train, bus, and flight options with time and cost analysis
- **Point of Interest Discovery**: Find restaurants, attractions, gas stations, and accommodations along routes
- **Real-time Recommendations**: Get personalized travel advice based on preferences and context
- **Comprehensive Knowledge Base**: Access extensive travel information and route data

## Architecture

- **LlamaIndex**: RAG (Retrieval Augmented Generation) framework for intelligent query processing
- **Couchbase Capella**: Vector database for semantic search of travel knowledge
- **Capella AI Services**: Embedding and LLM services for natural language understanding
- **Agent Catalog**: Tool orchestration and prompt management

## Setup

### Prerequisites

1. **Couchbase Capella Account**: Create a cluster with Data, Query, Index, and Search services
2. **Capella AI Services**: Deploy embedding and LLM models in your Capella instance
3. **Python 3.12+**: Required for the application

### Installation

1. **Clone the repository**:
   ```bash
   cd examples/route_planner_agent
   ```

2. **Install dependencies**:
   ```bash
   pip install -e .
   # or with poetry
   poetry install
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your Couchbase and Capella AI credentials
   ```

### Environment Variables

- `CB_CONN_STRING`: Couchbase cluster connection string
- `CB_USERNAME`: Database username
- `CB_PASSWORD`: Database password  
- `CB_BUCKET`: Bucket name for route data
- `SCOPE_NAME`: Scope name (default: shared)
- `COLLECTION_NAME`: Collection name (default: route_data)
- `INDEX_NAME`: Vector search index name
- `CAPELLA_AI_ENDPOINT`: Capella AI Services endpoint URL

## Usage

### Running the Agent

```bash
python main.py
```

### Example Interactions

**Simple Route Query**:
```
🧭 You: What's the best way to get from San Francisco to Los Angeles?

🤖 Route Planner: I'll help you find the best route from San Francisco to Los Angeles! Let me search for route options...

Based on my search, here are the main options:

**Option 1: Highway 1 (Pacific Coast Highway) - Scenic Route**
- Distance: ~400 miles
- Duration: 8-10 hours driving
- Highlights: Breathtaking ocean views, Big Sur, Monterey, Santa Barbara
- Best for: Scenic road trip with time to spare

**Option 2: Highway 101 - Faster Route**  
- Distance: ~380 miles
- Duration: 6-7 hours driving
- Highlights: Faster inland route through central valley
- Best for: Direct travel with minimal stops

Would you like me to find specific attractions, restaurants, or accommodations along either route?
```

**Multi-Stop Planning**:
```
🧭 You: Plan a 5-day road trip from Seattle to San Francisco with stops at national parks

🤖 Route Planner: I'll create an amazing 5-day Pacific Northwest to California road trip for you!

**Day 1: Seattle to Mount Rainier National Park**
- Drive: 2 hours to Paradise area
- Activities: Hiking trails, wildflower meadows
- Overnight: Paradise Inn or nearby Ashford

**Day 2: Mount Rainier to Crater Lake National Park**
- Drive: 4 hours through Oregon
- Activities: Rim Drive, lake viewpoints
- Overnight: Crater Lake Lodge

**Day 3: Crater Lake to Redwood National Park**
- Drive: 4 hours to coastal redwoods
- Activities: Tall Trees Grove, coastal drives
- Overnight: Eureka or Crescent City

**Day 4: Redwoods to San Francisco**
- Drive: 5 hours via Highway 101
- Stop: Sonoma wine country (optional)
- Arrive: San Francisco evening

**Day 5: Explore San Francisco**
- Local attractions and activities

Would you like me to find specific accommodations, restaurants, or calculate total costs for this itinerary?
```

## Tools Available

### Route Planning Tools
- `search_routes`: Semantic search for route information  
- `search_routes_by_cities`: Specific city-to-city planning
- `search_scenic_routes`: Beautiful and interesting routes
- `calculate_distance`: Distance, time, and cost calculations
- `compare_transport_options`: Multi-modal transportation analysis

### Point of Interest Tools  
- `find_restaurants`: Dining recommendations along routes
- `find_attractions`: Tourist attractions and landmarks
- `find_gas_stations`: Fuel stops for road trips
- `find_accommodations`: Hotels and lodging options
- `plan_stops_along_route`: Comprehensive stop planning

## Development

### Adding New Tools

1. Create tool file in `tools/` directory
2. Use `@agentc.catalog.tool` decorator
3. Follow existing tool patterns for error handling and responses

### Extending Knowledge Base

1. Add data to `data/route_data.py`
2. Follow the existing document structure with metadata
3. Re-run ingestion to update vector store

### Testing

```bash
pytest tests/
```

## Troubleshooting

### Connection Issues
- Verify Couchbase cluster is running and accessible
- Check network connectivity and firewall settings
- Ensure database credentials are correct

### Vector Search Issues  
- Confirm search index exists and is online
- Check embedding model deployment in Capella AI
- Verify index configuration matches data structure

### Performance Optimization
- Adjust `similarity_top_k` for query results
- Tune chunk size and overlap for document ingestion
- Monitor Capella AI service usage and scaling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

See LICENSE file in the repository root. 