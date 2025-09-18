# Flight Search Agent - Architecture Documentation

## 🏗️ **Core Architecture Stack**

- **Agent Catalog**: Tool management and orchestration framework
- **Couchbase**: Vector database for airline reviews + NoSQL for bookings
- **LangGraph**: Agent workflow orchestration with ReAct pattern
- **OpenAI/Capella AI**: LLM backend with 4-tier priority system

## 📊 **Data Layer Components**

### **1. Airline Reviews (Vector Store)**

- **Source**: Kaggle Indian Airlines Customer Reviews dataset (via kagglehub)
- **Processing**: Converts reviews to structured text with airline, rating, title, content
- **Storage**: Couchbase vector store with embeddings for semantic search
- **Index**: Custom vector search index (`airline_reviews_index`)
- **Usage**: Powers `search_airline_reviews` tool for customer feedback queries

### **2. Flight Bookings (NoSQL)**

- **Scope**: `agentc_bookings`
- **Collection**: Daily collections (`user_bookings_YYYYMMDD`)
- **Schema**: booking_id, airports, date, passengers, class, price, status
- **Features**: Duplicate detection, automatic pricing, booking confirmations

### **3. Flight Routes (External)**

- **Source**: Couchbase `travel-sample` bucket (demo data)
- **Data**: Routes with airline codes, aircraft types, airport pairs
- **Usage**: Powers `lookup_flight_info` for flight availability

## 🔧 **Agent Tools (4 Core Functions)**

### **1. `lookup_flight_info`** - Flight Search

- **Input**: source_airport, destination_airport (3-letter codes)
- **Function**: Queries Couchbase travel-sample for available routes
- **Output**: Formatted list of flights with airline codes and aircraft types
- **Example**: "JFK,LAX" → Lists 8 airlines (AA, DL, UA, etc.) with equipment

### **2. `save_flight_booking`** - Flight Booking

- **Input**: Structured or natural language booking request
- **Processing**: Parses airports, dates, passengers, class; validates inputs
- **Features**: Duplicate detection, automatic pricing, booking ID generation
- **Output**: Confirmation with booking ID and details
- **Example**: Creates booking FL08061563CACD with full details

### **3. `retrieve_flight_bookings`** - Booking Management

- **Input**: Empty for all bookings, or "SOURCE,DEST,DATE" for specific
- **Function**: Queries daily booking collections with status filtering
- **Output**: Formatted list of current bookings with all details
- **Features**: Date-based collection partitioning, status management

### **4. `search_airline_reviews`** - Customer Feedback

- **Input**: Natural language query about airline services
- **Function**: Vector similarity search on embedded airline reviews
- **Output**: Top 5 relevant reviews with ratings and details
- **Example**: "SpiceJet service" → Returns customer feedback with ratings

## 💭 **Agent Prompt System (ReAct Pattern)**

### **Prompt Structure** (`flight_search_assistant.yaml`):

- **Framework**: ReAct (Reason + Act) pattern with strict formatting
- **Task Classification**: Automatically identifies flight search, booking, retrieval, or review tasks
- **Tool Mapping**: Direct tool calls without intermediate extraction steps
- **Error Recovery**: Built-in fallback strategies and alternative approaches

### **Key Behavior Rules**:

1. **Immediate Tool Execution**: No intermediate steps - calls tools directly
2. **Format Compliance**: Strict ReAct format (Question → Thought → Action → Observation → Final Answer)
3. **Error Handling**: Robust input parsing with multiple fallback strategies
4. **Completion Focus**: Always completes user requests successfully

### **Agent Flow**:

```
Query → Task Classification → Tool Selection → Parameter Parsing → Tool Execution → Response Formatting
```

### **Input Handling Examples**:

- "Find flights JFK to LAX" → `lookup_flight_info(JFK, LAX)`
- "Book 2 business class LAX to JFK tomorrow" → `save_flight_booking` with parsed details
- Natural language → Structured parameters automatically

## 📈 **Evaluation Framework (Arize Phoenix)**

### **Phoenix Observability**:

- **Tracing**: Full LangGraph execution traces with tool calls
- **UI Dashboard**: Real-time monitoring at http://localhost:6006
- **Instrumentation**: OpenTelemetry for LangChain + OpenAI integrations

### **Evaluation Metrics** (4 Phoenix Evaluators):

1. **Relevance**: Does response address the flight query?
2. **QA Correctness**: Is flight information accurate and helpful?
3. **Hallucination**: Does response contain fabricated information?
4. **Toxicity**: Is response harmful or inappropriate?

### **Reference Answers**:

Pre-defined expected outputs in `data/queries.py` for consistent evaluation

### **Arize Dataset Integration**:

- Automatic dataset creation from evaluation results
- Timestamped dataset names for version tracking
- Integration with Arize AI platform for production monitoring

### **Test Queries** (Standard Evaluation Set):

- Flight search: "Find flights from JFK to LAX"
- Booking: "Book a flight from LAX to JFK for tomorrow, 2 passengers, business class"
- Retrieval: "Show me my current flight bookings"
- Reviews: "What do passengers say about SpiceJet's service quality?"

## 📊 **System Architecture Flowchart**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FLIGHT SEARCH AGENT                          │
│                        (Agent Catalog + LangGraph)                     │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                             USER QUERY                                 │
│              ("Find flights JFK to LAX", "Book a flight")             │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       LANGRAPH WORKFLOW ENGINE                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
│  │ FlightSearch    │  │   ReAct Agent   │  │    Agent Catalog        │ │
│  │ Graph State     │→ │   (Reasoning)   │→ │   Tool Discovery        │ │
│  │                 │  │                 │  │                         │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            TOOL SELECTION                              │
│  ┌──────────────────┐ ┌──────────────────┐ ┌─────────────────────────┐ │
│  │ lookup_flight_   │ │ save_flight_     │ │ retrieve_flight_        │ │
│  │ info             │ │ booking          │ │ bookings                │ │
│  └──────────────────┘ └──────────────────┘ └─────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                  search_airline_reviews                            │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          COUCHBASE DATABASE                            │
│  ┌─────────────────────┐            ┌─────────────────────────────────┐ │
│  │   VECTOR STORE      │            │         NoSQL STORE             │ │
│  │                     │            │                                 │ │
│  │ • Airline Reviews   │            │ • Flight Bookings               │ │
│  │ • Vector Embeddings │            │ • User Sessions                 │ │
│  │ • Similarity Search │            │ • Daily Partitions              │ │
│  │                     │            │                                 │ │
│  └─────────────────────┘            └─────────────────────────────────┘ │
│                    │                              │                     │
│         (Capella AI Embeddings)        (SQL++ Queries)                 │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         RESPONSE GENERATION                            │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                    LLM BACKEND (4-TIER)                            │ │
│  │  1. Capella AI (Priority)  2. OpenAI  3. Fallback  4. Local       │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        FORMATTED RESPONSE                              │
│            (Flight listings, Booking confirmations, etc.)              │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🔄 **Data Flow Workflow**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FLIGHT SEARCH │    │  FLIGHT BOOKING │    │  REVIEW SEARCH  │
│                 │    │                 │    │                 │
│ JFK → LAX       │    │ Book JFK→MIA    │    │ SpiceJet service│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│lookup_flight_   │    │save_flight_     │    │search_airline_  │
│info()           │    │booking()        │    │reviews()        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   SQL++ Query   │    │  Input Parsing  │    │  Vector Search  │
│                 │    │                 │    │                 │
│ travel-sample   │    │ • Airports      │    │ • Embeddings    │
│ .inventory.route│    │ • Date parsing  │    │ • Similarity    │
│                 │    │ • Passenger cnt │    │ • Top-K results │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ 8 Flight Routes │    │ Booking Record  │    │  5 Reviews      │
│ • Airlines      │    │ • FL08061563... │    │ • Ratings       │
│ • Aircraft      │    │ • Confirmation  │    │ • Customer      │
│ • Route info    │    │ • Price calc    │    │ • Experience    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 **Evaluation Pipeline Flow**

```
┌─────────────────────────────────────────────────────────────────────┐
│                      EVALUATION PIPELINE                           │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐
│  Test Queries   │  │  Agent Setup    │  │   Phoenix Observability │
│                 │  │                 │  │                         │
│ • Flight search │→ │ • Clear data    │→ │ • Launch UI (port 6006) │
│ • Booking       │  │ • Initialize    │  │ • OTEL instrumentation │
│ • Retrieval     │  │ • Load reviews  │  │ • Trace collection      │
│ • Reviews       │  │                 │  │                         │
└─────────────────┘  └─────────────────┘  └─────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PHOENIX EVALUATORS                             │
│  ┌────────────────┐ ┌────────────────┐ ┌─────────────────────────┐ │
│  │   Relevance    │ │ QA Correctness │ │      Hallucination      │ │
│  │                │ │                │ │                         │ │
│  │ Does response  │ │ Is information │ │ Contains fabricated     │ │
│  │ address query? │ │ accurate?      │ │ information?            │ │
│  └────────────────┘ └────────────────┘ └─────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                        Toxicity                                │ │
│  │                 Is response harmful?                           │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐
│ Results DataFrame│→ │ Arize Dataset   │→ │   Performance Report    │
│                 │  │                 │  │                         │
│ • Query/Response│  │ • Timestamped   │  │ • Success rates         │
│ • Eval scores   │  │ • Versioned     │  │ • Execution times       │
│ • Explanations  │  │ • Exportable    │  │ • Quality metrics       │
└─────────────────┘  └─────────────────┘  └─────────────────────────┘
```

## 🚀 **Key Technical Implementation Details**

### **Agent Catalog Integration**

- **Prompt-Embedded Tools**: Tools declared directly in prompt YAML for single source of truth
- **Automatic Tool Discovery**: `prompt_resource.tools` provides direct access to embedded tools
- **Prompt Management**: YAML-based prompt templates with dynamic content injection
- **Session Tracking**: Built-in observability and activity logging
- **Multi-framework Support**: Works with LangGraph, LangChain, and LlamaIndex

### **Couchbase Integration**

- **Dual Database Pattern**: Vector store for reviews + NoSQL for transactional data
- **Connection Pooling**: Shared cluster connections across tool modules
- **Automatic Setup**: Dynamic collection/scope creation with proper indexing
- **Data Partitioning**: Daily collections for bookings with cleanup strategies

### **Tool Wrapper Architecture**

- **Interface Translation**: Bridges Agent Catalog tools (structured parameters) with LangChain ReAct agent (single string inputs)
- **Input Sanitization**: Removes ReAct format artifacts (`\nObservation`, `Action:`, etc.) from tool inputs
- **Multi-format Parsing**: Supports various input formats:
  - Key-value: `source_airport="JFK", destination_airport="LAX"`
  - Comma-separated: `"JFK,LAX"`  
  - Natural language: `"JFK to LAX"`
- **Parameter Mapping**: Converts single strings to named function parameters
- **Error Handling**: Provides user-friendly error messages and graceful degradation

**Why Tool Wrappers Are Needed**:
```python
# Agent Catalog tools expect:
lookup_flight_info(source_airport="JFK", destination_airport="LAX")

# LangChain ReAct provides:
tool_input = "JFK to LAX\nObservation: ..."  # Messy string

# Wrapper converts:  string → structured parameters → tool call
```

### **Error Handling & Robustness**

- **Input Parsing**: Multiple fallback strategies for natural language processing
- **Connection Recovery**: Automatic reconnection and timeout handling
- **Validation Layers**: Airport codes, date formats, passenger counts
- **Graceful Degradation**: Meaningful error messages for users

### **Performance Optimizations**

- **Batch Processing**: Embeddings created in configurable batch sizes
- **Connection Reuse**: Global cluster instances prevent connection overhead
- **Caching Strategies**: Processed data caching in memory for repeated loads
- **Query Optimization**: Parameterized queries prevent SQL injection

### **Production Considerations**

- **Environment Configuration**: 12-factor app pattern with .env files
- **Logging Integration**: Structured logging with configurable levels
- **Monitoring Ready**: Phoenix traces + Arize dataset exports
- **Scalability**: Stateless design supports horizontal scaling

## 🎯 **Interview Talking Points**

### **Architecture Strengths**:

1. **Modular Design**: Clean separation between tools, data, and orchestration
2. **Technology Integration**: Demonstrates modern AI stack (Agent Catalog + Couchbase + LangGraph)
3. **Production Ready**: Comprehensive error handling, monitoring, and evaluation
4. **Extensible**: Easy to add new tools or modify existing functionality

### **Technical Depth**:

1. **Vector Search**: Semantic search implementation with embedding strategies
2. **Database Design**: Multi-modal data storage (vector + NoSQL) patterns
3. **Agent Workflows**: ReAct pattern implementation with LangGraph state management
4. **Evaluation Framework**: Comprehensive testing with LLM-as-a-judge metrics

### **Business Value**:

1. **User Experience**: Natural language to structured operations
2. **Data Integration**: Real customer review data enhances booking decisions
3. **Observability**: Full traceability for debugging and optimization
4. **Scalability**: Architecture supports production deployment scenarios
