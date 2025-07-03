# Route Planner Agent - Simplified Task Master

## Project Overview
Building a LlamaIndex-based route planner agent with Couchbase vector store and Capella AI model services for intelligent route planning and travel recommendations. **SIMPLIFIED VERSION - Bare minimum working implementation.**

## ✅ PRIORITY 1: Core Infrastructure (Bare Minimum)
- [x] Create project configuration
  - [x] `/examples/route_planner_agent/pyproject.toml` - Poetry dependencies for LlamaIndex, Couchbase, Agent Catalog
  - [ ] `/examples/route_planner_agent/.env.example` - Environment variables template
  - [x] `/examples/route_planner_agent/README.md` - Project documentation and setup instructions

- [x] Set up Couchbase vector search infrastructure  
  - [x] `/examples/route_planner_agent/fts_index.json` - Full-text search index configuration for vector search
  - [x] `/examples/route_planner_agent/main.py` - Basic main implementation (needs simplification)

- [x] Create data pipeline and knowledge base
  - [x] `/examples/route_planner_agent/data/route_data.py` - Travel and route knowledge base data

## ✅ PRIORITY 2: Minimal Agent Implementation
- [x] Build essential route planning tools (KEEP MINIMAL - 2-3 tools max)
  - [x] `/examples/route_planner_agent/tools/search_routes.py` - Basic semantic search for route information
  - [x] `/examples/route_planner_agent/tools/calculate_distance.py` - Simple distance calculations
  - [ ] `/examples/route_planner_agent/tools/find_points_of_interest.py` - **MOVE TO ADVANCED** 
  - [ ] `/examples/route_planner_agent/tools/compare_transport_options.py` - **MOVE TO ADVANCED**

- [x] Design conversation prompts (KEEP SIMPLE)
  - [x] `/examples/route_planner_agent/prompts/route_planner_assistant.yaml` - **SIMPLIFY** - remove advanced features

- [x] Build LlamaIndex RAG agent (SIMPLIFY)
  - [x] `/examples/route_planner_agent/main.py` - **SIMPLIFY** main route planner agent

## 🔧 PRIORITY 3: Basic Testing
- [ ] Create minimal tests
  - [ ] `/examples/route_planner_agent/tests/test_basic_functionality.py` - Basic functionality test
  - [ ] `/examples/route_planner_agent/tests/test_vector_search.py` - Vector search test

## 📚 PRIORITY 4: Basic Documentation
- [ ] Create simple usage example
  - [ ] `/examples/route_planner_agent/examples/basic_route_planning.py` - Simple route planning example

## 🚀 ADVANCED FEATURES (MOVED TO OPTIONAL)
- [ ] **Advanced tools (Optional - implement later)**
  - [ ] `/examples/route_planner_agent/tools/find_points_of_interest.py` - Search POIs along routes 
  - [ ] `/examples/route_planner_agent/tools/get_traffic_info.py` - Real-time traffic data
  - [ ] `/examples/route_planner_agent/tools/save_route_plan.py` - Save user preferences
  - [ ] `/examples/route_planner_agent/tools/search_accommodations.py` - Find hotels and lodging
  - [ ] `/examples/route_planner_agent/tools/get_weather_info.py` - Weather data
  - [ ] `/examples/route_planner_agent/tools/find_transport_options.py` - Multi-modal transport
  - [ ] `/examples/route_planner_agent/tools/compare_transport_options.py` - Transport comparisons

- [ ] **Advanced prompts (Optional)**
  - [ ] `/examples/route_planner_agent/prompts/multi_stop_planner.yaml` - Complex multi-destination planning
  - [ ] `/examples/route_planner_agent/prompts/travel_advisor.yaml` - Travel recommendations

- [ ] **Advanced agent features (Optional)**
  - [ ] `/examples/route_planner_agent/state/route_state.py` - Route planning conversation state
  - [ ] `/examples/route_planner_agent/state/user_preferences.py` - User preference tracking
  - [ ] `/examples/route_planner_agent/agents/route_agent.py` - Advanced agent class
  - [ ] `/examples/route_planner_agent/workflows/planning_workflow.py` - Multi-step workflow
  - [ ] `/examples/route_planner_agent/workflows/recommendation_engine.py` - Personalized recommendations

- [ ] **Enhanced capabilities (Optional)**
  - [ ] `/examples/route_planner_agent/features/real_time_updates.py` - Real-time updates
  - [ ] `/examples/route_planner_agent/features/group_travel.py` - Group planning
  - [ ] `/examples/route_planner_agent/features/budget_optimizer.py` - Cost optimization
  - [ ] `/examples/route_planner_agent/features/eco_routes.py` - Eco-friendly routes

## 🎯 SIMPLIFIED SUCCESS CRITERIA
- [ ] Agent can plan basic routes between two locations
- [ ] Vector search works with Couchbase and Capella AI embeddings  
- [ ] Basic route recommendations with simple POI suggestions
- [ ] Simple conversation interface similar to flight search agent
- [ ] Minimal working demo that can be extended later

## 📋 IMMEDIATE ACTION ITEMS
1. **Simplify main.py** - Remove complex features, focus on basic RAG
2. **Simplify prompt** - Remove advanced tool references, keep it basic
3. **Create .env.example** - Document required environment variables
4. **Create basic test** - One simple test to verify functionality
5. **Create simple README** - Basic setup and usage instructions

## 🎨 ARCHITECTURE GOAL
**Follow flight search agent pattern:**
- Simple main.py with clear setup functions
- 2-3 essential tools maximum
- Basic prompt configuration
- Minimal but working vector search
- Clear documentation and examples