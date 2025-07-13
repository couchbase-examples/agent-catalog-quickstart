#!/usr/bin/env python3
"""
Simplified Flight Data for Vector Search Ingestion

Only contains flight policies as per project requirements.
Flight routes are sourced from travel-sample.inventory.route database.
Flight bookings are handled by save_flight_booking.py tool.
"""

# Flight Policy Documents (ONLY data we keep as mock)
FLIGHT_POLICIES = [
    {
        "policy_id": "CANCEL_001",
        "title": "Flight Cancellation Policy",
        "category": "cancellation",
        "content": """Flight cancellation policies vary by ticket type and timing. Economy tickets cancelled within 24 hours of booking receive full refunds. Cancellations made 7+ days before departure incur a $200 fee for domestic flights and $400 for international flights. Cancellations within 7 days of departure forfeit 50% of ticket value. Premium and business class tickets offer more flexible cancellation terms with reduced fees. No-show passengers forfeit entire ticket value unless extenuating circumstances apply.""",
    },
    {
        "policy_id": "BAGGAGE_001",
        "title": "Carry-on Baggage Restrictions",
        "category": "baggage",
        "content": """Carry-on bags must not exceed 22x14x9 inches and 15 pounds. Personal items like purses, laptops, and small backpacks are permitted in addition to carry-on bags. Liquids must be in containers of 3.4oz or less and fit in a quart-sized clear bag. Prohibited items include sharp objects, flammable materials, and large electronics. Medical equipment and baby formula are exempt from liquid restrictions with proper documentation.""",
    },
    {
        "policy_id": "BAGGAGE_002",
        "title": "Checked Baggage Fees and Limits",
        "category": "baggage",
        "content": """First checked bag costs $35 domestic, $60 international. Second bag costs $45 domestic, $100 international. Maximum weight is 50 pounds per bag. Overweight bags (51-70 lbs) incur $100 fee. Oversized bags exceeding 62 linear inches cost additional $200. Premium passengers receive one free checked bag. Military personnel with orders travel with free checked bags.""",
    },
    {
        "policy_id": "CHECKIN_001",
        "title": "Online and Mobile Check-in",
        "category": "check-in",
        "content": """Online check-in opens 24 hours before departure and closes 1 hour before domestic flights, 2 hours before international flights. Mobile boarding passes are accepted at all airports. Passengers with checked bags must visit counter or kiosk for bag drop. Seat selection is available during check-in for a fee on economy tickets. Check-in reminders are sent via email and SMS 24 hours prior to departure.""",
    },
    {
        "policy_id": "WEATHER_001",
        "title": "Weather Delay and Cancellation Policy",
        "category": "weather",
        "content": """Weather-related delays and cancellations are considered extraordinary circumstances beyond airline control. Passengers are rebooked on next available flight at no additional cost. Hotel accommodations not provided for weather delays, but airport vouchers may be issued for extended delays. Travel insurance recommended to cover weather-related expenses. Real-time flight status updates provided via app, email, and SMS.""",
    },
    {
        "policy_id": "UPGRADE_001",
        "title": "Flight Upgrade and Change Policy",
        "category": "upgrade",
        "content": """Flight upgrades available based on availability and passenger status. Elite members receive complimentary upgrades 24-72 hours before departure. Paid upgrades can be purchased at booking, check-in, or at the gate. Change fees apply: $200 domestic, $400 international for economy tickets. Same-day changes available for $75 fee. Premium class tickets allow free changes up to 24 hours before departure.""",
    },
    {
        "policy_id": "SPECIAL_001",
        "title": "Special Assistance and Medical Equipment",
        "category": "special_assistance",
        "content": """Special assistance available for passengers with disabilities, unaccompanied minors, and medical needs. Wheelchair assistance provided free of charge with advance notice. Medical equipment like CPAP machines, oxygen concentrators allowed with proper documentation. Service animals travel free in cabin. Emotional support animals require advance approval and documentation. Medical prescriptions exempt from liquid restrictions.""",
    },
    {
        "policy_id": "LOYALTY_001",
        "title": "Frequent Flyer and Elite Benefits",
        "category": "loyalty",
        "content": """Frequent flyer program offers miles for flights, upgrades, and partner purchases. Elite status levels: Silver (25k miles), Gold (50k miles), Platinum (75k miles), Diamond (100k miles). Benefits include priority boarding, free checked bags, seat upgrades, and lounge access. Miles expire after 24 months of inactivity. Elite members receive bonus miles and priority customer service.""",
    }
]


def get_flight_policies():
    """Return flight policies for vector search ingestion."""
    return FLIGHT_POLICIES


def get_policy_by_category(category):
    """Get policies by category (e.g., 'baggage', 'cancellation')."""
    return [policy for policy in FLIGHT_POLICIES if policy["category"] == category]


def get_policy_by_id(policy_id):
    """Get a specific policy by ID."""
    for policy in FLIGHT_POLICIES:
        if policy["policy_id"] == policy_id:
            return policy
    return None


def search_policies(query_text):
    """Simple text search in policy content."""
    results = []
    query_lower = query_text.lower()
    
    for policy in FLIGHT_POLICIES:
        if (query_lower in policy["title"].lower() or 
            query_lower in policy["content"].lower() or 
            query_lower in policy["category"].lower()):
            results.append(policy)
    
    return results


# Legacy function for compatibility (now returns only policies)
def get_all_flight_data():
    """Return all flight-related data for vector search ingestion."""
    return {
        "policies": FLIGHT_POLICIES,
        "routes": [],  # Now sourced from travel-sample database
        "bookings": []  # Now handled by booking tools
    }


def load_flight_policies_to_couchbase(cluster, bucket_name, scope_name, collection_name, embeddings, index_name=None):
    """Load flight policies into Couchbase vector store.
    
    Args:
        cluster: Connected Couchbase cluster
        bucket_name: Name of the bucket
        scope_name: Name of the scope
        collection_name: Name of the collection
        embeddings: Configured embeddings model
        index_name: Optional index name (defaults to collection_name + '_index')
    """
    from langchain_couchbase.vectorstores import CouchbaseVectorStore
    import time
    
    if not index_name:
        index_name = f"{collection_name}_index"
    
    print(f"Loading flight policies to {bucket_name}.{scope_name}.{collection_name}")
    
    def add_documents_with_retry(vector_store, texts, metadatas, max_retries=3, batch_size=3):
        """Add documents with retry logic and batching to handle timeouts."""
        total_docs = len(texts)
        successful_docs = 0
        
        # Process in smaller batches to avoid timeouts
        for i in range(0, total_docs, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            for attempt in range(max_retries):
                try:
                    print(f"Processing batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} (docs {i+1}-{min(i+batch_size, total_docs)})")
                    
                    # Add batch with longer timeout
                    vector_store.add_texts(texts=batch_texts, metadatas=batch_metadatas)
                    successful_docs += len(batch_texts)
                    print(f"✓ Successfully added batch of {len(batch_texts)} documents")
                    break  # Success, move to next batch
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        print(f"⚠️ Batch failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ Batch failed after {max_retries} attempts: {e}")
                        # Continue with next batch instead of failing completely
                        continue
            
            # Small delay between batches to avoid overwhelming the connection
            if i + batch_size < total_docs:
                time.sleep(0.5)
        
        return successful_docs
    
    try:
        # Initialize vector store with provided parameters
        vector_store = CouchbaseVectorStore(
            cluster=cluster,
            bucket_name=bucket_name,
            scope_name=scope_name,
            collection_name=collection_name,
            embedding=embeddings,
            index_name=index_name,
        )
        print("✓ Vector store initialized")
        
        # Prepare documents for ingestion
        texts = []
        metadatas = []
        
        for policy in FLIGHT_POLICIES:
            # Create searchable text combining title and content
            text = f"{policy['title']}: {policy['content']}"
            texts.append(text)
            
            # Create metadata
            metadata = {
                "policy_id": policy["policy_id"],
                "title": policy["title"],
                "category": policy["category"],
                "type": "policy",
                "source": "flight_policies"
            }
            metadatas.append(metadata)
        
        print(f"Prepared {len(texts)} policy documents for ingestion")
        
        # Add documents with retry logic and batching
        successful_count = add_documents_with_retry(vector_store, texts, metadatas)
        
        if successful_count == len(texts):
            print(f"✅ Successfully loaded all {successful_count} flight policies into vector store")
        else:
            print(f"⚠️ Loaded {successful_count}/{len(texts)} flight policies (some failed due to timeouts)")
        
        return vector_store
        
    except Exception as e:
        print(f"❌ Error loading flight policies: {e}")
        raise
