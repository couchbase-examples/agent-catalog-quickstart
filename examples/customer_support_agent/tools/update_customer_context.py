"""
Customer Context Management Tool

This tool manages customer context and interaction history
to provide personalized support experiences.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional

# Agent Catalog tool metadata
TOOL_METADATA = {
    "name": "update_customer_context",
    "description": "Update and manage customer context, preferences, and interaction history for personalized support",
    "input": {
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string",
                "description": "Unique customer identifier"
            },
            "context_update": {
                "type": "object",
                "properties": {
                    "preferences": {
                        "type": "object",
                        "description": "Customer preferences (seating, meals, etc.)"
                    },
                    "interaction_type": {
                        "type": "string",
                        "enum": ["inquiry", "complaint", "booking", "modification", "cancellation"]
                    },
                    "resolution_status": {
                        "type": "string", 
                        "enum": ["pending", "in_progress", "resolved", "escalated"]
                    },
                    "satisfaction_score": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "Customer satisfaction rating (1-5)"
                    },
                    "notes": {
                        "type": "string",
                        "description": "Additional notes about the interaction"
                    }
                }
            }
        },
        "required": ["customer_id", "context_update"]
    },
    "annotations": {
        "privacy": "pii_handling",
        "retention": "customer_data",
        "gdpr_compliant": "true"
    }
}


def update_customer_context(customer_id: str, context_update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update customer context and return the updated customer profile.
    
    Args:
        customer_id: Unique identifier for the customer
        context_update: Dictionary containing context updates
        
    Returns:
        Updated customer context dictionary
    """
    
    # In a real implementation, this would connect to a customer database
    # For demo purposes, we'll simulate context management
    
    timestamp = datetime.now().isoformat()
    
    # Simulate fetching existing customer context
    existing_context = {
        "customer_id": customer_id,
        "preferences": {},
        "interaction_history": [],
        "current_status": "active",
        "last_updated": None
    }
    
    # Update preferences if provided
    if "preferences" in context_update:
        existing_context["preferences"].update(context_update["preferences"])
    
    # Add new interaction to history
    interaction_record = {
        "timestamp": timestamp,
        "type": context_update.get("interaction_type", "inquiry"),
        "status": context_update.get("resolution_status", "pending"),
        "notes": context_update.get("notes", ""),
        "satisfaction_score": context_update.get("satisfaction_score")
    }
    
    existing_context["interaction_history"].append(interaction_record)
    existing_context["last_updated"] = timestamp
    
    # Calculate customer satisfaction trend
    recent_scores = [
        i.get("satisfaction_score") 
        for i in existing_context["interaction_history"][-5:] 
        if i.get("satisfaction_score")
    ]
    
    if recent_scores:
        avg_satisfaction = sum(recent_scores) / len(recent_scores)
        existing_context["satisfaction_trend"] = round(avg_satisfaction, 2)
    
    # Determine customer tier based on interaction history
    interaction_count = len(existing_context["interaction_history"])
    if interaction_count >= 10:
        existing_context["customer_tier"] = "premium"
    elif interaction_count >= 5:
        existing_context["customer_tier"] = "standard"
    else:
        existing_context["customer_tier"] = "new"
    
    return {
        "success": True,
        "customer_context": existing_context,
        "update_summary": {
            "updated_fields": list(context_update.keys()),
            "timestamp": timestamp,
            "interaction_count": len(existing_context["interaction_history"])
        }
    }


def get_customer_insights(customer_id: str) -> Dict[str, Any]:
    """
    Get insights about customer behavior and preferences.
    
    Args:
        customer_id: Customer identifier
        
    Returns:
        Customer insights dictionary
    """
    
    # This would typically query analytics data
    return {
        "customer_id": customer_id,
        "insights": {
            "preferred_contact_method": "chat",
            "typical_inquiry_types": ["flight_changes", "baggage"],
            "resolution_time_preference": "immediate",
            "communication_style": "detailed"
        },
        "recommendations": {
            "proactive_services": ["flight_alerts", "check_in_reminders"],
            "upsell_opportunities": ["premium_seating", "priority_boarding"]
        }
    }


# Main function for testing
if __name__ == "__main__":
    # Test the customer context update
    test_update = {
        "preferences": {
            "seating": "aisle",
            "meal": "vegetarian"
        },
        "interaction_type": "booking",
        "resolution_status": "resolved", 
        "satisfaction_score": 4,
        "notes": "Customer successfully booked round-trip flight to Paris"
    }
    
    result = update_customer_context("CUST_001", test_update)
    print(json.dumps(result, indent=2))