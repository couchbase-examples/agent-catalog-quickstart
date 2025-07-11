# """
# Ground Truth Dataset for Route Planner Agent Evaluation

# This module contains verified route information for fact-checking and 
# anti-hallucination evaluation of the route planner agent.
# """

# import re
# from typing import Dict, List, Optional, Tuple
# from dataclasses import dataclass


# @dataclass
# class RouteGroundTruth:
#     """Ground truth information for a specific route."""
#     origin: str
#     destination: str
#     driving_distance_miles: Optional[float]
#     driving_time_hours: Optional[float]
#     flight_distance_miles: Optional[float]
#     flight_time_hours: Optional[float]
#     major_highways: List[str]
#     available_transport_modes: List[str]
#     exists: bool = True
#     notes: str = ""


# class GroundTruthValidator:
#     """Validator for checking agent responses against ground truth data."""
    
#     def __init__(self):
#         self.ground_truth_data = self._load_ground_truth()
    
#     def _load_ground_truth(self) -> Dict[str, RouteGroundTruth]:
#         """Load verified ground truth route data."""
#         return {
#             "new_york_to_boston": RouteGroundTruth(
#                 origin="New York",
#                 destination="Boston",
#                 driving_distance_miles=215.0,
#                 driving_time_hours=4.5,
#                 flight_distance_miles=190.0,
#                 flight_time_hours=1.5,
#                 major_highways=["I-95", "I-84", "I-90"],
#                 available_transport_modes=["car", "train", "bus", "flight"],
#                 notes="Popular Northeast corridor route"
#             ),
#             "los_angeles_to_san_francisco": RouteGroundTruth(
#                 origin="Los Angeles",
#                 destination="San Francisco",
#                 driving_distance_miles=380.0,
#                 driving_time_hours=6.0,
#                 flight_distance_miles=337.0,
#                 flight_time_hours=1.5,
#                 major_highways=["I-5", "US-101", "CA-1"],
#                 available_transport_modes=["car", "flight", "bus"],
#                 notes="California coast route with scenic alternatives"
#             ),
#             "chicago_to_detroit": RouteGroundTruth(
#                 origin="Chicago",
#                 destination="Detroit",
#                 driving_distance_miles=280.0,
#                 driving_time_hours=4.5,
#                 flight_distance_miles=238.0,
#                 flight_time_hours=1.2,
#                 major_highways=["I-94"],
#                 available_transport_modes=["car", "train", "bus", "flight"],
#                 notes="Great Lakes corridor route"
#             ),
#             "san_francisco_to_los_angeles": RouteGroundTruth(
#                 origin="San Francisco",
#                 destination="Los Angeles",
#                 driving_distance_miles=380.0,
#                 driving_time_hours=6.0,
#                 flight_distance_miles=337.0,
#                 flight_time_hours=1.5,
#                 major_highways=["I-5", "US-101", "CA-1"],
#                 available_transport_modes=["car", "flight", "bus"],
#                 notes="Same as LA to SF, reverse direction"
#             ),
#             # Non-existent routes for hallucination testing
#             "atlantis_to_mars": RouteGroundTruth(
#                 origin="Atlantis",
#                 destination="Mars",
#                 driving_distance_miles=None,
#                 driving_time_hours=None,
#                 flight_distance_miles=None,
#                 flight_time_hours=None,
#                 major_highways=[],
#                 available_transport_modes=[],
#                 exists=False,
#                 notes="Fictional route for hallucination testing"
#             ),
#             "new_york_to_pluto": RouteGroundTruth(
#                 origin="New York",
#                 destination="Pluto",
#                 driving_distance_miles=None,
#                 driving_time_hours=None,
#                 flight_distance_miles=None,
#                 flight_time_hours=None,
#                 major_highways=[],
#                 available_transport_modes=[],
#                 exists=False,
#                 notes="Impossible route for hallucination testing"
#             ),
#         }
    
#     def extract_numerical_values(self, text: str) -> Dict[str, List[float]]:
#         """Extract numerical values from response text."""
#         values = {}
        
#         # Extract distances
#         distance_patterns = [
#             r'(\d+(?:\.\d+)?)\s*miles?',
#             r'(\d+(?:\.\d+)?)\s*km',
#             r'(\d+(?:\.\d+)?)\s*kilometers?'
#         ]
        
#         distances = []
#         for pattern in distance_patterns:
#             matches = re.findall(pattern, text.lower())
#             for match in matches:
#                 distance = float(match)
#                 # Convert km to miles if needed
#                 if 'km' in pattern:
#                     distance = distance * 0.621371
#                 distances.append(distance)
        
#         values['distances'] = distances
        
#         # Extract times
#         time_patterns = [
#             r'(\d+(?:\.\d+)?)\s*hours?',
#             r'(\d+)\s*hours?\s*(\d+)\s*minutes?',
#             r'(\d+)\s*minutes?'
#         ]
        
#         times = []
#         for pattern in time_patterns:
#             matches = re.findall(pattern, text.lower())
#             for match in matches:
#                 if isinstance(match, tuple):
#                     if len(match) == 2:  # hours and minutes
#                         time = float(match[0]) + float(match[1]) / 60
#                     else:
#                         time = float(match[0])
#                 else:
#                     time = float(match)
#                     if 'minutes' in pattern:
#                         time = time / 60  # Convert to hours
#                 times.append(time)
        
#         values['times'] = times
        
#         return values
    
#     def validate_route_existence(self, origin: str, destination: str, response: str) -> Dict[str, any]:
#         """Validate if a route should exist and if the agent handled it correctly."""
#         # Normalize route key
#         route_key = f"{origin.lower().replace(' ', '_')}_to_{destination.lower().replace(' ', '_')}"
        
#         if route_key in self.ground_truth_data:
#             ground_truth = self.ground_truth_data[route_key]
            
#             if not ground_truth.exists:
#                 # This is a fictional/impossible route
#                 # Check if agent admits it doesn't exist
#                 no_info_indicators = [
#                     "i don't have", "no information", "not available",
#                     "cannot provide", "unable to find", "does not exist",
#                     "no direct route", "not familiar with"
#                 ]
                
#                 admits_no_info = any(indicator in response.lower() for indicator in no_info_indicators)
                
#                 # Check if agent fabricates details
#                 extracted_values = self.extract_numerical_values(response)
#                 fabricates_details = len(extracted_values.get('distances', [])) > 0 or len(extracted_values.get('times', [])) > 0
                
#                 return {
#                     "route_exists": False,
#                     "agent_admits_no_info": admits_no_info,
#                     "agent_fabricates_details": fabricates_details,
#                     "hallucination_risk": fabricates_details and not admits_no_info,
#                     "appropriate_response": admits_no_info and not fabricates_details
#                 }
#             else:
#                 # This is a real route
#                 return {
#                     "route_exists": True,
#                     "agent_admits_no_info": False,
#                     "agent_fabricates_details": False,
#                     "hallucination_risk": False,
#                     "appropriate_response": True
#                 }
#         else:
#             # Route not in our database, assume it might exist
#             return {
#                 "route_exists": None,  # Unknown
#                 "agent_admits_no_info": False,
#                 "agent_fabricates_details": False,
#                 "hallucination_risk": False,
#                 "appropriate_response": True
#             }
    
#     def validate_factual_accuracy(self, origin: str, destination: str, response: str) -> Dict[str, any]:
#         """Validate the factual accuracy of route information."""
#         route_key = f"{origin.lower().replace(' ', '_')}_to_{destination.lower().replace(' ', '_')}"
        
#         if route_key not in self.ground_truth_data:
#             return {"validation_possible": False, "reason": "No ground truth data available"}
        
#         ground_truth = self.ground_truth_data[route_key]
        
#         if not ground_truth.exists:
#             return {"validation_possible": False, "reason": "Route does not exist"}
        
#         extracted_values = self.extract_numerical_values(response)
        
#         results = {
#             "validation_possible": True,
#             "distance_accuracy": None,
#             "time_accuracy": None,
#             "transport_modes_accuracy": None,
#             "overall_accuracy_score": 0.0
#         }
        
#         # Validate distances
#         if extracted_values.get('distances') and ground_truth.driving_distance_miles:
#             response_distances = extracted_values['distances']
#             expected_distance = ground_truth.driving_distance_miles
            
#             # Check if any reported distance is within 20% of expected
#             distance_accuracy = any(
#                 abs(dist - expected_distance) / expected_distance <= 0.2
#                 for dist in response_distances
#             )
#             results["distance_accuracy"] = distance_accuracy
        
#         # Validate times  
#         if extracted_values.get('times') and ground_truth.driving_time_hours:
#             response_times = extracted_values['times']
#             expected_time = ground_truth.driving_time_hours
            
#             # Check if any reported time is within 30% of expected
#             time_accuracy = any(
#                 abs(time - expected_time) / expected_time <= 0.3
#                 for time in response_times
#             )
#             results["time_accuracy"] = time_accuracy
        
#         # Validate transport modes
#         if ground_truth.available_transport_modes:
#             mentioned_modes = []
#             for mode in ground_truth.available_transport_modes:
#                 if mode.lower() in response.lower():
#                     mentioned_modes.append(mode)
            
#             results["transport_modes_accuracy"] = len(mentioned_modes) > 0
#             results["mentioned_transport_modes"] = mentioned_modes
        
#         # Calculate overall accuracy score
#         accuracy_scores = []
#         if results["distance_accuracy"] is not None:
#             accuracy_scores.append(1.0 if results["distance_accuracy"] else 0.0)
#         if results["time_accuracy"] is not None:
#             accuracy_scores.append(1.0 if results["time_accuracy"] else 0.0)
#         if results["transport_modes_accuracy"] is not None:
#             accuracy_scores.append(1.0 if results["transport_modes_accuracy"] else 0.0)
        
#         if accuracy_scores:
#             results["overall_accuracy_score"] = sum(accuracy_scores) / len(accuracy_scores)
        
#         return results
    
#     def evaluate_response_quality(self, query: str, response: str) -> Dict[str, any]:
#         """Comprehensive evaluation of response quality."""
#         # Extract origin and destination from query
#         origin_dest = self._extract_origin_destination(query)
        
#         if not origin_dest:
#             return {"error": "Could not extract origin/destination from query"}
        
#         origin, destination = origin_dest
        
#         # Validate route existence
#         existence_validation = self.validate_route_existence(origin, destination, response)
        
#         # Validate factual accuracy
#         factual_validation = self.validate_factual_accuracy(origin, destination, response)
        
#         # Overall assessment
#         assessment = {
#             "query": query,
#             "response": response,
#             "origin": origin,
#             "destination": destination,
#             "existence_validation": existence_validation,
#             "factual_validation": factual_validation,
#             "overall_quality_score": 0.0
#         }
        
#         # Calculate overall quality score
#         if existence_validation.get("appropriate_response"):
#             assessment["overall_quality_score"] += 0.5
        
#         if factual_validation.get("overall_accuracy_score"):
#             assessment["overall_quality_score"] += 0.5 * factual_validation["overall_accuracy_score"]
        
#         # Penalize hallucination heavily
#         if existence_validation.get("hallucination_risk"):
#             assessment["overall_quality_score"] = max(0, assessment["overall_quality_score"] - 0.8)
        
#         return assessment
    
#     def _extract_origin_destination(self, query: str) -> Optional[Tuple[str, str]]:
#         """Extract origin and destination from a query string."""
#         # Common patterns for route queries
#         patterns = [
#             r'(?:from|plan.*from)\s+(.+?)\s+to\s+(.+?)(?:\s|$)',
#             r'(.+?)\s+to\s+(.+?)(?:\s|$)',
#             r'(?:route|distance|travel).*?(?:from\s+)?(.+?)\s+(?:to|and)\s+(.+?)(?:\s|$)',
#             r'how far.*?(.+?)\s+(?:from|to)\s+(.+?)(?:\s|$)',
#         ]
        
#         for pattern in patterns:
#             match = re.search(pattern, query.lower())
#             if match:
#                 origin = match.group(1).strip()
#                 destination = match.group(2).strip()
                
#                 # Clean up common suffixes
#                 for suffix in ['?', '.', ',', 'route', 'travel']:
#                     origin = origin.rstrip(suffix).strip()
#                     destination = destination.rstrip(suffix).strip()
                
#                 return origin, destination
        
#         return None