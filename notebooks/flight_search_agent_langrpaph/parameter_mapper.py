"""
Intelligent Parameter Mapper for Flight Search Agent

This module provides LLM-based parameter mapping with guardrail-safe prompts
and robust error handling with fallback mechanisms.
"""

import inspect
import json
import logging
import re
from typing import Any, Optional

import langchain_openai.chat_models
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class ParameterMapper:
    """Intelligent parameter mapper using LLM with guardrail-safe prompts."""

    def __init__(self, chat_model: langchain_openai.chat_models.ChatOpenAI):
        self.chat_model = chat_model

        # Common parameter synonyms for flight tools
        self.parameter_synonyms = {
            "source_airport": ["departure_airport", "origin", "from", "origin_airport", "start"],
            "destination_airport": ["arrival_airport", "destination", "to", "dest", "end"],
            "departure_date": ["date", "travel_date", "dep_date", "when"],
            "return_date": ["return", "return_date", "back_date"],
            "passengers": ["pax", "travelers", "people", "passenger_count"],
            "flight_class": ["class", "cabin", "service_class", "ticket_class"],
        }

    def get_function_parameters(self, func) -> set[str]:
        """Extract parameter names from function signature."""
        try:
            sig = inspect.signature(func)
            return set(sig.parameters.keys())
        except Exception as e:
            logger.exception("Error getting function parameters")
            return set()

    def map_parameters_smart(
        self, tool_name: str, raw_args: dict[str, Any], func
    ) -> dict[str, Any]:
        """
        Smart parameter mapping using LLM to understand parameter intent.

        Args:
            tool_name: Name of the tool being called
            raw_args: Raw arguments from LLM
            func: Function object to get expected parameters

        Returns:
            Mapped parameters ready for function call
        """
        try:
            # Get expected parameters from function signature
            expected_params = self.get_function_parameters(func)

            # If parameters already match, return as-is
            if set(raw_args.keys()).issubset(expected_params):
                return raw_args

            # Use LLM to map parameters intelligently with guardrail-safe prompts
            mapped_params = self._llm_parameter_mapping(tool_name, raw_args, expected_params)

            # Add tool-specific defaults
            mapped_params = self._add_tool_defaults(tool_name, mapped_params)

            # Filter to only valid parameters
            final_params = {k: v for k, v in mapped_params.items() if k in expected_params}

            return final_params

        except Exception:
            logger.exception("Error in smart parameter mapping")
            # Fallback to synonym-based mapping
            return self._fallback_synonym_mapping(raw_args, expected_params)

    def _llm_parameter_mapping(
        self, tool_name: str, raw_args: dict[str, Any], expected_params: set[str]
    ) -> dict[str, Any]:
        """Use LLM to map parameters with minimal, guardrail-safe prompts."""

        # Ultra-minimal prompt to avoid guardrail violations
        system_prompt = f"""Map parameters to: {list(expected_params)}
Output only valid JSON."""

        user_prompt = f"""Input: {json.dumps(raw_args)}
Tool: {tool_name}"""

        try:
            # Primary LLM call with minimal prompt
            mapped_params = self._safe_llm_call(system_prompt, user_prompt)
            if mapped_params:
                return mapped_params

        except Exception as e:
            logger.warning(f"LLM parameter mapping failed: {e}")

        # Fallback to synonym-based mapping
        return self._fallback_synonym_mapping(raw_args, expected_params)

    def _safe_llm_call(self, system_prompt: str, user_prompt: str) -> Optional[dict]:
        """Make safe LLM call with robust JSON parsing."""
        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

            response = self.chat_model.invoke(messages)
            content = response.content.strip()

            # Try multiple JSON parsing methods
            return self._parse_json_response(content)

        except Exception as e:
            logger.warning(f"Safe LLM call failed: {e}")
            return None

    def _parse_json_response(self, content: str) -> Optional[dict]:
        """Parse JSON response with multiple fallback methods."""

        logger.debug(f"Parsing JSON response: '{content[:200]}...' (length: {len(content)})")

        # Method 1: Direct JSON parsing
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parsing failed: {e}")
            pass

        # Method 2: Extract from code blocks
        try:
            if "```json" in content:
                json_content = content.split("```json")[1].split("```")[0].strip()
                return json.loads(json_content)
            elif "```" in content:
                json_content = content.split("```")[1].strip()
                return json.loads(json_content)
        except (json.JSONDecodeError, IndexError):
            pass

        # Method 3: Find JSON-like patterns
        try:
            # Look for balanced {.*} patterns
            json_pattern = r"\{[^{}]*\}"
            matches = re.findall(json_pattern, content)
            if matches:
                return json.loads(matches[0])
        except (json.JSONDecodeError, IndexError):
            pass

        # Method 3b: More complex JSON pattern with nested braces
        try:
            # Find JSON with proper brace matching
            start = content.find("{")
            if start != -1:
                brace_count = 0
                for i, char in enumerate(content[start:], start):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_content = content[start : i + 1]
                            return json.loads(json_content)
        except (json.JSONDecodeError, ValueError):
            pass

        # Method 4: Handle truncated JSON (common at character 168)
        try:
            # If content appears truncated, try to find the last complete JSON object
            if len(content) >= 160:  # Near the problematic character range
                # Look for the last complete brace pair
                last_close_brace = content.rfind("}")
                if last_close_brace > 0:
                    # Find the matching opening brace
                    brace_count = 0
                    for i in range(last_close_brace, -1, -1):
                        if content[i] == "}":
                            brace_count += 1
                        elif content[i] == "{":
                            brace_count -= 1
                            if brace_count == 0:
                                truncated_json = content[i : last_close_brace + 1]
                                logger.debug(f"Trying truncated JSON: '{truncated_json}'")
                                return json.loads(truncated_json)
        except (json.JSONDecodeError, ValueError):
            pass

        # Method 5: Clean and retry
        try:
            # Remove extra whitespace and try again
            cleaned = re.sub(r"\s+", " ", content.strip())
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Method 6: Emergency fallback - try to construct basic JSON from visible patterns
        try:
            # Look for key-value pairs and construct basic JSON
            if "source_airport" in content and "destination_airport" in content:
                # Try to extract airport codes with regex
                source_match = re.search(r'"source_airport"\s*:\s*"([^"]*)"', content)
                dest_match = re.search(r'"destination_airport"\s*:\s*"([^"]*)"', content)

                if source_match and dest_match:
                    fallback_json = {
                        "source_airport": source_match.group(1),
                        "destination_airport": dest_match.group(1),
                    }
                    logger.debug(f"Emergency fallback JSON: {fallback_json}")
                    return fallback_json
        except Exception:
            pass

        logger.warning(
            f"Failed to parse JSON response (length: {len(content)}): {content[:200]}..."
        )
        return None

    def _fallback_synonym_mapping(
        self, raw_args: dict[str, Any], expected_params: set[str]
    ) -> dict[str, Any]:
        """Fallback to synonym-based mapping if LLM fails."""
        mapped = {}

        for expected_param in expected_params:
            # Check if parameter exists directly
            if expected_param in raw_args:
                mapped[expected_param] = raw_args[expected_param]
                continue

            # Check synonyms
            synonyms = self.parameter_synonyms.get(expected_param, [])
            for synonym in synonyms:
                if synonym in raw_args:
                    mapped[expected_param] = raw_args[synonym]
                    break

        return mapped

    def _add_tool_defaults(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Add tool-specific defaults for single user system."""

        if tool_name == "save_flight_booking":
            # Default departure date if missing
            if "departure_date" not in params:
                params["departure_date"] = "tomorrow"

            # Default passengers if missing
            if "passengers" not in params:
                params["passengers"] = 1

            # Default flight class if missing
            if "flight_class" not in params:
                params["flight_class"] = "economy"

        elif tool_name == "retrieve_flight_bookings":
            # No customer_id needed for single user system
            pass

        return params

    def extract_airports_from_text(self, text: str) -> dict[str, Optional[str]]:
        """Extract airport codes from text using guardrail-safe LLM calls."""

        # Ultra-minimal prompt for location code extraction
        system_prompt = (
            "Extract location codes. Return JSON with source_airport and destination_airport."
        )
        user_prompt = f"Text: {text}"

        try:
            # Try LLM extraction with minimal prompt
            result = self._safe_llm_call(system_prompt, user_prompt)
            if result and isinstance(result, dict):
                # Validate and clean the result
                cleaned_result = {}
                for key in ["source_airport", "destination_airport"]:
                    value = result.get(key)
                    if value and isinstance(value, str) and len(value) == 3:
                        cleaned_result[key] = value.upper()
                    else:
                        cleaned_result[key] = None
                return cleaned_result

        except Exception as e:
            logger.warning(f"LLM airport extraction failed: {e}")

        # Fallback to pattern matching
        return self._fallback_airport_extraction(text)

    def _fallback_airport_extraction(self, text: str) -> dict[str, Optional[str]]:
        """Fallback airport extraction using regex patterns."""
        result = {"source_airport": None, "destination_airport": None}

        # Find 3-letter codes
        airport_codes = re.findall(r"\b[A-Z]{3}\b", text.upper())

        if len(airport_codes) >= 2:
            result["source_airport"] = airport_codes[0]
            result["destination_airport"] = airport_codes[1]
        elif len(airport_codes) == 1:
            # Try to determine if it's source or destination
            if any(word in text.lower() for word in ["from", "origin"]):
                result["source_airport"] = airport_codes[0]
            elif any(word in text.lower() for word in ["to", "destination"]):
                result["destination_airport"] = airport_codes[0]

        return result

    def map_positional_args(self, tool_name: str, args: tuple, func) -> dict[str, Any]:
        """Map positional arguments from ReAct agent to function parameters."""
        try:
            expected_params = self.get_function_parameters(func)

            # Map positional args to expected parameter names
            mapped = {}

            if tool_name == "lookup_flight_info":
                # Expects source_airport, destination_airport
                if len(args) >= 2:
                    mapped["source_airport"] = args[0]
                    mapped["destination_airport"] = args[1]
                elif len(args) == 1:
                    # Try to extract both from single string
                    airports = self.extract_airports_from_text(args[0])
                    mapped.update(airports)

            elif tool_name == "save_flight_booking":
                # Map based on position and add defaults
                if len(args) >= 2:
                    mapped["source_airport"] = args[0]
                    mapped["destination_airport"] = args[1]
                if len(args) >= 3:
                    mapped["departure_date"] = args[2]

                # Add defaults for missing parameters
                mapped = self._add_tool_defaults(tool_name, mapped)

            elif tool_name == "search_airline_reviews":
                # Single query parameter
                if len(args) >= 1:
                    mapped["query"] = " ".join(args)

            # Filter to only valid parameters
            final_mapped = {k: v for k, v in mapped.items() if k in expected_params}
            return final_mapped

        except Exception:
            logger.exception("Error mapping positional args for %s", tool_name)
            return {}

    def map_string_input(self, tool_name: str, input_str: str, func) -> dict[str, Any]:
        """Map single string input to function parameters."""
        try:
            expected_params = self.get_function_parameters(func)
            mapped = {}

            logger.debug(
                f"Parameter mapping for {tool_name}: input='{input_str}', expected={expected_params}"
            )

            # Enhanced cleaning for ReAct parsing artifacts
            clean_input = self._clean_react_artifacts(input_str)

            # Debug logging for edge cases
            if input_str != clean_input:
                logger.debug(
                    f"Cleaned ReAct artifacts for {tool_name}: '{input_str}' -> '{clean_input}'"
                )

            if tool_name == "lookup_flight_info":
                # Handle comma-separated format: "JFK,LAX,tomorrow"
                parts = [part.strip() for part in clean_input.split(",")]

                if len(parts) >= 2:
                    # Direct airport codes
                    mapped["source_airport"] = parts[0].upper()
                    mapped["destination_airport"] = parts[1].upper()
                else:
                    # Try to extract airports from string
                    airports = self.extract_airports_from_text(clean_input)
                    mapped.update(airports)

            elif tool_name == "save_flight_booking":
                # Handle comma-separated format: "SOURCE,DEST,DATE,PASSENGERS,CLASS"
                parts = [part.strip() for part in clean_input.split(",")]

                if len(parts) >= 2:
                    mapped["source_airport"] = parts[0].upper()
                    mapped["destination_airport"] = parts[1].upper()

                    # Handle positional parameters
                    if len(parts) >= 3:
                        mapped["departure_date"] = parts[2]
                    if len(parts) >= 4:
                        # Try to parse passengers as integer
                        try:
                            mapped["passengers"] = int(parts[3])
                        except ValueError:
                            # If not integer, check if it contains a number
                            numbers = re.findall(r"\d+", parts[3])
                            if numbers:
                                mapped["passengers"] = int(numbers[0])
                    if len(parts) >= 5:
                        # Enhanced cleaning for flight class with ReAct artifacts
                        flight_class = self._clean_flight_class(parts[4])
                        if flight_class:
                            mapped["flight_class"] = flight_class
                else:
                    # Try to extract flight info from text
                    airports = self.extract_airports_from_text(clean_input)
                    mapped.update(airports)

                # Add defaults
                mapped = self._add_tool_defaults(tool_name, mapped)

            elif tool_name == "search_airline_reviews":
                # Use cleaned string as query
                mapped["query"] = clean_input

            elif tool_name == "retrieve_flight_bookings":
                # No parameters needed for single user system
                # Handle "None" input from ReAct agent
                if clean_input.lower() in ["none", "null", ""]:
                    pass  # Return empty dict - no parameters needed
                else:
                    pass  # Also no parameters needed for any other input

            # Filter to only valid parameters
            final_mapped = {
                k: v for k, v in mapped.items() if k in expected_params and v is not None
            }

            logger.debug(f"Parameter mapping result for {tool_name}: {final_mapped}")

            if not final_mapped and tool_name != "retrieve_flight_bookings":
                logger.warning(
                    f"No valid parameters mapped for {tool_name} with input '{input_str}'"
                )

            return final_mapped

        except Exception:
            logger.exception("Error mapping string input for %s", tool_name)
            return {}

    def _clean_react_artifacts(self, input_str: str) -> str:
        """Clean ReAct parsing artifacts from input string."""
        if not input_str:
            return ""

        # Remove common ReAct artifacts
        clean_str = input_str

        # Enhanced cleaning for ReAct artifacts - handle multi-line patterns
        # Remove trailing quotes and observation artifacts (case insensitive)
        clean_str = re.sub(
            r'["\']?\s*\n?\s*observation.*$', "", clean_str, flags=re.IGNORECASE | re.DOTALL
        )

        # Remove newlines followed by any text (common ReAct artifact)
        clean_str = re.sub(r"\n.*$", "", clean_str, flags=re.DOTALL)

        # Remove leading/trailing quotes and whitespace
        clean_str = clean_str.strip().strip("\"'").strip()

        # Handle specific ReAct patterns like "None\nObservation"
        if clean_str.lower().startswith("none"):
            # Extract just "none" if it starts with none followed by artifacts
            clean_str = "none"

        return clean_str

    def _clean_flight_class(self, flight_class_str: str) -> str:
        """Clean flight class parameter with enhanced artifact removal."""
        if not flight_class_str:
            return ""

        # Start with basic cleaning
        cleaned = flight_class_str.strip().lower()

        # Remove quotes
        cleaned = cleaned.strip("\"'")

        # Remove observation artifacts (case insensitive)
        cleaned = re.sub(r'\s*["\']?\s*observation.*$', "", cleaned, flags=re.IGNORECASE)

        # Remove newlines and everything after
        cleaned = re.sub(r"\n.*$", "", cleaned)

        # Remove any remaining special characters at the end
        cleaned = re.sub(r"[^a-zA-Z]+$", "", cleaned)

        # Final trim
        cleaned = cleaned.strip()

        # Validate against known flight classes
        valid_classes = ["economy", "business", "first", "premium"]
        if cleaned in valid_classes:
            return cleaned

        # If not exact match, try to find closest match
        for valid_class in valid_classes:
            if valid_class in cleaned:
                return valid_class

        return cleaned  # Return as-is if no match found
