"""
Intelligent Parameter Mapper for Flight Search Agent

This module provides smart parameter mapping to replace hardcoded if-else statements
with LLM-based parameter extraction and mapping.
"""

import inspect
import json
import logging
from typing import Any, Optional

import langchain_openai.chat_models
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class ParameterMapper:
    """Intelligent parameter mapper using LLM to handle parameter variations."""

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

            # Use LLM to map parameters intelligently
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
        """Use LLM to intelligently map parameters."""

        system_prompt = f"""
        You are a parameter mapper for flight booking tools. Map parameters to expected names.
        
        Tool: {tool_name}
        Expected parameters: {list(expected_params)}
        
        Common mappings:
        - source_airport: departure_airport, origin, from, origin_airport
        - destination_airport: arrival_airport, destination, to, dest
        - departure_date: date, travel_date, dep_date, when
        - return_date: return, back_date
        - passengers: pax, travelers, people, passenger_count
        - flight_class: class, cabin, service_class
        
        Rules:
        1. Map parameter names to expected names
        2. Extract values from text if needed
        3. Add reasonable defaults for missing required parameters
        4. Return valid JSON with mapped parameters
        5. For single user system, don't include customer_id
        """

        user_prompt = f"""
        Raw parameters: {json.dumps(raw_args, indent=2)}
        
        Map these to the expected parameters and return valid JSON:
        """

        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

            response = self.chat_model.invoke(messages)

            # Extract JSON from response
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()

            mapped_params = json.loads(content)
            return mapped_params

        except Exception:
            logger.exception("LLM parameter mapping failed")
            return raw_args

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
        """Extract airport codes from free text using LLM."""

        system_prompt = """
        Extract airport codes from text. Return JSON with source_airport and destination_airport.
        Look for IATA codes (3 letters) or city names. Return null if not found.
        """

        user_prompt = f"Extract airports from: {text}"

        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

            response = self.chat_model.invoke(messages)
            content = response.content.strip()

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()

            return json.loads(content)

        except Exception:
            logger.exception("Airport extraction failed")
            return {"source_airport": None, "destination_airport": None}

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

            elif tool_name == "search_flight_policies":
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

            if tool_name == "lookup_flight_info":
                # Try to extract airports from string
                airports = self.extract_airports_from_text(input_str)
                mapped.update(airports)

            elif tool_name == "save_flight_booking":
                # Try to extract flight info and add defaults
                airports = self.extract_airports_from_text(input_str)
                mapped.update(airports)
                mapped = self._add_tool_defaults(tool_name, mapped)

            elif tool_name == "search_flight_policies":
                # Use string as query
                mapped["query"] = input_str

            elif tool_name == "retrieve_flight_bookings":
                # No parameters needed for single user system
                pass

            # Filter to only valid parameters
            final_mapped = {k: v for k, v in mapped.items() if k in expected_params}
            return final_mapped

        except Exception:
            logger.exception("Error mapping string input for %s", tool_name)
            return {}
