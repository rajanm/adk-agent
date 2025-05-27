import os
import datetime
import requests
import pandas as pd
import json # Added as per instructions, though not strictly necessary for response.json() if it's always valid JSON
from zoneinfo import ZoneInfo
from google.adk.tools.tool_context import ToolContext

# Functions to be moved from agent.py

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city using OpenWeatherMap API.

    Returns:
        dict: A dictionary containing the weather information with a 'status' key ('success' or 'error')
              and a 'report' key with the weather details if successful, or an 'error_message' if an error occurred.
    """
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return {"status": "error", "error_message": "OpenWeatherMap API key not found. Please set the OPENWEATHERMAP_API_KEY environment variable."}

    api_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if response.status_code == 200:
            if data.get("cod") == 200:
                main_weather = data.get("weather", [{}])[0]
                description = main_weather.get("description", "N/A")
                temp = data.get("main", {}).get("temp", "N/A")
                humidity = data.get("main", {}).get("humidity", "N/A")
                wind_speed = data.get("wind", {}).get("speed", "N/A")
                report = (f"The weather in {city} is {description} with a temperature of {temp}°C, "
                          f"humidity of {humidity}%, and wind speed of {wind_speed} m/s.")
                return {"status": "success", "report": report}
            elif data.get("cod") == "404": 
                 return {"status": "error", "error_message": f"Weather information for '{city}' could not be found."}
            else: 
                return {"status": "error", "error_message": f"Failed to retrieve weather data for '{city}'. API message: {data.get('message', 'Unknown error')}"}

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return {"status": "error", "error_message": f"Weather information for '{city}' could not be found."}
        elif e.response.status_code == 401: 
             return {"status": "error", "error_message": "Failed to retrieve weather data. Invalid API key."}
        else:
            return {"status": "error", "error_message": f"Failed to retrieve weather data. HTTP error: {e.response.status_code}"}
    except requests.exceptions.RequestException:
        return {"status": "error", "error_message": "Failed to retrieve weather data. Please check your API key or network connection."}
    except Exception: 
        return {"status": "error", "error_message": "An unexpected error occurred while fetching weather data."}

def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Returns:
        dict: A dictionary containing the current time for a specified city information with a 'status' key ('success' or 'error') and a 'report' key with the current time details in a city if successful, or an 'error_message' if an error occurred.
    """
    # This function uses datetime and ZoneInfo, which are imported at the top of the file.
    city_timezones = {
        "new york": "America/New_York",
        "london": "Europe/London",
        "paris": "Europe/Paris",
        "tokyo": "Asia/Tokyo",
        "sydney": "Australia/Sydney",
        "los angeles": "America/Los_Angeles",
        "chicago": "America/Chicago",
        "toronto": "America/Toronto",
        "berlin": "Europe/Berlin",
        "moscow": "Europe/Moscow",
        "dubai": "Asia/Dubai",
        "singapore": "Asia/Singapore",
        "hong kong": "Asia/Hong_Kong",
        "shanghai": "Asia/Shanghai",
        "beijing": "Asia/Shanghai",
        "mumbai": "Asia/Kolkata",
        "sao paulo": "America/Sao_Paulo",
    }

    city_lower = city.lower()

    if city_lower in city_timezones:
        tz_identifier = city_timezones[city_lower]
    else:
        return {
            "status": "error",
            "error_message": f"Timezone information for '{city}' is not available in my current list. Please specify a major city or a standard timezone name.",
        }

    try:
        tz = ZoneInfo(tz_identifier)
        # Use datetime.datetime.now(tz) which is part of the datetime module imported at the top
        current_time_in_city = datetime.datetime.now(tz) 
    except Exception as e:
        return {"status": "error", "error_message": f"Failed to process timezone for {city}: {e}"}
    return {"status": "success",
            "report": f"""The current time in {city} is {current_time_in_city.strftime("%Y-%m-%d %H:%M:%S %Z%z")}"""}

def get_exchange_rate(base_currency: str, target_currency: str) -> float or None:
    """
    Retrieves the current exchange rate between two currencies using the exchangerate-api.com free service.
    Args:
        base_currency (str): The currency to convert from (e.g., 'USD').
        target_currency (str): The currency to convert to (e.g., 'EUR').
    Returns:
        float or None: The exchange rate as a float if successful, None on failure or if the rate is not found.
    """
    base_currency = base_currency.upper()
    target_currency = target_currency.upper()
    api_url = f"https://api.exchangerate-api.com/v4/latest/{base_currency}"
    print(api_url)

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        if 'rates' in data and target_currency in data['rates']:
            return float(data['rates'][target_currency])
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching exchange rate: {e}")
        return None

def get_historical_exchange_rate(base_currency: str, target_currency: str, date_string: str) -> dict or None:
    """
    Retrieves historical exchange rate data for a specific date from the Exchange Rate API.
    Args:
        base_currency (str): The three-letter ISO 4217 currency code for the base currency. (Unused in current body, but kept for signature consistency)
        target_currency (str): The currency to convert to (e.g., 'EUR'). (Unused in current body, but kept for signature consistency)
        date_string (str): The date for which to retrieve historical data, in "YYYY-MM-DD" format.
    Returns:
        dict: A dictionary containing the historical exchange rate data, or None if the request fails.
    """
    api_key = os.getenv("EXCHANGERATE_API_KEY")
    if not api_key: # Added check for API key similar to get_weather
        return {"status": "error", "error_message": "EXCHANGERATE_API_KEY not found."}
    
    try:
        # Validate date format before making API call
        datetime.datetime.strptime(date_string, '%Y-%m-%d')
    except ValueError:
        return {"status": "error", "error_message": f"Error: Invalid date format. Please use 'YYYY-MM-DD'. Received: {date_string}"}

    url = f"https://openexchangerates.org/api/historical/{date_string}.json?app_id={api_key}"
    print(url)

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        # Ensure the response structure is as expected before returning
        if 'rates' in data and 'base' in data:
            return data 
        else:
            # Added more specific error if structure is unexpected
            return {"status": "error", "error_message": "Unexpected response structure from historical exchange rate API."}
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        # Return a dict for consistency in error reporting
        return {"status": "error", "error_message": f"Error making API request: {e}"}
    except json.JSONDecodeError as e: # In case response is not valid JSON
        print(f"Error decoding JSON response: {e}. Response text was: {response.text if response else 'No response'}")
        return {"status": "error", "error_message": f"Error decoding JSON response: {e}"}


async def get_exchange_rate_trend(base_currency: str, target_currency: str, start_date: str, end_date: str, tool_context: ToolContext) -> dict:
    """
    Get the trend of exchange rates between two currencies for a given date range. 
    Args:
        base_currency (str): The currency to convert from (e.g., 'USD').
        target_currency (str): The currency to convert to (e.g., 'EUR').
        start_date (str): The start date for the trend in 'YYYY-MM-DD' format.
        end_date (str): The end date for the trend in 'YYYY-MM-DD' format.
        tool_context (ToolContext): The context for the tool. (Unused in current body, but kept for signature consistency)
    Returns:
        dict: A dictionary containing the exchange rate trend data or an error message.
    """
    try:
        dates = pd.date_range(start=start_date, end=end_date)
    except ValueError as e:
        return {"status": "error", "error_message": f"Invalid date format or range: {e}"}
        
    exchange_rates = []

    for date_obj in dates:
        date_str = date_obj.strftime('%Y-%m-%d')
        # get_historical_exchange_rate now returns a dict, so need to adjust how we access rates
        historical_data_response = get_historical_exchange_rate(base_currency, target_currency, date_str)
        
        # Check if the response was successful and data is present
        if isinstance(historical_data_response, dict) and historical_data_response.get("status") == "success": # Assuming success status for direct data
            data = historical_data_response # If it's a direct data dict
        elif isinstance(historical_data_response, dict) and "rates" in historical_data_response and "base" in historical_data_response: # Direct data dict from API
            data = historical_data_response
        elif isinstance(historical_data_response, dict) and historical_data_response.get("status") == "error":
            print(f"Error fetching historical rate for {date_str}: {historical_data_response.get('error_message')}")
            exchange_rates.append(None)
            continue # Skip to next date
        else: # Unexpected response or error not in dict format
            print(f"Unexpected response or error fetching historical rate for {date_str}: {historical_data_response}")
            exchange_rates.append(None)
            continue

        if target_currency.upper() in data.get('rates', {}):
            exchange_rates.append(data['rates'][target_currency.upper()])
        else:
            exchange_rates.append(None)

    df = pd.DataFrame({'Date': dates, 'Exchange Rate': exchange_rates})
    df.dropna(subset=['Exchange Rate'], inplace=True)

    if df.empty:
        return {"status": "success", "report": "Could not retrieve sufficient exchange rate data for the specified date range to show a trend."}
    else:
        # Returning a structured success response
        return {"status": "success", "report": f"Exchange rate trend data is \n {df.to_string()}"}
