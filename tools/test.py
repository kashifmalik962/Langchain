from dotenv import load_dotenv
import os
import requests

load_dotenv()

API_KEY = os.getenv("EXCHANGE_API_KEY", "7a4d583a08382ea1fb20f91c50451094")


def convert_currency(base_currency: str, target_currency: str) -> dict:
    url = f"https://v6.exchangerate-api.com/v6/c754eab14ffab33112e380ca/pair/{base_currency}/{target_currency}"
    response = requests.get(url).json()

    if "error" in response:
        return {"error": response["error"]["message"]}

    conversion_rate = response.get("conversion_rate")


    return {
        "base": base_currency,
        "target": target_currency,
        "conversion_rate": conversion_rate
    }

# Example usage:
print(convert_currency("USD", "INR"))  # Convert 10 USD → INR
