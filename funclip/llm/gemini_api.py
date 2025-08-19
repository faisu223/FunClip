import os
import logging
import google.generativeai as genai

def call_gemini_model(api_key, model, prompt, system_prompt=None):
    logging.info(f"Calling Gemini model {model}")
    try:
        genai.configure(api_key=api_key)

        generation_config = {
            "temperature": 0.5,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
        }

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        model = genai.GenerativeModel(model_name=model,
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)

        # Gemini API doesn't have a dedicated system prompt. It's part of the conversation history.
        # We will prepend it to the user's prompt.
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        response = model.generate_content(full_prompt)

        if response.parts:
            return response.text
        else:
            # Handle cases where the response might be empty or blocked
            logging.warning("Gemini response was empty or blocked.")
            return "Error: The response from the Gemini API was empty or blocked. This might be due to safety settings."

    except Exception as e:
        logging.error(f"An error occurred while calling Gemini API: {e}")
        return f"An error occurred: {e}"
