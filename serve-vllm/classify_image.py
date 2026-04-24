#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests",
# ]
# ///

import base64
import requests
import json
import sys

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def main():
    # Use the first argument as the image path, or default to arable-land.jpg
    image_path = sys.argv[1] if len(sys.argv) > 1 else "arable-land.jpg"
    
    try:
        base64_image = encode_image(image_path)
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)

    prompt = "Classify the given image in one of the following classes. Classes: non-irrigated arable land, dump sites, peatbogs, pastures, coniferous forest, agro-forestry areas, broad-leaved forest, sparsely vegetated areas, industrial or commercial units, airports, bare rock, vineyards, water courses, rice fields, salt marshes, sport and leisure facilities, sea and ocean, water bodies, inland marshes, annual crops associated with permanent crops, mixed forest, beaches, dunes, sands, complex cultivation patterns, road and rail networks and associated land, land principally occupied by agriculture with significant areas of natural vegetation, moors and heathland, discontinuous urban fabric, continuous urban fabric, olive groves, intertidal flats, burnt areas, mineral extraction sites, permanently irrigated land, estuaries, green urban areas, construction sites, sclerophyllous vegetation, fruit trees and berry plantations, coastal lagoons, natural grassland, port areas, salines, transitional woodland/shrub. Answer in one short phrase."

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "model": "haruiz/gemmaearth",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 50
    }

    try:
        response = requests.post("http://localhost:8000/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            print("Classification:", data["choices"][0]["message"]["content"])
        else:
            print("Unexpected response:", json.dumps(data, indent=2))
            
    except requests.exceptions.RequestException as e:
        print("Error connecting to vLLM server:", e)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
