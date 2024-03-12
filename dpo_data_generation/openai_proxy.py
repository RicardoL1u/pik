
import openai
import logging
# logging.basicConfig(level=logging.DEBUG)
message = "Hello, I'm a chatbot created by OpenAI. How can I help you today?"
def gpt4(messages):
    openai.api_type = "azure"
    openai.api_key = "9a4745458eea4dff856ad8bf0db7733c"
    openai.api_base = "https://openaiexpcanada.openai.azure.com/"
    openai.api_version = "2023-05-15"

    response = openai.ChatCompletion.create(
        # engine="Moderation",
        engine="gpt-3.5-turbo",
        # model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )
    return response


import requests

def chat_completion_use_cache(prompt,
                              model='gpt-3.5-turbo',
                              temperature=0.7
                              ):
    url = 'http://103.238.162.37:9072/chat_completion/use_cache'
    headers = {'Content-Type': 'application/json'}
    data = {
        'user_token': 'eba1ef7a-2bbf-11ee-a59a-9cc2c4278efc',
        'model': model,
        'messages': [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        'temperature': temperature
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        return None


if __name__ == '__main__':
    response = gpt4([{"role": "user", "content": message}])
    print(chat_completion_use_cache(message))