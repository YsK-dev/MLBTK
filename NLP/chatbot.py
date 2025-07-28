# %%

import openai

# Set your OpenAI API key
openai.api_key = 'your_api_key_here'

def chat_with_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']
# Example usage
if __name__ == "__main__":
    prompt = "What is the capital of France?"
    response = chat_with_gpt(prompt)
    print("Response from GPT-3.5:", response)

# %%
