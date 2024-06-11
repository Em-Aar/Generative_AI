# Introduction

## Text-Generation Models

Trained to understand natural language, code, and images

- Generative Pre-trained Transformers (GPTs)
- Large Language Models (LLMs)

The input provided to these models is called prompt. These models provide output in the form of text (natural language).

### As a developer, how we can benefit from these models

Developers can build various applications to

- Draft documents
- Write computer code
- Answer questions about a knowledge base
- Analyze texts
- Give software a natural language interface
- Tutor in a range of subjects
- Translate languages
- Simulate characters for games

### Procedure to use LLMs in our application code

1. Create an Account (Signup on OpenAI or Gemini)
2. Familiarize oourself with the API/SDK: Each provider offers documentation and tutorials on how to interact with their LLMs using their APIs or SDKs. Explore these resources to understand the functionalities and available models.
3. Choose an LLM Model: Select an appropriate LLM model based on our application's needs. Consider factors like the model's size, capabilities, and specific tasks we want to perform (e.g., text generation, translation, code completion).
4. Obtain Credentials: Depending on the provider, we might need API keys, access tokens, or other credentials to authenticate our application and access the LLM models.
5. Integrate the LLM into our Code: Use the chosen provider's API or SDK to send requests to the LLM model from within our application code.
6. Handle Responses: Process the LLM's responses appropriately within our application logic.
7. Error Handling and Monitoring: Implement proper error handling mechanisms to address potential issues with LLM requests or unexpected responses.

### Important Considerations

1. Security
2. Cost
3. Responsible Usage

# References
1. [Api Reference](https://platform.openai.com/docs/api-reference/chat/create)
2. [Cookbook](https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models)

# Chat Completion
1. Create a new package with `poetry new chat_completion`
2. Create a new file named `main.ipynb` in the new package
3. Install packages 
    - openai: `poetry add openai`
    - ipykernel: `poetry add ipykernel` **NOTE: ipykernel is installed because we are using jupyter notebook**
    - python-dotenv: `poetry add python-dotenv`

## What is Chat Completion by OpenAI?
Chat completion by OpenAI refers to the capability of its language models, like GPT-4, to generate coherent and contextually relevant responses in a conversational format. It allows for interactive dialogues where the model can understand and reply to user prompts, maintaining context over multiple exchanges. This technology is used in various applications, including customer service, virtual assistants, and interactive chatbots, enabling them to provide human-like interactions and responses.

Chat completion in OpenAI's language models inherently uses Natural Language Processing (NLP) techniques to understand and generate human-like text. The model processes input text, understands the context and meaning, and generates appropriate responses based on its training data. By default, it performs various NLP tasks such as language understanding, context management, and coherent text generation, enabling it to engage in meaningful and relevant conversations. This makes it suitable for applications requiring sophisticated text interactions and natural language understanding. However, we can get response format in different formats (JSON, HTML, etc) by instructing the model.

## Applications of Openai's chat completion

Chat completion by OpenAI can be utilized in a wide range of applications, enhancing user interactions with technology. Here are some key uses and applications:

1. **Customer Support**: Automating responses to common customer inquiries, providing quick and accurate information, and handling large volumes of requests efficiently.

2. **Virtual Assistants**: Enabling personal assistants to understand and respond to user queries, manage schedules, set reminders, and perform tasks through natural language conversation.

3. **Education and Tutoring**: Assisting students with their studies by answering questions, explaining concepts, and providing resources, thereby offering personalized learning experiences.

4. **Content Creation**: Aiding writers, marketers, and content creators by generating ideas, drafting text, and refining content, thus improving productivity and creativity.

5. **Entertainment**: Developing interactive storylines and characters in video games, creating engaging chatbot experiences, and facilitating dynamic storytelling.

6. **Healthcare**: Offering preliminary medical advice, scheduling appointments, and providing information on medications and treatments, thus supporting healthcare professionals and patients.

These applications showcase how chat completion can streamline processes, enhance user engagement, and improve the efficiency and effectiveness of various services.

## Chat Completion Deep Dive

### What is `role` in OpenAI's chat completion?

```python
{"role": "system", "content": "You are a helpful assistant."},
{"role": "user", "content": "Who won the world series in 2020?"},
{"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
{"role": "user", "content": "Where was it played?"}
```

**system:** to modify the personality of the assistant (Model) or provide specific instructions about how it should behave throughout the conversation.

**user:** prompt/input from user.

**assistant:** response from AI Model.

#### Why we are providing repponse back to AI Model?

```python
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
```

The models have no memory of past requests, so we have to provide the context by providing previous responses so that the models can response to current prompt as per the previous context. However, the assistnat message can also be written by us to give examples of desired behavior.

### finish_reason in Response

**stop:** API returned complete message, or a message terminated by one of the stop sequences provided via the stop parameter

**length:** Incomplete model output due to max_tokens parameter or token limit

**function_call:** The model decided to call a function

**content_filter:** Omitted content due to a flag from our content filters

**null:** API response still in progress or incomplete

### Some Other Parameters with usage and explanation

REFERENCES:

- [Api Reference](https://platform.openai.com/docs/api-reference/chat/create)
- [Cookbook](https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models)

#### 1. `frequency_penalty`

**Value**
number or null, Defaults to 0

**Usage:**

```python
response = openai.Completion.create(
model="gpt-3.5-turbo",
prompt="Once upon a time",
frequency_penalty=0.5
)

```

**Range:**
Typically, the frequency penalty can range from 0 to 2. A value of 0 means no penalty is applied, while higher values apply a stronger penalty on repeated tokens.

**Why It Is Used:**

- Reduce Repetition: In long texts, language models might tend to repeat words or phrases. The frequency penalty helps to reduce this repetition, making the generated text more diverse and natural.
- Improve Coherence: By penalizing frequent tokens, the model is encouraged to use a wider vocabulary, which can improve the coherence and readability of the text.
- Enhance Creativity: For creative writing tasks, a higher frequency penalty can help generate more unique and varied responses, which can be useful for tasks like storytelling or brainstorming.

**Importance**

- Context-Specific: The importance of the frequency penalty depends on the specific application. For tasks where repetition is undesirable, such as essay writing or dialogue generation, it is very useful.
- Balancing Act: Setting the right frequency penalty requires balancing. Too high a penalty might result in less coherent text, while too low a penalty might lead to excessive repetition.
  **Benefits**
- Natural Language Generation: Produces more human-like text by avoiding repetitive patterns.
- Improved User Experience: In applications like chatbots or virtual assistants, it helps in maintaining engaging and varied conversations.
- Enhanced Output Quality: For content creation, marketing copy, and other professional writing tasks, it improves the overall quality of the generated content.

**Example**

a. With lower value of **frequency_penalty**

`Once upon a time, there was a brave knight. The knight went to the castle. The knight fought the dragon. The knight saved the princess.`

b. With higher value of **frequency_penalty**

`Once upon a time, there was a courageous knight. He journeyed to the majestic castle. In a fierce battle, he vanquished the mighty dragon and heroically rescued the princess.`


#### 2. `logit_bias`

**Value**
map, Defaults to null

**Usage:**

```python
# Token ID for "spoiler" is hypothetically 12345

logit_bias = {12345: -100}  # Strongly discourage the token "spoiler"

response = openai.Completion.create(
    model="gpt-3.5-turbo",
    prompt="Tell me about the plot of the movie.",
    logit_bias=logit_bias
)

```

**Range:**
-100 to 100: -100 means strongly discourages the use of specific token(word) in the response. 100 means encourages the use of specific token i.e. 'spoiler' in this example.

**Explanation**

The logit_bias parameter allows you to adjust the likelihood of specific tokens appearing in the model's output by directly modifying the logits before the model samples tokens to generate text. This can be useful for fine-tuning the behavior of the model for specific tasks or to encourage/discourage the use of certain words or phrases.

**Use Cases**

- Content Filtering: Preventing the model from generating inappropriate or undesired content by penalizing specific tokens.
- Task-Specific Tuning: Adjusting token probabilities to better align the model's output with specific tasks or domain-specific language.
- Creative Control: Steering the model's creative outputs by encouraging or discouraging certain words or phrases.

#### 3. `max_tokens`

**Value**
integer or null

**Usage:**

```python
response = openai.Completion.create(
model = "gpt-3.5-turbo",
prompt = "Once upon a time",
max_token = 50,
)

```

**Explanation:**

- Sets the maximum number of generated tokens in chat completion.

#### 4. `logprobs`

**Value**
boolean or null, defaults to False

**Usage**:

- **When `logprobs` is True**: The response includes the log probabilities for each token in the generated text. This provides detailed information about the model's decision-making process.
- **When `logprobs` is False**: The response does not include this information, resulting in a simpler output focused solely on the generated text.

**Key Concepts**

The `logprobs` parameter in the OpenAI API allows you to get the log probabilities of the tokens generated by the model. This can be useful for understanding the model's confidence in its predictions, analyzing the text generation process, or for advanced applications such as modifying the output based on the probabilities.

**Log Probabilities**:

- **Definition**: Log probabilities are the logarithm of the probabilities assigned to each token by the model. They provide a measure of the model's confidence in selecting a particular token.
- **Range**: Log probabilities are typically negative values (since probabilities are between 0 and 1, and the log of a number between 0 and 1 is negative). Higher (less negative) log probabilities indicate higher confidence.

##### Example of Usage

##### Requesting Log Probabilities

When we set `logprobs` to `true`, the API returns additional information in the response, including the log probabilities of each token. Here's an example:

```python

# Define a function to get a completion with log probabilities
def get_completion_with_logprobs(prompt):
    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=50,
        logprobs=5  # Return log probabilities for the top 5 tokens at each step
    )
    return response

# Example prompt
prompt = "What is the capital of France?"

# Get the response
response = get_completion_with_logprobs(prompt)

# Print the response and log probabilities
for choice in response.choices:
    print(f"Text: {choice.text}")
    if 'logprobs' in choice:
        print("Log Probabilities:", choice.logprobs)

```

##### Interpreting the Response

The response object will include detailed information about the log probabilities for each token. Here's what to look for:

1. **Generated Text**:

   - The main content of the generated response.

2. **Log Probabilities (`logprobs`)**:
   - **Tokens**: The tokens generated by the model.
   - **Token Logprobs**: The log probabilities of each token.
   - **Top Logprobs**: The log probabilities for the top N tokens considered at each step (if specified).

##### Example Response

Here is an example of how the response might look:

```json
{
  "id": "cmpl-xxxxxxxxxxxxxxxxxxxxxxx",
  "object": "text_completion",
  "created": 1614807350,
  "model": "gpt-3.5-turbo",
  "choices": [
    {
      "text": "\n\nThe capital of France is Paris.",
      "index": 0,
      "logprobs": {
        "tokens": [
          "\n\n",
          "The",
          " capital",
          " of",
          " France",
          " is",
          " Paris",
          "."
        ],
        "token_logprobs": [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
        "top_logprobs": [
          { "\n\n": -0.1, " ": -2.3, "": -5.1 },
          { "The": -0.2, "A": -1.9, "It": -2.5 },
          { " capital": -0.3, " city": -1.4, " name": -3.2 },
          { " of": -0.4, " in": -1.7, " for": -3.0 },
          { " France": -0.5, " Paris": -1.2, " Europe": -3.5 },
          { " is": -0.6, " was": -1.8, " has": -3.3 },
          { " Paris": -0.7, " Lyon": -2.1, " France": -3.7 },
          { ".": -0.8, "!": -2.0, "?": -3.6 }
        ],
        "text_offset": [0, 1, 2, 3, 4, 5, 6, 7]
      },
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 7,
    "completion_tokens": 8,
    "total_tokens": 15
  }
}
```

##### Components of the Logprobs Response

- **tokens**: The individual tokens that make up the generated text.
- **token_logprobs**: The log probabilities for each of these tokens.
- **top_logprobs**: The log probabilities for the top tokens considered at each position.
- **text_offset**: The character offset for each token in the original text.

##### Why Use `logprobs`?

1. **Understanding Model Decisions**: Provides insights into why the model chose specific tokens, helping in debugging or improving prompts.
2. **Advanced Control**: Enables fine-tuning and adjusting responses by understanding the likelihood of different tokens.
3. **Probability Analysis**: Useful in applications where understanding the confidence of the model's output is critical, such as in scientific or data analysis contexts.

##### Conclusion

The `logprobs` parameter is a powerful tool for gaining deeper insights into the language model's behavior. By setting it to `true`, we can access detailed log probabilities for the tokens in the generated output, allowing for a more granular analysis and better control over the text generation process. This can be particularly valuable for debugging, fine-tuning, and applications requiring a high level of precision and understanding of the model's decision-making process.

#### 5. ` top_logprobs`

**Description:**
An integer between 0 and 20 specifying the number of most likely tokens to return at each token position, each with an associated log probability. logprobs must be set to true if this parameter is used.
This can be particularly useful for understanding the model's decision-making process and for gaining insights into the various options the model considered at each step of text generation.

**Usage:**

- When top_logprobs is Set: The response will include detailed information about the top N tokens considered by the model at each step, and their respective log probabilities.
- When top_logprobs is Not Set or is Null: This information will not be included in the response.

**Key Concepts:**

- The top_logprobs parameter returns a list of the most likely tokens (up to a specified number) along with their log probabilities for each position in the generated text.
- Range: We can specify an integer between 0 and 20. This determines how many of the top tokens and their log probabilities will be included in the response.

**Example**

```python
import openai

# Initialize your API key
openai.api_key = 'your-api-key-here'

# Define a function to get a completion with top log probabilities
def get_completion_with_top_logprobs(prompt, top_logprobs):
    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=50,
        top_logprobs=top_logprobs  # Return log probabilities for the top N tokens at each step
    )
    return response

# Example prompt
prompt = "What is the capital of France?"

# Get the response with top 5 log probabilities
response = get_completion_with_top_logprobs(prompt, top_logprobs=5)

# Print the response and top log probabilities
for choice in response.choices:
    print(f"Text: {choice.text}")
    if 'logprobs' in choice:
        print("Top Log Probabilities:", choice.logprobs)


```

**Interpreting the Response**

```python
{
  "id": "cmpl-xxxxxxxxxxxxxxxxxxxxxxx",
  "object": "text_completion",
  "created": 1614807350,
  "model": "gpt-3.5-turbo",
  "choices": [
    {
      "text": "\n\nThe capital of France is Paris.",
      "index": 0,
      "logprobs": {
        "tokens": ["\n\n", "The", " capital", " of", " France", " is", " Paris", "."],
        "token_logprobs": [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
        "top_logprobs": [
          {"\n\n": -0.1, " ": -2.3, "": -5.1},
          {"The": -0.2, "A": -1.9, "It": -2.5},
          {" capital": -0.3, " city": -1.4, " name": -3.2},
          {" of": -0.4, " in": -1.7, " for": -3.0},
          {" France": -0.5, " Paris": -1.2, " Europe": -3.5},
          {" is": -0.6, " was": -1.8, " has": -3.3},
          {" Paris": -0.7, " Lyon": -2.1, " France": -3.7},
          {".": -0.8, "!": -2.0, "?": -3.6}
        ],
        "text_offset": [0, 1, 2, 3, 4, 5, 6, 7]
      },
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 7,
    "completion_tokens": 8,
    "total_tokens": 15
  }
}


```

##### Components of the Top Logprobs Response

_tokens:_ The individual tokens that make up the generated text.
_token_logprobs:_ The log probabilities for each of these tokens.
_top_logprobs:_ The log probabilities for the top tokens considered at each position.
For example, at the first position, the model considered "\n\n" with a log probability of -0.1, " " with -2.3, and "" with -5.1.
_text_offset:_ The character offset for each token in the original text.
#####Why Use top_logprobs?

- Understanding Model Decisions: By seeing the top tokens and their log probabilities, you can gain insights into why the model chose a specific token over others.
- Fine-Tuning: This information can help in adjusting prompts or settings to steer the model's output more effectively.
- Advanced Analysis: For research or detailed analysis, knowing the model's alternative choices at each step can be invaluable.

##### Conclusion

The top_logprobs parameter is a valuable tool for understanding the inner workings of the model's decision-making process. By specifying an integer value between 0 and 20, you can retrieve the top N tokens and their log probabilities at each step of text generation. This provides a detailed view of the model's confidence and alternatives, enabling better analysis, debugging, and fine-tuning of the model's behavior.

#### 6. ` n`

integer or null

**Description:**

- An integer specifying the number of chat completion choices to generate for each input message. If set to '2', the model will generate two chat completion choices for each prompt.
- Optional, Default value is '1'
- Note that we will be charged based on the number of generated tokens across all of the choices. Keep n as 1 to minimize costs.

#### 7. `presence_penalty`

[Reference](https://platform.openai.com/docs/guides/text-generation/parameter-details)

**Description:**

- A number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
- number or null.Optional, Default value is '0.0'

**Key Concepts:**

- The frequency and presence penalties in the OpenAI API are used to reduce the likelihood of the model generating repetitive sequences of tokens. They work by adjusting the logits, which are the un-normalized log-probabilities that the model uses to decide which token to generate next.
- **Frequency Penalty:** Reduces the logit of a token proportionally to how often it has already been used.
- **Presence Penalty:** Applies a fixed reduction to the logit of a token if it has been used at least once before.

##### **Practical Explanation**

**Frequency Penalty:** If a token has been used multiple times, its logit is reduced more each time it appears again. This makes the model less likely to repeat the same token frequently.
**Presence Penalty:** If a token has been used at least once, its logit is reduced by a fixed amount. This discourages the model from repeating any token that has already appeared.

##### Choosing Penalty Values

**Small Values (0.1 to 1):** These will slightly reduce repetition, making the text less redundant but still maintaining coherence.
**Large Values (up to 2):** These will strongly suppress repetition, but might degrade the overall quality of the text, making it less coherent or natural.
**Negative Values:** These can be used if you want to encourage repetition, which might be useful in specific cases like poetry or certain stylistic writing.

#### 8. `stop`

[Reference](https://platform.openai.com/docs/guides/text-generation/parameter-details)

**Description:**

- A string or list of strings that will be used to control the generation process. If provided, the model will stop when it encounters one of the stop sequences.
- Optional, Default value is None

#### 9. `top_p`

**Description:**

- number or null.Optional Defaults to 1, Range 0.1 to 1

##### Understanding `top_p` (Nucleus Sampling)

`top_p` is a parameter used in language models to control the diversity of the generated text. It works through a technique called nucleus sampling, which focuses on a subset of the most probable tokens when generating the next token in a sequence.

##### How `top_p` Works

- **Nucleus Sampling**: Instead of considering all possible tokens when generating text, nucleus sampling restricts the selection to a subset of tokens that collectively have the highest cumulative probability.
- **Probability Mass**: The sum of the probabilities of a group of tokens. For example, if `top_p` is set to 0.1, the model only considers the tokens that make up the top 10% of the probability mass.

##### Example

Let's say the model has the following probabilities for the next token:

- Token A: 30%
- Token B: 25%
- Token C: 15%
- Token D: 10%
- Token E: 5%
- Other Tokens: Remaining 15%

If `top_p` is set to 0.5, the model will consider only the tokens whose probabilities sum up to 50%. In this case, it would likely consider Tokens A, B, and C (30% + 25% + 15% = 70%, which exceeds 50%).

##### Adjusting Diversity with `top_p`

- **High `top_p` (close to 1)**: The model considers more tokens, leading to more diverse and creative text.
- **Low `top_p` (close to 0)**: The model considers fewer tokens, focusing on the most probable ones, leading to more deterministic and focused text.

##### Relationship with Temperature

- **Temperature**: Another parameter that affects text diversity by scaling the logits before applying softmax to get probabilities. Higher temperatures make the model more random, while lower temperatures make it more deterministic.
- **Recommendation**: It's usually best to adjust either `top_p` or temperature, but not both, as both parameters influence the randomness and diversity of the output. Adjusting both can lead to unpredictable results.

##### Practical Usage

##### Setting `top_p`

- **Creative Writing**: Use a higher `top_p` (e.g., 0.9) to allow for more variety and creativity in the text.
- **Technical or Factual Writing**: Use a lower `top_p` (e.g., 0.1 to 0.3) to keep the output focused and deterministic.

##### Example Code

Here’s an example of using `top_p` with the OpenAI API:

```python
import openai

# Initialize your API key
openai.api_key = 'your-api-key-here'

# Define the function to get a response with top_p
def generate_text(prompt, top_p_value):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        top_p=top_p_value
    )
    return response.choices[0].message.content

# Example prompt
prompt = "Write a short story about a space adventure."

# Generate text with top_p = 0.9
response = generate_text(prompt, 0.9)
print(response)
```

##### Summary

- **`top_p`**: Controls the diversity of the output by considering only the top tokens that make up a certain probability mass.
- **Higher `top_p`**: More diverse and creative output.
- **Lower `top_p`**: More focused and deterministic output.
- **Adjusting `top_p` vs. Temperature**: It’s recommended to adjust one but not both to maintain predictable control over the model’s behavior.

Using `top_p` allows you to fine-tune the balance between creativity and focus in the generated text, making it a powerful tool for different applications.


#### 10. `temperature`

number or null, Optional, Defaults to 1
**Description:**

- A number between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
- What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic

##### Understanding the Temperature Parameter in Chat Completion

The `temperature` parameter in language models like OpenAI's GPT series is a key factor that controls the randomness and creativity of the generated text. It adjusts the model's output by scaling the logits, which are the un-normalized probabilities of the next token.

##### How `temperature` Works

- **Logits Scaling**: The temperature parameter scales the logits before they are converted into probabilities using the softmax function. This scaling affects how confident the model is in its predictions.
- **Effect on Diversity**:
  - **Higher Temperature**: Increases randomness by making the probability distribution flatter. This means the model is more likely to choose less probable tokens, leading to more varied and creative outputs.
  - **Lower Temperature**: Decreases randomness by making the probability distribution sharper. The model tends to choose the most probable tokens, resulting in more focused and deterministic outputs.

##### Mathematical Perspective

When generating the next token, the model computes logits for all possible tokens. The temperature modifies these logits as follows:

\[ \text{logits}\_i = \frac{\text{logits}\_i}{\text{temperature}} \]

- **Temperature > 1**: Flattens the distribution, increasing the chances of picking less probable tokens.
- **Temperature < 1**: Sharpens the distribution, making the model more confident in its top choices.

##### Practical Usage

##### Setting the Temperature

- **High Creativity (e.g., 0.7 to 1.0 or higher)**: Use higher temperature values to encourage the model to produce more diverse and creative text. Suitable for tasks like creative writing, brainstorming, or generating varied responses.
- **Focused and Deterministic Output (e.g., 0.1 to 0.3)**: Use lower temperature values to get more predictable and coherent text. Suitable for technical writing, factual content, or applications where consistency is crucial.

##### Example Code

Here’s an example of how to use the `temperature` parameter with the OpenAI API:

```python
import openai

# Initialize your API key
openai.api_key = 'your-api-key-here'

# Define the function to get a response with temperature
def generate_text(prompt, temperature_value):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature_value
    )
    return response.choices[0].message.content

# Example prompt
prompt = "Write a poem about the ocean."

# Generate text with different temperature values
response_low_temp = generate_text(prompt, 0.2)
response_high_temp = generate_text(prompt, 0.8)

print("Low Temperature Output:\n", response_low_temp)
print("High Temperature Output:\n", response_high_temp)
```

##### Comparing Outputs

- **Low Temperature (e.g., 0.2)**: The output will be more predictable, sticking closely to common patterns and highly probable tokens.
- **High Temperature (e.g., 0.8)**: The output will be more varied, with the model exploring less common tokens and potentially producing more creative and unexpected text.

##### Summary

- **Temperature Parameter**: Controls the randomness and creativity of the model's output by scaling logits.
- **Higher Temperature**: Increases diversity and creativity by making the probability distribution flatter.
- **Lower Temperature**: Increases focus and determinism by making the probability distribution sharper.
- **Practical Usage**: Adjust the temperature based on the desired level of creativity and predictability in the generated text.

By tuning the temperature parameter, you can balance between generating creative, varied outputs and maintaining coherent, predictable responses, making it a versatile tool for different use cases.

#### 10. `tools`

array, Optional

- A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for. A max of 128 functions are supported.

##### properties

**`type`** _string, required_
The type of the tool. Currently, only function is supported.

**`function`** _object, required_

**`description`** _string, optional_
A description of what the function does, used by the model to choose when and how to call the function.

**`name`** _string, required_
The name of the function to be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.

**`parameters`** _object, optional_
The parameters the functions accepts, described as a JSON Schema object. See the guide for examples, and the JSON Schema reference for documentation about the format.

Omitting parameters defines a function with an empty parameter list.

11. `tool_choice` _string or object, Optional_
    Controls which (if any) tool is called by the model. none means the model will not call any tool and instead generates a message. auto means the model can pick between generating a message or calling one or more tools. required means the model must call one or more tools. Specifying a particular tool via {"type": "function", "function": {"name": "my_function"}} forces the model to call that tool.

none is the default when no tools are present. auto is the default if tools are present. 12. `seed`
for deterministic response. 13. `user`
[For AI Safetry - Safety Best Practices](https://platform.openai.com/docs/guides/safety-best-practices)


## Bonus: How to find Token ID from Token Name
#### How to find the Token ID

```
pip install transformers

```

```python
from transformers import GPT2Tokenizer

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Encode the word
tokens = tokenizer.encode("spoiler", add_special_tokens=False)

# Print the token ID
print(f"Token ID for 'spoiler': {tokens[0]}")

# In this example, the GPT2Tokenizer is used because GPT-3 is a larger version of the GPT-2 architecture, and their tokenizers are compatible.

```
