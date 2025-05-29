from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import os
import pandas as pd

data = pd.read_csv("cleaned.csv")
# Initialize the OpenAI Chat model
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
)


# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["email"],
    template="Email: {email}\nSentiment:"
)

# Define few-shot 
examples = [
    {"email": "I love this product! It has changed my life for the better.", "sentiment": "positive"},
    {"email": "The service was okay, nothing special.", "sentiment": "neutral"},
    {"email": "I am very disappointed with the quality of the product.", "sentiment": "negative"},
    {"email": "I am happy with the product, but the delivery was late.", "sentiment": "mixed"},
    {"email": "This is urgent! I need a replacement immediately.", "sentiment": "urgent"}
]

system_message = """
You are a sentiment classification assistant working on CPF-related queries. Your task is to classify the overall sentiment of the member's message as either **positive**, **neutral**, or **negative**. Also, provide a **confidence score from 0 to 1** to reflect how certain you are.

Interpretations should consider:
- If the member expresses frustration, confusion, or a complaint about CPF policies or access to funds → likely **negative**
- If the member acknowledges it’s their own fault, or seems resigned without strong emotion → likely **neutral**
- If the member expresses appreciation or satisfaction, or that the resquest is successful → **positive**

Format your response strictly as:
Sentiment: <positive/neutral/negative>
Confidence: <score between 0-1>
"""
def analyze_sentiment(email):
    # Format the prompt string
    prompt_text = prompt_template.format(email=email)
    
    # Wrap the prompt in a HumanMessage
    response = llm.invoke([
        SystemMessage(content=system_message), 
        HumanMessage(content=prompt_text)])
    
    # Return the content (it's already a string)
    return response.content.strip()

# Example usage
if __name__ == "__main__":
    # Apply sentiment analysis
    data["sentiment_response"] = data["Review"].apply(analyze_sentiment)

    # Extract sentiment + confidence into separate columns
    data[["Sentiment_response", "Confidence_response"]] = data["sentiment_response"].str.extract(r'Sentiment:\s*(\w+)\s*Confidence:\s*([\d\.]+)')
    data.drop(columns=["sentiment_response"], inplace=True)
    # Save everything to CSV
    data.to_csv("sentiment_response.csv", index=False)
    print("Done: CSV with sentiment and confidence saved.")


