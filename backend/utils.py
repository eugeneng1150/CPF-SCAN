
import pandas as pd
import matplotlib.pyplot as plt

def read_csv_head(file_path):
    try:
        df = pd.read_csv(file_path)
        return df.head()
    except FileNotFoundError:
        return "File not found. Please check the path."
    except Exception as e:
        return f"An error occurred: {e}"

def get_sentiment_distribution(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'sentiment' in df.columns:
            return df['sentiment'].value_counts()
        else:
            return "The 'sentiment' column was not found in the provided file."
    except FileNotFoundError:
        return "File not found. Please check the path."
    except Exception as e:
        return f"An error occurred: {e}"

def create_sentiment_barplot(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'sentiment' not in df.columns:
            return "The 'sentiment' column was not found in the provided file."

        sentiment_counts = df['sentiment'].value_counts()
        fig, ax = plt.subplots()
        bars = ax.bar(sentiment_counts.index, sentiment_counts.values)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center')

        plt.xticks(rotation='horizontal')
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        ax.set_title("Sentiment Distribution")

        return fig

    except FileNotFoundError:
        return "File not found. Please check the path."
    except Exception as e:
        return f"An error occurred: {e}"
