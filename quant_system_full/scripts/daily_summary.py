import os
import subprocess
import json
from dotenv import load_dotenv
from datetime import datetime

def get_daily_log_content():
    """Reads the content of the bot.log file for the current day."""
    try:
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_filename = f"bot_{date_str}.log"
        log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dashboard', 'state', 'daily', log_filename))
        with open(log_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # This is not an error, just means no logs for today yet.
        return ""
    except Exception as e:
        print(f"Error reading log file: {e}")
        return None

def summarize_with_gemini(api_key, log_content):
    """Uses curl to call the Gemini API and summarize the log content."""
    if not api_key:
        return "Error: GEMINI_API_KEY not found. Please check your .env file."
    if not log_content or log_content.strip() == "":
        return "Log file is empty. Nothing to summarize."

    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}'
    
    prompt = f"""Please summarize the following trading bot log from today. 
    Provide a clear, concise summary in markdown format. 
    The summary should include:
    1.  A list of securities that were analyzed.
    2.  A list of all BUY orders executed (symbol and quantity).
    3.  A list of all SELL orders executed (symbol and quantity).
    4.  A brief conclusion on the overall trading activity for the day.

    Log Content:
    ---
    {log_content}
    ---
    """

    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        process = subprocess.run(
            ['curl', '-s', url, '-H', 'Content-Type: application/json', '-d', json.dumps(data)],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8' # Explicitly set encoding for output
        )
        response_json = json.loads(process.stdout)
        summary = response_json['candidates'][0]['content']['parts'][0]['text']
        return summary
    except subprocess.CalledProcessError as e:
        return f"Error calling Gemini API: {e}\nResponse: {e.stderr}"
    except (KeyError, IndexError) as e:
        return f"Error parsing Gemini response: {e}\nResponse: {str(response_json)}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def main():
    """Main function to generate the daily summary."""
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
    load_dotenv(dotenv_path=dotenv_path)
    
    api_key = os.getenv("GEMINI_API_KEY")
    log_content = get_daily_log_content()

    print(f"--- Daily Trading Summary for {datetime.now().strftime('%Y-%m-%d')} ---")
    if log_content:
        summary = summarize_with_gemini(api_key, log_content)
        
        # --- Save the report to a file first ---
        try:
            reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reports'))
            os.makedirs(reports_dir, exist_ok=True)
            report_filename = f"trading_report_{datetime.now().strftime('%Y-%m-%d')}.md"
            report_path = os.path.join(reports_dir, report_filename)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"\nSuccessfully saved report to: {report_path}")
        except Exception as e:
            print(f"\nError saving report: {e}")
        
        # Now, print the summary to the console
        try:
            print(summary)
        except UnicodeEncodeError:
            print("(Could not print summary to console due to encoding issues, but it was saved to the file.)")
    else:
        print("Could not generate summary because log file was empty or could not be read.")

if __name__ == "__main__":
    main()