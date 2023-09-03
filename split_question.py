import re

def transform_text(original_text):
    # Extract the base question
    base_question_match = re.search(r'For which of these two scenarios (.+)? Scenario 1', original_text)
    if base_question_match:
        base_question = base_question_match.group(1)
    else:
        return "Could not find base question."
    
    # Remove any trailing 'does' or 'do' from the base question to avoid duplication
    base_question = re.sub(r'^(does|do)\s+', '', base_question)
    
    # Split the original text into parts, then extract the scenarios
    # Exclude the answer choices and the "Answer:" line at the end
    parts = original_text.split("Scenario ")
    scenarios = [part.split("|")[1].split("\n")[0].strip() for part in parts[1:]]
    
    # Prepare the transformed text
    transformed_text = ""
    for scenario in scenarios:
        transformed_text += f"Does {base_question}\n{scenario}\nA. No\nB. Yes\nAnswer:\n\n"
    
    return transformed_text