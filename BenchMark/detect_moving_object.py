import spacy

# Load the pre-trained English language model
nlp = spacy.load("en_core_web_sm")

# Define a function to extract the object being moved and the direction from a text prompt
def extract_object_moved(text):
    # Parse the text prompt using the Spacy nlp object
    doc = nlp(text)
    
    # Initialize variables to store the object and direction
    object_moved = None
    direction = None
    
    # Iterate through the parsed output to find the verb and its direct object
    for token in doc:
        # Identify the root verb of the sentence that indicates movement
        if token.pos_ == "VERB" and (token.dep_ == "ROOT" or token.dep_ == "acl"):
            verb = token
            # Iterate through the children of the verb to find the direct object
            for child in verb.children:
                if child.dep_ == "dobj":
                    # Store the text of the direct object as the object being moved
                    object_moved = child.text
                elif child.dep_ == "prep":
                    # Identify the preposition that indicates the direction
                    prep = child
                    # Iterate through the children of the preposition to find the direction
                    for grandchild in prep.children:
                        if grandchild.dep_ == "pobj":
                            # Store the text of the preposition's object as the direction
                            direction = grandchild.text
    
    # Return the object being moved and direction as a tuple
    return object_moved, direction

if __name__ == "__main__":
    # Example usage
    text = "Move the orange to the top."
    object_moved, direction = extract_object_moved(text)
    print(object_moved) # Output: orange
    print(direction) # Output: right
