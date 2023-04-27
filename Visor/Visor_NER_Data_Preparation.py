#Importing necesasary packages
import os
import pandas as pd
import numpy as np
import spacy
import random

# positional_words = ["above",
#                     "below",
#                     "beside",
#                     "between",
#                     "beyond",
#                     "near",
#                     "far from",
#                     "outside",
#                     "inside",
#                     "in front of",
#                     "behind",
#                     "on top of",
#                     "underneath",
#                     "next to",
#                     "adjacent to",
#                     "to the right of",
#                     "to the left of"]

positional_words = ["above",
                    "below",
                    "between",
                    "right",
                    "left"]

# read classes_copy.txt and save it as a list
with open('Visor/classes_copy.txt') as f:
    object_words = f.readlines()
object_words = [x.strip() for x in object_words]

# object_words = ['Chair', 
#                 'Lamp', 
#                 'Plant', 
#                 'Book', 
#                 'Cup', 
#                 'Phone', 
#                 'Computer', 
#                 'Keyboard', 
#                 'Mouse',
#                 'Pen',
#                 'Pencil',
#                 'Paper',
#                 'Clock', 
#                 'Sofa', 
#                 'Pillow', 
#                 'Blanket', 
#                 'Rug', 
#                 'Mirror', 
#                 'Picture frame', 
#                 'Television', 
#                 'Remote control', 
#                 'Stereo', 
#                 'Headphones', 
#                 'Speakers', 
#                 'Trash can', 
#                 'Fridge', 
#                 'Oven', 
#                 'Stove', 
#                 'Microwave', 
#                 'Dishwasher', 
#                 'Toaster', 
#                 'Kettle', 
#                 'Coffee maker', 
#                 'Blender', 
#                 'Mixer', 
#                 'Iron', 
#                 'Vacuum cleaner', 
#                 'Broom', 
#                 'Dustpan', 
#                 'Mop', 
#                 'Bucket', 
#                 'Sponge', 
#                 'Towel', 
#                 'Toothbrush', 
#                 'Soap dispenser', 
#                 'Shampoo bottle', 
#                 'Conditioner bottle', 
#                 'Razor', 
#                 'Tissue box', 
#                 'Hairbrush', 
#                 'Comb', 
#                 'Perfume bottle', 
#                 'Nail polish', 
#                 'Lipstick', 
#                 'Eye shadow', 
#                 'Mascara', 
#                 'Deodorant', 
#                 'Hanger', 
#                 'Coat rack', 
#                 'Shoe rack', 
#                 'Umbrella', 
#                 'Key', 
#                 'Wallet', 
#                 'Sunglasses', 
#                 'Hat', 
#                 'Scarf', 
#                 'Gloves', 
#                 'Backpack',
#                 'Purse', 
#                 'Briefcase', 
#                 'Passport', 
#                 'Camera', 
#                 'Tripod', 
#                 'Guitar', 
#                 'Drum', 
#                 'Microphone', 
#                 'Lights', 
#                 'Cables', 
#                 'DVD player', 
#                 'Console', 
#                 'Controller', 
#                 'Board game', 
#                 'Sports ball', 
#                 'Jump rope', 
#                 'Yoga mat', 
#                 'Dumbbell', 
#                 'Treadmill', 
#                 'Exercise bike', 
#                 'Weight bench', 
#                 'Medicine ball', 
#                 'Resistance bands', 
#                 'Water bottle', 
#                 'Protein shake', 
#                 'Running shoes', 
#                 'Gym bag', 
#                 'Exercise clothes', 
#                 'Car', 
#                 'Truck', 
#                 'Motorcycle', 
#                 'Bicycle', 
#                 'Bus', 
#                 'Train', 
#                 'Airplane', 
#                 'Helicopter', 
#                 'Boat', 
#                 'Kayak', 
#                 'Cat', 
#                 'Dog', 
#                 'Bird', 
#                 'Fish', 
#                 'Hamster', 
#                 'Rabbit', 
#                 'Guinea pig', 
#                 'Turtle', 
#                 'Lizard', 
#                 'Snake', 
#                 'Spider', 
#                 'Frog', 
#                 'Horse', 
#                 'Cow', 
#                 'Pig', 
#                 'Sheep', 
#                 'Goat', 
#                 'Chicken', 
#                 'Rooster', 
#                 'Duck', 
#                 'Goose', 
#                 'Turkey', 
#                 'Deer', 
#                 'Bear', 
#                 'Lion', 
#                 'Tiger', 
#                 'Leopard', 
#                 'Giraffe', 
#                 'Elephant', 
#                 'Rhino', 
#                 'Hippo', 
#                 'Kangaroo', 
#                 'Koala', 
#                 'Monkey', 
#                 'Gorilla', 
#                 'Chimpanzee', 
#                 'Orangutan', 
#                 'Panda', 
#                 'Sloth', 
#                 'Whale', 
#                 'Dolphin', 
#                 'Octopus', 
#                 'Jellyfish']


#Sentence format lists
template_sentences ={'2': "<object1> <direction1> the <object2>",
                     '3': "<object1> <direction1> the <object2> <direction2> the <object3>",
                     '4': "<object1> <direction1> the <object2> <direction2> the <object3> <direction3> the <object4>"}


#Function to generate prompts and annotate them for NER
def generate_prompts(positional_words, object_words, num_prompts, num_objects, template_sentences):
    prompts = []

    for i in range(num_prompts):
        template_sentence = template_sentences[str(num_objects)]
        random.seed()
        for i in range(num_objects):
            object = random.choice(object_words)
            if i == 0:
                if object[0] in ['a','e','i','o','u']:
                    object = 'an ' + object
                else:
                    object = 'a ' + object
            template_sentence = template_sentence.replace('<object'+str(i+1)+'>', object)
        for i in range(num_objects-1):
            direction = random.choice(positional_words)
            template_sentence = template_sentence.replace('<direction'+str(i+1)+'>', direction)

        prompts.append(template_sentence)
    
    return prompts   


#Function to annotate prompts for NER
def annotate_prompts(prompts, positional_words=positional_words, object_words=object_words):
    annotated_prompts = []
    objects, directions = list(), list()
    for prompt in prompts:
        prompt = prompt.lower()
        entities = []
        words = prompt.split()
        for i in range(len(words)):
            if words[i] in object_words:
                start = prompt.find(words[i])
                end = start + len(words[i])
                output_tuple = (start, end, 'OBJECT')
                print("OBJECT", words[i], output_tuple)
                entities.append(output_tuple)
                objects.append(words[i])
            elif words[i] in positional_words:
                start = prompt.find(words[i])
                end = start + len(words[i])
                output_tuple = (start, end, 'POSITION')
                print("POSITION", words[i], output_tuple)
                entities.append(output_tuple)
                directions.append(words[i])
            else:
                if i != len(words)-1:
                    if words[i] + " " + words[i+1] in object_words:
                        start = prompt.find(words[i] + " " + words[i+1])
                        end = start + len(words[i] + " " + words[i+1])
                        output_tuple = (start, end, 'OBJECT')
                        print("OBJECT", words[i] + " " + words[i+1], output_tuple)
                        entities.append(output_tuple)
                        objects.append(words[i] + " " + words[i+1])

        annotated_prompts.append((prompt, {'entities': entities}))
    
    return annotated_prompts, objects, directions


def main():
    object_words = [word.lower() for word in object_words]
    two_object_prompts = generate_prompts(positional_words, object_words, 25, 2, template_sentences)
    np.savetxt('two_object_prompts.txt', two_object_prompts, fmt='%s')

    three_object_prompts = generate_prompts(positional_words, object_words, 25, 3, template_sentences)


    four_object_prompts = generate_prompts(positional_words, object_words, 25, 4, template_sentences)


    #Append all prompts to a single list
    # prompts = two_object_prompts + three_object_prompts + four_object_prompts


    prompts = ['a Towel on top of the Dog above the Medicine ball']
    annotated_prompts = annotate_prompts(prompts, positional_words, object_words)

    print(annotated_prompts)