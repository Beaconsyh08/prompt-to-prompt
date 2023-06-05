import json
from tqdm import tqdm 

SCENE = "foggy"
ORI_JSON_PATH = "/mnt/ve_share/generation/data/train/diffusions/comb_cls/index.json"
NEW_JSON_PATH = "/mnt/ve_share/generation/data/p2p/ori_jsons/%s_replace.json" % SCENE
street_words = ["street", "road", "highway"]
selected_word = ["night", "rain", "fog", "day", "snow", "cloud",]
# replace_scene_word = ["night", "rainy", "snowy", "foggy", "cloudy"]
replace_scene_word = [SCENE,]
STREET = False


def add_words(sentence, words_to_check, word_to_add):
    words = sentence.split()
    new_sentence = sentence
    flag = False

    for word in words_to_check:
        for i, current_word in enumerate(words):
            if current_word.lower() == word.lower():
                # Found the word, add another word after it
                new_sentence = ' '.join(words[:i+1] + [word_to_add] + words[i+1:])
                words = new_sentence.split()  # Update the words list
                flag = True
                break

    return new_sentence if flag else "%s, %s" % (word_to_add, new_sentence)


with open(ORI_JSON_PATH, 'r') as file:
    data = json.load(file)

ori_prompts = []
for each in tqdm(data):
    prompt = each["prompt"]
    save = True
    for word in selected_word:
        if word in prompt:
            save = False
            break
    if save:
        ori_prompts.append(prompt)
        save = False
    
print(len(ori_prompts))
print(ori_prompts[:10])
print(ori_prompts[-10:])


result = []
count = 0 
for scene in tqdm(replace_scene_word):
    for prompt in tqdm(ori_prompts):
        if "/" in prompt:
            continue
        res_dict = dict()
        if scene == "night":
            if STREET:
                prompt_1 = add_words(prompt, street_words, "at daytime")
                prompt_2 = add_words(prompt, street_words, "at nighttime")
                
            else:
                prompt_1 = "daytime, %s" % prompt
                prompt_2 = "nighttime, %s" % prompt
        elif scene == "snowy":
            if STREET:
                prompt_1 = prompt
                prompt_2 = add_words(prompt, street_words, "covered by heavy snow")
                
            else:
                prompt_1 = "sunny, %s" % prompt
                prompt_2 = "snowy, %s" % prompt
        else:
            prompt_1 = "sunny, %s" % prompt
            prompt_2 = "%s, %s" % (scene, prompt)
        res_dict["id"] = count
        res_dict["prompt_1"] = prompt_1
        res_dict["prompt_2"] = prompt_2
        res_dict["scene"] = scene
        count += 1
        result.append(res_dict)
        
print(len(result))
        
with open(NEW_JSON_PATH, 'w') as json_file:
    json.dump(result, json_file, indent=4)
    
print(NEW_JSON_PATH)
            
        
        
    

    
    
