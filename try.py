def add_word(sentence, word_to_check, word_to_add):
    words = sentence.split()
    new_sentence = sentence

    for i, word in enumerate(words):
        if word.lower() == word_to_check.lower():
            # Found the word, add another word after it
            new_sentence = ' '.join(words[:i+1] + [word_to_add] + words[i+1:])
            break

    return new_sentence

# Example usage
sentence = "The quick brown fox jumps over the lazy dog."
word_to_check = "fox"
word_to_add = "sly"

new_sentence = add_word(sentence, word_to_check, word_to_add)
print("Original sentence:", sentence)
print("New sentence:", new_sentence)