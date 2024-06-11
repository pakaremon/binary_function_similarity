import streamlit as st
import json
import re
import numpy as np
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

TYPE_ONE = 1
TYPE_TWO = 2
TYPE_THREE = 3
TYPE_FOUR = 4
TYPE_FIVE = 5
TYPE_SIX = 6
TYPE_SEVEN = 7
TYPE_EIGHT = 8

register = [
    'rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rbp', 'rsp',
    'eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'ebp', 'esp',
    'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15',
    'ah', 'al', 'ax', 'bh', 'bl', 'bx', 'ch', 'cl', 'cx',
    'dh', 'dl', 'dx', 'sp', 'bp', 'si', 'di'
]
jmp_call = [
    'jnc', 'jl', 'jmp', 'jb', 'je', 'jns', 'jnle', 'jno', 'jpo', 'jp',
    'jpe', 'js', 'jne', 'jz', 'jo', 'jnl', 'jg', 'jrcxz', 'jc', 'jge',
    'jnz', 'jae', 'jng', 'jnp', 'jcxz', 'jnae', 'jnge', 'jbe', 'jna',
    'jle', 'ja', 'jecxz', 'jnb', 'jnbe'
]

type_mapping = {
    1: "TypeOne",
    2: "TypeTwo",
    3: "TypeThree",
    4: "TypeFour",
    5: "TypeFive",
    6: "TypeSix",
    7: "TypeSeven",
    8: "TypeEight"
}

def remove_noise(instruction):
    new_instruction = re.sub(r'((qword|short|dword|byte)(\sptr)?)|,', '', instruction)
    new_instruction = re.sub(r'\s*;.*$', '', new_instruction)
    return new_instruction

def check_operand_type(operand, opcode):
    if matches := re.search(r'\[(.*)\]', operand):
        if matches.group(1).count('+') == 2:
            return TYPE_FOUR
        elif matches.group(1).count('+') == 1:
            return TYPE_THREE
        else:
            return TYPE_TWO
    elif operand in register:
        return TYPE_ONE
    elif opcode == "call":
        return TYPE_SIX
    elif opcode in jmp_call:
        return TYPE_SEVEN
    elif (matches := re.search(r'\d+h?', operand)):
        return TYPE_FIVE
    return TYPE_EIGHT

def normalized_instruction(instruction):
    instruction = remove_noise(instruction)
    new_instruction = []
    list_operands = instruction.split()
    opcode = list_operands[0]
    new_instruction.append(opcode)
    if len(list_operands) == 3:
        new_instruction.append(type_mapping[check_operand_type(list_operands[1], opcode)])
        new_instruction.append(type_mapping[check_operand_type(list_operands[2], opcode)])
    elif len(list_operands) == 2:
        new_instruction.append(type_mapping[check_operand_type(list_operands[1], opcode)])
    else:
        new_instruction.append(type_mapping[TYPE_EIGHT])
    return " ".join(new_instruction)

def normalize_function(function):
    return [normalized_instruction(instruction) for instruction in function]

def pad_or_truncate_instructions(instruction_sequence, max_length=1000):
    if len(instruction_sequence) < max_length:
        instruction_sequence += ["nop"] * (max_length - len(instruction_sequence))
    elif len(instruction_sequence) > max_length:
        instruction_sequence = instruction_sequence[:max_length]
    return instruction_sequence

def instructions_to_indices(instructions, word_index):
    return [word_index[word] if word in word_index else 0 for word in instructions]

def main():
    st.title("Function Similarity Detector")
    st.write("Upload two JSON files containing function instructions.")

    uploaded_file1 = st.file_uploader("Choose the first JSON file", type="json")
    uploaded_file2 = st.file_uploader("Choose the second JSON file", type="json")

    if uploaded_file1 and uploaded_file2:
        data1 = json.load(uploaded_file1)
        data2 = json.load(uploaded_file2)

        st.write("Functions in the first file:")
        st.write(list(data1.keys()))

        st.write("Functions in the second file:")
        st.write(list(data2.keys()))

        function_name1 = st.selectbox("Select function from first file", list(data1.keys()))
        function_name2 = st.selectbox("Select function from second file", list(data2.keys()))

        if st.button("Compare Functions"):
            function1 = normalize_function(data1[function_name1])
            function2 = normalize_function(data2[function_name2])

            # Load Word2Vec model (assuming the model is in the same directory)
            word2vec_model = Word2Vec.load("word2vec_model_path")

            word_index = {word: idx for idx, word in enumerate(word2vec_model.wv.index_to_key)}

            x1 = pad_or_truncate_instructions(function1)
            x2 = pad_or_truncate_instructions(function2)

            x1 = instructions_to_indices(x1, word_index)
            x2 = instructions_to_indices(x2, word_index)

            x1 = pad_sequences([x1], maxlen=1000, padding='post', truncating='post').tolist()
            x2 = pad_sequences([x2], maxlen=1000, padding='post', truncating='post').tolist()

            x1 = np.array(x1)
            x2 = np.array(x2)

            # Load Siamese model (assuming the model is in the same directory)
            model = tf.keras.models.load_model("siamese_model.h5")

            y_pred = model.predict([x1, x2])
            threshold = 0.5
            y_pred_binary = (y_pred > threshold).astype(int)

            if y_pred_binary[0][0] == 1:
                st.write("The functions are similar.")
            else:
                st.write("The functions are not similar.")

if __name__ == '__main__':
    main()
