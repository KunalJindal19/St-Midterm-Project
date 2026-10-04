import sys
import os

# Add the project directory to the path so we can import llm_client
sys.path.append("/Users/kunaljindal/Desktop/ST/Midterm_Project")

from config import Config
from llm_client import call_llm
from test_case_generator import TestCaseGenerator

config = Config()
gen = TestCaseGenerator(config)

code = """
def remove_Occ(s, ch):                                                                                      
    first = s.find(ch)                                                                                      
    if first == -1:                                                                                         
        return s                                                                                            
    last = s.rfind(ch)                                                                                      
    if first == last:                                                                                       
        # Only one occurrence, remove it                                                                    
        return s[:first] + s[first+1:]                                                                      
    else:                                                                                                   
        # Remove first and last occurrences                                                                 
        return s[:first] + s[first+1:last] + s[last+1:]  
"""
problem = {
    "text": "Write a python function to remove first and last occurrence of a given character from the string.",
    "code": code
}

prompt = gen._build_user_prompt(code, problem, "prime_path")
system_prompt = gen.SYSTEM_PROMPTS["prime_path"]

print("Calling LLM...")
raw_response, model = call_llm(config, system_prompt, prompt)
print("RAW RESPONSE:")
print("-------------")
print(repr(raw_response))
print("-------------")
