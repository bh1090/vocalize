from dotenv import load_dotenv
load_dotenv()

from modal import Cls

def test_modal():
    CodingCompanion = Cls.from_name("coding-companion", "CodingCompanion")
    model = CodingCompanion()

    print("Sending test message to Modal endpoint...")

    test_messages = [
        {
            "role": "system",
            "content": "You are a helpful coding assistant."
        },
        {
            "role": "user", 
            "content": "In one sentence, what is a hash map?"
        }
    ]

    response = model.generate.remote(test_messages, max_new_tokens=100)
    
    print("\n✅ Modal responded:")
    print(response)

if __name__ == "__main__":
    test_modal()