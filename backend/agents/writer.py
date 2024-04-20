from datetime import datetime
from langchain.adapters.openai import convert_openai_messages
from langchain_openai import ChatOpenAI
import json5 as json

sample_json = """
{
    "subject": subject of the email,
    "email_content": "email content",
"""

sample_revise_json = """
{
    "subject": subject of the email,
    "email_content": "email content",
    "message": "message to the critique",
    "number_of_revisions": "number of revisions made to the email"
}
"""


class WriterAgent:
    def __init__(self):
        pass

    def writer(self, email: dict):

        prompt = [{
            "role": "system",
            "content": "You are a marketing email writer. Your sole purpose is to write a well-written personalized "
                       "marketing email about my product based on provided context and sources. "
                       "Your emails are engaging and are the perfect balance between fun, informing, and serious."
                       "Your emails are built like this: "
                       "First of all, start the email with 'Hi {first name},'."
                       "1. Intro - The intro introduces the user to the lead."
                       "2. Pitch- The pitch should be no longer than 80 words and should be do the following:"
                       "a. Concisely explain what the product does."
                       "b. Clearly state what the recipient gains from accepting your pitch. You’ll be most effective "
                       " when you tie your value proposition to their pain points."
                       "3. Include a strong call to action and make it super easy and simple for the recipient to do."
                       " It can either be a link or a simple question to schedule a meeting. "
                       "4. Write a compelling Subject Line: You want people to open your email. For that, you need a "
                       "clear and compelling subject line."
                       "Here are a few tips:"
                       "a. Keep your subject lines accurate (clickbait will irritate your recipient and might get your "
                       "message marked as junk)"
                       "b. Use around 45 characters—or less—to avoid truncation on mobile (this happens when the "
                       "subject line is shortened due "
                       "to space limitations)"
                       "c. You can but dont have to make it personal by adding a name or referring to the industry "
                       "your recipient works in."
                       "d. Avoid spammy tactics like capitalized words or excessive exclamation points"
                       "e. You can add an emoji but dont have to.\n"
                       "Write less than 150 words, and add new line tagging so the text would be styled for HTML"
        }, {
            "role": "user",
            "content": f"{str(email)}\n"

                       f"Your task is to write a personalized and engaging email about a product based on the "
                       f"given context and news sources.\n"
                       f"please return nothing but a JSON in the following format:\n"
                       f"{sample_json}\n"

        }]

        lc_messages = convert_openai_messages(prompt)
        optional_params = {
            "response_format": {"type": "json_object"}
        }

        response = ChatOpenAI(model='gpt-4-0125-preview', max_retries=1, model_kwargs=optional_params).invoke(
            lc_messages).content
        return json.loads(response)

    def revise(self, email: dict):
        prompt = [{
            "role": "system",
            "content": "You are editing a marketing email. Your sole purpose is to edit a personalized and "
                       "engaging marketing email about a product based on given critique."
                       "Some important points, remind the writer if needed that the email must not be longer than 130"
                       "words. The email should be a good balance of informal, informing and business oriented."
                       "Remind the writer to connect the product to the sources of the target's company or talk about "
                       "why this will help the target and their firm. Try to see what creates the most coherent "
                       "email. Remind the writer that he may never make things up, he must stay true to what he was "
                       "told, and try not to make claims. Remind him to include a direct call to action.\n "
                       "Remind it to Write less than 150 words, and add new line tagging so the text would be styled"
                       " for HTML"
        }, {
            "role": "user",
            "content": f"subject: {email['subject']}\n"
                        f"email_content: {email['email_content']}\n"
                        f"message: {email.get('message')}\n"
                       f"number_of_revisions: {email.get('number_of_revisions', 0)}\n"
                       f"Your task is to edit the email based on the critique given and explain the changes made in "
                       f"the message field.\n"
                       f"if you cannot change the email based on the critique, please return the same email and "
                       f"explain why in the message field\n"
                       f"Also, please increment number_of_revisions by 1\n"
                       f"Please return nothing but a JSON in the following format:\n"
                       f"{sample_revise_json}\n "

        }]

        lc_messages = convert_openai_messages(prompt)
        optional_params = {
            "response_format": {"type": "json_object"}
        }

        response = ChatOpenAI(model='gpt-4-0125-preview', max_retries=1, model_kwargs=optional_params).invoke(
            lc_messages).content
        response = json.loads(response)
        print(f"For article: {email['title']}")
        print(f"Writer Revision Message: {response['message']}\n")
        return response

    def run(self, email: dict):
        critique = email.get("critique")
        if critique is not None:
            email.update(self.revise(email))
        else:
            email.update(self.writer(email))
            print(email)
        return email
