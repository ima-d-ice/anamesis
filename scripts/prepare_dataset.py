#!/usr/bin/env python3
"""Generate a synthetic support-ticket dataset for Anamesis benchmarking."""

import json
import os
import random

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

TEMPLATES = [
    ("How do I reset my password?", "Follow the password reset link on the login page or contact support."),
    ("How do I update my email address?", "Go to Account Settings > Profile and click Edit Email."),
    ("How do I cancel my subscription?", "Visit the Billing page and select Cancel Subscription."),
    ("How do I request a refund?", "Submit a refund request within 30 days of purchase via Help Center."),
    ("How do I contact customer support?", "Email support@example.com or use the live chat widget."),
    ("How do I track my order?", "Use the tracking number in your confirmation email on the Order Status page."),
    ("How do I change my billing address?", "Update it in Account Settings > Billing > Address."),
    ("How do I delete my account?", "Go to Account Settings > Privacy > Delete Account."),
    ("How do I enable two-factor authentication?", "Enable 2FA in Account Settings > Security > Two-Factor."),
    ("How do I upgrade my plan?", "Go to Account Settings > Subscription > Upgrade Plan."),
]

PARAPHRASES = [
    "How can I reset my password?",
    "I forgot my password, how do I reset it?",
    "Password reset instructions please",
    "What's the process to change my email?",
    "How do I update my email address on file?",
    "Change my email in account settings",
    "I want to cancel my subscription",
    "How to cancel my recurring billing?",
    "Cancel my plan",
    "Refund request process",
    "Can I get my money back?",
    "How to request a refund for my purchase?",
    "Contact support team",
    "How do I reach customer service?",
    "Support contact information",
    "Track my package",
    "Where is my order?",
    "Order tracking number",
    "Change billing address",
    "Update my address for billing",
    "Delete my account permanently",
    "How to close my account?",
    "Enable two-factor auth",
    "Turn on 2FA for my account",
    "Two-factor authentication setup",
    "Upgrade my subscription plan",
    "How to move to a higher tier?",
    "Plan upgrade instructions",
]

NEGATIVES = [
    "What is the weather in Tokyo today?",
    "How do I bake a chocolate cake?",
    "Who won the 1998 World Cup?",
    "What is the capital of France?",
    "How do I fix a leaky faucet?",
    "What is the stock price of Apple?",
    "How do I train a dog?",
    "What is the best programming language?",
    "How do I make coffee?",
    "What is the meaning of life?",
]

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    positives = []
    for i, (query, response) in enumerate(TEMPLATES):
        positives.append({
            "id": f"q{i}",
            "query": query,
            "response": response,
            "namespace": 1,
            "paraphrases": [],
        })

    # Assign paraphrases to templates (round-robin with variation)
    for i, para in enumerate(PARAPHRASES):
        tmpl_idx = i % len(TEMPLATES)
        positives[tmpl_idx]["paraphrases"].append(para)

    negatives = [{"id": f"n{i}", "query": q, "namespace": 1} for i, q in enumerate(NEGATIVES)]

    # Save
    with open(os.path.join(DATA_DIR, "positives.jsonl"), "w") as f:
        for p in positives:
            f.write(json.dumps(p) + "\n")

    with open(os.path.join(DATA_DIR, "negatives.jsonl"), "w") as f:
        for n in negatives:
            f.write(json.dumps(n) + "\n")

    print(f"Dataset created in {DATA_DIR}")
    print(f"  Positives (cached queries): {len(positives)}")
    print(f"  Paraphrase variants: {sum(len(p['paraphrases']) for p in positives)}")
    print(f"  Negatives: {len(negatives)}")

if __name__ == "__main__":
    main()
