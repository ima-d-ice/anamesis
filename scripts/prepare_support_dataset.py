#!/usr/bin/env python3
"""
Create a high-quality support dataset with near-identical paraphrases for
release-gate calibration. Paraphrase pairs are designed to have cosine
similarity >= 0.95 to test if the release gate can be met at high thresholds.
"""

import json
import os
import random

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")
SEED = 42
random.seed(SEED)

# Support topics with near-identical paraphrase pairs
# Each anchor query has multiple paraphrases that are semantically identical
# but with minor wording variations (should have cosine similarity 0.95+)
SUPPORT_DATA = {
    "password_reset": {
        "answer": "To reset your password, go to the login page and click 'Forgot Password'. Enter your email address and we'll send you a reset link. Check your spam folder if you don't see it within 5 minutes.",
        "anchor": "How do I reset my password?",
        "paraphrases": [
            "How do I reset my password?",
            "How can I reset my password?",
            "I need to reset my password",
            "password reset how to",
            "reset my login password",
        ],
    },
    "billing_refund": {
        "answer": "Refunds are processed within 5-7 business days after approval. You'll receive an email confirmation once your refund has been issued.",
        "anchor": "How do I request a refund?",
        "paraphrases": [
            "How do I request a refund?",
            "How can I get a refund?",
            "I want a refund",
            "refund request how to",
            "get money back refund",
        ],
    },
    "account_delete": {
        "answer": "To delete your account, go to Account Settings > Privacy > Delete Account. This action is permanent.",
        "anchor": "How do I delete my account?",
        "paraphrases": [
            "How do I delete my account?",
            "How can I delete my account?",
            "I want to delete my account",
            "delete my account please",
            "account deletion how to",
        ],
    },
    "shipping_tracking": {
        "answer": "You can track your order using the tracking number in your confirmation email. Log in to your account and go to Orders > Track Order.",
        "anchor": "How do I track my order?",
        "paraphrases": [
            "How do I track my order?",
            "How can I track my order?",
            "I want to track my order",
            "track my order please",
            "order tracking how to",
        ],
    },
    "billing_update": {
        "answer": "To update your payment method, go to Account Settings > Billing > Payment Methods. You can add a new credit card or PayPal account.",
        "anchor": "How do I update my payment method?",
        "paraphrases": [
            "How do I update my payment method?",
            "How can I update my payment method?",
            "I want to update my payment method",
            "update payment method please",
            "payment method update how to",
        ],
    },
    "technical_login": {
        "answer": "If you can't log in, first try clearing your browser cache and cookies. Make sure you're using the correct email address and password.",
        "anchor": "I can't log in to my account",
        "paraphrases": [
            "I can't log in to my account",
            "I cannot log in to my account",
            "login not working",
            "can't sign in to account",
            "login problem help",
        ],
    },
    "subscription_cancel": {
        "answer": "To cancel your subscription, go to Account Settings > Subscription > Cancel Subscription. Your access will continue until the end of your current billing period.",
        "anchor": "How do I cancel my subscription?",
        "paraphrases": [
            "How do I cancel my subscription?",
            "How can I cancel my subscription?",
            "I want to cancel my subscription",
            "cancel subscription please",
            "subscription cancellation how to",
        ],
    },
    "feature_request": {
        "answer": "You can submit feature ideas through our feedback portal at feedback.example.com. Our product team reviews all submissions.",
        "anchor": "How do I request a new feature?",
        "paraphrases": [
            "How do I request a new feature?",
            "How can I request a new feature?",
            "I want to request a feature",
            "feature request how to",
            "submit feature idea",
        ],
    },
    "data_export": {
        "answer": "You can export your data by going to Account Settings > Privacy > Data Export. Select the date range and data types you want, then click Export.",
        "anchor": "How do I export my data?",
        "paraphrases": [
            "How do I export my data?",
            "How can I export my data?",
            "I want to export my data",
            "export my data please",
            "data export how to",
        ],
    },
    "two_factor": {
        "answer": "To enable two-factor authentication, go to Account Settings > Security > Two-Factor Authentication. You can use an authenticator app like Google Authenticator or Authy.",
        "anchor": "How do I enable two-factor authentication?",
        "paraphrases": [
            "How do I enable two-factor authentication?",
            "How can I enable two-factor authentication?",
            "I want to enable 2FA",
            "enable 2fa please",
            "two factor authentication setup",
        ],
    },
    "billing_invoice": {
        "answer": "You can download invoices from Account Settings > Billing > Invoices. Select the invoice you need and click Download.",
        "anchor": "How do I download my invoice?",
        "paraphrases": [
            "How do I download my invoice?",
            "How can I download my invoice?",
            "I want to download my invoice",
            "download invoice please",
            "invoice download how to",
        ],
    },
}

# Negative questions from unrelated topics (should NOT match cache)
NEGATIVE_QUESTIONS = [
    "What are your business hours?",
    "Do you ship internationally?",
    "What payment methods do you accept?",
    "Is there a mobile app?",
    "Do you offer student discounts?",
    "Can I change my shipping address?",
    "What is your return policy?",
    "Do you have a loyalty program?",
    "How long does shipping take?",
    "Can I track multiple orders at once?",
    "Do you offer gift wrapping?",
    "What countries do you ship to?",
    "Can I change my email address?",
    "How do I create an account?",
    "Do you have a physical store?",
]

# Hard negatives: same topic area but different answer
HARD_NEGATIVES = [
    ("How do I reset my password?", "How do I change my email address?"),
    ("How do I request a refund?", "How do I dispute a charge?"),
    ("How do I delete my account?", "How do I deactivate my account temporarily?"),
    ("How do I track my order?", "How do I change my shipping address?"),
    ("How do I update my payment method?", "How do I add a new credit card?"),
    ("I can't log in to my account", "I can't log in to my account on mobile"),
    ("How do I cancel my subscription?", "How do I downgrade my subscription?"),
    ("How do I request a new feature?", "How do I report a bug?"),
    ("How do I export my data?", "How do I import my data?"),
    ("How do I enable two-factor authentication?", "How do I disable two-factor authentication?"),
    ("How do I download my invoice?", "How do I download my receipt?"),
]

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Build cache: one anchor per topic
    cache_anchors = []
    for topic_name, topic_data in SUPPORT_DATA.items():
        cache_anchors.append({
            "id": f"c{len(cache_anchors):05d}",
            "query": topic_data["anchor"],
            "response": topic_data["answer"],
            "namespace": 1,
            "topic": topic_name,
        })

    # Build positives: each paraphrase of each anchor
    pos_pairs = []
    for topic_name, topic_data in SUPPORT_DATA.items():
        anchor = topic_data["anchor"]
        for para in topic_data["paraphrases"]:
            pos_pairs.append({
                "query": para,
                "expected_key": anchor,
                "topic": topic_name,
            })

    # Build negatives: completely different topics
    neg_pairs = []
    for i, q in enumerate(NEGATIVE_QUESTIONS):
        # Expect an answer from a different topic
        topic_list = list(SUPPORT_DATA.keys())
        other_topic = random.choice(topic_list)
        other_query = SUPPORT_DATA[other_topic]["anchor"]
        neg_pairs.append({
            "query": q,
            "expected_key": other_query,
            "topic": "negative",
        })

    # Build hard negatives: same topic area but different answer
    hard_pairs = []
    for q1, q2 in HARD_NEGATIVES:
        hard_pairs.append({
            "query": q1,
            "expected_key": q2,
            "topic": "hard_negative",
        })

    print(f"Cache anchors: {len(cache_anchors)}")
    print(f"Positive pairs: {len(pos_pairs)}")
    print(f"Negative pairs: {len(neg_pairs)}")
    print(f"Hard negative pairs: {len(hard_pairs)}")

    # Write cache
    with open(os.path.join(DATA_DIR, "support_cache.jsonl"), "w") as f:
        for entry in cache_anchors:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Write positives
    with open(os.path.join(DATA_DIR, "support_pos.jsonl"), "w") as f:
        for entry in pos_pairs:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Write negatives
    with open(os.path.join(DATA_DIR, "support_neg.jsonl"), "w") as f:
        for entry in neg_pairs:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Write hard negatives
    with open(os.path.join(DATA_DIR, "support_hard.jsonl"), "w") as f:
        for entry in hard_pairs:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Collect all texts for embedding
    all_texts = set()
    for entry in cache_anchors:
        all_texts.add(entry["query"])
    for entry in pos_pairs:
        all_texts.add(entry["query"])
        all_texts.add(entry["expected_key"])
    for entry in neg_pairs:
        all_texts.add(entry["query"])
        all_texts.add(entry["expected_key"])
    for entry in hard_pairs:
        all_texts.add(entry["query"])
        all_texts.add(entry["expected_key"])

    print(f"Total unique texts: {len(all_texts)}")

    # Write text list for embedding
    with open(os.path.join(DATA_DIR, "support_texts.jsonl"), "w") as f:
        for text in sorted(all_texts):
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
