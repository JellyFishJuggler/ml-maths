import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from NaiveBayes import NaiveBayes, SPAM_TRAIN, HAM_TRAIN

TEST_CASES = [
    ("spam", "You are a big winner today claim your reward immediately"),
    ("spam", "Limited time only get your free gift voucher now"),
    ("spam", "Your bank account is frozen contact us fast to restore access"),
    ("spam", "Make huge profits today with our new crypto investment plan"),
    ("spam", "Special offer just for you buy the miracle pill today"),
    ("spam", "Congratulations you qualify for the big cash jackpot"),
    ("spam", "Click now to confirm your credit card details securely"),
    ("spam", "Last chance to double your money in just minutes"),
    ("spam", "Hurry up redeem your free ticket before the deadline"),
    ("spam", "Your payment failed please update your billing information"),
    ("spam", "You have been selected for a huge discount on our store"),
    ("spam", "Start earning passive income from home today"),
    ("spam", "Guaranteed returns on your savings with zero effort"),
    ("spam", "Act fast the exclusive deal ends at midnight"),
    ("spam", "You won the mega prize call us to release your winnings"),
    ("spam", "Free trial of our weight loss supplement shipping today"),
    ("spam", "Verify your identity now to keep your account active"),
    ("spam", "Buy cheap gadgets directly from our online market today"),
    ("spam", "Your number was randomly chosen for a gift card"),
    ("spam", "Open the attachment to claim your refund immediately"),
    ("spam", "Severe security warning your password must be changed"),
    ("spam", "Lowest rates on loans approved for you instantly"),
    ("spam", "Your subscription is expiring renew today for a bonus"),
    ("spam", "Unlimited chances to win a luxury car enter now"),
    ("spam", "Once in a lifetime business opportunity is waiting for you"),

    ("ham", "Can we reschedule our call to later this evening please"),
    ("ham", "I liked the presentation you made yesterday great job"),
    ("ham", "Will you be home before dinner tonight"),
    ("ham", "Please send me the bill from last weeks shopping"),
    ("ham", "The team enjoyed the picnic at the park yesterday"),
    ("ham", "Remember to take your keys and lock the door"),
    ("ham", "Thanks for the birthday wishes everyone was so kind"),
    ("ham", "The lab class got extended by one hour today"),
    ("ham", "Call me when you reach the station safely"),
    ("ham", "I bought the groceries you asked me to get"),
    ("ham", "The new cafe near the office has really nice coffee"),
    ("ham", "Please confirm the dates for the family get together"),
    ("ham", "Our flight got delayed by thirty minutes check schedule"),
    ("ham", "I will share the meeting notes with you after lunch"),
    ("ham", "The dog had a great time at the beach today"),
    ("ham", "Are we still on for the movie this weekend"),
    ("ham", "Mother asked us to visit the store on our way"),
    ("ham", "The assignment deadline was moved to next friday"),
    ("ham", "Hope you are feeling better after resting yesterday"),
    ("ham", "Looking forward to seeing you at the ceremony tonight"),
    ("ham", "The electricity bill is due by the end of the week"),
    ("ham", "Please bring the snacks for the class party tomorrow"),
    ("ham", "I finished reading the book you recommended good one"),
    ("ham", "We should fix a day to clean the garage together"),
    ("ham", "The internet was down so I could not reply earlier"),
]


def accuracy_report():
    model = NaiveBayes()
    model.fit(SPAM_TRAIN, HAM_TRAIN)

    correct = 0
    total = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for label, text in TEST_CASES:
        total = total + 1
        result = model.predict(text)
        if label == "spam" and result == "spam":
            correct = correct + 1
            tp = tp + 1
        elif label == "spam" and result == "ham":
            fn = fn + 1
        elif label == "ham" and result == "ham":
            correct = correct + 1
            tn = tn + 1
        else:
            fp = fp + 1

    accuracy = correct / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("Test accuracy: " + str(accuracy))
    print("TP: " + str(tp) + "  FP: " + str(fp) + "  TN: " + str(tn) + "  FN: " + str(fn))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1: " + str(f1))

    for label, text in TEST_CASES:
        result = model.predict(text)
        if result != label:
            print("MISMATCH -> expected: " + label + " got: " + result + " | " + text)


if __name__ == "__main__":
    accuracy_report()