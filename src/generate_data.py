import pandas as pd

# Define the dataset
arab_terms = [
    ("arab", "names", "أحمد"),
    ("arab", "names", "فاطمة"),
    ("arab", "food", "المنسف"),
    ("arab", "food", "الحمص"),
    ("arab", "symbols", "الكعبة"),
    ("arab", "symbols", "الهلال"),
    ("arab", "clothing", "عباءة"),
    ("arab", "clothing", "غترة"),
    ("arab", "literature", "المتنبي"),
    ("arab", "literature", "ألف ليلة وليلة")
]

western_terms = [
    ("western", "names", "جون"),
    ("western", "names", "ماري"),
    ("western", "food", "البيتزا"),
    ("western", "food", "البرغر"),
    ("western", "symbols", "الصليب"),
    ("western", "symbols", "تمثال الحرية"),
    ("western", "clothing", "البدلة"),
    ("western", "clothing", "القميص"),
    ("western", "literature", "شكسبير"),
    ("western", "literature", "هاري بوتر")
]

sentiment_words = [
    ("positive", "جميل"),
    ("positive", "رائع"),
    ("positive", "لذيذ"),
    ("positive", "ملهم"),
    ("positive", "ناجح"),
    ("negative", "قبيح"),
    ("negative", "سيء"),
    ("negative", "ممل"),
    ("negative", "فاشل"),
    ("negative", "عديم الفائدة")
]

context_sentences = [
    ("arab", "names", "[اسم] يحب القهوة."),
    ("arab", "food", "[طعام] طعام تقليدي."),
    ("arab", "symbols", "يوجد [رمز] على العلم."),
    ("arab", "clothing", "[ملابس] لباس شهير."),
    ("arab", "literature", "[أدب] مشهور في العالم."),
    ("western", "names", "[اسم] يحب القهوة."),
    ("western", "food", "[طعام] طعام تقليدي."),
    ("western", "symbols", "يوجد [رمز] على العلم."),
    ("western", "clothing", "[ملابس] لباس شهير."),
    ("western", "literature", "[أدب] مشهور في العالم.")
]

# Combine Arab and Western terms into a single dataset
data_culture_terms = arab_terms + western_terms

# Save combined culture terms to CSV
df_culture_terms = pd.DataFrame(data_culture_terms, columns=["Culture", "Entity", "Term"])
df_culture_terms.to_csv("data/culture_terms.csv", index=False, encoding="utf-8-sig")

# Save Sentiment words to CSV
df_sentiment_words = pd.DataFrame(sentiment_words, columns=["Sentiment", "Term"])
df_sentiment_words.to_csv("data/sentiment_words.csv", index=False, encoding="utf-8-sig")

# Save Context sentences to CSV
context_data = [
    (culture, entity, sentence.replace("[اسم]", term) if entity == "names" else
     sentence.replace("[طعام]", term) if entity == "food" else
     sentence.replace("[رمز]", term) if entity == "symbols" else
     sentence.replace("[ملابس]", term) if entity == "clothing" else
     sentence.replace("[أدب]", term))
    for culture, entity, sentence in context_sentences
    for _, cat, term in (arab_terms if culture == "arab" else western_terms)
    if entity == cat
]

df_context_sentences = pd.DataFrame(context_data, columns=["Culture", "Entity", "Sentence"])
df_context_sentences.to_csv("data/context_sentences.csv", index=False, encoding="utf-8-sig")

print("Datasets saved as separate CSV files in 'data/' directory: culture_terms.csv, sentiment_words.csv, context_sentences.csv")
