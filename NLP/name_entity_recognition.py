
# %%
# %%
import pandas as pd

import spacy


# %%
try:
    data = spacy.load("en_core_web_sm")
except OSError:
    print("Model 'en_core_web_sm' not found. Install it with:")
    print("    python -m spacy download en_core_web_sm")
    raise

txt = "In Kanada riding mountain biking is a popular sport. The weather is great for outdoor activities On the other hand, in the UK, people prefer cycling on roads and paths. The UK has a rich history of cycling, with many famous cyclists originating from there."
doc = data(txt)

for ent in doc.ents:
    print(ent.text, ent.label_)
    
entities = [(ent.text, ent.label_) for ent in doc.ents]
print("Entities found:", entities)
df = pd.DataFrame(entities, columns=['Entity', 'Label'])
print(df)

# %%
# morfology analysis
for token in doc:
    print(f"{token.text} - {token.pos_} - {token.dep_} - {token.lemma_} - {token.is_stop}") 
