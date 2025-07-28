# %%
"""
problem statement:Recommendation system 
This code implements a recommendation system using collaborative filtering and content-based filtering techniques.
"""
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
# %%
user_ids = np.array([1, 2, 3, 4, 5, 0, 1, 2, 3, 4])
item_ids = np.array([0, 1, 2, 3, 4,7, 8, 9, 10, 11])
ratings = np.array([5, 4, 3, 2, 1, 5, 4, 3, 2, 1])

train_test_split = train_test_split(np.arange(len(user_ids)), test_size=0.2, random_state=42)
# %%

def create_model(num_users, num_items, embedding_size=8):
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')

    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size, name='user_embedding')(user_input)
    item_embedding = Embedding(input_dim=num_items, output_dim=embedding_size, name='item_embedding')(item_input)

    user_vector = Flatten()(user_embedding)
    item_vector = Flatten()(item_embedding)

    dot_product = Dot(axes=1)([user_vector, item_vector])
    output = Dense(1, activation='linear')(dot_product)

    model = Model(inputs=[user_input, item_input], outputs=output)
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    
    return model

#train and test 

def train_and_evaluate_model(user_ids, item_ids, ratings, test_size=0.2):
    num_users = len(np.unique(user_ids))
    num_items = len(np.unique(item_ids))

    model = create_model(num_users, num_items)

    X_train, X_test, y_train, y_test = train_test_split(
        np.array(list(zip(user_ids, item_ids))), ratings, test_size=test_size, random_state=42
    )

    model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=10, verbose=1)

    loss = model.evaluate([X_test[:, 0], X_test[:, 1]], y_test)
    print(f'Test Loss: {loss}')

    return model

model =create_model(5, 12)

# show the model summary
model.summary() 



# %%
X_train, X_test, y_train, y_test = train_test_split(
        np.array(list(zip(user_ids, item_ids))), ratings, test_size=0.8, random_state=42
    )
loss = model.evaluate([X_test[:, 0], X_test[:, 1]], y_test)
print(f'Test Loss: {loss}')

# %%
example_user_id = 1
example_item_id = 2
predicted_rating = model.predict([np.array([example_user_id]), np.array([example_item_id])])
print(f'Predicted rating for user {example_user_id} on item {example_item_id}: {predicted_rating[0][0]}')
#%%
! pip install numpy==1.24.4 scikit-surprise==1.1.3
# %%
import pandas as pd
import numpy
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, KNNBasic , accuracy, Reader
from sklearn.model_selection import train_test_split

# Load the MovieLens dataset
data = Dataset.load_builtin('ml-100k')
df = pd.DataFrame(data.raw_ratings, columns=['user_id', 'item_id', 'rating', 'timestamp'])
df['user_id'] = df['user_id'].astype(int)
df['item_id'] = df['item_id'].astype(int)   

# train-test split
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

model_options = {
    'name': 'cosine',
    'user_based': True  # True for user-based, False for item-based
}

model = KNNBasic(sim_options=model_options)
model.fit(train_data)

predictions = model.test(test_data)
accuracy.rmse(predictions)


# %%
def get_top_n_recommendations(user_id, n=5):
    user_items = df[df['user_id'] == user_id]['item_id'].tolist()
    all_items = df['item_id'].unique().tolist()
    items_to_predict = [item for item in all_items if item not in user_items]

    if not items_to_predict:
        return []

    predictions = []
    for item in items_to_predict:
        pred_rating = model.predict(user_id, item).est
        predictions.append((item, pred_rating))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]

# Example usage
user_id = 1
top_n_recommendations = get_top_n_recommendations(user_id, n=5)
print(f"Top {len(top_n_recommendations)} recommendations for user {user_id}:")
for item, rating in top_n_recommendations:
    print(f"Item ID: {item}, Predicted Rating: {rating:.2f}")   



# %%
