import SequenceModel
import tensorflow as tf

# model = tf.keras.models.load_model(SequenceModel.CHECKPOINT_PATH)

model = SequenceModel.make_model()
model.load_weights(SequenceModel.CHECKPOINT_PATH+"/variables/variables")

print("Loaded model", flush=True)

#model.summary()

alphabet_embedding = model.layers[0].get_weights()

print(alphabet_embedding)
