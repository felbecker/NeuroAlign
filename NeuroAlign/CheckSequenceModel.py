import SequenceModel

model = SequenceModel.make_model()
model.load_weights(SequenceModel.CHECKPOINT_PATH)

model.summary()

print(model.layers)

alphabet_embedding = model.layers[0].get_weights()

print(alphabet_embedding)
