config = {
    'pre_trained_model_name': 'galactic_nexus_pre_trained_model',
    'num_classes': 10,
    'epochs': 10,
    'batch_size': 32,
    'optimizer': Adam(lr=0.001),
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy']
}
