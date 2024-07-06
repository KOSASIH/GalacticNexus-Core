require 'tensorflow'

class MLService
  def initialize
    @model = TensorFlow::Keras::Sequential.new([...])
  end

  def train_model(data)
    x_train, x_test, y_train, y_test = data.split(test_size: 0.2)
    @model.fit(x_train, y_train, epochs: 10)
  end
end
