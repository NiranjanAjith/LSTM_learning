import tensorflow as tf

class LSTMCell(tf.keras.layers.Layer):
    def __init__(self, units):
        super(LSTMCell, self).__init__()
        self.units = units
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Initialize weights
        self.W_f = self.add_weight(shape=(input_dim + self.units, self.units),
                                   initializer='glorot_uniform',
                                   name='W_f')
        self.W_i = self.add_weight(shape=(input_dim + self.units, self.units),
                                   initializer='glorot_uniform',
                                   name='W_i')
        self.W_c = self.add_weight(shape=(input_dim + self.units, self.units),
                                   initializer='glorot_uniform',
                                   name='W_c')
        self.W_o = self.add_weight(shape=(input_dim + self.units, self.units),
                                   initializer='glorot_uniform',
                                   name='W_o')
        
        # Initialize biases
        self.b_f = self.add_weight(shape=(self.units,),
                                   initializer='zeros',
                                   name='b_f')
        self.b_i = self.add_weight(shape=(self.units,),
                                   initializer='zeros',
                                   name='b_i')
        self.b_c = self.add_weight(shape=(self.units,),
                                   initializer='zeros',
                                   name='b_c')
        self.b_o = self.add_weight(shape=(self.units,),
                                   initializer='zeros',
                                   name='b_o')
        
    def call(self, inputs, states):
        h_prev, c_prev = states
        
        # Concatenate input and previous hidden state
        z = tf.concat([inputs, h_prev], axis=-1)
        
        # Forget gate
        f = tf.sigmoid(tf.matmul(z, self.W_f) + self.b_f)
        
        # Input gate
        i = tf.sigmoid(tf.matmul(z, self.W_i) + self.b_i)
        
        # Candidate memory cell
        c_tilde = tf.tanh(tf.matmul(z, self.W_c) + self.b_c)
        
        # Update memory cell
        c = f * c_prev + i * c_tilde
        
        # Output gate
        o = tf.sigmoid(tf.matmul(z, self.W_o) + self.b_o)
        
        # Update hidden state
        h = o * tf.tanh(c)
        
        return h, [h, c]

class LSTMLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(LSTMLayer, self).__init__()
        self.units = units
        self.cell = LSTMCell(units)
    
    def call(self, inputs):
        # Assume inputs shape is (batch_size, time_steps, features)
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        
        # Initialize hidden state and cell state
        h = tf.zeros((batch_size, self.units))
        c = tf.zeros((batch_size, self.units))
        
        # List to store outputs for each time step
        outputs = []
        
        # Process each time step
        for t in range(time_steps):
            h, [h, c] = self.cell(inputs[:, t, :], [h, c])
            outputs.append(h)
        
        # Stack outputs to create a tensor
        return tf.stack(outputs, axis=1)

# Example usage:
input_dim = 10  # Number of features in each input
time_steps = 5  # Number of time steps in the sequence
batch_size = 32  # Batch size
lstm_units = 64  # Number of LSTM units

# Create input tensor
inputs = tf.random.normal((batch_size, time_steps, input_dim))

# Create LSTM layer
lstm_layer = LSTMLayer(lstm_units)

# Apply LSTM layer to inputs
outputs = lstm_layer(inputs)

print(f"Input shape: {inputs.shape}")
print(f"Output shape: {outputs.shape}")