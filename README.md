# Forecasting using Machine Learning

This project implements time series forecasting using:
- CNN (Convolutional Neural Networks)
- RNN (Recurrent Neural Networks) 
- Window Functions

## Models

### Window Functions
A window function in time series analysis is a sliding frame that processes sequential data points in fixed-size chunks. Here's why it's important:
#### Key Benefits:
- Pattern Recognition: Enables detection of temporal patterns and trends
- Feature Engineering: Creates structured input features for ML models
- Prediction: Helps forecast future values based on past observations
- Memory Handling: Efficiently processes large sequential datasets

```python
def create_windowed_dataset(data, window_size=5, shift=1):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    # Create windows of size 5, shifting by 1 step
    dataset = dataset.window(window_size, shift=shift, drop_remainder=True)
    # Convert windows to tensors of size 5
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    # Split into features (first 4 values) and label (last value)
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    return dataset
```
Parameters

- window_size: Size of sliding window (default: 5)
- shift: Number of steps to shift window (default: 1)
- drop_remainder: Drop incomplete windows

##### Example Output
-Input: [0,1,2,3,4,5,6,7,8,9]
- Output:
Features: [0,1,2,3] Label: [4]
Features: [1,2,3,4] Label: [5]
Features: [2,3,4,5] Label: [6]



