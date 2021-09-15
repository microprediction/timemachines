# SPDX-License-Identifier: Apache-2.0

from timemachines.skaters.tcn.tcninclusiontraining import using_tcntraining

if using_tcntraining:
    from onnxruntime import InferenceSession
    import numpy as np
    from tensorflow import keras
    from tensorflow.keras import layers, Input

    def test_keras_onnx_runtime():
        """
        :return: test if onnx and keras seem to be working
        """
        # adapted from https://github.com/microprediction/tensorflow-onnx/blob/master/examples/end2end_tfkeras.py

        # Creates the model.
        model = keras.Sequential()
        model.add(Input((4, 4)))
        model.add(layers.SimpleRNN(8))
        model.add(layers.Dense(2))
        print(model.summary())
        input_names = [n.name for n in model.inputs]
        output_names = [n.name for n in model.outputs]
        print('inputs:', input_names)
        print('outputs:', output_names)

        ########################################
        # Training
        # ....
        # Skipped.

        ########################################
        # Testing the model.
        input = np.random.randn(2, 4, 4).astype(np.float32)
        expected = model.predict(input)
        print(expected)

        ########################################
        # Serialize but do not save the model
        from tf2onnx.keras2onnx_api import convert_keras
        onnx_model = convert_keras(model=model,name='example')
        onnx_model_as_byte_string = onnx_model.SerializeToString()

        ########################################
        # Runs onnxruntime.
        session = InferenceSession(onnx_model_as_byte_string)
        got = session.run(None, {'input_1': input})
        print(got[0])

        ########################################
        # Measures the differences.
        assert (np.abs(got[0] - expected).max())<1e-5
