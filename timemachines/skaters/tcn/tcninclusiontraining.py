
try:
    import tcn
    import keras
    import microprediction
    import microprediction
    from tf2onnx.keras2onnx_api import convert_keras
    using_tcntraining = True
except ImportError:
    using_tcntraining = False


if __name__=='__main__':
    print(using_tcntraining)