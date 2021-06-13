try:
    from neuralprophet import NeuralProphet
    using_neuralprophet = True
except ImportError:
    class NeuralMock:
        whatever=False
    using_neuralprophet = False
    NeuralProphet = NeuralMock()