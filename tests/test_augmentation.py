from timemachines.skatertools.evaluation.evaluators import chunk_to_end


def test_chunk_from_end():
    ys = [1,2,3,4,5,6,7,8]
    chunks = chunk_to_end(ys,5)
    assert len(chunks[0])==5
    assert len(chunks)==1
    assert chunks[0][0]==4